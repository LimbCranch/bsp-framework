//! Real-time EMG signal streaming for live visualization

use bsp_core::{SignalEntity, BspResult};
use crate::emg_simulator::{EMGSimulator, EMGConfig};
use tokio::time::{interval, Duration, Instant};
use tokio::sync::{broadcast, mpsc};
use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};

/// Configuration for real-time streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// EMG simulation configuration
    pub emg_config: EMGConfig,
    /// Chunk duration in seconds (e.g., 0.1 for 100ms chunks)
    pub chunk_duration: f32,
    /// Buffer size for the stream (number of chunks to keep)
    pub buffer_size: usize,
    /// Update rate in Hz (how often to send new data)
    pub update_rate: f32,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            emg_config: EMGConfig::default(),
            chunk_duration: 0.1, // 100ms chunks
            buffer_size: 50,     // 5 seconds of history at 100ms chunks
            update_rate: 10.0,   // 10 Hz updates
        }
    }
}

/// Real-time EMG signal stream
pub struct RealTimeEMGStream {
    config: StreamConfig,
    simulator: Arc<Mutex<EMGSimulator>>,
    data_sender: broadcast::Sender<SignalEntity>,
    control_receiver: mpsc::Receiver<StreamCommand>,
    control_sender: mpsc::Sender<StreamCommand>,
    is_running: Arc<Mutex<bool>>,
    current_chunk: Arc<Mutex<Option<SignalEntity>>>,
}

/// Commands for controlling the stream
#[derive(Debug, Clone)]
pub enum StreamCommand {
    Start,
    Stop,
    Pause,
    Resume,
    UpdateConfig(StreamConfig),
    UpdatePattern(crate::signal_patterns::SignalPattern),
    SetActivationLevel(f32),
}

/// Stream statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamStats {
    pub is_running: bool,
    pub chunks_generated: u64,
    pub total_duration: f32,
    pub current_activation: f32,
    pub average_chunk_time: f32,
    pub last_update: u64,
}

impl RealTimeEMGStream {
    /// Create new real-time EMG stream
    pub fn new(config: StreamConfig) -> BspResult<Self> {
        let simulator = EMGSimulator::new(config.emg_config.clone())?;
        let (data_sender, _) = broadcast::channel(config.buffer_size);
        let (control_sender, control_receiver) = mpsc::channel(32);

        Ok(RealTimeEMGStream {
            config,
            simulator: Arc::new(Mutex::new(simulator)),
            data_sender,
            control_receiver,
            control_sender,
            is_running: Arc::new(Mutex::new(false)),
            current_chunk: Arc::new(Mutex::new(None)),
        })
    }

    /// Get a receiver for data updates
    pub fn subscribe(&self) -> broadcast::Receiver<SignalEntity> {
        self.data_sender.subscribe()
    }

    /// Get control sender for sending commands
    pub fn control_handle(&self) -> mpsc::Sender<StreamCommand> {
        self.control_sender.clone()
    }

    /// Start the streaming task
    pub async fn run(&mut self) -> BspResult<()> {
        let update_interval = Duration::from_secs_f32(1.0 / self.config.update_rate);
        let mut interval_timer = interval(update_interval);

        let mut stats = StreamStats {
            is_running: false,
            chunks_generated: 0,
            total_duration: 0.0,
            current_activation: 0.0,
            average_chunk_time: 0.0,
            last_update: 0,
        };

        println!("EMG Stream started - Update rate: {:.1}Hz, Chunk duration: {:.0}ms",
                 self.config.update_rate,
                 self.config.chunk_duration * 1000.0);

        loop {
            tokio::select! {
                // Handle timer ticks for data generation
                _ = interval_timer.tick() => {
                    let is_running = *self.is_running.lock().await;
                    if is_running {
                        let start_time = Instant::now();
                        
                        // Generate new chunk
                        let chunk = {
                            let mut sim = self.simulator.lock().await;
                            sim.generate_chunk(self.config.chunk_duration)?
                        };

                        let generation_time = start_time.elapsed();
                        
                        // Update statistics
                        stats.chunks_generated += 1;
                        stats.total_duration += self.config.chunk_duration;
                        stats.average_chunk_time = generation_time.as_secs_f32();
                        stats.last_update = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64;

                        // Update current activation level from chunk metadata
                        if let bsp_core::EMGSignal::Surface { activation_level, .. } = chunk.metadata.signal_type {
                            stats.current_activation = activation_level;
                        }

                        // Store current chunk
                        {
                            let mut current = self.current_chunk.lock().await;
                            *current = Some(chunk.clone());
                        }

                        // Send to subscribers (ignore if no receivers)
                        let _ = self.data_sender.send(chunk);
                        
                        // Log performance warning if generation is slow
                        if generation_time.as_millis() > (self.config.chunk_duration * 1000.0) as u128 {
                            println!("Warning: Chunk generation took {:.1}ms, longer than chunk duration {:.0}ms",
                                     generation_time.as_millis(),
                                     self.config.chunk_duration * 1000.0);
                        }
                    }
                }
                
                // Handle control commands
                command = self.control_receiver.recv() => {
                    match command {
                        Some(StreamCommand::Start) => {
                            *self.is_running.lock().await = true;
                            stats.is_running = true;
                            println!("EMG Stream started");
                        }
                        Some(StreamCommand::Stop) => {
                            *self.is_running.lock().await = false;
                            stats.is_running = false;
                            stats.chunks_generated = 0;
                            stats.total_duration = 0.0;
                            
                            // Reset simulator time
                            {
                                let mut sim = self.simulator.lock().await;
                                sim.reset_time();
                            }
                            println!("EMG Stream stopped");
                        }
                        Some(StreamCommand::Pause) => {
                            *self.is_running.lock().await = false;
                            stats.is_running = false;
                            println!("EMG Stream paused");
                        }
                        Some(StreamCommand::Resume) => {
                            *self.is_running.lock().await = true;
                            stats.is_running = true;
                            println!("EMG Stream resumed");
                        }
                        Some(StreamCommand::UpdateConfig(new_config)) => {
                            self.config = new_config.clone();
                            
                            // Update simulator
                            {
                                let mut sim = self.simulator.lock().await;
                                sim.update_config(new_config.emg_config)?;
                            }
                            
                            // Update interval timer
                            let new_interval = Duration::from_secs_f32(1.0 / self.config.update_rate);
                            interval_timer = interval(new_interval);
                            
                            println!("EMG Stream configuration updated");
                        }
                        Some(StreamCommand::UpdatePattern(pattern)) => {
                            let mut config = self.config.clone();
                            config.emg_config.pattern = crate::emg_simulator::PatternConfig::from_pattern(pattern);
                            
                            {
                                let mut sim = self.simulator.lock().await;
                                sim.update_config(config.emg_config.clone())?;
                            }
                            
                            self.config = config;
                            println!("EMG Stream pattern updated: {}", pattern.description());
                        }
                        Some(StreamCommand::SetActivationLevel(level)) => {
                            let pattern = crate::signal_patterns::SignalPattern::Constant { level };
                            let mut config = self.config.clone();
                            config.emg_config.pattern = crate::emg_simulator::PatternConfig::from_pattern(pattern);
                            
                            {
                                let mut sim = self.simulator.lock().await;
                                sim.update_config(config.emg_config.clone())?;
                            }
                            
                            self.config = config;
                            println!("EMG Stream activation level set to {:.1}%", level * 100.0);
                        }
                        None => {
                            println!("EMG Stream control channel closed");
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get current stream statistics
    pub async fn stats(&self) -> StreamStats {
        StreamStats {
            is_running: *self.is_running.lock().await,
            chunks_generated: 0, // Would need to track this properly
            total_duration: 0.0,
            current_activation: 0.0,
            average_chunk_time: 0.0,
            last_update: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    /// Get the most recent chunk (for immediate access)
    pub async fn current_chunk(&self) -> Option<SignalEntity> {
        self.current_chunk.lock().await.clone()
    }

    /// Check if stream is running
    pub async fn is_running(&self) -> bool {
        *self.is_running.lock().await
    }

    /// Get current configuration
    pub fn config(&self) -> &StreamConfig {
        &self.config
    }
}

/// Helper function to create and start a stream in the background
pub async fn start_emg_stream(config: StreamConfig) -> BspResult<(
    broadcast::Receiver<SignalEntity>,
    mpsc::Sender<StreamCommand>,
)> {
    let mut stream = RealTimeEMGStream::new(config)?;
    let data_receiver = stream.subscribe();
    let control_sender = stream.control_handle();

    // Start the stream in a background task
    tokio::spawn(async move {
        if let Err(e) = stream.run().await {
            eprintln!("EMG Stream error: {}", e);
        }
    });

    Ok((data_receiver, control_sender))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_real_time_stream_basic() {
        let config = StreamConfig {
            chunk_duration: 0.05, // 50ms chunks for faster testing
            update_rate: 20.0,    // 20Hz updates
            ..Default::default()
        };

        let (mut data_receiver, control_sender) = start_emg_stream(config).await.unwrap();

        // Start the stream
        control_sender.send(StreamCommand::Start).await.unwrap();

        // Wait a bit and collect some chunks
        sleep(Duration::from_millis(200)).await;

        let mut chunk_count = 0;
        while let Ok(chunk) = data_receiver.try_recv() {
            chunk_count += 1;
            assert_eq!(chunk.duration(), 0.05);
            assert_eq!(chunk.sampling_rate(), 1000.0);

            if chunk_count >= 3 {
                break;
            }
        }

        assert!(chunk_count >= 3, "Should have received at least 3 chunks");

        // Stop the stream
        control_sender.send(StreamCommand::Stop).await.unwrap();
    }

    #[tokio::test]
    async fn test_stream_control_commands() {
        let config = StreamConfig::default();
        let (mut data_receiver, control_sender) = start_emg_stream(config).await.unwrap();

        // Test start/pause/resume cycle
        control_sender.send(StreamCommand::Start).await.unwrap();
        sleep(Duration::from_millis(100)).await;

        control_sender.send(StreamCommand::Pause).await.unwrap();
        sleep(Duration::from_millis(100)).await;

        control_sender.send(StreamCommand::Resume).await.unwrap();
        sleep(Duration::from_millis(100)).await;

        // Test activation level change
        control_sender.send(StreamCommand::SetActivationLevel(0.8)).await.unwrap();
        sleep(Duration::from_millis(100)).await;

        // Should receive some data
        let chunk = data_receiver.recv().await.unwrap();
        assert!(chunk.len() > 0);

        control_sender.send(StreamCommand::Stop).await.unwrap();
    }
}