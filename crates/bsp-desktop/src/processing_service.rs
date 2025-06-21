//! Processing service for real-time signal processing integration

use bsp_core::{SignalEntity, BspResult};
use bsp_processing::{Pipeline, PipelineBuilder, ProcessingResult, FeatureSet};
use tokio::sync::{broadcast, mpsc};
use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};

/// Commands for controlling processing
#[derive(Debug, Clone)]
pub enum ProcessingCommand {
    Start,
    Stop,
    Pause,
    Resume,
    UpdatePipeline(ProcessingPipelineConfig),
    SetBypassMode(bool),
    ResetPipeline,
}

/// Processing pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingPipelineConfig {
    /// Pipeline type selection
    pub pipeline_type: PipelineType,
    /// Enable/disable specific processors
    pub enable_filters: bool,
    pub enable_features: bool,
    /// Filter parameters
    pub highpass_cutoff: f32,
    pub lowpass_cutoff: f32,
    pub notch_frequency: f32,
    /// Feature extraction parameters
    pub feature_window_size: usize,
    pub feature_overlap: f32,
}

/// Available pipeline types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineType {
    /// Minimal processing for real-time display
    RealTime,
    /// Comprehensive processing for analysis
    Research,
    /// Balanced processing
    Standard,
    /// No processing (passthrough)
    Bypass,
}

impl Default for ProcessingPipelineConfig {
    fn default() -> Self {
        Self {
            pipeline_type: PipelineType::Standard,
            enable_filters: true,
            enable_features: false, // Disabled by default for performance
            highpass_cutoff: 20.0,
            lowpass_cutoff: 450.0,
            notch_frequency: 50.0,
            feature_window_size: 256,
            feature_overlap: 0.5,
        }
    }
}

/// Processing results for UI consumption
#[derive(Debug, Clone)]
pub struct ProcessedSignalData {
    /// Original signal
    pub raw_signal: SignalEntity,
    /// Processed signal
    pub processed_signal: SignalEntity,
    /// Extracted features (if enabled)
    pub features: Option<Vec<FeatureSet>>,
    /// Processing metrics
    pub processing_time_us: u64,
    /// Processing success
    pub success: bool,
    /// Any warnings
    pub warnings: Vec<String>,
}

/// Statistics about processing performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    pub is_running: bool,
    pub signals_processed: u64,
    pub total_processing_time_us: u64,
    pub average_latency_us: u64,
    pub success_rate: f32,
    pub last_update: u64,
    pub pipeline_type: PipelineType,
}

/// Real-time signal processing service
pub struct ProcessingService {
    config: ProcessingPipelineConfig,
    pipeline: Arc<Mutex<Option<Pipeline>>>,

    // Communication channels
    input_receiver: broadcast::Receiver<SignalEntity>,
    output_sender: broadcast::Sender<ProcessedSignalData>,
    command_receiver: mpsc::Receiver<ProcessingCommand>,
    command_sender: mpsc::Sender<ProcessingCommand>,

    // State management
    is_running: Arc<Mutex<bool>>,
    stats: Arc<Mutex<ProcessingStats>>,

    // Performance tracking
    signals_processed: u64,
    total_processing_time: u64,
    successful_processing: u64,
}

impl ProcessingService {
    /// Create new processing service
    pub fn new(
        input_receiver: broadcast::Receiver<SignalEntity>,
        config: ProcessingPipelineConfig,
    ) -> BspResult<Self> {
        let (output_sender, _) = broadcast::channel(50);
        let (command_sender, command_receiver) = mpsc::channel(32);

        let stats = ProcessingStats {
            is_running: false,
            signals_processed: 0,
            total_processing_time_us: 0,
            average_latency_us: 0,
            success_rate: 1.0,
            last_update: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            pipeline_type: config.pipeline_type,
        };

        Ok(ProcessingService {
            config: config.clone(),
            pipeline: Arc::new(Mutex::new(None)),
            input_receiver,
            output_sender,
            command_receiver,
            command_sender,
            is_running: Arc::new(Mutex::new(false)),
            stats: Arc::new(Mutex::new(stats)),
            signals_processed: 0,
            total_processing_time: 0,
            successful_processing: 0,
        })
    }

    /// Get output receiver for processed signals
    pub fn subscribe_output(&self) -> broadcast::Receiver<ProcessedSignalData> {
        self.output_sender.subscribe()
    }

    /// Get command sender for controlling processing
    pub fn command_handle(&self) -> mpsc::Sender<ProcessingCommand> {
        self.command_sender.clone()
    }

    /// Create pipeline based on configuration
    fn create_pipeline(&self, config: &ProcessingPipelineConfig) -> BspResult<Pipeline> {
        match config.pipeline_type {
            PipelineType::Bypass => {
                // Minimal pipeline that just passes through
                let mut builder = PipelineBuilder::new("Bypass");
                builder.set_bypass_enabled(true);
                Ok(builder.build())
            },

            PipelineType::RealTime => {
                let mut builder = PipelineBuilder::new("Real-time");
                builder.set_bypass_enabled(true); // Allow bypass for real-time

                if config.enable_filters {
                    // Minimal filtering for low latency
                    builder.add_highpass(config.highpass_cutoff, 2)?; // Lower order
                    builder.add_notch(config.notch_frequency, 10.0); // Lower Q
                }

                Ok(builder.build())
            },

            PipelineType::Standard => {
                let mut builder = PipelineBuilder::new("Standard");

                if config.enable_filters {
                    builder.add_highpass(config.highpass_cutoff, 4)?;
                    builder.add_lowpass(config.lowpass_cutoff, 4)?;
                    builder.add_notch(config.notch_frequency, 30.0);
                }

                if config.enable_features {
                    builder.add_emg_features(config.feature_window_size);
                }

                Ok(builder.build())
            },

            PipelineType::Research => {
                let mut builder = PipelineBuilder::new("Research");

                if config.enable_filters {
                    // High-quality filters
                    builder.add_highpass(10.0, 6)?; // Lower cutoff, higher order
                    builder.add_lowpass(config.lowpass_cutoff, 6)?;
                    builder.add_notch(50.0, 50.0); // High Q
                    builder.add_notch(60.0, 50.0); // Also filter 60Hz
                }

                if config.enable_features {
                    builder.add_emg_features(512); // Larger window for better frequency resolution
                }

                Ok(builder.build())
            },
        }
    }

    /// Main processing loop
    pub async fn run(&mut self) -> BspResult<()> {
        println!("Processing Service started - Pipeline: {:?}", self.config.pipeline_type);

        // Initialize pipeline
        let pipeline = self.create_pipeline(&self.config)?;
        {
            let mut pipeline_lock = self.pipeline.lock().await;
            *pipeline_lock = Some(pipeline);
        }

        loop {
            tokio::select! {
                // Handle incoming signals
                signal_result = self.input_receiver.recv() => {
                    match signal_result {
                        Ok(signal) => {
                            let is_running = *self.is_running.lock().await;
                            if is_running {
                                self.process_signal(signal).await;
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(skipped)) => {
                            println!("Warning: Processing lagged, skipped {} signals", skipped);
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            println!("Input channel closed, stopping processing service");
                            break;
                        }
                    }
                }
                
                // Handle control commands
                command = self.command_receiver.recv() => {
                    match command {
                        Some(ProcessingCommand::Start) => {
                            *self.is_running.lock().await = true;
                            self.update_stats(|stats| stats.is_running = true).await;
                            println!("Processing started");
                        }
                        Some(ProcessingCommand::Stop) => {
                            *self.is_running.lock().await = false;
                            self.update_stats(|stats| {
                                stats.is_running = false;
                                stats.signals_processed = 0;
                                stats.total_processing_time_us = 0;
                            }).await;
                            self.reset_metrics();
                            println!("Processing stopped");
                        }
                        Some(ProcessingCommand::Pause) => {
                            *self.is_running.lock().await = false;
                            self.update_stats(|stats| stats.is_running = false).await;
                            println!("Processing paused");
                        }
                        Some(ProcessingCommand::Resume) => {
                            *self.is_running.lock().await = true;
                            self.update_stats(|stats| stats.is_running = true).await;
                            println!("Processing resumed");
                        }
                        Some(ProcessingCommand::UpdatePipeline(new_config)) => {
                            if let Err(e) = self.update_pipeline(new_config).await {
                                println!("Failed to update pipeline: {}", e);
                            }
                        }
                        Some(ProcessingCommand::SetBypassMode(enabled)) => {
                            if let Some(pipeline) = self.pipeline.lock().await.as_mut() {
                                pipeline.set_bypass_enabled(enabled);
                                println!("Bypass mode: {}", if enabled { "enabled" } else { "disabled" });
                            }
                        }
                        Some(ProcessingCommand::ResetPipeline) => {
                            if let Some(pipeline) = self.pipeline.lock().await.as_mut() {
                                pipeline.reset();
                                println!("Pipeline reset");
                            }
                        }
                        None => {
                            println!("Command channel closed");
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Process a single signal through the pipeline
    async fn process_signal(&mut self, signal: SignalEntity) {
        let start_time = std::time::Instant::now();

        let result = {
            let mut pipeline_lock = self.pipeline.lock().await;
            if let Some(ref mut pipeline) = *pipeline_lock {
                pipeline.process(&signal)
            } else {
                // No pipeline, just pass through
                Ok(bsp_processing::PipelineResult {
                    output_signal: signal.clone(),
                    processing_metrics: Vec::new(),
                    total_latency_us: 0,
                    success: true,
                    warnings: vec!["No pipeline configured - passthrough mode".to_string()],
                    features: None,
                })
            }
        };

        let processing_time = start_time.elapsed().as_micros() as u64;

        match result {
            Ok(pipeline_result) => {
                let processed_data = ProcessedSignalData {
                    raw_signal: signal,
                    processed_signal: pipeline_result.output_signal,
                    features: pipeline_result.features,
                    processing_time_us: processing_time,
                    success: true,
                    warnings: pipeline_result.warnings,
                };

                // Send to subscribers (ignore if no receivers)
                let _ = self.output_sender.send(processed_data);

                // Update metrics
                self.update_metrics(processing_time, true).await;
            }
            Err(e) => {
                println!("Processing error: {}", e);

                // Send raw signal as fallback
                let processed_data = ProcessedSignalData {
                    raw_signal: signal.clone(),
                    processed_signal: signal,
                    features: None,
                    processing_time_us: processing_time,
                    success: false,
                    warnings: vec![format!("Processing failed: {}", e)],
                };

                let _ = self.output_sender.send(processed_data);
                self.update_metrics(processing_time, false).await;
            }
        }
    }

    /// Update pipeline configuration
    async fn update_pipeline(&mut self, new_config: ProcessingPipelineConfig) -> BspResult<()> {
        println!("Updating pipeline configuration: {:?}", new_config.pipeline_type);

        let new_pipeline = self.create_pipeline(&new_config)?;

        {
            let mut pipeline_lock = self.pipeline.lock().await;
            *pipeline_lock = Some(new_pipeline);
        }

        self.config = new_config.clone();
        self.update_stats(|stats| stats.pipeline_type = new_config.pipeline_type).await;

        Ok(())
    }

    /// Update processing metrics
    async fn update_metrics(&mut self, processing_time_us: u64, success: bool) {
        self.signals_processed += 1;
        self.total_processing_time += processing_time_us;

        if success {
            self.successful_processing += 1;
        }

        // Update stats periodically
        if self.signals_processed % 10 == 0 {
            self.update_stats(|stats| {
                stats.signals_processed = self.signals_processed;
                stats.total_processing_time_us = self.total_processing_time;
                stats.average_latency_us = if self.signals_processed > 0 {
                    self.total_processing_time / self.signals_processed
                } else {
                    0
                };
                stats.success_rate = if self.signals_processed > 0 {
                    self.successful_processing as f32 / self.signals_processed as f32
                } else {
                    1.0
                };
                stats.last_update = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64;
            }).await;
        }
    }

    /// Reset metrics
    fn reset_metrics(&mut self) {
        self.signals_processed = 0;
        self.total_processing_time = 0;
        self.successful_processing = 0;
    }

    /// Update stats with a closure
    async fn update_stats<F>(&self, update_fn: F)
    where
        F: FnOnce(&mut ProcessingStats),
    {
        let mut stats = self.stats.lock().await;
        update_fn(&mut *stats);
    }

    /// Get current processing statistics
    pub async fn get_stats(&self) -> ProcessingStats {
        self.stats.lock().await.clone()
    }

    /// Check if processing is running
    pub async fn is_running(&self) -> bool {
        *self.is_running.lock().await
    }

    /// Get current configuration
    pub fn config(&self) -> &ProcessingPipelineConfig {
        &self.config
    }
}

/// Helper function to start processing service in background
pub async fn start_processing_service(
    input_receiver: broadcast::Receiver<SignalEntity>,
    config: ProcessingPipelineConfig,
) -> BspResult<(
    broadcast::Receiver<ProcessedSignalData>,
    mpsc::Sender<ProcessingCommand>,
    Arc<Mutex<ProcessingStats>>,
)> {
    let mut service = ProcessingService::new(input_receiver, config)?;

    let output_receiver = service.subscribe_output();
    let command_sender = service.command_handle();
    let stats_handle = service.stats.clone();

    // Start service in background task
    tokio::spawn(async move {
        if let Err(e) = service.run().await {
            eprintln!("Processing service error: {}", e);
        }
    });

    Ok((output_receiver, command_sender, stats_handle))
}