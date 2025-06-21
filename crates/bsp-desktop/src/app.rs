//! Main application state and logic

use bsp_core::{SignalEntity, MuscleGroup};
use bsp_simulation::{
    RealTimeEMGStream, StreamConfig, StreamCommand, EMGConfig, SignalPattern,
    start_emg_stream
};
use tokio::sync::{broadcast, mpsc};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use crate::ui::{UIState, PlotData, ControlPanel};

/// Main application state
pub struct BSPApp {
    // Streaming components
    data_receiver: Option<broadcast::Receiver<SignalEntity>>,
    control_sender: Option<mpsc::Sender<StreamCommand>>,

    // Tokio runtime for async operations
    runtime: tokio::runtime::Runtime,

    // UI state - make public for access from UI module
    pub ui_state: UIState,

    // Data storage for visualization
    signal_buffer: VecDeque<SignalEntity>,
    pub plot_data: PlotData, // Make public for UI access

    // Application state
    is_running: bool,
    last_update: Instant,
    frame_count: u64,

    // Configuration
    stream_config: StreamConfig,

    // Initialization state
    initialized: bool,
}

impl BSPApp {
    /// Create new application instance
    pub fn new() -> anyhow::Result<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| anyhow::anyhow!("Failed to create tokio runtime: {}", e))?;

        let ui_state = UIState::new();
        let plot_data = PlotData::new();
        let stream_config = StreamConfig::default();

        Ok(BSPApp {
            data_receiver: None,
            control_sender: None,
            runtime,
            ui_state,
            signal_buffer: VecDeque::with_capacity(100),
            plot_data,
            is_running: false,
            last_update: Instant::now(),
            frame_count: 0,
            stream_config,
            initialized: false,
        })
    }

    /// Initialize the streaming components (called on first frame)
    fn initialize_stream(&mut self) {
        if self.initialized {
            return;
        }

        println!("Initializing EMG stream...");

        let stream_config = self.stream_config.clone();
        match self.runtime.block_on(start_emg_stream(stream_config)) {
            Ok((data_receiver, control_sender)) => {
                self.data_receiver = Some(data_receiver);
                self.control_sender = Some(control_sender);
                self.initialized = true;
                println!("EMG stream initialized successfully");
            }
            Err(e) => {
                eprintln!("Failed to initialize EMG stream: {}", e);
            }
        }
    }

    /// Update application state (called every frame)
    fn update_data(&mut self) {
        // Initialize stream on first call
        if !self.initialized {
            self.initialize_stream();
            return;
        }

        // Try to receive new signal data (non-blocking)
        if let Some(ref mut receiver) = self.data_receiver {
            while let Ok(signal) = receiver.try_recv() {
                // Add to buffer
                self.signal_buffer.push_back(signal.clone());

                // Keep buffer size manageable
                if self.signal_buffer.len() > 100 {
                    self.signal_buffer.pop_front();
                }

                // Update plot data
                self.plot_data.add_signal(signal);
            }
        }

        self.frame_count += 1;
    }

    /// Send control command to stream
    fn send_command(&self, command: StreamCommand) {
        if let Some(ref sender) = self.control_sender {
            if let Err(e) = sender.try_send(command) {
                eprintln!("Failed to send stream command: {}", e);
            }
        } else {
            eprintln!("Stream not initialized, cannot send command");
        }
    }

    /// Start the EMG stream
    pub fn start_stream(&mut self) {
        self.send_command(StreamCommand::Start);
        self.is_running = true;
    }

    /// Stop the EMG stream
    pub fn stop_stream(&mut self) {
        self.send_command(StreamCommand::Stop);
        self.is_running = false;
        self.signal_buffer.clear();
        self.plot_data.clear();
    }

    /// Pause the EMG stream
    pub fn pause_stream(&mut self) {
        self.send_command(StreamCommand::Pause);
        self.is_running = false;
    }

    /// Resume the EMG stream
    pub fn resume_stream(&mut self) {
        self.send_command(StreamCommand::Resume);
        self.is_running = true;
    }

    /// Update activation level
    pub fn set_activation_level(&mut self, level: f32) {
        self.send_command(StreamCommand::SetActivationLevel(level));
        self.ui_state.activation_level = level;
    }

    /// Update muscle group
    pub fn set_muscle_group(&mut self, muscle_group: MuscleGroup) {
        let mut config = self.stream_config.clone();
        config.emg_config.muscle_group = muscle_group;
        self.send_command(StreamCommand::UpdateConfig(config.clone()));
        self.stream_config = config;
        self.ui_state.selected_muscle = muscle_group;
    }

    /// Update signal pattern
    pub fn set_signal_pattern(&mut self, pattern: SignalPattern) {
        self.send_command(StreamCommand::UpdatePattern(pattern));
        self.ui_state.selected_pattern = pattern;
    }

    /// Update sampling rate
    pub fn set_sampling_rate(&mut self, rate: f32) {
        let mut config = self.stream_config.clone();
        config.emg_config.sampling_rate = rate;
        self.send_command(StreamCommand::UpdateConfig(config.clone()));
        self.stream_config = config;
        self.ui_state.sampling_rate = rate;
    }

    /// Get current statistics
    pub fn get_stats(&self) -> AppStats {
        let total_samples = self.signal_buffer.iter()
            .map(|s| s.len())
            .sum::<usize>();

        let total_duration = self.signal_buffer.iter()
            .map(|s| s.duration())
            .sum::<f32>();

        AppStats {
            is_running: self.is_running,
            buffer_chunks: self.signal_buffer.len(),
            total_samples,
            total_duration,
            frame_count: self.frame_count,
            fps: self.calculate_fps(),
            initialized: self.initialized,
        }
    }

    fn calculate_fps(&self) -> f32 {
        let elapsed = self.last_update.elapsed().as_secs_f32();
        if elapsed > 0.0 {
            1.0 / elapsed
        } else {
            0.0
        }
    }

    /// Get current stream configuration
    pub fn stream_config(&self) -> &StreamConfig {
        &self.stream_config
    }

    /// Get most recent signal from buffer
    pub fn latest_signal(&self) -> Option<&SignalEntity> {
        self.signal_buffer.back()
    }

    /// Get signal buffer for UI access
    pub fn signal_buffer(&self) -> &VecDeque<SignalEntity> {
        &self.signal_buffer
    }
}

impl eframe::App for BSPApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update data from stream
        self.update_data();
        self.last_update = Instant::now();

        // Request continuous repaints for real-time updates
        ctx.request_repaint();

        // Show initialization message if not ready
        if !self.initialized {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(200.0);
                    ui.heading("Initializing BSP-Framework...");
                    ui.add_space(20.0);
                    ui.spinner();
                    ui.label("Setting up EMG simulation stream...");
                });
            });
            return;
        }

        // Main UI layout
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Export Data").clicked() {
                        // TODO: Implement data export
                        ui.close_menu();
                    }
                    if ui.button("Settings").clicked() {
                        self.ui_state.show_settings = !self.ui_state.show_settings;
                        ui.close_menu();
                    }
                });

                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.ui_state.show_controls, "Show Controls");
                    ui.checkbox(&mut self.ui_state.show_stats, "Show Statistics");
                    ui.checkbox(&mut self.ui_state.show_plot, "Show Plot");
                });

                ui.separator();

                // Status indicator
                let status_color = if self.is_running {
                    egui::Color32::GREEN
                } else {
                    egui::Color32::RED
                };

                ui.colored_label(status_color, if self.is_running { "● RUNNING" } else { "● STOPPED" });
            });
        });

        // Control panel
        if self.ui_state.show_controls {
            egui::SidePanel::left("control_panel")
                .resizable(true)
                .default_width(300.0)
                .show(ctx, |ui| {
                    ControlPanel::show(ui, self);
                });
        }

        // Statistics panel
        if self.ui_state.show_stats {
            egui::SidePanel::right("stats_panel")
                .resizable(true)
                .default_width(250.0)
                .show(ctx, |ui| {
                    ui.heading("Statistics");
                    ui.separator();

                    let stats = self.get_stats();
                    ui.label(format!("Status: {}", if stats.is_running { "Running" } else { "Stopped" }));
                    ui.label(format!("Initialized: {}", if stats.initialized { "Yes" } else { "No" }));
                    ui.label(format!("Buffer chunks: {}", stats.buffer_chunks));
                    ui.label(format!("Total samples: {}", stats.total_samples));
                    ui.label(format!("Duration: {:.1}s", stats.total_duration));
                    ui.label(format!("Frame count: {}", stats.frame_count));
                    ui.label(format!("FPS: {:.1}", stats.fps));

                    ui.separator();

                    if let Some(latest_signal) = self.latest_signal() {
                        ui.label(format!("Latest signal:"));
                        ui.label(format!("  Channels: {}", latest_signal.channel_count()));
                        ui.label(format!("  Samples: {}", latest_signal.samples_per_channel()));
                        ui.label(format!("  Rate: {:.0}Hz", latest_signal.sampling_rate()));

                        if let Ok(stats) = latest_signal.channel_stats(0) {
                            ui.label(format!("  RMS: {:.3}mV", stats.rms));
                            ui.label(format!("  Peak-to-peak: {:.3}mV", stats.peak_to_peak));
                        }
                    }
                });
        }

        // Main plot area
        if self.ui_state.show_plot {
            egui::CentralPanel::default().show(ctx, |ui| {
                self.plot_data.show_plot(ui, &self.ui_state);
            });
        }

        // Settings window
        if self.ui_state.show_settings {
            egui::Window::new("Settings")
                .resizable(true)
                .default_size([400.0, 300.0])
                .show(ctx, |ui| {
                    ui.heading("Stream Configuration");

                    ui.horizontal(|ui| {
                        ui.label("Sampling Rate:");
                        if ui.add(egui::Slider::new(&mut self.ui_state.sampling_rate, 500.0..=4000.0)
                            .suffix("Hz")).changed() {
                            self.set_sampling_rate(self.ui_state.sampling_rate);
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Update Rate:");
                        ui.add(egui::Slider::new(&mut self.ui_state.update_rate, 1.0..=60.0)
                            .suffix("Hz"));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Chunk Duration:");
                        ui.add(egui::Slider::new(&mut self.ui_state.chunk_duration, 0.01..=1.0)
                            .suffix("s"));
                    });

                    ui.separator();

                    if ui.button("Apply Settings").clicked() {
                        // Apply configuration changes
                        let mut config = self.stream_config.clone();
                        config.update_rate = self.ui_state.update_rate;
                        config.chunk_duration = self.ui_state.chunk_duration;
                        self.send_command(StreamCommand::UpdateConfig(config.clone()));
                        self.stream_config = config;
                    }

                    if ui.button("Close").clicked() {
                        self.ui_state.show_settings = false;
                    }
                });
        }
    }
}

/// Application statistics
#[derive(Debug, Clone)]
pub struct AppStats {
    pub is_running: bool,
    pub buffer_chunks: usize,
    pub total_samples: usize,
    pub total_duration: f32,
    pub frame_count: u64,
    pub fps: f32,
    pub initialized: bool,
}