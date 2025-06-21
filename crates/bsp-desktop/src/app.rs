//! Main application state and logic with processing integration

use bsp_core::{SignalEntity, MuscleGroup};
use bsp_simulation::{
    RealTimeEMGStream, StreamConfig, StreamCommand, EMGConfig, SignalPattern,
    start_emg_stream
};
use tokio::sync::{broadcast, mpsc};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::ui::{UIState, PlotData, ControlPanel};
use crate::processing_service::{
    ProcessingService, ProcessingCommand, ProcessingPipelineConfig, PipelineType,
    ProcessedSignalData, ProcessingStats, start_processing_service
};

/// Main application state
pub struct BSPApp {
    // Streaming components
    data_receiver: Option<broadcast::Receiver<SignalEntity>>,
    control_sender: Option<mpsc::Sender<StreamCommand>>,

    // Processing components
    processed_data_receiver: Option<broadcast::Receiver<ProcessedSignalData>>,
    processing_command_sender: Option<mpsc::Sender<ProcessingCommand>>,
    processing_stats: Option<Arc<Mutex<ProcessingStats>>>,

    // Tokio runtime for async operations
    runtime: tokio::runtime::Runtime,

    // UI state - make public for access from UI module
    pub ui_state: UIState,

    // Data storage for visualization
    signal_buffer: VecDeque<SignalEntity>,
    processed_signal_buffer: VecDeque<ProcessedSignalData>,
    pub plot_data: PlotData, // Make public for UI access
    pub processed_plot_data: PlotData, // Plot data for processed signals

    // Application state
    is_running: bool,
    last_update: Instant,
    frame_count: u64,

    // Configuration
    stream_config: StreamConfig,
    pub processing_config: ProcessingPipelineConfig,

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
        let processed_plot_data = PlotData::new();
        let stream_config = StreamConfig::default();
        let processing_config = ProcessingPipelineConfig::default();

        Ok(BSPApp {
            data_receiver: None,
            control_sender: None,
            processed_data_receiver: None,
            processing_command_sender: None,
            processing_stats: None,
            runtime,
            ui_state,
            signal_buffer: VecDeque::with_capacity(100),
            processed_signal_buffer: VecDeque::with_capacity(100),
            plot_data,
            processed_plot_data,
            is_running: false,
            last_update: Instant::now(),
            frame_count: 0,
            stream_config,
            processing_config,
            initialized: false,
        })
    }

    /// Initialize the streaming and processing components (called on first frame)
    fn initialize_services(&mut self) {
        if self.initialized {
            return;
        }

        println!("Initializing EMG stream and processing pipeline...");

        let stream_config = self.stream_config.clone();
        let processing_config = self.processing_config.clone();

        match self.runtime.block_on(async {
            // Start EMG simulation stream
            let (data_receiver, control_sender) = start_emg_stream(stream_config).await?;

            // Start processing service with the simulation stream as input
            let (processed_data_receiver, processing_command_sender, processing_stats) =
                start_processing_service(data_receiver.resubscribe(), processing_config).await?;

            Ok::<_, anyhow::Error>((
                data_receiver,
                control_sender,
                processed_data_receiver,
                processing_command_sender,
                processing_stats,
            ))
        }) {
            Ok((data_receiver, control_sender, processed_data_receiver, processing_command_sender, processing_stats)) => {
                self.data_receiver = Some(data_receiver);
                self.control_sender = Some(control_sender);
                self.processed_data_receiver = Some(processed_data_receiver);
                self.processing_command_sender = Some(processing_command_sender);
                self.processing_stats = Some(processing_stats);
                self.initialized = true;
                println!("Services initialized successfully");
            }
            Err(e) => {
                eprintln!("Failed to initialize services: {}", e);
            }
        }
    }

    /// Update application state (called every frame)
    fn update_data(&mut self) {
        // Initialize services on first call
        if !self.initialized {
            self.initialize_services();
            return;
        }

        // Try to receive new raw signal data (non-blocking)
        if let Some(ref mut receiver) = self.data_receiver {
            while let Ok(signal) = receiver.try_recv() {
                // Add to raw signal buffer
                self.signal_buffer.push_back(signal.clone());

                // Keep buffer size manageable
                if self.signal_buffer.len() > 100 {
                    self.signal_buffer.pop_front();
                }

                // Add to raw plot data
                self.plot_data.add_signal(signal);
            }
        }

        // Try to receive processed signal data (non-blocking)
        if let Some(ref mut receiver) = self.processed_data_receiver {
            while let Ok(processed_data) = receiver.try_recv() {
                // Add to processed signal buffer
                self.processed_signal_buffer.push_back(processed_data.clone());

                // Keep buffer size manageable
                if self.processed_signal_buffer.len() > 100 {
                    self.processed_signal_buffer.pop_front();
                }

                // Add to processed plot data
                self.processed_plot_data.add_signal(processed_data.processed_signal);
            }
        }

        self.frame_count += 1;
    }

    /// Send control command to stream
    fn send_stream_command(&self, command: StreamCommand) {
        if let Some(ref sender) = self.control_sender {
            if let Err(e) = sender.try_send(command) {
                eprintln!("Failed to send stream command: {}", e);
            }
        } else {
            eprintln!("Stream not initialized, cannot send command");
        }
    }

    /// Send control command to processing
    fn send_processing_command(&self, command: ProcessingCommand) {
        if let Some(ref sender) = self.processing_command_sender {
            if let Err(e) = sender.try_send(command) {
                eprintln!("Failed to send processing command: {}", e);
            }
        } else {
            eprintln!("Processing not initialized, cannot send command");
        }
    }

    /// Start the EMG stream and processing
    pub fn start_stream(&mut self) {
        self.send_stream_command(StreamCommand::Start);
        self.send_processing_command(ProcessingCommand::Start);
        self.is_running = true;
    }

    /// Stop the EMG stream and processing
    pub fn stop_stream(&mut self) {
        self.send_stream_command(StreamCommand::Stop);
        self.send_processing_command(ProcessingCommand::Stop);
        self.is_running = false;
        self.signal_buffer.clear();
        self.processed_signal_buffer.clear();
        self.plot_data.clear();
        self.processed_plot_data.clear();
    }

    /// Pause the EMG stream and processing
    pub fn pause_stream(&mut self) {
        self.send_stream_command(StreamCommand::Pause);
        self.send_processing_command(ProcessingCommand::Pause);
        self.is_running = false;
    }

    /// Resume the EMG stream and processing
    pub fn resume_stream(&mut self) {
        self.send_stream_command(StreamCommand::Resume);
        self.send_processing_command(ProcessingCommand::Resume);
        self.is_running = true;
    }

    /// Update activation level
    pub fn set_activation_level(&mut self, level: f32) {
        self.send_stream_command(StreamCommand::SetActivationLevel(level));
        self.ui_state.activation_level = level;
    }

    /// Update muscle group
    pub fn set_muscle_group(&mut self, muscle_group: MuscleGroup) {
        let mut config = self.stream_config.clone();
        config.emg_config.muscle_group = muscle_group;
        self.send_stream_command(StreamCommand::UpdateConfig(config.clone()));
        self.stream_config = config;
        self.ui_state.selected_muscle = muscle_group;
    }

    /// Update signal pattern
    pub fn set_signal_pattern(&mut self, pattern: SignalPattern) {
        self.send_stream_command(StreamCommand::UpdatePattern(pattern));
        self.ui_state.selected_pattern = pattern;
    }

    /// Update sampling rate
    pub fn set_sampling_rate(&mut self, rate: f32) {
        let mut config = self.stream_config.clone();
        config.emg_config.sampling_rate = rate;
        self.send_stream_command(StreamCommand::UpdateConfig(config.clone()));
        self.stream_config = config;
        self.ui_state.sampling_rate = rate;
    }

    /// Update processing pipeline configuration
    pub fn update_processing_config(&mut self, new_config: ProcessingPipelineConfig) {
        self.send_processing_command(ProcessingCommand::UpdatePipeline(new_config.clone()));
        self.processing_config = new_config;
    }

    /// Set processing pipeline type
    pub fn set_pipeline_type(&mut self, pipeline_type: PipelineType) {
        let mut config = self.processing_config.clone();
        config.pipeline_type = pipeline_type;
        self.update_processing_config(config);
    }

    /// Enable/disable processing bypass mode
    pub fn set_processing_bypass(&mut self, enabled: bool) {
        self.send_processing_command(ProcessingCommand::SetBypassMode(enabled));
    }

    /// Reset processing pipeline
    pub fn reset_processing(&mut self) {
        self.send_processing_command(ProcessingCommand::ResetPipeline);
    }

    /// Get current statistics
    pub fn get_stats(&self) -> AppStats {
        let total_samples = self.signal_buffer.iter()
            .map(|s| s.len())
            .sum::<usize>();

        let total_duration = self.signal_buffer.iter()
            .map(|s| s.duration())
            .sum::<f32>();

        let processed_samples = self.processed_signal_buffer.iter()
            .map(|s| s.processed_signal.len())
            .sum::<usize>();

        AppStats {
            is_running: self.is_running,
            buffer_chunks: self.signal_buffer.len(),
            processed_chunks: self.processed_signal_buffer.len(),
            total_samples,
            processed_samples,
            total_duration,
            frame_count: self.frame_count,
            fps: self.calculate_fps(),
            initialized: self.initialized,
        }
    }

    /// Get processing statistics
    pub fn get_processing_stats(&self) -> Option<ProcessingStats> {
        if let Some(ref stats_handle) = self.processing_stats {
            // Try to get stats without blocking the UI thread
            if let Ok(stats) = stats_handle.try_lock() {
                Some(stats.clone())
            } else {
                None
            }
        } else {
            None
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

    /// Get most recent processed signal
    pub fn latest_processed_signal(&self) -> Option<&ProcessedSignalData> {
        self.processed_signal_buffer.back()
    }

    /// Get signal buffer for UI access
    pub fn signal_buffer(&self) -> &VecDeque<SignalEntity> {
        &self.signal_buffer
    }

    /// Get processed signal buffer for UI access
    pub fn processed_signal_buffer(&self) -> &VecDeque<ProcessedSignalData> {
        &self.processed_signal_buffer
    }
}

impl eframe::App for BSPApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update data from streams
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
                    ui.label("Setting up EMG simulation and processing pipeline...");
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
                    ui.checkbox(&mut self.ui_state.show_processing, "Show Processing");
                });

                ui.menu_button("Processing", |ui| {
                    if ui.button("Reset Pipeline").clicked() {
                        self.reset_processing();
                        ui.close_menu();
                    }

                    ui.separator();

                    ui.label("Pipeline Type:");
                    let current_type = self.processing_config.pipeline_type;

                    if ui.radio_value(&mut self.processing_config.pipeline_type, PipelineType::Bypass, "Bypass").clicked() && current_type != PipelineType::Bypass {
                        self.set_pipeline_type(PipelineType::Bypass);
                    }
                    if ui.radio_value(&mut self.processing_config.pipeline_type, PipelineType::RealTime, "Real-time").clicked() && current_type != PipelineType::RealTime {
                        self.set_pipeline_type(PipelineType::RealTime);
                    }
                    if ui.radio_value(&mut self.processing_config.pipeline_type, PipelineType::Standard, "Standard").clicked() && current_type != PipelineType::Standard {
                        self.set_pipeline_type(PipelineType::Standard);
                    }
                    if ui.radio_value(&mut self.processing_config.pipeline_type, PipelineType::Research, "Research").clicked() && current_type != PipelineType::Research {
                        self.set_pipeline_type(PipelineType::Research);
                    }
                });

                ui.separator();

                // Status indicator
                let status_color = if self.is_running {
                    egui::Color32::GREEN
                } else {
                    egui::Color32::RED
                };

                ui.colored_label(status_color, if self.is_running { "● RUNNING" } else { "● STOPPED" });

                // Pipeline type indicator
                ui.separator();
                ui.label(format!("Pipeline: {:?}", self.processing_config.pipeline_type));
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

        // Statistics and processing panel
        if self.ui_state.show_stats || self.ui_state.show_processing {
            egui::SidePanel::right("stats_panel")
                .resizable(true)
                .default_width(300.0)
                .show(ctx, |ui| {
                    if self.ui_state.show_stats {
                        ui.heading("Signal Statistics");
                        ui.separator();

                        let stats = self.get_stats();
                        ui.label(format!("Status: {}", if stats.is_running { "Running" } else { "Stopped" }));
                        ui.label(format!("Initialized: {}", if stats.initialized { "Yes" } else { "No" }));
                        ui.label(format!("Raw chunks: {}", stats.buffer_chunks));
                        ui.label(format!("Processed chunks: {}", stats.processed_chunks));
                        ui.label(format!("Total samples: {}", stats.total_samples));
                        ui.label(format!("Processed samples: {}", stats.processed_samples));
                        ui.label(format!("Duration: {:.1}s", stats.total_duration));
                        ui.label(format!("Frame count: {}", stats.frame_count));
                        ui.label(format!("FPS: {:.1}", stats.fps));

                        if let Some(latest_signal) = self.latest_signal() {
                            ui.separator();
                            ui.label("Latest raw signal:");
                            ui.label(format!("  Channels: {}", latest_signal.channel_count()));
                            ui.label(format!("  Samples: {}", latest_signal.samples_per_channel()));
                            ui.label(format!("  Rate: {:.0}Hz", latest_signal.sampling_rate()));

                            if let Ok(stats) = latest_signal.channel_stats(0) {
                                ui.label(format!("  RMS: {:.3}mV", stats.rms));
                                ui.label(format!("  Peak-to-peak: {:.3}mV", stats.peak_to_peak));
                            }
                        }
                    }

                    if self.ui_state.show_processing {
                        ui.separator();
                        ui.heading("Processing Statistics");
                        ui.separator();

                        if let Some(proc_stats) = self.get_processing_stats() {
                            ui.label(format!("Processing: {}", if proc_stats.is_running { "Running" } else { "Stopped" }));
                            ui.label(format!("Pipeline: {:?}", proc_stats.pipeline_type));
                            ui.label(format!("Signals processed: {}", proc_stats.signals_processed));
                            ui.label(format!("Avg latency: {:.1}μs", proc_stats.average_latency_us));
                            ui.label(format!("Success rate: {:.1}%", proc_stats.success_rate * 100.0));
                            ui.label(format!("Total processing time: {:.1}ms", proc_stats.total_processing_time_us as f32 / 1000.0));
                        } else {
                            ui.label("Processing stats unavailable");
                        }

                        if let Some(latest_processed) = self.latest_processed_signal() {
                            ui.separator();
                            ui.label("Latest processed signal:");
                            ui.label(format!("  Success: {}", if latest_processed.success { "Yes" } else { "No" }));
                            ui.label(format!("  Processing time: {:.1}μs", latest_processed.processing_time_us));

                            if !latest_processed.warnings.is_empty() {
                                ui.label("  Warnings:");
                                for warning in &latest_processed.warnings {
                                    ui.label(format!("    • {}", warning));
                                }
                            }

                            if let Some(ref features) = latest_processed.features {
                                ui.label(format!("  Features extracted: {} sets", features.len()));
                            }
                        }
                    }
                });
        }

        // Main plot area
        if self.ui_state.show_plot {
            egui::CentralPanel::default().show(ctx, |ui| {
                // Plot selection tabs
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.ui_state.plot_mode, PlotMode::Raw, "Raw Signal");
                    ui.selectable_value(&mut self.ui_state.plot_mode, PlotMode::Processed, "Processed Signal");
                    ui.selectable_value(&mut self.ui_state.plot_mode, PlotMode::Comparison, "Comparison");
                });

                ui.separator();

                match self.ui_state.plot_mode {
                    PlotMode::Raw => {
                        ui.label("Raw EMG Signal");
                        self.plot_data.show_plot(ui, &self.ui_state);
                    }
                    PlotMode::Processed => {
                        ui.label("Processed EMG Signal");
                        self.processed_plot_data.show_plot(ui, &self.ui_state);
                    }
                    PlotMode::Comparison => {
                        ui.label("Raw vs Processed Signal Comparison");

                        // Split the area for comparison
                        ui.columns(2, |columns| {
                            columns[0].group(|ui| {
                                ui.label("Raw Signal");
                                self.plot_data.show_plot(ui, &self.ui_state);
                            });

                            columns[1].group(|ui| {
                                ui.label("Processed Signal");
                                self.processed_plot_data.show_plot(ui, &self.ui_state);
                            });
                        });
                    }
                }
            });
        }

        // Settings window
        if self.ui_state.show_settings {
            egui::Window::new("Settings")
                .resizable(true)
                .default_size([500.0, 400.0])
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
                    ui.heading("Processing Configuration");

                    ui.horizontal(|ui| {
                        ui.label("Pipeline Type:");
                        egui::ComboBox::from_id_source("pipeline_type")
                            .selected_text(format!("{:?}", self.processing_config.pipeline_type))
                            .show_ui(ui, |ui| {
                                if ui.selectable_value(&mut self.processing_config.pipeline_type, PipelineType::Bypass, "Bypass").clicked() {
                                    self.set_pipeline_type(PipelineType::Bypass);
                                }
                                if ui.selectable_value(&mut self.processing_config.pipeline_type, PipelineType::RealTime, "Real-time").clicked() {
                                    self.set_pipeline_type(PipelineType::RealTime);
                                }
                                if ui.selectable_value(&mut self.processing_config.pipeline_type, PipelineType::Standard, "Standard").clicked() {
                                    self.set_pipeline_type(PipelineType::Standard);
                                }
                                if ui.selectable_value(&mut self.processing_config.pipeline_type, PipelineType::Research, "Research").clicked() {
                                    self.set_pipeline_type(PipelineType::Research);
                                }
                            });
                    });

                    ui.checkbox(&mut self.processing_config.enable_filters, "Enable Filters");
                    ui.checkbox(&mut self.processing_config.enable_features, "Enable Feature Extraction");

                    if self.processing_config.enable_filters {
                        ui.horizontal(|ui| {
                            ui.label("Highpass Cutoff:");
                            ui.add(egui::Slider::new(&mut self.processing_config.highpass_cutoff, 1.0..=100.0)
                                .suffix("Hz"));
                        });

                        ui.horizontal(|ui| {
                            ui.label("Lowpass Cutoff:");
                            ui.add(egui::Slider::new(&mut self.processing_config.lowpass_cutoff, 100.0..=1000.0)
                                .suffix("Hz"));
                        });

                        ui.horizontal(|ui| {
                            ui.label("Notch Frequency:");
                            ui.add(egui::Slider::new(&mut self.processing_config.notch_frequency, 45.0..=65.0)
                                .suffix("Hz"));
                        });
                    }

                    if self.processing_config.enable_features {
                        ui.horizontal(|ui| {
                            ui.label("Feature Window Size:");
                            ui.add(egui::Slider::new(&mut self.processing_config.feature_window_size, 64..=1024));
                        });

                        ui.horizontal(|ui| {
                            ui.label("Window Overlap:");
                            ui.add(egui::Slider::new(&mut self.processing_config.feature_overlap, 0.0..=0.9));
                        });
                    }

                    ui.separator();

                    ui.horizontal(|ui| {
                        if ui.button("Apply Settings").clicked() {
                            // Apply stream configuration changes
                            let mut config = self.stream_config.clone();
                            config.update_rate = self.ui_state.update_rate;
                            config.chunk_duration = self.ui_state.chunk_duration;
                            self.send_stream_command(StreamCommand::UpdateConfig(config.clone()));
                            self.stream_config = config;

                            // Apply processing configuration changes
                            self.update_processing_config(self.processing_config.clone());
                        }

                        if ui.button("Reset to Defaults").clicked() {
                            self.processing_config = ProcessingPipelineConfig::default();
                            self.ui_state.sampling_rate = 1000.0;
                            self.ui_state.update_rate = 10.0;
                            self.ui_state.chunk_duration = 0.1;
                        }

                        if ui.button("Close").clicked() {
                            self.ui_state.show_settings = false;
                        }
                    });
                });
        }
    }
}

/// Application statistics
#[derive(Debug, Clone)]
pub struct AppStats {
    pub is_running: bool,
    pub buffer_chunks: usize,
    pub processed_chunks: usize,
    pub total_samples: usize,
    pub processed_samples: usize,
    pub total_duration: f32,
    pub frame_count: u64,
    pub fps: f32,
    pub initialized: bool,
}

/// Plot modes for signal visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlotMode {
    Raw,
    Processed,
    Comparison,
}