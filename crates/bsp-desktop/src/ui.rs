//! UI components and state management

use bsp_core::{SignalEntity, MuscleGroup};
use bsp_simulation::SignalPattern;
use egui_plot::{Line, Plot, PlotPoints, Legend, Corner};
use std::collections::VecDeque;
use crate::app::BSPApp;

/// UI state management
#[derive(Debug)]
pub struct UIState {
    // Panel visibility
    pub show_controls: bool,
    pub show_stats: bool,
    pub show_plot: bool,
    pub show_settings: bool,

    // Control values
    pub activation_level: f32,
    pub selected_muscle: MuscleGroup,
    pub selected_pattern: SignalPattern,
    pub sampling_rate: f32,
    pub update_rate: f32,
    pub chunk_duration: f32,

    // Plot settings
    pub plot_window_duration: f32,
    pub plot_auto_scale: bool,
    pub plot_y_range: [f32; 2],
    pub show_multiple_channels: bool,
    pub selected_channel: usize,
}

impl UIState {
    pub fn new() -> Self {
        Self {
            show_controls: true,
            show_stats: true,
            show_plot: true,
            show_settings: false,

            activation_level: 0.4,
            selected_muscle: MuscleGroup::Biceps,
            selected_pattern: SignalPattern::Realistic {
                base_activation: 0.4,
                tremor_frequency: 8.0,
                tremor_amplitude: 0.05,
            },
            sampling_rate: 1000.0,
            update_rate: 10.0,
            chunk_duration: 0.1,

            plot_window_duration: 3.0,
            plot_auto_scale: true,
            plot_y_range: [-2.0, 2.0],
            show_multiple_channels: true,
            selected_channel: 0,
        }
    }
}

/// Plot data management
pub struct PlotData {
    // Time series data for plotting
    time_series: VecDeque<(f32, Vec<f32>)>, // (time, channel_values)
    max_points: usize,
    current_time: f32,
}

impl PlotData {
    pub fn new() -> Self {
        Self {
            time_series: VecDeque::new(),
            max_points: 3000, // 3 seconds at 1kHz
            current_time: 0.0,
        }
    }

    /// Check if plot data is empty
    pub fn is_empty(&self) -> bool {
        self.time_series.is_empty()
    }

    /// Add new signal data to the plot
    pub fn add_signal(&mut self, signal: SignalEntity) {
        let time_vector = signal.time_vector();
        let channels = match signal.all_channels() {
            Ok(channels) => channels,
            Err(_) => return,
        };

        // Add each sample to the time series
        for (i, &time_offset) in time_vector.iter().enumerate() {
            let absolute_time = self.current_time + time_offset;
            let mut channel_values = Vec::new();

            for channel_data in &channels {
                if i < channel_data.len() {
                    channel_values.push(channel_data[i]);
                }
            }

            self.time_series.push_back((absolute_time, channel_values));

            // Keep buffer size manageable
            if self.time_series.len() > self.max_points {
                self.time_series.pop_front();
            }
        }

        self.current_time += signal.duration();
    }

    /// Clear all plot data
    pub fn clear(&mut self) {
        self.time_series.clear();
        self.current_time = 0.0;
    }

    /// Show the EMG plot
    pub fn show_plot(&self, ui: &mut egui::Ui, ui_state: &UIState) {
        ui.heading("EMG Signal Visualization");

        // Plot controls
        ui.horizontal(|ui| {
            ui.label("Window:");
            ui.label(format!("{:.1}s", ui_state.plot_window_duration));

            ui.separator();

            ui.label("Auto-scale:");
            ui.label(if ui_state.plot_auto_scale { "ON" } else { "OFF" });

            if !ui_state.plot_auto_scale {
                ui.separator();
                ui.label(format!("Range: [{:.1}, {:.1}]mV",
                                 ui_state.plot_y_range[0], ui_state.plot_y_range[1]));
            }
        });

        // Create the plot
        let plot = Plot::new("emg_plot")
            .legend(Legend::default().position(Corner::LeftTop))
            .height(400.0)
            .allow_zoom(true)
            .allow_drag(true);

        let plot = if ui_state.plot_auto_scale {
            plot
        } else {
            plot.include_y(ui_state.plot_y_range[0])
                .include_y(ui_state.plot_y_range[1])
        };

        plot.show(ui, |plot_ui| {
            // Filter data to show only the recent window
            let window_start = self.current_time - ui_state.plot_window_duration;
            let recent_data: Vec<_> = self.time_series.iter()
                .filter(|(time, _)| *time >= window_start)
                .cloned()
                .collect();

            if recent_data.is_empty() {
                return;
            }

            // Determine number of channels
            let num_channels = recent_data.iter()
                .map(|(_, channels)| channels.len())
                .max()
                .unwrap_or(1);

            // Plot each channel
            for ch in 0..num_channels {
                let points: PlotPoints = recent_data.iter()
                    .filter_map(|(time, channels)| {
                        if ch < channels.len() {
                            Some([*time as f64, channels[ch] as f64])
                        } else {
                            None
                        }
                    })
                    .collect();

                if !points.points().is_empty() {
                    let color = match ch {
                        0 => egui::Color32::from_rgb(255, 100, 100), // Red
                        1 => egui::Color32::from_rgb(100, 255, 100), // Green
                        2 => egui::Color32::from_rgb(100, 100, 255), // Blue
                        3 => egui::Color32::from_rgb(255, 255, 100), // Yellow
                        _ => egui::Color32::from_rgb(150, 150, 150), // Gray
                    };

                    let line = Line::new(points)
                        .color(color)
                        .name(format!("Channel {}", ch + 1));

                    plot_ui.line(line);
                }
            }
        });

        // Plot statistics
        if !self.time_series.is_empty() {
            ui.separator();
            ui.horizontal(|ui| {
                ui.label(format!("Data points: {}", self.time_series.len()));
                ui.separator();
                ui.label(format!("Duration: {:.1}s", self.current_time));

                if let Some((_, latest_channels)) = self.time_series.back() {
                    ui.separator();
                    ui.label(format!("Latest values:"));
                    for (i, &value) in latest_channels.iter().enumerate() {
                        ui.label(format!("CH{}: {:.3}mV", i + 1, value));
                    }
                }
            });
        }
    }
}

/// Control panel UI
pub struct ControlPanel;

impl ControlPanel {
    pub fn show(ui: &mut egui::Ui, app: &mut BSPApp) {
        ui.heading("EMG Simulation Controls");
        ui.separator();

        // Playback controls
        ui.group(|ui| {
            ui.label("Playback");

            ui.horizontal(|ui| {
                if ui.button("▶ Start").clicked() {
                    app.start_stream();
                }

                if ui.button("⏸ Pause").clicked() {
                    app.pause_stream();
                }

                if ui.button("⏹ Stop").clicked() {
                    app.stop_stream();
                }

                if ui.button("▶ Resume").clicked() {
                    app.resume_stream();
                }
            });
        });

        ui.separator();

        // Muscle group selection
        ui.group(|ui| {
            ui.label("Muscle Group");

            egui::ComboBox::from_id_source("muscle_group_combo")
                .selected_text(format!("{}", app.ui_state.selected_muscle))
                .show_ui(ui, |ui| {
                    let muscles = [
                        MuscleGroup::Biceps,
                        MuscleGroup::Triceps,
                        MuscleGroup::Forearm,
                        MuscleGroup::Quadriceps,
                        MuscleGroup::Hamstring,
                        MuscleGroup::Calf,
                    ];

                    for muscle in muscles {
                        if ui.selectable_value(&mut app.ui_state.selected_muscle, muscle, format!("{}", muscle)).clicked() {
                            app.set_muscle_group(muscle);
                        }
                    }
                });
        });

        ui.separator();

        // Signal pattern selection
        ui.group(|ui| {
            ui.label("Signal Pattern");

            let patterns = SignalPattern::presets();
            let current_description = app.ui_state.selected_pattern.description();

            egui::ComboBox::from_id_source("signal_pattern_combo")
                .selected_text(current_description)
                .show_ui(ui, |ui| {
                    for (name, pattern) in patterns {
                        if ui.selectable_label(
                            pattern.description() == current_description,
                            name
                        ).clicked() {
                            app.ui_state.selected_pattern = pattern;
                            app.set_signal_pattern(pattern);
                        }
                    }
                });
        });

        ui.separator();

        // Manual activation control
        ui.group(|ui| {
            ui.label("Manual Activation");

            let mut activation = app.ui_state.activation_level;
            ui.horizontal(|ui| {
                ui.label("Level:");
                if ui.add(egui::Slider::new(&mut activation, 0.0..=1.0)
                    .text("Activation")).changed() {
                    app.set_activation_level(activation);
                }
            });

            ui.label(format!("{:.0}%", activation * 100.0));

            // Quick preset buttons
            ui.horizontal(|ui| {
                if ui.small_button("Rest").clicked() {
                    app.set_activation_level(0.1);
                }
                if ui.small_button("Light").clicked() {
                    app.set_activation_level(0.3);
                }
                if ui.small_button("Moderate").clicked() {
                    app.set_activation_level(0.6);
                }
                if ui.small_button("High").clicked() {
                    app.set_activation_level(0.9);
                }
            });
        });

        ui.separator();

        // Sampling rate control
        ui.group(|ui| {
            ui.label("Signal Parameters");

            let mut rate = app.ui_state.sampling_rate;
            ui.horizontal(|ui| {
                ui.label("Sampling Rate:");
                if ui.add(egui::Slider::new(&mut rate, 500.0..=4000.0)
                    .suffix("Hz")).changed() {
                    app.set_sampling_rate(rate);
                }
            });
        });

        ui.separator();

        // Plot controls
        ui.group(|ui| {
            ui.label("Plot Settings");

            ui.horizontal(|ui| {
                ui.label("Window:");
                ui.add(egui::Slider::new(&mut app.ui_state.plot_window_duration, 0.5..=10.0)
                    .suffix("s"));
            });

            ui.checkbox(&mut app.ui_state.plot_auto_scale, "Auto-scale Y axis");

            if !app.ui_state.plot_auto_scale {
                ui.horizontal(|ui| {
                    ui.label("Y Range:");
                    ui.add(egui::DragValue::new(&mut app.ui_state.plot_y_range[0])
                        .speed(0.1).prefix("Min: "));
                    ui.add(egui::DragValue::new(&mut app.ui_state.plot_y_range[1])
                        .speed(0.1).prefix("Max: "));
                });
            }
        });

        // Help section
        ui.separator();
        ui.collapsing("Help", |ui| {
            ui.label("Controls:");
            ui.label("• Use Start/Stop to control signal generation");
            ui.label("• Select different muscle groups to see varying patterns");
            ui.label("• Adjust activation level for different signal amplitudes");
            ui.label("• Choose signal patterns for realistic EMG simulation");
            ui.label("• Modify sampling rate to see effect on signal quality");
        });
    }
}