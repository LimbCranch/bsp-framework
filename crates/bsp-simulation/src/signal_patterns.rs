//! Pre-defined EMG signal patterns for realistic simulation

use std::f32::consts::PI;

/// Predefined EMG signal patterns
#[derive(Debug, Clone, Copy)]
pub enum SignalPattern {
    /// Constant activation level
    Constant { level: f32 },
    /// Sinusoidal muscle contraction pattern
    Sinusoidal {
        frequency: f32,
        amplitude: f32,
        baseline: f32,
    },
    /// Ramp increase/decrease pattern
    Ramp {
        start_level: f32,
        end_level: f32,
        duration: f32,
    },
    /// Burst pattern (on/off cycles)
    Burst {
        on_duration: f32,
        off_duration: f32,
        amplitude: f32,
    },
    /// Fatigue pattern (decreasing amplitude over time)
    Fatigue {
        initial_amplitude: f32,
        decay_rate: f32,
    },
    /// Realistic muscle activation with tremor
    Realistic {
        base_activation: f32,
        tremor_frequency: f32,
        tremor_amplitude: f32,
    },
}

impl SignalPattern {
    /// Generate activation level at given time
    pub fn activation_at_time(&self, time: f32) -> f32 {
        match self {
            SignalPattern::Constant { level } => *level,

            SignalPattern::Sinusoidal { frequency, amplitude, baseline } => {
                baseline + amplitude * (2.0 * PI * frequency * time).sin()
            },

            SignalPattern::Ramp { start_level, end_level, duration } => {
                if time >= *duration {
                    *end_level
                } else {
                    start_level + (end_level - start_level) * (time / duration)
                }
            },

            SignalPattern::Burst { on_duration, off_duration, amplitude } => {
                let cycle_duration = on_duration + off_duration;
                let phase = time % cycle_duration;
                if phase < *on_duration {
                    *amplitude
                } else {
                    0.0
                }
            },

            SignalPattern::Fatigue { initial_amplitude, decay_rate } => {
                initial_amplitude * (-decay_rate * time).exp()
            },

            SignalPattern::Realistic { base_activation, tremor_frequency, tremor_amplitude } => {
                let tremor = tremor_amplitude * (2.0 * PI * tremor_frequency * time).sin();
                (base_activation + tremor).max(0.0).min(1.0)
            },
        }
    }

    /// Get pattern description
    pub fn description(&self) -> &'static str {
        match self {
            SignalPattern::Constant { .. } => "Constant activation",
            SignalPattern::Sinusoidal { .. } => "Sinusoidal contraction",
            SignalPattern::Ramp { .. } => "Gradual ramp",
            SignalPattern::Burst { .. } => "Burst pattern",
            SignalPattern::Fatigue { .. } => "Muscle fatigue",
            SignalPattern::Realistic { .. } => "Realistic with tremor",
        }
    }

    /// Create common preset patterns
    pub fn presets() -> Vec<(&'static str, SignalPattern)> {
        vec![
            ("Rest", SignalPattern::Constant { level: 0.1 }),
            ("Light Activity", SignalPattern::Constant { level: 0.3 }),
            ("Moderate Activity", SignalPattern::Constant { level: 0.6 }),
            ("High Activity", SignalPattern::Constant { level: 0.9 }),
            ("Slow Contraction", SignalPattern::Sinusoidal {
                frequency: 0.5, amplitude: 0.4, baseline: 0.2
            }),
            ("Fast Contraction", SignalPattern::Sinusoidal {
                frequency: 2.0, amplitude: 0.3, baseline: 0.1
            }),
            ("Warmup", SignalPattern::Ramp {
                start_level: 0.1, end_level: 0.7, duration: 10.0
            }),
            ("Cooldown", SignalPattern::Ramp {
                start_level: 0.8, end_level: 0.1, duration: 15.0
            }),
            ("Exercise Bursts", SignalPattern::Burst {
                on_duration: 2.0, off_duration: 1.0, amplitude: 0.8
            }),
            ("Fatigue Test", SignalPattern::Fatigue {
                initial_amplitude: 0.9, decay_rate: 0.1
            }),
            ("Natural Movement", SignalPattern::Realistic {
                base_activation: 0.4, tremor_frequency: 8.0, tremor_amplitude: 0.05
            }),
        ]
    }
}