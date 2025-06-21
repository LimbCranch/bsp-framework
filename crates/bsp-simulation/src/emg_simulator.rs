//! EMG signal simulator with realistic muscle activation patterns

use bsp_core::{SignalEntity, EMGMetadata, EMGSignal, MuscleGroup, BspResult};
use crate::signal_patterns::SignalPattern;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Configuration for EMG simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EMGConfig {
    /// Muscle group being simulated
    pub muscle_group: MuscleGroup,
    /// Sampling rate in Hz
    pub sampling_rate: f32,
    /// Number of channels to simulate
    pub channel_count: usize,
    /// Signal pattern to generate
    pub pattern: PatternConfig,
    /// Noise configuration
    pub noise: NoiseConfig,
    /// Power line interference (50/60Hz)
    pub powerline_freq: Option<f32>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

/// Pattern configuration wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternConfig {
    pub pattern_type: String,
    pub parameters: Vec<f32>,
}

impl PatternConfig {
    pub fn from_pattern(pattern: SignalPattern) -> Self {
        match pattern {
            SignalPattern::Constant { level } => PatternConfig {
                pattern_type: "constant".to_string(),
                parameters: vec![level],
            },
            SignalPattern::Sinusoidal { frequency, amplitude, baseline } => PatternConfig {
                pattern_type: "sinusoidal".to_string(),
                parameters: vec![frequency, amplitude, baseline],
            },
            SignalPattern::Realistic { base_activation, tremor_frequency, tremor_amplitude } => {
                PatternConfig {
                    pattern_type: "realistic".to_string(),
                    parameters: vec![base_activation, tremor_frequency, tremor_amplitude],
                }
            },
            _ => PatternConfig {
                pattern_type: "constant".to_string(),
                parameters: vec![0.3],
            },
        }
    }

    pub fn to_pattern(&self) -> SignalPattern {
        match self.pattern_type.as_str() {
            "constant" => SignalPattern::Constant {
                level: self.parameters.get(0).copied().unwrap_or(0.3)
            },
            "sinusoidal" => SignalPattern::Sinusoidal {
                frequency: self.parameters.get(0).copied().unwrap_or(1.0),
                amplitude: self.parameters.get(1).copied().unwrap_or(0.3),
                baseline: self.parameters.get(2).copied().unwrap_or(0.2),
            },
            "realistic" => SignalPattern::Realistic {
                base_activation: self.parameters.get(0).copied().unwrap_or(0.4),
                tremor_frequency: self.parameters.get(1).copied().unwrap_or(8.0),
                tremor_amplitude: self.parameters.get(2).copied().unwrap_or(0.05),
            },
            _ => SignalPattern::Constant { level: 0.3 },
        }
    }
}

/// Noise configuration for realistic EMG simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseConfig {
    /// Gaussian noise standard deviation (0.0 = no noise)
    pub gaussian_std: f32,
    /// Baseline wander amplitude
    pub baseline_wander: f32,
    /// Motion artifact probability (0.0 to 1.0)
    pub motion_artifact_prob: f32,
    /// Motion artifact amplitude
    pub motion_artifact_amp: f32,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            gaussian_std: 0.05,
            baseline_wander: 0.02,
            motion_artifact_prob: 0.01,
            motion_artifact_amp: 0.3,
        }
    }
}

impl Default for EMGConfig {
    fn default() -> Self {
        Self {
            muscle_group: MuscleGroup::Biceps,
            sampling_rate: 1000.0,
            channel_count: 1,
            pattern: PatternConfig::from_pattern(SignalPattern::Realistic {
                base_activation: 0.4,
                tremor_frequency: 8.0,
                tremor_amplitude: 0.05,
            }),
            noise: NoiseConfig::default(),
            powerline_freq: Some(50.0),
            seed: None,
        }
    }
}

/// EMG signal simulator
pub struct EMGSimulator {
    config: EMGConfig,
    rng: rand::rngs::StdRng,
    normal_dist: Normal<f32>,
    time_offset: f32,
}

impl EMGSimulator {
    /// Create new EMG simulator with configuration
    pub fn new(config: EMGConfig) -> BspResult<Self> {
        // Validate configuration
        EMGMetadata::validate_sampling_rate(config.sampling_rate)?;
        EMGMetadata::validate_channel_count(config.channel_count)?;

        let seed = config.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        });

        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal_dist = Normal::new(0.0, config.noise.gaussian_std)
            .map_err(|e| bsp_core::BspError::SimulationError {
                message: format!("Failed to create normal distribution: {}", e),
            })?;

        Ok(EMGSimulator {
            config,
            rng,
            normal_dist,
            time_offset: 0.0,
        })
    }

    /// Generate EMG signal for specified duration
    pub fn generate(&mut self, duration: f32) -> BspResult<SignalEntity> {
        let samples_per_channel = (duration * self.config.sampling_rate) as usize;
        let total_samples = samples_per_channel * self.config.channel_count;
        let mut data = Vec::with_capacity(total_samples);

        let dt = 1.0 / self.config.sampling_rate;
        let pattern = self.config.pattern.to_pattern();

        // Generate interleaved channel data
        for sample_idx in 0..samples_per_channel {
            let time = self.time_offset + sample_idx as f32 * dt;

            for channel_idx in 0..self.config.channel_count {
                let mut signal_value = self.generate_emg_sample(time, channel_idx, &pattern);

                // Add noise components
                signal_value += self.add_noise(time);

                // Add powerline interference if configured
                if let Some(powerline_freq) = self.config.powerline_freq {
                    signal_value += self.add_powerline_interference(time, powerline_freq);
                }

                // Clamp to reasonable EMG range
                signal_value = signal_value.max(-5.0).min(5.0);

                data.push(signal_value);
            }
        }

        // Update time offset for continuous generation
        self.time_offset += duration;

        // Create metadata
        let metadata = EMGMetadata::new(
            EMGSignal::Surface {
                muscle_group: self.config.muscle_group,
                activation_level: self.calculate_average_activation(&pattern, duration),
            },
            self.config.sampling_rate,
            self.config.channel_count,
            duration,
            self.config.noise.gaussian_std,
        )?;

        SignalEntity::new(data, metadata)
    }

    /// Generate single EMG sample
    fn generate_emg_sample(&mut self, time: f32, channel_idx: usize, pattern: &SignalPattern) -> f32 {
        let activation = pattern.activation_at_time(time);

        // Generate realistic EMG signal based on muscle activation
        let base_frequency = 80.0 + (channel_idx as f32 * 10.0); // Slight frequency variation per channel
        let signal_amplitude = activation * 2.0; // Scale activation to mV range

        // Multiple frequency components for realistic EMG
        let mut emg_signal = 0.0;

        // Primary muscle firing frequency
        emg_signal += signal_amplitude * (2.0 * std::f32::consts::PI * base_frequency * time).sin();

        // Higher harmonics
        emg_signal += signal_amplitude * 0.3 * (2.0 * std::f32::consts::PI * base_frequency * 2.0 * time).sin();
        emg_signal += signal_amplitude * 0.1 * (2.0 * std::f32::consts::PI * base_frequency * 3.0 * time).sin();

        // Random muscle fiber recruitment
        let recruitment_noise = activation * self.rng.gen_range(-0.2..0.2);
        emg_signal += recruitment_noise;

        emg_signal
    }

    /// Add various noise components
    fn add_noise(&mut self, time: f32) -> f32 {
        let mut noise = 0.0;

        // Gaussian noise
        noise += self.normal_dist.sample(&mut self.rng);

        // Baseline wander (slow drift)
        noise += self.config.noise.baseline_wander *
            (2.0 * std::f32::consts::PI * 0.1 * time).sin();

        // Motion artifacts (random spikes)
        if self.rng.gen::<f32>() < self.config.noise.motion_artifact_prob {
            noise += self.config.noise.motion_artifact_amp * self.rng.gen_range(-1.0..1.0);
        }

        noise
    }

    /// Add powerline interference
    fn add_powerline_interference(&mut self, time: f32, frequency: f32) -> f32 {
        let amplitude = 0.05; // Small interference
        amplitude * (2.0 * std::f32::consts::PI * frequency * time).sin()
    }

    /// Calculate average activation level for metadata
    fn calculate_average_activation(&self, pattern: &SignalPattern, duration: f32) -> f32 {
        let num_samples = 100; // Sample pattern at 100 points
        let dt = duration / num_samples as f32;

        let mut sum = 0.0;
        for i in 0..num_samples {
            sum += pattern.activation_at_time(i as f32 * dt);
        }

        (sum / num_samples as f32).max(0.0).min(1.0)
    }

    /// Reset time offset (useful for restarting simulation)
    pub fn reset_time(&mut self) {
        self.time_offset = 0.0;
    }

    /// Get current configuration
    pub fn config(&self) -> &EMGConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: EMGConfig) -> BspResult<()> {
        EMGMetadata::validate_sampling_rate(config.sampling_rate)?;
        EMGMetadata::validate_channel_count(config.channel_count)?;

        self.config = config;
        Ok(())
    }

    /// Generate continuous chunks for streaming
    pub fn generate_chunk(&mut self, chunk_duration: f32) -> BspResult<SignalEntity> {
        self.generate(chunk_duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal_patterns::SignalPattern;

    #[test]
    fn test_emg_simulator_basic() {
        let config = EMGConfig::default();
        let mut simulator = EMGSimulator::new(config).unwrap();

        let signal = simulator.generate(1.0).unwrap();

        assert_eq!(signal.duration(), 1.0);
        assert_eq!(signal.sampling_rate(), 1000.0);
        assert_eq!(signal.channel_count(), 1);
        assert_eq!(signal.samples_per_channel(), 1000);
    }

    #[test]
    fn test_multichannel_simulation() {
        let mut config = EMGConfig::default();
        config.channel_count = 4;

        let mut simulator = EMGSimulator::new(config).unwrap();
        let signal = simulator.generate(0.5).unwrap();

        assert_eq!(signal.channel_count(), 4);
        assert_eq!(signal.len(), 2000); // 500 samples * 4 channels

        // Test that channels have different but reasonable values
        let channels = signal.all_channels().unwrap();
        assert_eq!(channels.len(), 4);

        for channel_data in channels {
            assert_eq!(channel_data.len(), 500);
            // Basic sanity check - signal should have some variation
            let stats = bsp_core::ChannelStats::calculate(&channel_data);
            assert!(stats.std_dev > 0.0);
        }
    }

    #[test]
    fn test_different_patterns() {
        let patterns = vec![
            SignalPattern::Constant { level: 0.5 },
            SignalPattern::Sinusoidal { frequency: 1.0, amplitude: 0.3, baseline: 0.2 },
            SignalPattern::Realistic { base_activation: 0.6, tremor_frequency: 8.0, tremor_amplitude: 0.05 },
        ];

        for pattern in patterns {
            let mut config = EMGConfig::default();
            config.pattern = PatternConfig::from_pattern(pattern);

            let mut simulator = EMGSimulator::new(config).unwrap();
            let signal = simulator.generate(1.0).unwrap();

            assert_eq!(signal.len(), 1000);

            // Verify signal has reasonable range
            let channel_data = signal.channel_data(0).unwrap();
            let stats = bsp_core::ChannelStats::calculate(&channel_data);
            assert!(stats.min >= -5.0);
            assert!(stats.max <= 5.0);
        }
    }
}