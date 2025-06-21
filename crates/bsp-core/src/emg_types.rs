//! EMG-specific signal types and metadata

use serde::{Deserialize, Serialize};
use crate::error::{BspError, BspResult};

/// EMG signal classification
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EMGSignal {
    /// Surface EMG from muscle groups
    Surface {
        muscle_group: MuscleGroup,
        activation_level: f32, // 0.0 to 1.0
    },
    /// Intramuscular EMG (for future)
    Intramuscular {
        muscle_group: MuscleGroup,
    },
}

/// Major muscle groups for EMG classification
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MuscleGroup {
    Biceps,
    Triceps,
    Forearm,
    Quadriceps,
    Hamstring,
    Calf,
    Other(u8), // For extensibility
}

/// EMG signal metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EMGMetadata {
    /// Type of EMG signal
    pub signal_type: EMGSignal,
    /// Sampling rate in Hz
    pub sampling_rate: f32,
    /// Number of channels
    pub channel_count: usize,
    /// Signal duration in seconds
    pub duration: f32,
    /// Noise level (0.0 = clean, 1.0 = very noisy)
    pub noise_level: f32,
    /// Creation timestamp
    pub timestamp: u64,
}

impl EMGMetadata {
    /// Create new EMG metadata
    pub fn new(
        signal_type: EMGSignal,
        sampling_rate: f32,
        channel_count: usize,
        duration: f32,
        noise_level: f32,
    ) -> BspResult<Self> {
        Self::validate_sampling_rate(sampling_rate)?;
        Self::validate_channel_count(channel_count)?;

        if duration <= 0.0 {
            return Err(BspError::InvalidSignalData {
                reason: "Duration must be positive".to_string(),
            });
        }

        if !(0.0..=1.0).contains(&noise_level) {
            return Err(BspError::InvalidSignalData {
                reason: "Noise level must be between 0.0 and 1.0".to_string(),
            });
        }

        Ok(EMGMetadata {
            signal_type,
            sampling_rate,
            channel_count,
            duration,
            noise_level,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        })
    }

    /// Validate sampling rate for EMG signals
    pub fn validate_sampling_rate(rate: f32) -> BspResult<()> {
        const MIN_RATE: f32 = 500.0;
        const MAX_RATE: f32 = 4000.0;

        if rate < MIN_RATE || rate > MAX_RATE {
            Err(BspError::InvalidSamplingRate {
                rate,
                valid_range: format!("{}-{}Hz", MIN_RATE, MAX_RATE),
            })
        } else {
            Ok(())
        }
    }

    /// Validate channel count for EMG signals
    pub fn validate_channel_count(count: usize) -> BspResult<()> {
        const MAX_CHANNELS: usize = 16;

        if count == 0 || count > MAX_CHANNELS {
            Err(BspError::InvalidChannelCount {
                count,
                max: MAX_CHANNELS,
            })
        } else {
            Ok(())
        }
    }

    /// Get expected number of samples for this signal
    pub fn expected_samples(&self) -> usize {
        (self.sampling_rate * self.duration) as usize * self.channel_count
    }
}

impl Default for EMGMetadata {
    fn default() -> Self {
        EMGMetadata {
            signal_type: EMGSignal::Surface {
                muscle_group: MuscleGroup::Biceps,
                activation_level: 0.5,
            },
            sampling_rate: 1000.0,
            channel_count: 1,
            duration: 1.0,
            noise_level: 0.1,
            timestamp: 0,
        }
    }
}

impl std::fmt::Display for MuscleGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MuscleGroup::Biceps => write!(f, "Biceps"),
            MuscleGroup::Triceps => write!(f, "Triceps"),
            MuscleGroup::Forearm => write!(f, "Forearm"),
            MuscleGroup::Quadriceps => write!(f, "Quadriceps"),
            MuscleGroup::Hamstring => write!(f, "Hamstring"),
            MuscleGroup::Calf => write!(f, "Calf"),
            MuscleGroup::Other(id) => write!(f, "Other({})", id),
        }
    }
}

impl std::fmt::Display for EMGSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EMGSignal::Surface { muscle_group, activation_level } => {
                write!(f, "Surface EMG - {} ({}%)", muscle_group, (activation_level * 100.0) as u8)
            }
            EMGSignal::Intramuscular { muscle_group } => {
                write!(f, "Intramuscular EMG - {}", muscle_group)
            }
        }
    }
}