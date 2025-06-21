//! SignalEntity: Core container for EMG signal data

use crate::emg_types::EMGMetadata;
use crate::error::{BspError, BspResult};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Universal container for EMG signal data
#[derive(Debug, Clone)]
pub struct SignalEntity {
    /// Unique identifier for this signal entity
    pub id: Uuid,
    /// EMG signal data (interleaved channels)
    pub data: Vec<f32>,
    /// Signal metadata
    pub metadata: EMGMetadata,
    /// Creation timestamp
    pub created_at: u64,
}

impl SignalEntity {
    /// Create new signal entity with data and metadata
    pub fn new(data: Vec<f32>, metadata: EMGMetadata) -> BspResult<Self> {
        // Validate data length matches metadata expectations
        let expected_samples = metadata.expected_samples();
        if data.len() != expected_samples {
            return Err(BspError::InvalidSignalData {
                reason: format!(
                    "Data length {} doesn't match expected {} samples",
                    data.len(),
                    expected_samples
                ),
            });
        }

        Ok(SignalEntity {
            id: Uuid::new_v4(),
            data,
            metadata,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        })
    }

    /// Get total number of samples across all channels
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if entity is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get number of samples per channel
    pub fn samples_per_channel(&self) -> usize {
        if self.metadata.channel_count == 0 {
            0
        } else {
            self.data.len() / self.metadata.channel_count
        }
    }

    /// Get data for a specific channel
    pub fn channel_data(&self, channel_index: usize) -> BspResult<Vec<f32>> {
        if channel_index >= self.metadata.channel_count {
            return Err(BspError::InvalidSignalData {
                reason: format!(
                    "Channel index {} out of bounds (0-{})",
                    channel_index,
                    self.metadata.channel_count - 1
                ),
            });
        }

        let samples_per_channel = self.samples_per_channel();
        let mut channel_data = Vec::with_capacity(samples_per_channel);

        // Extract interleaved channel data
        for sample_idx in 0..samples_per_channel {
            let data_idx = sample_idx * self.metadata.channel_count + channel_index;
            channel_data.push(self.data[data_idx]);
        }

        Ok(channel_data)
    }

    /// Get all channel data as separate vectors
    pub fn all_channels(&self) -> BspResult<Vec<Vec<f32>>> {
        let mut channels = Vec::with_capacity(self.metadata.channel_count);

        for ch in 0..self.metadata.channel_count {
            channels.push(self.channel_data(ch)?);
        }

        Ok(channels)
    }

    /// Get signal duration in seconds
    pub fn duration(&self) -> f32 {
        self.metadata.duration
    }

    /// Get sampling rate
    pub fn sampling_rate(&self) -> f32 {
        self.metadata.sampling_rate
    }

    /// Get channel count
    pub fn channel_count(&self) -> usize {
        self.metadata.channel_count
    }

    /// Get time vector for plotting
    pub fn time_vector(&self) -> Vec<f32> {
        let samples = self.samples_per_channel();
        let dt = 1.0 / self.metadata.sampling_rate;

        (0..samples)
            .map(|i| i as f32 * dt)
            .collect()
    }

    /// Calculate basic statistics for a channel
    pub fn channel_stats(&self, channel_index: usize) -> BspResult<ChannelStats> {
        let data = self.channel_data(channel_index)?;
        Ok(ChannelStats::calculate(&data))
    }

    /// Slice the signal entity to a time range
    pub fn slice_time(&self, start_time: f32, end_time: f32) -> BspResult<SignalEntity> {
        if start_time < 0.0 || end_time > self.duration() || start_time >= end_time {
            return Err(BspError::InvalidSignalData {
                reason: format!(
                    "Invalid time range [{:.3}, {:.3}]s for signal duration {:.3}s",
                    start_time, end_time, self.duration()
                ),
            });
        }

        let start_sample = (start_time * self.metadata.sampling_rate) as usize;
        let end_sample = (end_time * self.metadata.sampling_rate) as usize;

        let samples_per_channel = end_sample - start_sample;
        let mut sliced_data = Vec::with_capacity(samples_per_channel * self.metadata.channel_count);

        // Extract sliced data maintaining channel interleaving
        for sample_idx in start_sample..end_sample {
            for ch in 0..self.metadata.channel_count {
                let data_idx = sample_idx * self.metadata.channel_count + ch;
                sliced_data.push(self.data[data_idx]);
            }
        }

        let mut new_metadata = self.metadata.clone();
        new_metadata.duration = end_time - start_time;

        SignalEntity::new(sliced_data, new_metadata)
    }
}

/// Basic statistics for a signal channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelStats {
    pub mean: f32,
    pub rms: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub peak_to_peak: f32,
}

impl ChannelStats {
    pub fn calculate(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self {
                mean: 0.0,
                rms: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                peak_to_peak: 0.0,
            };
        }

        let sum: f32 = data.iter().sum();
        let mean = sum / data.len() as f32;

        let sum_sq: f32 = data.iter().map(|x| x * x).sum();
        let rms = (sum_sq / data.len() as f32).sqrt();

        let variance: f32 = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / data.len() as f32;
        let std_dev = variance.sqrt();

        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let peak_to_peak = max - min;

        Self {
            mean,
            rms,
            std_dev,
            min,
            max,
            peak_to_peak,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emg_types::{EMGSignal, MuscleGroup};

    #[test]
    fn test_signal_entity_creation() {
        let metadata = EMGMetadata::new(
            EMGSignal::Surface {
                muscle_group: MuscleGroup::Biceps,
                activation_level: 0.5,
            },
            1000.0,
            1,
            1.0,
            0.1,
        ).unwrap();

        let data = vec![0.0; 1000]; // 1 second of data
        let entity = SignalEntity::new(data, metadata).unwrap();

        assert_eq!(entity.len(), 1000);
        assert_eq!(entity.samples_per_channel(), 1000);
        assert_eq!(entity.channel_count(), 1);
    }

    #[test]
    fn test_multichannel_entity() {
        let metadata = EMGMetadata::new(
            EMGSignal::Surface {
                muscle_group: MuscleGroup::Biceps,
                activation_level: 0.5,
            },
            1000.0,
            2, // 2 channels
            1.0,
            0.1,
        ).unwrap();

        // Interleaved data: [ch0_sample0, ch1_sample0, ch0_sample1, ch1_sample1, ...]
        let data = (0..2000).map(|i| i as f32).collect();
        let entity = SignalEntity::new(data, metadata).unwrap();

        assert_eq!(entity.len(), 2000);
        assert_eq!(entity.samples_per_channel(), 1000);
        assert_eq!(entity.channel_count(), 2);

        // Test channel extraction
        let ch0_data = entity.channel_data(0).unwrap();
        let ch1_data = entity.channel_data(1).unwrap();

        assert_eq!(ch0_data.len(), 1000);
        assert_eq!(ch1_data.len(), 1000);

        // Check interleaving
        assert_eq!(ch0_data[0], 0.0);
        assert_eq!(ch1_data[0], 1.0);
        assert_eq!(ch0_data[1], 2.0);
        assert_eq!(ch1_data[1], 3.0);
    }
}