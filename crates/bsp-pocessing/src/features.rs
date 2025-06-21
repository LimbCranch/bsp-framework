//! Feature extraction for biosignal analysis

use crate::processor::{SignalProcessor, ProcessorConfig, ProcessorType, ProcessingMetrics};
use bsp_core::{SignalEntity, BspResult, BspError};
use serde::{Deserialize, Serialize};
use rustfft::{FftPlanner, num_complex::Complex};
use std::collections::HashMap;

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Window size for feature calculation (in samples)
    pub window_size: usize,
    /// Window overlap (0.0 to 1.0)
    pub window_overlap: f32,
    /// Enabled feature types
    pub enabled_features: Vec<FeatureType>,
    /// Frequency bands for spectral features
    pub frequency_bands: Vec<FrequencyBand>,
}

/// Types of features that can be extracted
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureType {
    // Time domain features
    Mean,
    RMS,
    MAV,           // Mean Absolute Value
    ZeroCrossings,
    WaveformLength,
    SlopeSignChanges,

    // Frequency domain features
    MeanFrequency,
    MedianFrequency,
    PeakFrequency,
    SpectralCentroid,
    SpectralSpread,
    SpectralEntropy,

    // Time-frequency features
    InstantaneousFrequency,
    SpectralEdgeFrequency,

    // Complexity features
    SampleEntropy,
    ApproximateEntropy,
    LempelZivComplexity,
}

/// Frequency band definition for spectral analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyBand {
    pub name: String,
    pub low_freq: f32,
    pub high_freq: f32,
}

impl FrequencyBand {
    /// Create EMG frequency bands
    pub fn emg_bands() -> Vec<FrequencyBand> {
        vec![
            FrequencyBand { name: "Low".to_string(), low_freq: 10.0, high_freq: 60.0 },
            FrequencyBand { name: "Mid".to_string(), low_freq: 60.0, high_freq: 150.0 },
            FrequencyBand { name: "High".to_string(), low_freq: 150.0, high_freq: 300.0 },
        ]
    }
}

/// Container for extracted features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSet {
    /// Time domain features
    pub time_features: TimeFeatures,
    /// Frequency domain features  
    pub frequency_features: FrequencyFeatures,
    /// Statistical features
    pub statistical_features: StatisticalFeatures,
    /// Feature extraction timestamp
    pub timestamp: u64,
    /// Window information
    pub window_info: WindowInfo,
}

/// Time domain features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeFeatures {
    pub mean: f32,
    pub rms: f32,
    pub mav: f32,
    pub zero_crossings: u32,
    pub waveform_length: f32,
    pub slope_sign_changes: u32,
    pub variance: f32,
    pub std_dev: f32,
    pub min_value: f32,
    pub max_value: f32,
    pub peak_to_peak: f32,
}

/// Frequency domain features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyFeatures {
    pub mean_frequency: f32,
    pub median_frequency: f32,
    pub peak_frequency: f32,
    pub spectral_centroid: f32,
    pub spectral_spread: f32,
    pub spectral_entropy: f32,
    pub total_power: f32,
    pub band_powers: HashMap<String, f32>,
    pub peak_magnitude: f32,
}

/// Statistical features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalFeatures {
    pub skewness: f32,
    pub kurtosis: f32,
    pub entropy: f32,
    pub energy: f32,
    pub power: f32,
}

/// Window information for feature context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowInfo {
    pub start_sample: usize,
    pub end_sample: usize,
    pub window_size: usize,
    pub sampling_rate: f32,
    pub channel: usize,
}

/// Feature extractor implementation
pub struct FeatureExtractor {
    config: ProcessorConfig,
    feature_config: FeatureConfig,
    fft_planner: FftPlanner<f32>,
    window_buffer: Vec<Vec<f32>>, // Buffer for each channel
    sample_count: usize,
}

impl FeatureExtractor {
    /// Create new feature extractor
    pub fn new(feature_config: FeatureConfig) -> Self {
        let mut config = ProcessorConfig::new("feature_extractor", ProcessorType::FeatureExtractor);
        config.set_parameter("window_size", (feature_config.window_size as i32).into());
        config.set_parameter("window_overlap", feature_config.window_overlap.into());

        FeatureExtractor {
            config,
            feature_config,
            fft_planner: FftPlanner::new(),
            window_buffer: Vec::new(),
            sample_count: 0,
        }
    }

    /// Create EMG feature extractor with typical configuration
    pub fn emg_features(window_size: usize) -> Self {
        let feature_config = FeatureConfig {
            window_size,
            window_overlap: 0.5,
            enabled_features: vec![
                FeatureType::Mean,
                FeatureType::RMS,
                FeatureType::MAV,
                FeatureType::ZeroCrossings,
                FeatureType::WaveformLength,
                FeatureType::MeanFrequency,
                FeatureType::MedianFrequency,
                FeatureType::SpectralCentroid,
            ],
            frequency_bands: FrequencyBand::emg_bands(),
        };

        Self::new(feature_config)
    }

    /// Initialize buffers for channels
    fn initialize(&mut self, channel_count: usize) {
        self.window_buffer = vec![Vec::with_capacity(self.feature_config.window_size); channel_count];
    }

    /// Extract features from a signal window
    pub fn extract_features(&mut self, window_data: &[f32], sampling_rate: f32, channel: usize) -> FeatureSet {
        let time_features = self.extract_time_features(window_data);
        let frequency_features = self.extract_frequency_features(window_data, sampling_rate);
        let statistical_features = self.extract_statistical_features(window_data);

        FeatureSet {
            time_features,
            frequency_features,
            statistical_features,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            window_info: WindowInfo {
                start_sample: self.sample_count,
                end_sample: self.sample_count + window_data.len(),
                window_size: window_data.len(),
                sampling_rate,
                channel,
            },
        }
    }

    /// Extract time domain features
    fn extract_time_features(&self, data: &[f32]) -> TimeFeatures {
        if data.is_empty() {
            return TimeFeatures::default();
        }

        let n = data.len() as f32;

        // Basic statistics
        let sum: f32 = data.iter().sum();
        let mean = sum / n;

        let sum_sq: f32 = data.iter().map(|x| x * x).sum();
        let rms = (sum_sq / n).sqrt();

        let mav = data.iter().map(|x| x.abs()).sum::<f32>() / n;

        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt();

        let min_value = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_value = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let peak_to_peak = max_value - min_value;

        // Zero crossings
        let mut zero_crossings = 0u32;
        for i in 1..data.len() {
            if (data[i] >= 0.0 && data[i-1] < 0.0) || (data[i] < 0.0 && data[i-1] >= 0.0) {
                zero_crossings += 1;
            }
        }

        // Waveform length
        let waveform_length = data.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f32>();

        // Slope sign changes
        let mut slope_sign_changes = 0u32;
        if data.len() > 2 {
            let mut prev_slope_positive = data[1] > data[0];
            for i in 2..data.len() {
                let current_slope_positive = data[i] > data[i-1];
                if current_slope_positive != prev_slope_positive {
                    slope_sign_changes += 1;
                }
                prev_slope_positive = current_slope_positive;
            }
        }

        TimeFeatures {
            mean,
            rms,
            mav,
            zero_crossings,
            waveform_length,
            slope_sign_changes,
            variance,
            std_dev,
            min_value,
            max_value,
            peak_to_peak,
        }
    }

    /// Extract frequency domain features
    fn extract_frequency_features(&mut self, data: &[f32], sampling_rate: f32) -> FrequencyFeatures {
        if data.len() < 4 {
            return FrequencyFeatures::default();
        }

        // Prepare FFT
        let fft_size = data.len().next_power_of_two();
        let mut fft = self.fft_planner.plan_fft_forward(fft_size);

        // Prepare input data (zero-padded)
        let mut fft_input: Vec<Complex<f32>> = data.iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        fft_input.resize(fft_size, Complex::new(0.0, 0.0));

        // Compute FFT
        fft.process(&mut fft_input);

        // Calculate magnitude spectrum (only positive frequencies)
        let magnitude_spectrum: Vec<f32> = fft_input[0..fft_size/2]
            .iter()
            .map(|c| c.norm())
            .collect();

        // Calculate power spectrum
        let power_spectrum: Vec<f32> = magnitude_spectrum.iter()
            .map(|m| m * m)
            .collect();

        let total_power: f32 = power_spectrum.iter().sum();

        if total_power == 0.0 {
            return FrequencyFeatures::default();
        }

        // Frequency resolution
        let freq_resolution = sampling_rate / fft_size as f32;

        // Find peak frequency
        let peak_idx = magnitude_spectrum.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let peak_frequency = peak_idx as f32 * freq_resolution;
        let peak_magnitude = magnitude_spectrum[peak_idx];

        // Calculate spectral centroid (weighted mean frequency)
        let mut weighted_sum = 0.0;
        for (i, &power) in power_spectrum.iter().enumerate() {
            let frequency = i as f32 * freq_resolution;
            weighted_sum += frequency * power;
        }
        let spectral_centroid = weighted_sum / total_power;

        // Calculate spectral spread (weighted standard deviation)
        let mut variance_sum = 0.0;
        for (i, &power) in power_spectrum.iter().enumerate() {
            let frequency = i as f32 * freq_resolution;
            variance_sum += (frequency - spectral_centroid).powi(2) * power;
        }
        let spectral_spread = (variance_sum / total_power).sqrt();

        // Calculate mean frequency
        let mean_frequency = {
            let mut sum_freq = 0.0;
            let mut sum_power = 0.0;
            for (i, &power) in power_spectrum.iter().enumerate() {
                if i > 0 { // Skip DC component
                    let frequency = i as f32 * freq_resolution;
                    sum_freq += frequency * power;
                    sum_power += power;
                }
            }
            if sum_power > 0.0 { sum_freq / sum_power } else { 0.0 }
        };

        // Calculate median frequency
        let median_frequency = {
            let half_power = total_power / 2.0;
            let mut cumulative_power = 0.0;
            let mut median_freq = 0.0;

            for (i, &power) in power_spectrum.iter().enumerate() {
                cumulative_power += power;
                if cumulative_power >= half_power {
                    median_freq = i as f32 * freq_resolution;
                    break;
                }
            }
            median_freq
        };

        // Calculate spectral entropy
        let spectral_entropy = {
            let mut entropy = 0.0;
            for &power in &power_spectrum {
                if power > 0.0 {
                    let prob = power / total_power;
                    entropy -= prob * prob.log2();
                }
            }
            entropy
        };

        // Calculate band powers
        let mut band_powers = HashMap::new();
        for band in &self.feature_config.frequency_bands {
            let low_bin = (band.low_freq / freq_resolution) as usize;
            let high_bin = ((band.high_freq / freq_resolution) as usize).min(power_spectrum.len() - 1);

            let band_power: f32 = power_spectrum[low_bin..=high_bin].iter().sum();
            band_powers.insert(band.name.clone(), band_power);
        }

        FrequencyFeatures {
            mean_frequency,
            median_frequency,
            peak_frequency,
            spectral_centroid,
            spectral_spread,
            spectral_entropy,
            total_power,
            band_powers,
            peak_magnitude,
        }
    }

    /// Extract statistical features
    fn extract_statistical_features(&self, data: &[f32]) -> StatisticalFeatures {
        if data.len() < 2 {
            return StatisticalFeatures::default();
        }

        let n = data.len() as f32;
        let mean = data.iter().sum::<f32>() / n;

        // Calculate moments
        let mut m2 = 0.0; // Second moment (variance)
        let mut m3 = 0.0; // Third moment
        let mut m4 = 0.0; // Fourth moment

        for &x in data {
            let diff = x - mean;
            let diff2 = diff * diff;
            let diff3 = diff2 * diff;
            let diff4 = diff3 * diff;

            m2 += diff2;
            m3 += diff3;
            m4 += diff4;
        }

        m2 /= n;
        m3 /= n;
        m4 /= n;

        let std_dev = m2.sqrt();

        // Skewness
        let skewness = if std_dev > 0.0 {
            m3 / (std_dev.powi(3))
        } else {
            0.0
        };

        // Kurtosis
        let kurtosis = if std_dev > 0.0 {
            m4 / (std_dev.powi(4)) - 3.0 // Excess kurtosis
        } else {
            0.0
        };

        // Shannon entropy (approximate)
        let entropy = {
            // Simple histogram-based entropy
            let mut hist = vec![0u32; 20]; // 20 bins
            let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let range = max_val - min_val;

            if range > 0.0 {
                for &x in data {
                    let bin = ((x - min_val) / range * 19.0) as usize;
                    let bin = bin.min(19);
                    hist[bin] += 1;
                }

                let mut entropy = 0.0;
                for &count in &hist {
                    if count > 0 {
                        let prob = count as f32 / n;
                        entropy -= prob * prob.log2();
                    }
                }
                entropy
            } else {
                0.0
            }
        };

        // Energy and power
        let energy = data.iter().map(|x| x * x).sum::<f32>();
        let power = energy / n;

        StatisticalFeatures {
            skewness,
            kurtosis,
            entropy,
            energy,
            power,
        }
    }
}

impl SignalProcessor for FeatureExtractor {
    fn process(&mut self, input: &SignalEntity) -> BspResult<SignalEntity> {
        let mut timer = ProcessingMetrics::start_timing();

        // Initialize if needed
        if self.window_buffer.len() != input.channel_count() {
            self.initialize(input.channel_count());
        }

        let channels = input.all_channels()?;
        let mut all_features = Vec::new();

        // Process each channel
        for (channel_idx, channel_data) in channels.iter().enumerate() {
            // Add samples to window buffer
            for &sample in channel_data {
                self.window_buffer[channel_idx].push(sample);

                // Check if window is full
                if self.window_buffer[channel_idx].len() >= self.feature_config.window_size {
                    let window_data = &self.window_buffer[channel_idx];
                    
                    // Extract features from window
                    let  features = self.extract_features(
                        &window_data.clone(),
                        input.sampling_rate(),
                        channel_idx,
                    );

                    all_features.push(features);

                    // Slide window (considering overlap)
                    let step_size = (self.feature_config.window_size as f32 *
                        (1.0 - self.feature_config.window_overlap)) as usize;
                    let step_size = step_size.max(1);

                    // Remove old samples
                    if step_size < self.window_buffer[channel_idx].len() {
                        self.window_buffer[channel_idx].drain(0..step_size);
                    } else {
                        self.window_buffer[channel_idx].clear();
                    }
                }
            }
        }

        self.sample_count += input.samples_per_channel();

        // For now, return the original signal (features would be used separately)
        // In a real implementation, you might create a new signal type for features
        timer.set_output_quality(1.0);
        let metrics = timer.finish();

        Ok(input.clone())
    }

    fn config(&self) -> &ProcessorConfig {
        &self.config
    }

    fn update_config(&mut self, config: ProcessorConfig) -> BspResult<()> {
        self.config = config;
        Ok(())
    }

    fn name(&self) -> &str {
        "Feature Extractor"
    }

    fn reset(&mut self) {
        for buffer in &mut self.window_buffer {
            buffer.clear();
        }
        self.sample_count = 0;
    }

    fn latency_estimate(&self) -> u64 {
        // Latency depends on window size
        (self.feature_config.window_size as u64 * 1000) // microseconds at 1kHz
    }

    fn processor_type(&self) -> ProcessorType {
        ProcessorType::FeatureExtractor
    }
}

// Default implementations
impl Default for TimeFeatures {
    fn default() -> Self {
        Self {
            mean: 0.0,
            rms: 0.0,
            mav: 0.0,
            zero_crossings: 0,
            waveform_length: 0.0,
            slope_sign_changes: 0,
            variance: 0.0,
            std_dev: 0.0,
            min_value: 0.0,
            max_value: 0.0,
            peak_to_peak: 0.0,
        }
    }
}

impl Default for FrequencyFeatures {
    fn default() -> Self {
        Self {
            mean_frequency: 0.0,
            median_frequency: 0.0,
            peak_frequency: 0.0,
            spectral_centroid: 0.0,
            spectral_spread: 0.0,
            spectral_entropy: 0.0,
            total_power: 0.0,
            band_powers: HashMap::new(),
            peak_magnitude: 0.0,
        }
    }
}

impl Default for StatisticalFeatures {
    fn default() -> Self {
        Self {
            skewness: 0.0,
            kurtosis: 0.0,
            entropy: 0.0,
            energy: 0.0,
            power: 0.0,
        }
    }
}

impl FeatureSet {
    /// Get a summary of key features for quick analysis
    pub fn summary(&self) -> FeatureSummary {
        FeatureSummary {
            rms: self.time_features.rms,
            mav: self.time_features.mav,
            zero_crossings: self.time_features.zero_crossings,
            mean_frequency: self.frequency_features.mean_frequency,
            spectral_centroid: self.frequency_features.spectral_centroid,
            total_power: self.frequency_features.total_power,
            window_size: self.window_info.window_size,
            channel: self.window_info.channel,
        }
    }
}

/// Compact feature summary for real-time display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSummary {
    pub rms: f32,
    pub mav: f32,
    pub zero_crossings: u32,
    pub mean_frequency: f32,
    pub spectral_centroid: f32,
    pub total_power: f32,
    pub window_size: usize,
    pub channel: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use bsp_core::{EMGMetadata, EMGSignal, MuscleGroup};

    #[test]
    fn test_time_features() {
        let mut extractor = FeatureExtractor::emg_features(256);

        // Create test signal (sine wave)
        let test_data: Vec<f32> = (0..256)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / 64.0).sin())
            .collect();

        let features = extractor.extract_time_features(&test_data);

        // For a sine wave, mean should be close to 0
        assert!((features.mean).abs() < 0.1);

        // RMS should be approximately 1/sqrt(2) ≈ 0.707
        assert!((features.rms - 0.707).abs() < 0.1);

        // Zero crossings should be present
        assert!(features.zero_crossings > 0);
    }

    #[test]
    fn test_feature_extractor_process() {
        let mut extractor = FeatureExtractor::emg_features(128);

        let test_data = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let metadata = EMGMetadata::new(
            EMGSignal::Surface {
                muscle_group: MuscleGroup::Biceps,
                activation_level: 0.5,
            },
            1000.0, 1, 0.256, 0.1
        ).unwrap();

        let input_signal = SignalEntity::new(test_data, metadata).unwrap();
        let output_signal = extractor.process(&input_signal).unwrap();

        // Output should have same structure as input
        assert_eq!(output_signal.len(), input_signal.len());
        assert_eq!(output_signal.channel_count(), input_signal.channel_count());
    }

    #[test]
    fn test_frequency_features() {
        let mut extractor = FeatureExtractor::emg_features(256);

        // Create test signal with known frequency content
        let test_data: Vec<f32> = (0..256)
            .map(|i| {
                let t = i as f32 / 256.0;
                (2.0 * std::f32::consts::PI * 10.0 * t).sin() + // 10 Hz
                    0.5 * (2.0 * std::f32::consts::PI * 50.0 * t).sin() // 50 Hz
            })
            .collect();

        let features = extractor.extract_frequency_features(&test_data, 256.0);

        // Should detect frequency content
        assert!(features.total_power > 0.0);
        assert!(features.peak_frequency > 0.0);
        assert!(features.mean_frequency > 0.0);
    }

    #[test]
    fn test_feature_config() {
        let config = FeatureConfig {
            window_size: 256,
            window_overlap: 0.5,
            enabled_features: vec![FeatureType::RMS, FeatureType::MAV],
            frequency_bands: FrequencyBand::emg_bands(),
        };

        let extractor = FeatureExtractor::new(config);
        assert_eq!(extractor.feature_config.window_size, 256);
        assert_eq!(extractor.feature_config.enabled_features.len(), 2);
    }

    #[test]
    fn test_statistical_features() {
        let extractor = FeatureExtractor::emg_features(256);

        // Create test data with known statistical properties
        let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Simple ascending sequence

        let features = extractor.extract_statistical_features(&test_data);

        // Basic sanity checks
        assert!(features.energy > 0.0);
        assert!(features.power > 0.0);
        assert!(!features.skewness.is_nan());
        assert!(!features.kurtosis.is_nan());
    }
}