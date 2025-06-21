//! Digital filters for biosignal processing

use crate::processor::{SignalProcessor, ProcessorConfig, ProcessorType, ProcessingMetrics, ParameterValue};
use bsp_core::{SignalEntity, BspResult, BspError, EMGMetadata};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Filter types supported by the framework
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterType {
    /// Butterworth lowpass filter
    ButterworthLowpass,
    /// Butterworth highpass filter
    ButterworthHighpass,
    /// Butterworth bandpass filter
    ButterworthBandpass,
    /// Notch filter for powerline interference
    Notch,
    /// Simple moving average filter
    MovingAverage,
    /// Median filter for artifact removal
    Median,
}

/// Filter configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    /// Filter type
    pub filter_type: FilterType,
    /// Filter order (for IIR filters)
    pub order: usize,
    /// Cutoff frequency for lowpass/highpass (Hz)
    pub cutoff_freq: Option<f32>,
    /// Low cutoff for bandpass (Hz)
    pub low_cutoff: Option<f32>,
    /// High cutoff for bandpass (Hz)
    pub high_cutoff: Option<f32>,
    /// Notch frequency (Hz) - typically 50 or 60
    pub notch_freq: Option<f32>,
    /// Notch quality factor
    pub notch_q: Option<f32>,
    /// Window size for moving average/median
    pub window_size: Option<usize>,
}

impl FilterConfig {
    /// Create lowpass filter configuration
    pub fn lowpass(cutoff_freq: f32, order: usize) -> Self {
        Self {
            filter_type: FilterType::ButterworthLowpass,
            order,
            cutoff_freq: Some(cutoff_freq),
            low_cutoff: None,
            high_cutoff: None,
            notch_freq: None,
            notch_q: None,
            window_size: None,
        }
    }

    /// Create highpass filter configuration
    pub fn highpass(cutoff_freq: f32, order: usize) -> Self {
        Self {
            filter_type: FilterType::ButterworthHighpass,
            order,
            cutoff_freq: Some(cutoff_freq),
            low_cutoff: None,
            high_cutoff: None,
            notch_freq: None,
            notch_q: None,
            window_size: None,
        }
    }

    /// Create bandpass filter configuration
    pub fn bandpass(low_cutoff: f32, high_cutoff: f32, order: usize) -> Self {
        Self {
            filter_type: FilterType::ButterworthBandpass,
            order,
            cutoff_freq: None,
            low_cutoff: Some(low_cutoff),
            high_cutoff: Some(high_cutoff),
            notch_freq: None,
            notch_q: None,
            window_size: None,
        }
    }

    /// Create notch filter configuration
    pub fn notch(freq: f32, q: f32) -> Self {
        Self {
            filter_type: FilterType::Notch,
            order: 2,
            cutoff_freq: None,
            low_cutoff: None,
            high_cutoff: None,
            notch_freq: Some(freq),
            notch_q: Some(q),
            window_size: None,
        }
    }

    /// Create moving average filter configuration
    pub fn moving_average(window_size: usize) -> Self {
        Self {
            filter_type: FilterType::MovingAverage,
            order: 0,
            cutoff_freq: None,
            low_cutoff: None,
            high_cutoff: None,
            notch_freq: None,
            notch_q: None,
            window_size: Some(window_size),
        }
    }
}

/// Butterworth filter implementation
pub struct ButterworthFilter {
    config: ProcessorConfig,
    filter_config: FilterConfig,
    // Filter coefficients
    b_coeffs: Vec<f32>, // Numerator coefficients
    a_coeffs: Vec<f32>, // Denominator coefficients
    // Filter state for each channel
    input_history: Vec<VecDeque<f32>>,  // x[n-1], x[n-2], ...
    output_history: Vec<VecDeque<f32>>, // y[n-1], y[n-2], ...
    sampling_rate: f32,
    initialized: bool,
}

impl ButterworthFilter {
    /// Create new Butterworth filter
    pub fn new(filter_config: FilterConfig) -> BspResult<Self> {
        let mut config = ProcessorConfig::new("butterworth", ProcessorType::Filter);
        config.set_parameter("filter_type", format!("{:?}", filter_config.filter_type).into());
        config.set_parameter("order", (filter_config.order as i32).into());

        if let Some(freq) = filter_config.cutoff_freq {
            config.set_parameter("cutoff_freq", freq.into());
        }
        if let Some(low) = filter_config.low_cutoff {
            config.set_parameter("low_cutoff", low.into());
        }
        if let Some(high) = filter_config.high_cutoff {
            config.set_parameter("high_cutoff", high.into());
        }

        Ok(ButterworthFilter {
            config,
            filter_config,
            b_coeffs: Vec::new(),
            a_coeffs: Vec::new(),
            input_history: Vec::new(),
            output_history: Vec::new(),
            sampling_rate: 0.0,
            initialized: false,
        })
    }

    /// Initialize filter coefficients for given sampling rate
    fn initialize(&mut self, sampling_rate: f32, channel_count: usize) -> BspResult<()> {
        self.sampling_rate = sampling_rate;

        // Calculate filter coefficients based on type
        match self.filter_config.filter_type {
            FilterType::ButterworthLowpass => {
                if let Some(cutoff) = self.filter_config.cutoff_freq {
                    self.calculate_lowpass_coefficients(cutoff, sampling_rate)?;
                } else {
                    return Err(BspError::ConfigurationError {
                        message: "Lowpass filter requires cutoff frequency".to_string(),
                    });
                }
            },
            FilterType::ButterworthHighpass => {
                if let Some(cutoff) = self.filter_config.cutoff_freq {
                    self.calculate_highpass_coefficients(cutoff, sampling_rate)?;
                } else {
                    return Err(BspError::ConfigurationError {
                        message: "Highpass filter requires cutoff frequency".to_string(),
                    });
                }
            },
            FilterType::ButterworthBandpass => {
                if let (Some(low), Some(high)) = (self.filter_config.low_cutoff, self.filter_config.high_cutoff) {
                    self.calculate_bandpass_coefficients(low, high, sampling_rate)?;
                } else {
                    return Err(BspError::ConfigurationError {
                        message: "Bandpass filter requires low and high cutoff frequencies".to_string(),
                    });
                }
            },
            _ => {
                return Err(BspError::ConfigurationError {
                    message: "Unsupported filter type for Butterworth filter".to_string(),
                });
            }
        }

        // Initialize history buffers for each channel
        let history_length = self.a_coeffs.len().max(self.b_coeffs.len());
        self.input_history = vec![VecDeque::with_capacity(history_length); channel_count];
        self.output_history = vec![VecDeque::with_capacity(history_length); channel_count];

        self.initialized = true;
        Ok(())
    }

    /// Calculate coefficients for lowpass Butterworth filter
    fn calculate_lowpass_coefficients(&mut self, cutoff: f32, fs: f32) -> BspResult<()> {
        // Normalized cutoff frequency (0 to 1, where 1 is Nyquist)
        let wn = cutoff / (fs / 2.0);

        if wn >= 1.0 {
            return Err(BspError::ConfigurationError {
                message: "Cutoff frequency must be less than Nyquist frequency".to_string(),
            });
        }

        // For simplicity, implement 2nd order Butterworth filter
        // In a full implementation, you'd use proper filter design algorithms
        let order = 2.min(self.filter_config.order);

        // Pre-warp frequency for bilinear transform
        let wd = 2.0 * (std::f32::consts::PI * wn / 2.0).tan();
        let wa = wd;

        // Butterworth polynomial roots for 2nd order
        let k = wa * wa;
        let a0 = 4.0 + 2.0 * std::f32::consts::SQRT_2 * wa + k;

        // Coefficients (normalized)
        self.b_coeffs = vec![k / a0, 2.0 * k / a0, k / a0];
        self.a_coeffs = vec![1.0, (2.0 * k - 8.0) / a0, (4.0 - 2.0 * std::f32::consts::SQRT_2 * wa + k) / a0];

        Ok(())
    }

    /// Calculate coefficients for highpass Butterworth filter
    fn calculate_highpass_coefficients(&mut self, cutoff: f32, fs: f32) -> BspResult<()> {
        let wn = cutoff / (fs / 2.0);

        if wn >= 1.0 {
            return Err(BspError::ConfigurationError {
                message: "Cutoff frequency must be less than Nyquist frequency".to_string(),
            });
        }

        // Pre-warp frequency
        let wd = 2.0 * (std::f32::consts::PI * wn / 2.0).tan();
        let wa = wd;

        // Highpass transformation
        let k = wa * wa;
        let a0 = 4.0 + 2.0 * std::f32::consts::SQRT_2 * wa + k;

        // Highpass coefficients
        self.b_coeffs = vec![4.0 / a0, -8.0 / a0, 4.0 / a0];
        self.a_coeffs = vec![1.0, (2.0 * k - 8.0) / a0, (4.0 - 2.0 * std::f32::consts::SQRT_2 * wa + k) / a0];

        Ok(())
    }

    /// Calculate coefficients for bandpass Butterworth filter
    fn calculate_bandpass_coefficients(&mut self, low: f32, high: f32, fs: f32) -> BspResult<()> {
        if low >= high {
            return Err(BspError::ConfigurationError {
                message: "Low cutoff must be less than high cutoff".to_string(),
            });
        }

        if high >= fs / 2.0 {
            return Err(BspError::ConfigurationError {
                message: "High cutoff must be less than Nyquist frequency".to_string(),
            });
        }

        // Simplified bandpass: cascade of highpass and lowpass
        // For a proper implementation, use bandpass transform
        let center = (low * high).sqrt();
        let bandwidth = high - low;

        let wn = center / (fs / 2.0);
        let bw = bandwidth / (fs / 2.0);

        // Simplified 4th order bandpass coefficients
        // This is a placeholder - real implementation would use proper design
        let a0 = 1.0;
        self.b_coeffs = vec![bw * bw, 0.0, -2.0 * bw * bw, 0.0, bw * bw];
        self.a_coeffs = vec![1.0, -4.0 * wn, 6.0 * wn * wn, -4.0 * wn * wn * wn, wn * wn * wn * wn];

        // Normalize
        let norm = self.b_coeffs.iter().sum::<f32>();
        for coeff in &mut self.b_coeffs {
            *coeff /= norm;
        }

        Ok(())
    }

    /// Apply filter to a single sample for one channel
    fn filter_sample(&mut self, sample: f32, channel: usize) -> f32 {
        // Ensure history buffers have correct length
        while self.input_history[channel].len() < self.b_coeffs.len() {
            self.input_history[channel].push_back(0.0);
        }
        while self.output_history[channel].len() < self.a_coeffs.len() - 1 {
            self.output_history[channel].push_back(0.0);
        }

        // Add current input sample
        self.input_history[channel].push_back(sample);
        if self.input_history[channel].len() > self.b_coeffs.len() {
            self.input_history[channel].pop_front();
        }

        // Calculate output using difference equation
        let mut output = 0.0;

        // Numerator (input) terms
        for (i, &b_coeff) in self.b_coeffs.iter().enumerate() {
            if i < self.input_history[channel].len() {
                let idx = self.input_history[channel].len() - 1 - i;
                output -= b_coeff * self.output_history[channel][idx];
            }
        }

        // Add current output to history
        self.output_history[channel].push_back(output);
        if self.output_history[channel].len() > self.a_coeffs.len() - 1 {
            self.output_history[channel].pop_front();
        }

        output
    }
}

impl SignalProcessor for ButterworthFilter {
    fn process(&mut self, input: &SignalEntity) -> BspResult<SignalEntity> {
        let mut timer = ProcessingMetrics::start_timing();

        // Initialize filter if needed
        if !self.initialized || self.sampling_rate != input.sampling_rate() {
            self.initialize(input.sampling_rate(), input.channel_count())?;
        }

        // Process each channel
        let mut processed_data = Vec::with_capacity(input.len());
        let channels = input.all_channels()?;

        // Process sample by sample to maintain filter state
        let samples_per_channel = input.samples_per_channel();

        for sample_idx in 0..samples_per_channel {
            for (channel_idx, channel_data) in channels.iter().enumerate() {
                let input_sample = channel_data[sample_idx];
                let filtered_sample = self.filter_sample(input_sample, channel_idx);
                processed_data.push(filtered_sample);
            }
        }

        // Create output signal entity
        let output_signal = SignalEntity::new(processed_data, input.metadata.clone())?;

        timer.set_output_quality(0.95); // Assume good quality after filtering
        let metrics = timer.finish();

        Ok(output_signal)
    }

    fn config(&self) -> &ProcessorConfig {
        &self.config
    }

    fn update_config(&mut self, config: ProcessorConfig) -> BspResult<()> {
        self.config = config;
        self.initialized = false; // Force re-initialization
        Ok(())
    }

    fn name(&self) -> &str {
        "Butterworth Filter"
    }

    fn reset(&mut self) {
        self.input_history.clear();
        self.output_history.clear();
        self.initialized = false;
    }

    fn latency_estimate(&self) -> u64 {
        // Filter latency is typically very low
        100 // 0.1ms
    }
}

/// Notch filter for powerline interference removal
pub struct NotchFilter {
    config: ProcessorConfig,
    notch_freq: f32,
    q_factor: f32,
    // Biquad coefficients
    b0: f32, b1: f32, b2: f32,
    a1: f32, a2: f32,
    // State variables for each channel
    x1: Vec<f32>, x2: Vec<f32>,
    y1: Vec<f32>, y2: Vec<f32>,
    sampling_rate: f32,
    initialized: bool,
}

impl NotchFilter {
    /// Create new notch filter
    pub fn new(notch_freq: f32, q_factor: f32) -> Self {
        let mut config = ProcessorConfig::new("notch", ProcessorType::Filter);
        config.set_parameter("notch_freq", notch_freq.into());
        config.set_parameter("q_factor", q_factor.into());

        NotchFilter {
            config,
            notch_freq,
            q_factor,
            b0: 0.0, b1: 0.0, b2: 0.0,
            a1: 0.0, a2: 0.0,
            x1: Vec::new(), x2: Vec::new(),
            y1: Vec::new(), y2: Vec::new(),
            sampling_rate: 0.0,
            initialized: false,
        }
    }

    /// Initialize notch filter coefficients
    fn initialize(&mut self, sampling_rate: f32, channel_count: usize) -> BspResult<()> {
        self.sampling_rate = sampling_rate;

        // Calculate biquad coefficients for notch filter
        let omega = 2.0 * std::f32::consts::PI * self.notch_freq / sampling_rate;
        let alpha = omega.sin() / (2.0 * self.q_factor);

        let cos_omega = omega.cos();

        // Notch filter coefficients
        let a0 = 1.0 + alpha;
        self.b0 = 1.0 / a0;
        self.b1 = -2.0 * cos_omega / a0;
        self.b2 = 1.0 / a0;
        self.a1 = -2.0 * cos_omega / a0;
        self.a2 = (1.0 - alpha) / a0;

        // Initialize state variables for each channel
        self.x1 = vec![0.0; channel_count];
        self.x2 = vec![0.0; channel_count];
        self.y1 = vec![0.0; channel_count];
        self.y2 = vec![0.0; channel_count];

        self.initialized = true;
        Ok(())
    }

    /// Apply notch filter to a single sample
    fn filter_sample(&mut self, sample: f32, channel: usize) -> f32 {
        // Biquad difference equation
        let output = self.b0 * sample + self.b1 * self.x1[channel] + self.b2 * self.x2[channel]
            - self.a1 * self.y1[channel] - self.a2 * self.y2[channel];

        // Update state variables
        self.x2[channel] = self.x1[channel];
        self.x1[channel] = sample;
        self.y2[channel] = self.y1[channel];
        self.y1[channel] = output;

        output
    }
}

impl SignalProcessor for NotchFilter {
    fn process(&mut self, input: &SignalEntity) -> BspResult<SignalEntity> {
        let mut timer = ProcessingMetrics::start_timing();

        if !self.initialized || self.sampling_rate != input.sampling_rate() {
            self.initialize(input.sampling_rate(), input.channel_count())?;
        }

        let mut processed_data = Vec::with_capacity(input.len());
        let channels = input.all_channels()?;
        let samples_per_channel = input.samples_per_channel();

        for sample_idx in 0..samples_per_channel {
            for (channel_idx, channel_data) in channels.iter().enumerate() {
                let input_sample = channel_data[sample_idx];
                let filtered_sample = self.filter_sample(input_sample, channel_idx);
                processed_data.push(filtered_sample);
            }
        }

        let output_signal = SignalEntity::new(processed_data, input.metadata.clone())?;

        timer.set_output_quality(0.98); // Notch filter typically preserves signal well
        let metrics = timer.finish();

        Ok(output_signal)
    }

    fn config(&self) -> &ProcessorConfig {
        &self.config
    }

    fn update_config(&mut self, config: ProcessorConfig) -> BspResult<()> {
        // Update parameters from config
        if let Some(freq) = config.get_parameter("notch_freq") {
            if let Some(f) = freq.as_float() {
                self.notch_freq = f;
            }
        }
        if let Some(q) = config.get_parameter("q_factor") {
            if let Some(q_val) = q.as_float() {
                self.q_factor = q_val;
            }
        }

        self.config = config;
        self.initialized = false;
        Ok(())
    }

    fn name(&self) -> &str {
        "Notch Filter"
    }

    fn reset(&mut self) {
        self.x1.fill(0.0);
        self.x2.fill(0.0);
        self.y1.fill(0.0);
        self.y2.fill(0.0);
    }

    fn latency_estimate(&self) -> u64 {
        50 // Very low latency
    }
}

/// Moving average filter for noise reduction
pub struct MovingAverageFilter {
    config: ProcessorConfig,
    window_size: usize,
    // Circular buffers for each channel
    buffers: Vec<VecDeque<f32>>,
    sums: Vec<f32>,
    initialized: bool,
}

impl MovingAverageFilter {
    /// Create new moving average filter
    pub fn new(window_size: usize) -> Self {
        let mut config = ProcessorConfig::new("moving_average", ProcessorType::Filter);
        config.set_parameter("window_size", (window_size as i32).into());

        MovingAverageFilter {
            config,
            window_size,
            buffers: Vec::new(),
            sums: Vec::new(),
            initialized: false,
        }
    }

    /// Initialize buffers for channels
    fn initialize(&mut self, channel_count: usize) {
        self.buffers = vec![VecDeque::with_capacity(self.window_size); channel_count];
        self.sums = vec![0.0; channel_count];
        self.initialized = true;
    }

    /// Apply moving average to a single sample
    fn filter_sample(&mut self, sample: f32, channel: usize) -> f32 {
        // Add new sample to buffer and sum
        self.buffers[channel].push_back(sample);
        self.sums[channel] += sample;

        // Remove oldest sample if buffer is full
        if self.buffers[channel].len() > self.window_size {
            if let Some(old_sample) = self.buffers[channel].pop_front() {
                self.sums[channel] -= old_sample;
            }
        }

        // Return average
        self.sums[channel] / self.buffers[channel].len() as f32
    }
}

impl SignalProcessor for MovingAverageFilter {
    fn process(&mut self, input: &SignalEntity) -> BspResult<SignalEntity> {
        let mut timer = ProcessingMetrics::start_timing();

        if !self.initialized {
            self.initialize(input.channel_count());
        }

        let mut processed_data = Vec::with_capacity(input.len());
        let channels = input.all_channels()?;
        let samples_per_channel = input.samples_per_channel();

        for sample_idx in 0..samples_per_channel {
            for (channel_idx, channel_data) in channels.iter().enumerate() {
                let input_sample = channel_data[sample_idx];
                let filtered_sample = self.filter_sample(input_sample, channel_idx);
                processed_data.push(filtered_sample);
            }
        }

        let output_signal = SignalEntity::new(processed_data, input.metadata.clone())?;

        timer.set_output_quality(0.90); // Moving average reduces noise but may blur signal
        let metrics = timer.finish();

        Ok(output_signal)
    }

    fn config(&self) -> &ProcessorConfig {
        &self.config
    }

    fn update_config(&mut self, config: ProcessorConfig) -> BspResult<()> {
        if let Some(size) = config.get_parameter("window_size") {
            if let Some(s) = size.as_int() {
                self.window_size = s as usize;
                self.initialized = false; // Force re-initialization
            }
        }

        self.config = config;
        Ok(())
    }

    fn name(&self) -> &str {
        "Moving Average Filter"
    }

    fn reset(&mut self) {
        for buffer in &mut self.buffers {
            buffer.clear();
        }
        self.sums.fill(0.0);
    }

    fn latency_estimate(&self) -> u64 {
        (self.window_size as u64 * 1000) / 2 // Half window delay in microseconds at 1kHz
    }
}

/// Filter bank for combining multiple filters
pub struct FilterBank {
    config: ProcessorConfig,
    filters: Vec<Box<dyn SignalProcessor>>,
}

impl FilterBank {
    /// Create new filter bank
    pub fn new() -> Self {
        let config = ProcessorConfig::new("filter_bank", ProcessorType::Filter);

        FilterBank {
            config,
            filters: Vec::new(),
        }
    }

    /// Add a filter to the bank
    pub fn add_filter(&mut self, filter: Box<dyn SignalProcessor>) {
        self.filters.push(filter);
    }

    /// Create a typical EMG preprocessing filter bank
    pub fn emg_preprocessing() -> BspResult<Self> {
        let mut bank = FilterBank::new();

        // High-pass filter to remove DC and low-frequency drift
        let highpass = ButterworthFilter::new(FilterConfig::highpass(10.0, 4))?;
        bank.add_filter(Box::new(highpass));

        // Low-pass filter to remove high-frequency noise
        let lowpass = ButterworthFilter::new(FilterConfig::lowpass(500.0, 4))?;
        bank.add_filter(Box::new(lowpass));

        // Notch filter to remove 50Hz powerline interference
        let notch = NotchFilter::new(50.0, 30.0);
        bank.add_filter(Box::new(notch));

        Ok(bank)
    }
}

impl SignalProcessor for FilterBank {
    fn process(&mut self, input: &SignalEntity) -> BspResult<SignalEntity> {
        let mut timer = ProcessingMetrics::start_timing();
        let mut current_signal = input.clone();

        // Apply filters sequentially
        for filter in &mut self.filters {
            current_signal = filter.process(&current_signal)?;
        }

        timer.set_output_quality(0.95);
        let metrics = timer.finish();

        Ok(current_signal)
    }

    fn config(&self) -> &ProcessorConfig {
        &self.config
    }

    fn update_config(&mut self, config: ProcessorConfig) -> BspResult<()> {
        self.config = config;
        Ok(())
    }

    fn name(&self) -> &str {
        "Filter Bank"
    }

    fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
    }

    fn latency_estimate(&self) -> u64 {
        self.filters.iter().map(|f| f.latency_estimate()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bsp_core::{EMGSignal, MuscleGroup};

    #[test]
    fn test_butterworth_lowpass() {
        let filter_config = FilterConfig::lowpass(100.0, 2);
        let mut filter = ButterworthFilter::new(filter_config).unwrap();

        // Create test signal
        let test_data = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let metadata = EMGMetadata::new(
            EMGSignal::Surface {
                muscle_group: MuscleGroup::Biceps,
                activation_level: 0.5,
            },
            1000.0, 1, 1.0, 0.1
        ).unwrap();

        let input_signal = SignalEntity::new(test_data, metadata).unwrap();
        let output_signal = filter.process(&input_signal).unwrap();

        assert_eq!(output_signal.len(), input_signal.len());
        assert_eq!(output_signal.channel_count(), input_signal.channel_count());
    }

    #[test]
    fn test_notch_filter() {
        let mut filter = NotchFilter::new(50.0, 30.0);

        // Create test signal with 50Hz component
        let mut test_data = Vec::new();
        for i in 0..1000 {
            let t = i as f32 / 1000.0;
            let signal = (2.0 * std::f32::consts::PI * 10.0 * t).sin() + // 10Hz signal
                0.5 * (2.0 * std::f32::consts::PI * 50.0 * t).sin(); // 50Hz interference
            test_data.push(signal);
        }

        let metadata = EMGMetadata::new(
            EMGSignal::Surface {
                muscle_group: MuscleGroup::Biceps,
                activation_level: 0.5,
            },
            1000.0, 1, 1.0, 0.1
        ).unwrap();

        let input_signal = SignalEntity::new(test_data, metadata).unwrap();
        let output_signal = filter.process(&input_signal).unwrap();

        assert_eq!(output_signal.len(), input_signal.len());
        // Notch filter should reduce 50Hz component while preserving 10Hz
    }

    #[test]
    fn test_moving_average() {
        let mut filter = MovingAverageFilter::new(10);

        let test_data = vec![1.0; 100]; // Constant signal
        let metadata = EMGMetadata::new(
            EMGSignal::Surface {
                muscle_group: MuscleGroup::Biceps,
                activation_level: 0.5,
            },
            1000.0, 1, 0.1, 0.1
        ).unwrap();

        let input_signal = SignalEntity::new(test_data, metadata).unwrap();
        let output_signal = filter.process(&input_signal).unwrap();

        let output_data = output_signal.channel_data(0).unwrap();

        // After the window fills up, output should be close to 1.0
        assert!((output_data[50] - 1.0).abs() < 0.01);
    }
}