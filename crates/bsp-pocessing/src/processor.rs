//! Core signal processor trait and types

use bsp_core::{SignalEntity, BspResult, BspError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Core trait for all signal processors
pub trait SignalProcessor: Send + Sync {
    /// Process a signal entity and return the processed result
    fn process(&mut self, input: &SignalEntity) -> BspResult<SignalEntity>;

    /// Get processor configuration
    fn config(&self) -> &ProcessorConfig;

    /// Update processor configuration
    fn update_config(&mut self, config: ProcessorConfig) -> BspResult<()>;

    /// Get processor name/identifier
    fn name(&self) -> &str;

    /// Reset processor internal state
    fn reset(&mut self);

    /// Get processing latency estimate in microseconds
    fn latency_estimate(&self) -> u64 {
        1000 // 1ms default
    }

    /// Check if processor can handle the given signal type
    fn can_process(&self, signal: &SignalEntity) -> bool {
        signal.channel_count() > 0 && signal.sampling_rate() > 0.0
    }

    /// Get processor type for pipeline organization
    fn processor_type(&self) -> ProcessorType {
        ProcessorType::Filter
    }
}

/// Types of signal processors for pipeline organization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProcessorType {
    /// Pre-processing filters (bandpass, notch, etc.)
    Filter,
    /// Feature extraction
    FeatureExtractor,
    /// Analysis and classification
    Analyzer,
    /// Post-processing and formatting
    PostProcessor,
}

/// Generic processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Processor name
    pub name: String,
    /// Processor type
    pub processor_type: ProcessorType,
    /// Enabled/disabled state
    pub enabled: bool,
    /// Processing parameters
    pub parameters: HashMap<String, ParameterValue>,
    /// Performance constraints
    pub constraints: ProcessingConstraints,
}

/// Parameter value types for flexible configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    Float(f32),
    Integer(i32),
    Boolean(bool),
    String(String),
    FloatArray(Vec<f32>),
    IntegerArray(Vec<i32>),
}

/// Processing constraints and requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConstraints {
    /// Maximum acceptable latency in microseconds
    pub max_latency_us: u64,
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Required sampling rate range
    pub sampling_rate_range: Option<(f32, f32)>,
    /// Required channel count range
    pub channel_count_range: Option<(usize, usize)>,
}

/// Result of signal processing operation
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    /// Processed signal entity
    pub signal: SignalEntity,
    /// Processing metrics
    pub metrics: ProcessingMetrics,
    /// Any warnings or notes
    pub warnings: Vec<String>,
}

/// Performance metrics for processing operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetrics {
    /// Actual processing time in microseconds
    pub processing_time_us: u64,
    /// Memory usage during processing
    pub memory_usage_bytes: usize,
    /// Input signal quality score (0.0-1.0)
    pub input_quality: f32,
    /// Output signal quality score (0.0-1.0)
    pub output_quality: f32,
    /// Success/failure status
    pub success: bool,
    /// Error message if processing failed
    pub error_message: Option<String>,
}

impl ProcessorConfig {
    /// Create new processor configuration
    pub fn new(name: &str, processor_type: ProcessorType) -> Self {
        Self {
            name: name.to_string(),
            processor_type,
            enabled: true,
            parameters: HashMap::new(),
            constraints: ProcessingConstraints::default(),
        }
    }

    /// Set a parameter value
    pub fn set_parameter(&mut self, key: &str, value: ParameterValue) {
        self.parameters.insert(key.to_string(), value);
    }

    /// Get a parameter value
    pub fn get_parameter(&self, key: &str) -> Option<&ParameterValue> {
        self.parameters.get(key)
    }

    /// Get float parameter with default
    pub fn get_float(&self, key: &str, default: f32) -> f32 {
        match self.get_parameter(key) {
            Some(ParameterValue::Float(value)) => *value,
            _ => default,
        }
    }

    /// Get integer parameter with default
    pub fn get_int(&self, key: &str, default: i32) -> i32 {
        match self.get_parameter(key) {
            Some(ParameterValue::Integer(value)) => *value,
            _ => default,
        }
    }

    /// Get boolean parameter with default
    pub fn get_bool(&self, key: &str, default: bool) -> bool {
        match self.get_parameter(key) {
            Some(ParameterValue::Boolean(value)) => *value,
            _ => default,
        }
    }

    /// Validate configuration against constraints
    pub fn validate(&self) -> BspResult<()> {
        if self.name.is_empty() {
            return Err(BspError::ConfigurationError {
                message: "Processor name cannot be empty".to_string(),
            });
        }

        if self.constraints.max_latency_us == 0 {
            return Err(BspError::ConfigurationError {
                message: "Maximum latency must be greater than 0".to_string(),
            });
        }

        Ok(())
    }
}

impl Default for ProcessingConstraints {
    fn default() -> Self {
        Self {
            max_latency_us: 10_000, // 10ms default
            max_memory_bytes: 1024 * 1024, // 1MB default
            sampling_rate_range: None,
            channel_count_range: None,
        }
    }
}

impl ProcessingMetrics {
    /// Create new processing metrics
    pub fn new() -> Self {
        Self {
            processing_time_us: 0,
            memory_usage_bytes: 0,
            input_quality: 1.0,
            output_quality: 1.0,
            success: true,
            error_message: None,
        }
    }

    /// Start timing a processing operation
    pub fn start_timing() -> ProcessingTimer {
        ProcessingTimer {
            start_time: Instant::now(),
            metrics: ProcessingMetrics::new(),
        }
    }

    /// Check if processing met the constraints
    pub fn meets_constraints(&self, constraints: &ProcessingConstraints) -> bool {
        self.success &&
            self.processing_time_us <= constraints.max_latency_us &&
            self.memory_usage_bytes <= constraints.max_memory_bytes
    }
}

/// Helper for timing processing operations
pub struct ProcessingTimer {
    start_time: Instant,
    metrics: ProcessingMetrics,
}

impl ProcessingTimer {
    /// Finish timing and return metrics
    pub fn finish(mut self) -> ProcessingMetrics {
        self.metrics.processing_time_us = self.start_time.elapsed().as_micros() as u64;
        self.metrics
    }

    /// Finish with error
    pub fn finish_with_error(mut self, error: &str) -> ProcessingMetrics {
        self.metrics.processing_time_us = self.start_time.elapsed().as_micros() as u64;
        self.metrics.success = false;
        self.metrics.error_message = Some(error.to_string());
        self.metrics
    }

    /// Set input quality score
    pub fn set_input_quality(&mut self, quality: f32) {
        self.metrics.input_quality = quality.clamp(0.0, 1.0);
    }

    /// Set output quality score
    pub fn set_output_quality(&mut self, quality: f32) {
        self.metrics.output_quality = quality.clamp(0.0, 1.0);
    }

    /// Set memory usage
    pub fn set_memory_usage(&mut self, bytes: usize) {
        self.metrics.memory_usage_bytes = bytes;
    }
}

impl ParameterValue {
    /// Convert to f32 if possible
    pub fn as_float(&self) -> Option<f32> {
        match self {
            ParameterValue::Float(v) => Some(*v),
            ParameterValue::Integer(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Convert to i32 if possible
    pub fn as_int(&self) -> Option<i32> {
        match self {
            ParameterValue::Integer(v) => Some(*v),
            ParameterValue::Float(v) => Some(*v as i32),
            _ => None,
        }
    }

    /// Convert to bool if possible
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParameterValue::Boolean(v) => Some(*v),
            _ => None,
        }
    }
}

/// Macro for easy parameter setting
#[macro_export]
macro_rules! set_params {
    ($config:expr, $($key:expr => $value:expr),+ $(,)?) => {
        $(
            $config.set_parameter($key, $value.into());
        )+
    };
}

/// Implement From for common parameter types
impl From<f32> for ParameterValue {
    fn from(value: f32) -> Self {
        ParameterValue::Float(value)
    }
}

impl From<i32> for ParameterValue {
    fn from(value: i32) -> Self {
        ParameterValue::Integer(value)
    }
}

impl From<bool> for ParameterValue {
    fn from(value: bool) -> Self {
        ParameterValue::Boolean(value)
    }
}

impl From<String> for ParameterValue {
    fn from(value: String) -> Self {
        ParameterValue::String(value)
    }
}

impl From<&str> for ParameterValue {
    fn from(value: &str) -> Self {
        ParameterValue::String(value.to_string())
    }
}

impl From<Vec<f32>> for ParameterValue {
    fn from(value: Vec<f32>) -> Self {
        ParameterValue::FloatArray(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bsp_core::{EMGMetadata, EMGSignal, MuscleGroup};

    #[test]
    fn test_processor_config() {
        let mut config = ProcessorConfig::new("test_filter", ProcessorType::Filter);

        config.set_parameter("cutoff_freq", ParameterValue::Float(100.0));
        config.set_parameter("order", ParameterValue::Integer(4));
        config.set_parameter("enabled", ParameterValue::Boolean(true));

        assert_eq!(config.get_float("cutoff_freq", 0.0), 100.0);
        assert_eq!(config.get_int("order", 0), 4);
        assert_eq!(config.get_bool("enabled", false), true);

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_processing_metrics() {
        let timer = ProcessingMetrics::start_timing();
        std::thread::sleep(Duration::from_millis(1));
        let metrics = timer.finish();

        assert!(metrics.processing_time_us > 0);
        assert!(metrics.success);
        assert!(metrics.error_message.is_none());
    }

    #[test]
    fn test_parameter_value_conversions() {
        let float_param = ParameterValue::Float(3.14);
        let int_param = ParameterValue::Integer(42);
        let bool_param = ParameterValue::Boolean(true);

        assert_eq!(float_param.as_float(), Some(3.14));
        assert_eq!(int_param.as_int(), Some(42));
        assert_eq!(bool_param.as_bool(), Some(true));

        // Cross-type conversions
        assert_eq!(int_param.as_float(), Some(42.0));
        assert_eq!(float_param.as_int(), Some(3));
    }
}