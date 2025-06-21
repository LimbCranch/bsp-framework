//! Configuration management for signal processing

use crate::processor::{ProcessorConfig, ProcessorType, ParameterValue};
use crate::filters::FilterConfig;
use crate::features::FeatureConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use bsp_core::{BspResult, BspError};

/// Global processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Configuration name/profile
    pub name: String,
    /// Target application (realtime, research, clinical)
    pub profile: ProcessingProfile,
    /// Global processing parameters
    pub global_params: GlobalParams,
    /// Individual processor configurations
    pub processors: HashMap<String, ProcessorConfig>,
    /// Pipeline configurations
    pub pipelines: HashMap<String, PipelineConfig>,
}

/// Processing profiles for different use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingProfile {
    /// Real-time processing with minimal latency
    RealTime,
    /// Research with high quality and comprehensive analysis
    Research,
    /// Clinical use with validated algorithms
    Clinical,
    /// Custom profile
    Custom,
}

/// Global processing parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalParams {
    /// Maximum acceptable total latency (ms)
    pub max_latency_ms: f32,
    /// Buffer size for processing (samples)
    pub buffer_size: usize,
    /// Enable parallel processing where possible
    pub parallel_processing: bool,
    /// Error handling strategy
    pub error_handling: ErrorHandling,
    /// Performance monitoring enabled
    pub performance_monitoring: bool,
    /// Debug logging level
    pub debug_level: DebugLevel,
}

/// Error handling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorHandling {
    /// Stop processing on any error
    StrictMode,
    /// Continue processing, bypassing failed processors
    BypassMode,
    /// Use fallback processors when primary fails
    FallbackMode,
}

/// Debug logging levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DebugLevel {
    None,
    Error,
    Warning,
    Info,
    Debug,
    Trace,
}

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Pipeline name
    pub name: String,
    /// Enabled state
    pub enabled: bool,
    /// Processor chain (in execution order)
    pub processor_chain: Vec<String>,
    /// Parallel execution groups
    pub parallel_groups: Vec<Vec<String>>,
    /// Pipeline-specific parameters
    pub parameters: HashMap<String, ParameterValue>,
}

/// Preset configurations for common scenarios
impl ProcessingConfig {
    /// Real-time EMG processing configuration
    pub fn realtime_emg() -> Self {
        let mut config = ProcessingConfig {
            name: "Real-time EMG".to_string(),
            profile: ProcessingProfile::RealTime,
            global_params: GlobalParams {
                max_latency_ms: 5.0,
                buffer_size: 128,
                parallel_processing: true,
                error_handling: ErrorHandling::BypassMode,
                performance_monitoring: true,
                debug_level: DebugLevel::Warning,
            },
            processors: HashMap::new(),
            pipelines: HashMap::new(),
        };

        // Add real-time optimized processors
        config.add_highpass_filter("rt_highpass", 20.0, 2);
        config.add_notch_filter("rt_notch", 50.0, 10.0);
        config.add_moving_average("rt_smooth", 3);

        // Add real-time pipeline
        config.add_pipeline("realtime", vec![
            "rt_highpass".to_string(),
            "rt_notch".to_string(),
            "rt_smooth".to_string(),
        ]);

        config
    }

    /// Research-grade EMG processing configuration
    pub fn research_emg() -> Self {
        let mut config = ProcessingConfig {
            name: "Research EMG".to_string(),
            profile: ProcessingProfile::Research,
            global_params: GlobalParams {
                max_latency_ms: 50.0,
                buffer_size: 1024,
                parallel_processing: true,
                error_handling: ErrorHandling::StrictMode,
                performance_monitoring: true,
                debug_level: DebugLevel::Info,
            },
            processors: HashMap::new(),
            pipelines: HashMap::new(),
        };

        // Add high-quality processors
        config.add_highpass_filter("research_highpass", 10.0, 6);
        config.add_lowpass_filter("research_lowpass", 450.0, 6);
        config.add_notch_filter("research_notch_50", 50.0, 50.0);
        config.add_notch_filter("research_notch_60", 60.0, 50.0);
        config.add_feature_extractor("research_features", 512, 0.75);

        // Add research pipeline
        config.add_pipeline("research", vec![
            "research_highpass".to_string(),
            "research_lowpass".to_string(),
            "research_notch_50".to_string(),
            "research_notch_60".to_string(),
            "research_features".to_string(),
        ]);

        config
    }

    /// Clinical EMG processing configuration
    pub fn clinical_emg() -> Self {
        let mut config = ProcessingConfig {
            name: "Clinical EMG".to_string(),
            profile: ProcessingProfile::Clinical,
            global_params: GlobalParams {
                max_latency_ms: 20.0,
                buffer_size: 256,
                parallel_processing: false, // Deterministic processing
                error_handling: ErrorHandling::StrictMode,
                performance_monitoring: true,
                debug_level: DebugLevel::Error,
            },
            processors: HashMap::new(),
            pipelines: HashMap::new(),
        };

        // Add clinically validated processors
        config.add_highpass_filter("clinical_highpass", 20.0, 4);
        config.add_lowpass_filter("clinical_lowpass", 500.0, 4);
        config.add_notch_filter("clinical_notch", 50.0, 30.0);
        config.add_feature_extractor("clinical_features", 256, 0.5);

        // Add clinical pipeline
        config.add_pipeline("clinical", vec![
            "clinical_highpass".to_string(),
            "clinical_lowpass".to_string(),
            "clinical_notch".to_string(),
            "clinical_features".to_string(),
        ]);

        config
    }

    /// Add highpass filter configuration
    pub fn add_highpass_filter(&mut self, name: &str, cutoff_freq: f32, order: usize) {
        let mut config = ProcessorConfig::new(name, ProcessorType::Filter);
        config.set_parameter("filter_type", "butterworth_highpass".into());
        config.set_parameter("cutoff_freq", cutoff_freq.into());
        config.set_parameter("order", (order as i32).into());

        self.processors.insert(name.to_string(), config);
    }

    /// Add lowpass filter configuration
    pub fn add_lowpass_filter(&mut self, name: &str, cutoff_freq: f32, order: usize) {
        let mut config = ProcessorConfig::new(name, ProcessorType::Filter);
        config.set_parameter("filter_type", "butterworth_lowpass".into());
        config.set_parameter("cutoff_freq", cutoff_freq.into());
        config.set_parameter("order", (order as i32).into());

        self.processors.insert(name.to_string(), config);
    }

    /// Add notch filter configuration
    pub fn add_notch_filter(&mut self, name: &str, notch_freq: f32, q_factor: f32) {
        let mut config = ProcessorConfig::new(name, ProcessorType::Filter);
        config.set_parameter("filter_type", "notch".into());
        config.set_parameter("notch_freq", notch_freq.into());
        config.set_parameter("q_factor", q_factor.into());

        self.processors.insert(name.to_string(), config);
    }

    /// Add moving average filter configuration
    pub fn add_moving_average(&mut self, name: &str, window_size: usize) {
        let mut config = ProcessorConfig::new(name, ProcessorType::Filter);
        config.set_parameter("filter_type", "moving_average".into());
        config.set_parameter("window_size", (window_size as i32).into());

        self.processors.insert(name.to_string(), config);
    }

    /// Add feature extractor configuration
    pub fn add_feature_extractor(&mut self, name: &str, window_size: usize, overlap: f32) {
        let mut config = ProcessorConfig::new(name, ProcessorType::FeatureExtractor);
        config.set_parameter("window_size", (window_size as i32).into());
        config.set_parameter("overlap", overlap.into());
        config.set_parameter("enabled_features", "time,frequency,statistical".into());

        self.processors.insert(name.to_string(), config);
    }

    /// Add pipeline configuration
    pub fn add_pipeline(&mut self, name: &str, processor_chain: Vec<String>) {
        let pipeline_config = PipelineConfig {
            name: name.to_string(),
            enabled: true,
            processor_chain,
            parallel_groups: Vec::new(),
            parameters: HashMap::new(),
        };

        self.pipelines.insert(name.to_string(), pipeline_config);
    }

    /// Validate entire configuration
    pub fn validate(&self) -> BspResult<()> {
        // Validate global parameters
        if self.global_params.max_latency_ms <= 0.0 {
            return Err(BspError::ConfigurationError {
                message: "Maximum latency must be positive".to_string(),
            });
        }

        if self.global_params.buffer_size == 0 {
            return Err(BspError::ConfigurationError {
                message: "Buffer size must be greater than 0".to_string(),
            });
        }

        // Validate individual processors
        for (name, processor_config) in &self.processors {
            processor_config.validate().map_err(|e| BspError::ConfigurationError {
                message: format!("Processor '{}' configuration invalid: {}", name, e),
            })?;
        }

        // Validate pipelines
        for (pipeline_name, pipeline_config) in &self.pipelines {
            self.validate_pipeline(pipeline_name, pipeline_config)?;
        }

        Ok(())
    }

    /// Validate individual pipeline
    fn validate_pipeline(&self, pipeline_name: &str, pipeline_config: &PipelineConfig) -> BspResult<()> {
        if pipeline_config.processor_chain.is_empty() {
            return Err(BspError::ConfigurationError {
                message: format!("Pipeline '{}' has no processors", pipeline_name),
            });
        }

        // Check that all referenced processors exist
        for processor_name in &pipeline_config.processor_chain {
            if !self.processors.contains_key(processor_name) {
                return Err(BspError::ConfigurationError {
                    message: format!(
                        "Pipeline '{}' references unknown processor '{}'",
                        pipeline_name, processor_name
                    ),
                });
            }
        }

        // Validate parallel groups
        for group in &pipeline_config.parallel_groups {
            for processor_name in group {
                if !self.processors.contains_key(processor_name) {
                    return Err(BspError::ConfigurationError {
                        message: format!(
                            "Pipeline '{}' parallel group references unknown processor '{}'",
                            pipeline_name, processor_name
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    /// Get processor configuration by name
    pub fn get_processor(&self, name: &str) -> Option<&ProcessorConfig> {
        self.processors.get(name)
    }

    /// Get pipeline configuration by name
    pub fn get_pipeline(&self, name: &str) -> Option<&PipelineConfig> {
        self.pipelines.get(name)
    }

    /// Update processor parameter
    pub fn update_processor_param(&mut self, processor_name: &str, param_name: &str, value: ParameterValue) -> BspResult<()> {
        if let Some(processor_config) = self.processors.get_mut(processor_name) {
            processor_config.set_parameter(param_name, value);
            Ok(())
        } else {
            Err(BspError::ConfigurationError {
                message: format!("Processor '{}' not found", processor_name),
            })
        }
    }

    /// Enable/disable processor
    pub fn set_processor_enabled(&mut self, processor_name: &str, enabled: bool) -> BspResult<()> {
        if let Some(processor_config) = self.processors.get_mut(processor_name) {
            processor_config.enabled = enabled;
            Ok(())
        } else {
            Err(BspError::ConfigurationError {
                message: format!("Processor '{}' not found", processor_name),
            })
        }
    }

    /// Enable/disable pipeline
    pub fn set_pipeline_enabled(&mut self, pipeline_name: &str, enabled: bool) -> BspResult<()> {
        if let Some(pipeline_config) = self.pipelines.get_mut(pipeline_name) {
            pipeline_config.enabled = enabled;
            Ok(())
        } else {
            Err(BspError::ConfigurationError {
                message: format!("Pipeline '{}' not found", pipeline_name),
            })
        }
    }

    /// Export configuration to JSON
    pub fn to_json(&self) -> BspResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| BspError::ConfigurationError {
            message: format!("Failed to serialize configuration: {}", e),
        })
    }

    /// Import configuration from JSON
    pub fn from_json(json: &str) -> BspResult<Self> {
        serde_json::from_str(json).map_err(|e| BspError::ConfigurationError {
            message: format!("Failed to deserialize configuration: {}", e),
        })
    }

    /// Create configuration suitable for given profile
    pub fn for_profile(profile: ProcessingProfile) -> Self {
        match profile {
            ProcessingProfile::RealTime => Self::realtime_emg(),
            ProcessingProfile::Research => Self::research_emg(),
            ProcessingProfile::Clinical => Self::clinical_emg(),
            ProcessingProfile::Custom => Self::realtime_emg(), // Default to real-time
        }
    }

    /// Get estimated total latency for a pipeline
    pub fn estimated_pipeline_latency(&self, pipeline_name: &str) -> Option<f32> {
        let pipeline = self.pipelines.get(pipeline_name)?;
        let mut total_latency = 0.0;

        for processor_name in &pipeline.processor_chain {
            if let Some(processor_config) = self.processors.get(processor_name) {
                // Estimate latency based on processor type and parameters
                let latency = match processor_config.processor_type {
                    ProcessorType::Filter => {
                        let order = processor_config.get_int("order", 2) as f32;
                        order * 0.1 // Rough estimate: 0.1ms per filter order
                    }
                    ProcessorType::FeatureExtractor => {
                        let window_size = processor_config.get_int("window_size", 256) as f32;
                        window_size / 1000.0 // Assume 1kHz sampling
                    }
                    _ => 1.0, // Default 1ms
                };
                total_latency += latency;
            }
        }

        Some(total_latency)
    }
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self::realtime_emg()
    }
}

impl Default for GlobalParams {
    fn default() -> Self {
        GlobalParams {
            max_latency_ms: 10.0,
            buffer_size: 256,
            parallel_processing: true,
            error_handling: ErrorHandling::BypassMode,
            performance_monitoring: true,
            debug_level: DebugLevel::Warning,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realtime_config() {
        let config = ProcessingConfig::realtime_emg();
        assert_eq!(config.profile, ProcessingProfile::RealTime);
        assert!(config.global_params.max_latency_ms <= 10.0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_research_config() {
        let config = ProcessingConfig::research_emg();
        assert_eq!(config.profile, ProcessingProfile::Research);
        assert!(config.processors.len() > 3); // Should have multiple processors
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let mut config = ProcessingConfig::realtime_emg();

        // Valid config should pass
        assert!(config.validate().is_ok());

        // Invalid latency should fail
        config.global_params.max_latency_ms = -1.0;
        assert!(config.validate().is_err());

        // Reset and test invalid buffer size
        config.global_params.max_latency_ms = 10.0;
        config.global_params.buffer_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_processor_management() {
        let mut config = ProcessingConfig::realtime_emg();

        // Test adding processor
        config.add_highpass_filter("test_filter", 25.0, 4);
        assert!(config.processors.contains_key("test_filter"));

        // Test updating parameter
        let result = config.update_processor_param("test_filter", "cutoff_freq", 30.0.into());
        assert!(result.is_ok());

        // Test enabling/disabling
        let result = config.set_processor_enabled("test_filter", false);
        assert!(result.is_ok());
        assert!(!config.processors["test_filter"].enabled);
    }

    #[test]
    fn test_pipeline_validation() {
        let mut config = ProcessingConfig::realtime_emg();

        // Add pipeline with valid processors
        config.add_pipeline("test_pipeline", vec!["rt_highpass".to_string()]);
        assert!(config.validate().is_ok());

        // Add pipeline with invalid processor reference
        config.add_pipeline("invalid_pipeline", vec!["nonexistent_processor".to_string()]);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_json_serialization() {
        let config = ProcessingConfig::realtime_emg();

        // Test serialization
        let json = config.to_json().unwrap();
        assert!(!json.is_empty());

        // Test deserialization
        let deserialized_config = ProcessingConfig::from_json(&json).unwrap();
        assert_eq!(config.name, deserialized_config.name);
        assert_eq!(config.profile, deserialized_config.profile);
    }

    #[test]
    fn test_latency_estimation() {
        let config = ProcessingConfig::realtime_emg();

        if let Some(latency) = config.estimated_pipeline_latency("realtime") {
            assert!(latency > 0.0);
            assert!(latency < config.global_params.max_latency_ms);
        }
    }

    #[test]
    fn test_profile_creation() {
        let rt_config = ProcessingConfig::for_profile(ProcessingProfile::RealTime);
        assert_eq!(rt_config.profile, ProcessingProfile::RealTime);

        let research_config = ProcessingConfig::for_profile(ProcessingProfile::Research);
        assert_eq!(research_config.profile, ProcessingProfile::Research);

        let clinical_config = ProcessingConfig::for_profile(ProcessingProfile::Clinical);
        assert_eq!(clinical_config.profile, ProcessingProfile::Clinical);
    }

    #[test]
    fn test_error_handling_modes() {
        let mut config = ProcessingConfig::realtime_emg();

        // Test different error handling modes
        config.global_params.error_handling = ErrorHandling::StrictMode;
        assert!(config.validate().is_ok());

        config.global_params.error_handling = ErrorHandling::BypassMode;
        assert!(config.validate().is_ok());

        config.global_params.error_handling = ErrorHandling::FallbackMode;
        assert!(config.validate().is_ok());
    }
}