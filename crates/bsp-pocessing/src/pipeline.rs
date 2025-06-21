//! Signal processing pipeline for chaining processors

use crate::processor::{SignalProcessor, ProcessorConfig, ProcessorType, ProcessingMetrics};
use crate::filters::{ButterworthFilter, NotchFilter, MovingAverageFilter, FilterConfig, FilterBank};
use crate::features::{FeatureExtractor, FeatureConfig, FeatureSet};
use bsp_core::{SignalEntity, BspResult, BspError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Processing pipeline that chains multiple processors
pub struct Pipeline {
    config: ProcessorConfig,
    processors: Vec<Box<dyn SignalProcessor>>,
    processing_metrics: Vec<ProcessingMetrics>,
    bypass_enabled: bool,
    name: String,
}

/// Pipeline builder for constructing processing chains
pub struct PipelineBuilder {
    processors: Vec<Box<dyn SignalProcessor>>,
    name: String,
    bypass_enabled: bool,
}

/// Configuration for the entire pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub name: String,
    pub enabled: bool,
    pub bypass_on_error: bool,
    pub max_total_latency_ms: f32,
    pub processors: Vec<ProcessorConfigEntry>,
}

/// Individual processor configuration entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfigEntry {
    pub name: String,
    pub processor_type: ProcessorType,
    pub enabled: bool,
    pub config: ProcessorConfig,
}

/// Processing chain for organizing processors by type
pub struct ProcessingChain {
    filters: Vec<Box<dyn SignalProcessor>>,
    feature_extractors: Vec<Box<dyn SignalProcessor>>,
    analyzers: Vec<Box<dyn SignalProcessor>>,
    post_processors: Vec<Box<dyn SignalProcessor>>,
}

/// Pipeline execution result with detailed metrics
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub output_signal: SignalEntity,
    pub processing_metrics: Vec<ProcessingMetrics>,
    pub total_latency_us: u64,
    pub success: bool,
    pub warnings: Vec<String>,
    pub features: Option<Vec<FeatureSet>>,
}

/// Pipeline performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelinePerformance {
    pub total_latency_us: u64,
    pub avg_latency_us: u64,
    pub success_rate: f32,
    pub processor_count: usize,
    pub last_update: u64,
}

impl Pipeline {
    /// Create new empty pipeline
    pub fn new(name: &str) -> Self {
        let config = ProcessorConfig::new(name, ProcessorType::Filter);

        Pipeline {
            config,
            processors: Vec::new(),
            processing_metrics: Vec::new(),
            bypass_enabled: false,
            name: name.to_string(),
        }
    }

    /// Add processor to pipeline
    pub fn add_processor(&mut self, processor: Box<dyn SignalProcessor>) {
        self.processors.push(processor);
    }

    /// Create real-time EMG pipeline optimized for low latency
    pub fn realtime_emg() -> BspResult<Self> {
        let mut builder = PipelineBuilder::new("Real-time EMG");
        builder.set_bypass_enabled(true); // Allow bypass for real-time

        // Minimal filtering for low latency
        let highpass = ButterworthFilter::new(FilterConfig::highpass(20.0, 2))?; // Lower order for speed
        builder.add_processor(Box::new(highpass));

        let notch = NotchFilter::new(50.0, 10.0); // Lower Q for faster response
        builder.add_processor(Box::new(notch));

        Ok(builder.build())
    }

    /// Create research-grade EMG pipeline with comprehensive processing
    pub fn research_emg() -> BspResult<Self> {
        let mut builder = PipelineBuilder::new("Research EMG");

        // High-quality filters
        let highpass = ButterworthFilter::new(FilterConfig::highpass(10.0, 6))?; // Higher order
        builder.add_processor(Box::new(highpass));

        let lowpass = ButterworthFilter::new(FilterConfig::lowpass(450.0, 6))?;
        builder.add_processor(Box::new(lowpass));

        // Multiple notch filters for different powerline frequencies
        let notch_50 = NotchFilter::new(50.0, 50.0); // High Q for sharp notch
        builder.add_processor(Box::new(notch_50));

        let notch_60 = NotchFilter::new(60.0, 50.0);
        builder.add_processor(Box::new(notch_60));

        // Feature extraction with larger window for better frequency resolution
        let feature_extractor = FeatureExtractor::emg_features(1024);
        builder.add_processor(Box::new(feature_extractor));

        Ok(builder.build())
    }

    /// Create comprehensive EMG analysis pipeline
    pub fn comprehensive_emg() -> BspResult<Self> {
        let mut builder = PipelineBuilder::new("Comprehensive EMG");

        // Full preprocessing chain
        let preprocessing = FilterBank::emg_preprocessing()?;
        builder.add_processor(Box::new(preprocessing));

        // Feature extraction with overlapping windows
        let feature_extractor = FeatureExtractor::emg_features(512);
        builder.add_processor(Box::new(feature_extractor));

        Ok(builder.build())
    }

    /// Process signal through the entire pipeline
    pub fn process(&mut self, input: &SignalEntity) -> BspResult<PipelineResult> {
        let start_time = Instant::now();
        let mut current_signal = input.clone();
        let mut all_metrics = Vec::new();
        let mut warnings = Vec::new();
        let mut features = Vec::new();

        // Process through each processor in sequence
        for (i, processor) in self.processors.iter_mut().enumerate() {
            // Skip disabled processors
            if !processor.config().enabled {
                continue;
            }

            let processor_start = Instant::now();

            match processor.process(&current_signal) {
                Ok(processed_signal) => {
                    current_signal = processed_signal;

                    // Record metrics
                    let processing_time = processor_start.elapsed().as_micros() as u64;
                    let mut metrics = ProcessingMetrics::new();
                    metrics.processing_time_us = processing_time;
                    metrics.success = true;
                    all_metrics.push(metrics);

                    // Check for latency warnings
                    let latency_estimate = processor.latency_estimate();
                    if processing_time > latency_estimate * 2 {
                        warnings.push(format!(
                            "Processor '{}' took {}μs, expected <{}μs",
                            processor.name(),
                            processing_time,
                            latency_estimate
                        ));
                    }
                }
                Err(e) => {
                    let mut metrics = ProcessingMetrics::new();
                    metrics.success = false;
                    metrics.error_message = Some(e.to_string());
                    metrics.processing_time_us = processor_start.elapsed().as_micros() as u64;
                    all_metrics.push(metrics);

                    if self.bypass_enabled {
                        warnings.push(format!(
                            "Processor '{}' failed: {}, bypassing",
                            processor.name(),
                            e
                        ));
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        let total_latency = start_time.elapsed().as_micros() as u64;

        // Store metrics for later analysis
        self.processing_metrics = all_metrics.clone();

        Ok(PipelineResult {
            output_signal: current_signal,
            processing_metrics: all_metrics,
            total_latency_us: total_latency,
            success: true,
            warnings,
            features: if features.is_empty() { None } else { Some(features) },
        })
    }

    /// Enable/disable bypass mode (continue on processor errors)
    pub fn set_bypass_enabled(&mut self, enabled: bool) {
        self.bypass_enabled = enabled;
    }

    /// Get pipeline performance summary
    pub fn performance_summary(&self) -> PipelinePerformance {
        let total_latency = self.processing_metrics.iter()
            .map(|m| m.processing_time_us)
            .sum();

        let success_rate = if self.processing_metrics.is_empty() {
            1.0
        } else {
            let successful = self.processing_metrics.iter()
                .filter(|m| m.success)
                .count();
            successful as f32 / self.processing_metrics.len() as f32
        };

        let avg_latency = if self.processing_metrics.is_empty() {
            0
        } else {
            total_latency / self.processing_metrics.len() as u64
        };

        PipelinePerformance {
            total_latency_us: total_latency,
            avg_latency_us: avg_latency,
            success_rate,
            processor_count: self.processors.len(),
            last_update: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    /// Reset all processors in the pipeline
    pub fn reset(&mut self) {
        for processor in &mut self.processors {
            processor.reset();
        }
        self.processing_metrics.clear();
    }

    /// Get processor by name
    pub fn get_processor(&self, name: &str) -> Option<&dyn SignalProcessor> {
        self.processors.iter()
            .find(|p| p.name() == name)
            .map(|p| p.as_ref())
    }

    /// Get mutable processor by name
    pub fn get_processor_mut(&mut self, name: &str) -> Option<&mut (dyn SignalProcessor + '_)> {
        self.processors.iter_mut()
            .find(|p| p.name() == name)
            .map(move |p| {
                p.as_mut()
            })
    }

    /// Get estimated total latency
    pub fn estimated_latency(&self) -> u64 {
        self.processors.iter()
            .filter(|p| p.config().enabled)
            .map(|p| p.latency_estimate())
            .sum()
    }

    /// Get list of all processor names
    pub fn processor_names(&self) -> Vec<String> {
        self.processors.iter()
            .map(|p| p.name().to_string())
            .collect()
    }

    /// Get processor types in pipeline
    pub fn processor_types(&self) -> Vec<ProcessorType> {
        self.processors.iter()
            .map(|p| p.processor_type())
            .collect()
    }

    /// Check if pipeline meets latency requirements
    pub fn meets_latency_requirement(&self, max_latency_us: u64) -> bool {
        self.estimated_latency() <= max_latency_us
    }

    /// Update processor configuration by name
    pub fn update_processor_config(&mut self, processor_name: &str, config: ProcessorConfig) -> BspResult<()> {
        if let Some(processor) = self.get_processor_mut(processor_name) {
            processor.update_config(config)?;
            Ok(())
        } else {
            Err(BspError::ConfigurationError {
                message: format!("Processor '{}' not found in pipeline", processor_name),
            })
        }
    }

    /// Enable/disable processor by name
    pub fn set_processor_enabled(&mut self, processor_name: &str, enabled: bool) -> BspResult<()> {
        if let Some(processor) = self.get_processor_mut(processor_name) {
            let mut config = processor.config().clone();
            config.enabled = enabled;
            processor.update_config(config)?;
            Ok(())
        } else {
            Err(BspError::ConfigurationError {
                message: format!("Processor '{}' not found in pipeline", processor_name),
            })
        }
    }

    /// Export pipeline configuration
    pub fn export_config(&self) -> PipelineConfig {
        let processors = self.processors.iter()
            .map(|p| ProcessorConfigEntry {
                name: p.name().to_string(),
                processor_type: p.processor_type(),
                enabled: p.config().enabled,
                config: p.config().clone(),
            })
            .collect();

        PipelineConfig {
            name: self.name.clone(),
            enabled: true,
            bypass_on_error: self.bypass_enabled,
            max_total_latency_ms: 10.0, // Default
            processors,
        }
    }
}

impl SignalProcessor for Pipeline {
    fn process(&mut self, input: &SignalEntity) -> BspResult<SignalEntity> {
        let result = Pipeline::process(self, input)?;
        Ok(result.output_signal)
    }

    fn config(&self) -> &ProcessorConfig {
        &self.config
    }

    fn update_config(&mut self, config: ProcessorConfig) -> BspResult<()> {
        self.config = config;
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn reset(&mut self) {
        Pipeline::reset(self);
    }

    fn latency_estimate(&self) -> u64 {
        self.estimated_latency()
    }
}

impl PipelineBuilder {
    /// Create new pipeline builder
    pub fn new(name: &str) -> Self {
        PipelineBuilder {
            processors: Vec::new(),
            name: name.to_string(),
            bypass_enabled: false,
        }
    }

    /// Add any processor to the pipeline
    pub fn add_processor(&mut self, processor: Box<dyn SignalProcessor>) -> &mut Self {
        self.processors.push(processor);
        self
    }

    /// Add filter to the pipeline
    pub fn add_filter(&mut self, filter: Box<dyn SignalProcessor>) -> &mut Self {
        self.processors.push(filter);
        self
    }

    /// Add feature extractor to the pipeline
    pub fn add_feature_extractor(&mut self, extractor: Box<dyn SignalProcessor>) -> &mut Self {
        self.processors.push(extractor);
        self
    }

    /// Set bypass mode for error handling
    pub fn set_bypass_enabled(&mut self, enabled: bool) -> &mut Self {
        self.bypass_enabled = enabled;
        self
    }

    /// Add EMG preprocessing chain
    pub fn add_emg_preprocessing(&mut self) -> BspResult<&mut Self> {
        let preprocessing = FilterBank::emg_preprocessing()?;
        self.processors.push(Box::new(preprocessing));
        Ok(self)
    }

    /// Add EMG feature extraction
    pub fn add_emg_features(&mut self, window_size: usize) -> &mut Self {
        let feature_extractor = FeatureExtractor::emg_features(window_size);
        self.processors.push(Box::new(feature_extractor));
        self
    }

    /// Add highpass filter
    pub fn add_highpass(&mut self, cutoff: f32, order: usize) -> BspResult<&mut Self> {
        let filter = ButterworthFilter::new(FilterConfig::highpass(cutoff, order))?;
        self.processors.push(Box::new(filter));
        Ok(self)
    }

    /// Add lowpass filter
    pub fn add_lowpass(&mut self, cutoff: f32, order: usize) -> BspResult<&mut Self> {
        let filter = ButterworthFilter::new(FilterConfig::lowpass(cutoff, order))?;
        self.processors.push(Box::new(filter));
        Ok(self)
    }

    /// Add bandpass filter
    pub fn add_bandpass(&mut self, low: f32, high: f32, order: usize) -> BspResult<&mut Self> {
        let filter = ButterworthFilter::new(FilterConfig::bandpass(low, high, order))?;
        self.processors.push(Box::new(filter));
        Ok(self)
    }

    /// Add notch filter
    pub fn add_notch(&mut self, freq: f32, q: f32) -> &mut Self {
        let filter = NotchFilter::new(freq, q);
        self.processors.push(Box::new(filter));
        self
    }

    /// Add moving average filter
    pub fn add_moving_average(&mut self, window_size: usize) -> &mut Self {
        let filter = MovingAverageFilter::new(window_size);
        self.processors.push(Box::new(filter));
        self
    }

    /// Build the pipeline
    pub fn build(self) -> Pipeline {
        let mut pipeline = Pipeline::new(&self.name);
        pipeline.set_bypass_enabled(self.bypass_enabled);

        for processor in self.processors {
            pipeline.add_processor(processor);
        }

        pipeline
    }
}

impl ProcessingChain {
    /// Create new processing chain
    pub fn new() -> Self {
        ProcessingChain {
            filters: Vec::new(),
            feature_extractors: Vec::new(),
            analyzers: Vec::new(),
            post_processors: Vec::new(),
        }
    }

    /// Add processor to appropriate chain based on type
    pub fn add_processor(&mut self, processor: Box<dyn SignalProcessor>) {
        match processor.processor_type() {
            ProcessorType::Filter => self.filters.push(processor),
            ProcessorType::FeatureExtractor => self.feature_extractors.push(processor),
            ProcessorType::Analyzer => self.analyzers.push(processor),
            ProcessorType::PostProcessor => self.post_processors.push(processor),
        }
    }

    /// Process signal through all chains in order
    pub fn process(&mut self, input: &SignalEntity) -> BspResult<SignalEntity> {
        let mut current_signal = input.clone();

        // Apply filters
        for processor in &mut self.filters {
            current_signal = processor.process(&current_signal)?;
        }

        // Extract features
        for processor in &mut self.feature_extractors {
            current_signal = processor.process(&current_signal)?;
        }

        // Apply analysis
        for processor in &mut self.analyzers {
            current_signal = processor.process(&current_signal)?;
        }

        // Post-processing
        for processor in &mut self.post_processors {
            current_signal = processor.process(&current_signal)?;
        }

        Ok(current_signal)
    }

    /// Reset all processors
    pub fn reset(&mut self) {
        for processor in &mut self.filters {
            processor.reset();
        }
        for processor in &mut self.feature_extractors {
            processor.reset();
        }
        for processor in &mut self.analyzers {
            processor.reset();
        }
        for processor in &mut self.post_processors {
            processor.reset();
        }
    }
}

/// Pipeline configuration management
impl PipelineConfig {
    /// Create default EMG processing configuration
    pub fn default_emg() -> Self {
        PipelineConfig {
            name: "Default EMG".to_string(),
            enabled: true,
            bypass_on_error: false,
            max_total_latency_ms: 10.0,
            processors: vec![
                ProcessorConfigEntry {
                    name: "Highpass Filter".to_string(),
                    processor_type: ProcessorType::Filter,
                    enabled: true,
                    config: ProcessorConfig::new("highpass", ProcessorType::Filter),
                },
                ProcessorConfigEntry {
                    name: "Lowpass Filter".to_string(),
                    processor_type: ProcessorType::Filter,
                    enabled: true,
                    config: ProcessorConfig::new("lowpass", ProcessorType::Filter),
                },
                ProcessorConfigEntry {
                    name: "Notch Filter".to_string(),
                    processor_type: ProcessorType::Filter,
                    enabled: true,
                    config: ProcessorConfig::new("notch", ProcessorType::Filter),
                },
                ProcessorConfigEntry {
                    name: "Feature Extractor".to_string(),
                    processor_type: ProcessorType::FeatureExtractor,
                    enabled: true,
                    config: ProcessorConfig::new("features", ProcessorType::FeatureExtractor),
                },
            ],
        }
    }

    /// Create real-time optimized configuration
    pub fn realtime_emg() -> Self {
        PipelineConfig {
            name: "Real-time EMG".to_string(),
            enabled: true,
            bypass_on_error: true, // Allow bypass for real-time processing
            max_total_latency_ms: 5.0, // Stricter latency requirement
            processors: vec![
                ProcessorConfigEntry {
                    name: "Highpass Filter".to_string(),
                    processor_type: ProcessorType::Filter,
                    enabled: true,
                    config: ProcessorConfig::new("highpass", ProcessorType::Filter),
                },
                ProcessorConfigEntry {
                    name: "Notch Filter".to_string(),
                    processor_type: ProcessorType::Filter,
                    enabled: true,
                    config: ProcessorConfig::new("notch", ProcessorType::Filter),
                },
            ],
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> BspResult<()> {
        if self.name.is_empty() {
            return Err(BspError::ConfigurationError {
                message: "Pipeline name cannot be empty".to_string(),
            });
        }

        if self.max_total_latency_ms <= 0.0 {
            return Err(BspError::ConfigurationError {
                message: "Maximum latency must be positive".to_string(),
            });
        }

        if self.processors.is_empty() {
            return Err(BspError::ConfigurationError {
                message: "Pipeline must have at least one processor".to_string(),
            });
        }

        // Validate individual processor configs
        for processor_config in &self.processors {
            processor_config.config.validate()?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bsp_core::{EMGSignal, MuscleGroup, EMGMetadata};

    #[test]
    fn test_pipeline_builder() {
        let mut builder = PipelineBuilder::new("Test Pipeline");

        builder.add_highpass(20.0, 4).unwrap();
        builder.add_notch(50.0, 30.0);

        let pipeline = builder.build();

        assert_eq!(pipeline.processors.len(), 2);
        assert_eq!(pipeline.name(), "Test Pipeline");
    }

    #[test]
    fn test_realtime_emg_pipeline() {
        let pipeline = Pipeline::realtime_emg().unwrap();
        assert!(pipeline.processors.len() > 0);
        assert!(pipeline.estimated_latency() > 0);
        assert!(pipeline.bypass_enabled); // Real-time should have bypass enabled
    }

    #[test]
    fn test_research_emg_pipeline() {
        let pipeline = Pipeline::research_emg().unwrap();
        assert!(pipeline.processors.len() >= 4); // Should have multiple processors
        assert!(!pipeline.bypass_enabled); // Research should be strict
    }

    #[test]
    fn test_pipeline_processing() {
        let mut pipeline = Pipeline::realtime_emg().unwrap();

        let test_data = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let metadata = EMGMetadata::new(
            EMGSignal::Surface {
                muscle_group: MuscleGroup::Biceps,
                activation_level: 0.5,
            },
            1000.0, 1, 1.0, 0.1
        ).unwrap();

        let input_signal = SignalEntity::new(test_data, metadata).unwrap();
        let result = pipeline.process(&input_signal).unwrap();

        assert!(result.success);
        assert_eq!(result.output_signal.len(), input_signal.len());
        assert!(result.total_latency_us > 0);
        assert_eq!(result.processing_metrics.len(), pipeline.processors.len());
    }

    #[test]
    fn test_pipeline_performance() {
        let mut pipeline = Pipeline::realtime_emg().unwrap();

        // Process multiple signals to get performance data
        for _ in 0..5 {
            let test_data = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
            let metadata = EMGMetadata::new(
                EMGSignal::Surface {
                    muscle_group: MuscleGroup::Biceps,
                    activation_level: 0.5,
                },
                1000.0, 1, 0.256, 0.1
            ).unwrap();

            let input_signal = SignalEntity::new(test_data, metadata).unwrap();
            let _ = pipeline.process(&input_signal).unwrap();
        }

        let performance = pipeline.performance_summary();
        assert!(performance.success_rate > 0.0);
        assert!(performance.processor_count > 0);
        assert!(performance.total_latency_us > 0);
    }

    #[test]
    fn test_processor_management() {
        let mut pipeline = Pipeline::realtime_emg().unwrap();

        let processor_names = pipeline.processor_names();
        assert!(!processor_names.is_empty());

        // Test finding processor
        if let Some(first_name) = processor_names.first() {
            assert!(pipeline.get_processor(first_name).is_some());
        }

        // Test latency check
        assert!(pipeline.meets_latency_requirement(1_000_000)); // 1 second should be plenty
    }

    #[test]
    fn test_pipeline_config_validation() {
        let config = PipelineConfig::default_emg();
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.name = String::new();
        assert!(invalid_config.validate().is_err());

        let mut invalid_latency = config.clone();
        invalid_latency.max_total_latency_ms = -1.0;
        assert!(invalid_latency.validate().is_err());
    }

    #[test]
    fn test_processing_chain() {
        let mut chain = ProcessingChain::new();

        let highpass = ButterworthFilter::new(FilterConfig::highpass(20.0, 4)).unwrap();
        chain.add_processor(Box::new(highpass));

        let feature_extractor = FeatureExtractor::emg_features(128);
        chain.add_processor(Box::new(feature_extractor));

        let test_data = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let metadata = EMGMetadata::new(
            EMGSignal::Surface {
                muscle_group: MuscleGroup::Biceps,
                activation_level: 0.5,
            },
            1000.0, 1, 0.256, 0.1
        ).unwrap();

        let input_signal = SignalEntity::new(test_data, metadata).unwrap();
        let output_signal = chain.process(&input_signal).unwrap();

        assert_eq!(output_signal.len(), input_signal.len());
    }

    #[test]
    fn test_bypass_mode() {
        // Create a pipeline that will have processing errors
        let mut builder = PipelineBuilder::new("Error Test");
        builder.set_bypass_enabled(true);

        // Add a filter that might fail with extreme parameters
        builder.add_highpass(10000.0, 20).unwrap(); // Unrealistic parameters

        let mut pipeline = builder.build();

        let test_data = vec![1.0f32; 100];
        let metadata = EMGMetadata::new(
            EMGSignal::Surface {
                muscle_group: MuscleGroup::Biceps,
                activation_level: 0.5,
            },
            1000.0, 1, 0.1, 0.1
        ).unwrap();

        let input_signal = SignalEntity::new(test_data, metadata).unwrap();

        // With bypass enabled, this should not fail even if processors fail
        let result = pipeline.process(&input_signal);
        assert!(result.is_ok());
    }
}