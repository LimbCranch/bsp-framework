//! BSP-Processing: Signal processing pipeline for biosignals
//!
//! Real-time signal processing with filters, feature extraction, and analysis.

pub mod pipeline;
pub mod filters;
pub mod features;
pub mod processor;
pub mod config;

pub use pipeline::*;
pub use processor::{SignalProcessor, ProcessorConfig, ProcessingResult};
pub use filters::{
    FilterType, ButterworthFilter, NotchFilter, MovingAverageFilter,
    FilterConfig, FilterBank
};
pub use features::{
    FeatureExtractor, TimeFeatures, FrequencyFeatures, FeatureSet,
    FeatureConfig, StatisticalFeatures
};
pub use config::{ProcessingConfig, PipelineConfig};