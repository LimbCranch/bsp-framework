
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, clippy::all)]

//! # BSP-Framework: High-Performance Biosignal Processing
//!
//! A real-time, embedded-friendly framework for processing biosignals with
//! strict performance guarantees and medical device compliance.
//!
//! ## Features
//!
//! - **Zero-cost abstractions**: Compile-time optimizations for real-time systems
//! - **no_std compatible**: Runs on embedded systems without heap allocation
//! - **Standards compliant**: IEEE 11073, EDF+, ISO 14155 compliance
//! - **Multi-channel support**: Handle up to 64 channels per signal entity
//! - **SIMD optimized**: Cache-friendly memory layouts for vectorized operations

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate core as std;

// Re-export commonly used types
pub use num_complex::Complex;
pub use uuid::Uuid;

// Core modules
pub mod signal_entity;
pub mod metadata;
pub mod signal_types;
pub mod quality;
pub mod timestamp;
pub mod format;
pub mod error;

// Re-exports for convenience
pub use signal_entity::{SignalEntity, PayloadData};
pub use metadata::{SignalMetadata, DeviceInfo, QualityMetrics};
pub use signal_types::{SignalType, PhysiologicalSignal, MechanicalSignal};
pub use quality::{QualityAssessment, QualityDimension};
pub use timestamp::{PrecisionTimestamp, TimestampProvider};
pub use error::{BspError, BspResult};

/// Complex number type alias for f32
pub type Complex32 = Complex<f32>;

/// Complex number type alias for f64  
pub type Complex64 = Complex<f64>;

/// Maximum number of supported channels per signal entity
pub const MAX_CHANNELS: usize = 64;

/// Maximum metadata size to avoid heap allocation
pub const MAX_METADATA_SIZE: usize = 256;

/// SIMD alignment requirement for payload data
pub const SIMD_ALIGNMENT: usize = 32;

/// Framework version for compatibility checking
pub const FRAMEWORK_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::*;

    // Compile-time assertions for critical constraints
    const_assert!(MAX_CHANNELS <= 64);
    const_assert!(MAX_METADATA_SIZE <= 256);
    const_assert!(SIMD_ALIGNMENT == 32);
}