//! Error handling for the BSP Framework
//!
//! Provides comprehensive error types for all framework operations,
//! designed to work in no_std environments.

use core::fmt;

/// Result type alias for BSP Framework operations
pub type BspResult<T> = Result<T, BspError>;

/// Comprehensive error type for all BSP Framework operations
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BspError {
    /// Invalid signal configuration
    InvalidSignalConfig {
        /// Description of the configuration error
        reason: &'static str,
    },

    /// Channel count exceeds maximum supported
    TooManyChannels {
        /// Requested channel count
        requested: usize,
        /// Maximum supported channels
        max_supported: usize,
    },

    /// Invalid sampling rate for signal type
    InvalidSamplingRate {
        /// Signal type that has invalid rate
        signal_type: &'static str,
        /// Provided sampling rate
        rate: f32,
        /// Valid range description
        valid_range: &'static str,
    },

    /// Metadata size exceeds limits
    MetadataTooLarge {
        /// Size of metadata in bytes
        size: usize,
        /// Maximum allowed size
        max_size: usize,
    },

    /// Timestamp validation error
    InvalidTimestamp {
        /// Description of timestamp issue
        reason: &'static str,
    },

    /// Quality assessment failure
    QualityAssessmentFailed {
        /// Quality dimension that failed
        dimension: &'static str,
        /// Measured value
        measured: f32,
        /// Required threshold
        threshold: f32,
    },

    /// Data alignment error for SIMD operations
    AlignmentError {
        /// Required alignment in bytes
        required: usize,
        /// Actual alignment found
        actual: usize,
    },

    /// Format parsing error
    FormatError {
        /// Description of format issue
        reason: &'static str,
    },

    /// Device identification error
    DeviceError {
        /// Device-related error description
        reason: &'static str,
    },

    /// Serialization/deserialization error
    #[cfg(feature = "serde-support")]
    SerializationError {
        /// Serialization error description
        reason: &'static str,
    },

    /// Cryptographic operation error
    #[cfg(feature = "crypto")]
    CryptoError {
        /// Cryptographic error description
        reason: &'static str,
    },

    /// Buffer capacity exceeded
    BufferOverflow {
        /// Available capacity
        capacity: usize,
        /// Requested size
        requested: usize,
    },

    /// Invalid data type conversion
    TypeConversionError {
        /// Source type description
        from: &'static str,
        /// Target type description
        to: &'static str,
    },
}

impl fmt::Display for BspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BspError::InvalidSignalConfig { reason } => {
                write!(f, "Invalid signal configuration: {}", reason)
            }
            BspError::TooManyChannels { requested, max_supported } => {
                write!(f, "Too many channels: requested {}, max supported {}",
                       requested, max_supported)
            }
            BspError::InvalidSamplingRate { signal_type, rate, valid_range } => {
                write!(f, "Invalid sampling rate for {}: {}Hz, valid range: {}",
                       signal_type, rate, valid_range)
            }
            BspError::MetadataTooLarge { size, max_size } => {
                write!(f, "Metadata too large: {} bytes, max allowed: {} bytes",
                       size, max_size)
            }
            BspError::InvalidTimestamp { reason } => {
                write!(f, "Invalid timestamp: {}", reason)
            }
            BspError::QualityAssessmentFailed { dimension, measured, threshold } => {
                write!(f, "Quality assessment failed for {}: measured {}, threshold {}",
                       dimension, measured, threshold)
            }
            BspError::AlignmentError { required, actual } => {
                write!(f, "Data alignment error: required {} bytes, actual {} bytes",
                       required, actual)
            }
            BspError::FormatError { reason } => {
                write!(f, "Format error: {}", reason)
            }
            BspError::DeviceError { reason } => {
                write!(f, "Device error: {}", reason)
            }
            #[cfg(feature = "serde-support")]
            BspError::SerializationError { reason } => {
                write!(f, "Serialization error: {}", reason)
            }
            #[cfg(feature = "crypto")]
            BspError::CryptoError { reason } => {
                write!(f, "Cryptographic error: {}", reason)
            }
            BspError::BufferOverflow { capacity, requested } => {
                write!(f, "Buffer overflow: capacity {}, requested {}",
                       capacity, requested)
            }
            BspError::TypeConversionError { from, to } => {
                write!(f, "Type conversion error: cannot convert from {} to {}",
                       from, to)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BspError {}

/// Convenience macro for creating configuration errors
#[macro_export]
macro_rules! config_error {
    ($reason:literal) => {
        $crate::error::BspError::InvalidSignalConfig { 
            reason: $reason 
        }
    };
}

/// Convenience macro for creating format errors
#[macro_export]
macro_rules! format_error {
    ($reason:literal) => {
        $crate::error::BspError::FormatError { 
            reason: $reason 
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let error = BspError::TooManyChannels {
            requested: 128,
            max_supported: 64
        };
        let display = format!("{}", error);
        assert!(display.contains("Too many channels"));
        assert!(display.contains("128"));
        assert!(display.contains("64"));
    }

    #[test]
    fn test_error_equality() {
        let error1 = BspError::InvalidSignalConfig {
            reason: "test"
        };
        let error2 = BspError::InvalidSignalConfig {
            reason: "test"
        };
        assert_eq!(error1, error2);
    }
}