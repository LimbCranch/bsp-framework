//! Error handling for BSP-Core

use std::fmt;

pub type BspResult<T> = Result<T, BspError>;

#[derive(Debug, Clone)]
pub enum BspError {
    InvalidSamplingRate { rate: f32, valid_range: String },
    InvalidChannelCount { count: usize, max: usize },
    InvalidSignalData { reason: String },
    SimulationError { message: String },
    ConfigurationError { message: String },
    ProcessingError { message: String },
}

impl fmt::Display for BspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BspError::InvalidSamplingRate { rate, valid_range } => {
                write!(f, "Invalid sampling rate {}Hz, valid range: {}", rate, valid_range)
            }
            BspError::InvalidChannelCount { count, max } => {
                write!(f, "Invalid channel count {}, max supported: {}", count, max)
            }
            BspError::InvalidSignalData { reason } => {
                write!(f, "Invalid signal data: {}", reason)
            }
            BspError::SimulationError { message } => {
                write!(f, "Simulation error: {}", message)
            }
            BspError::ConfigurationError { message } => {
                write!(f, "Configuration error: {}", message)
            }

            BspError::ProcessingError { message } => {
                write!(f, "Processing error: {}", message)
            }
        }
    }
}

impl std::error::Error for BspError {}