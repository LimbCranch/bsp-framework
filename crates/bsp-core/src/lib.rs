//! BSP-Core: Foundation types for biosignal processing
//!
//! Minimal core types for MVP focusing on EMG signals only.

pub mod signal_entity;
pub mod emg_types;
pub mod error;

pub use signal_entity::*;
pub use emg_types::*;
pub use error::{BspError, BspResult};