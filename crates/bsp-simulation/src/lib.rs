//! BSP-Simulation: EMG signal generation and simulation
//!
//! Provides realistic EMG signal simulation for testing and development.

pub mod real_time_stream;
pub mod signal_patterns;
pub mod emg_simulator;

pub use emg_simulator::*;
pub use real_time_stream::*;
pub use signal_patterns::*;