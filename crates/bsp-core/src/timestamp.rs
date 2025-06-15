//! High-precision timestamp system for biosignal data
//!
//! Provides microsecond-resolution monotonic timestamps essential for
//! real-time biosignal processing and medical device data integrity.

use crate::error::{BspError, BspResult};
use core::fmt;
use core::ops::{Add, Sub};

#[cfg(feature = "serde-support")]
use serde::{Deserialize, Serialize};

/// High-precision timestamp with microsecond resolution
///
/// Uses a 64-bit representation for nanosecond precision while maintaining
/// microsecond accuracy guarantees for biosignal applications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
#[repr(transparent)]
pub struct PrecisionTimestamp {
    /// Nanoseconds since Unix epoch (monotonic)
    nanos: u64,
}

impl PrecisionTimestamp {
    /// Create a new timestamp from nanoseconds since Unix epoch
    #[inline]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self { nanos }
    }

    /// Create a new timestamp from microseconds since Unix epoch
    #[inline]
    pub const fn from_micros(micros: u64) -> Self {
        Self {
            nanos: micros * 1_000
        }
    }

    /// Create a new timestamp from milliseconds since Unix epoch
    #[inline]
    pub const fn from_millis(millis: u64) -> Self {
        Self {
            nanos: millis * 1_000_000
        }
    }

    /// Create a new timestamp from seconds since Unix epoch
    #[inline]
    pub const fn from_secs(secs: u64) -> Self {
        Self {
            nanos: secs * 1_000_000_000
        }
    }

    /// Get nanoseconds since Unix epoch
    #[inline]
    pub const fn as_nanos(&self) -> u64 {
        self.nanos
    }

    /// Get microseconds since Unix epoch
    #[inline]
    pub const fn as_micros(&self) -> u64 {
        self.nanos / 1_000
    }

    /// Get milliseconds since Unix epoch
    #[inline]
    pub const fn as_millis(&self) -> u64 {
        self.nanos / 1_000_000
    }

    /// Get seconds since Unix epoch
    #[inline]
    pub const fn as_secs(&self) -> u64 {
        self.nanos / 1_000_000_000
    }

    /// Get fractional seconds as f64
    #[inline]
    pub fn as_secs_f64(&self) -> f64 {
        self.nanos as f64 / 1_000_000_000.0
    }

    /// Calculate duration since another timestamp
    #[inline]
    pub fn duration_since(&self, earlier: PrecisionTimestamp) -> BspResult<Duration> {
        if self.nanos >= earlier.nanos {
            Ok(Duration::from_nanos(self.nanos - earlier.nanos))
        } else {
            Err(BspError::InvalidTimestamp {
                reason: "timestamp is earlier than reference"
            })
        }
    }

    /// Add a duration to this timestamp
    #[inline]
    pub fn add_duration(&self, duration: Duration) -> BspResult<PrecisionTimestamp> {
        self.nanos.checked_add(duration.as_nanos())
            .map(PrecisionTimestamp::from_nanos)
            .ok_or(BspError::InvalidTimestamp {
                reason: "timestamp overflow"
            })
    }

    /// Subtract a duration from this timestamp
    #[inline]
    pub fn sub_duration(&self, duration: Duration) -> BspResult<PrecisionTimestamp> {
        self.nanos.checked_sub(duration.as_nanos())
            .map(PrecisionTimestamp::from_nanos)
            .ok_or(BspError::InvalidTimestamp {
                reason: "timestamp underflow"
            })
    }

    /// Check if timestamp is within valid range for biosignal data
    /// (between year 2000 and year 2100)
    pub fn validate(&self) -> BspResult<()> {
        const YEAR_2000_NANOS: u64 = 946_684_800 * 1_000_000_000;
        const YEAR_2100_NANOS: u64 = 4_102_444_800 * 1_000_000_000;

        if self.nanos < YEAR_2000_NANOS {
            Err(BspError::InvalidTimestamp {
                reason: "timestamp before year 2000"
            })
        } else if self.nanos > YEAR_2100_NANOS {
            Err(BspError::InvalidTimestamp {
                reason: "timestamp after year 2100"
            })
        } else {
            Ok(())
        }
    }
}

impl fmt::Display for PrecisionTimestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let secs = self.as_secs();
        let subsec_nanos = self.nanos % 1_000_000_000;
        write!(f, "{}.{:09}", secs, subsec_nanos)
    }
}

/// Duration type with nanosecond precision
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct Duration {
    nanos: u64,
}

impl Duration {
    /// Create duration from nanoseconds
    #[inline]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self { nanos }
    }

    /// Create duration from microseconds
    #[inline]
    pub const fn from_micros(micros: u64) -> Self {
        Self { nanos: micros * 1_000 }
    }

    /// Create duration from milliseconds
    #[inline]
    pub const fn from_millis(millis: u64) -> Self {
        Self { nanos: millis * 1_000_000 }
    }

    /// Create duration from seconds
    #[inline]
    pub const fn from_secs(secs: u64) -> Self {
        Self { nanos: secs * 1_000_000_000 }
    }

    /// Get duration as nanoseconds
    #[inline]
    pub const fn as_nanos(&self) -> u64 {
        self.nanos
    }

    /// Get duration as microseconds
    #[inline]
    pub const fn as_micros(&self) -> u64 {
        self.nanos / 1_000
    }

    /// Get duration as milliseconds
    #[inline]
    pub const fn as_millis(&self) -> u64 {
        self.nanos / 1_000_000
    }

    /// Get duration as seconds
    #[inline]
    pub const fn as_secs(&self) -> u64 {
        self.nanos / 1_000_000_000
    }

    /// Get duration as fractional seconds
    #[inline]
    pub fn as_secs_f64(&self) -> f64 {
        self.nanos as f64 / 1_000_000_000.0
    }
}

impl Add for Duration {
    type Output = Duration;

    #[inline]
    fn add(self, other: Duration) -> Duration {
        Duration::from_nanos(self.nanos + other.nanos)
    }
}

impl Sub for Duration {
    type Output = Duration;

    #[inline]
    fn sub(self, other: Duration) -> Duration {
        Duration::from_nanos(self.nanos - other.nanos)
    }
}

/// Trait for providing timestamps
pub trait TimestampProvider {
    /// Get current timestamp
    fn now() -> BspResult<PrecisionTimestamp>;

    /// Get a monotonic timestamp (guaranteed non-decreasing)
    fn now_monotonic() -> BspResult<PrecisionTimestamp>;
}

/// System timestamp provider using standard library
#[cfg(feature = "std")]
pub struct SystemTimestampProvider;

#[cfg(feature = "std")]
impl TimestampProvider for SystemTimestampProvider {
    fn now() -> BspResult<PrecisionTimestamp> {
        use std::time::{SystemTime, UNIX_EPOCH};

        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| PrecisionTimestamp::from_nanos(d.as_nanos() as u64))
            .map_err(|_| BspError::InvalidTimestamp {
                reason: "system time before Unix epoch"
            })
    }

    fn now_monotonic() -> BspResult<PrecisionTimestamp> {
        // Use Instant for monotonic time, but convert to timestamp
        // This is an approximation - real implementations should use
        // platform-specific monotonic clocks
        Self::now()
    }
}

/// Mock timestamp provider for testing
#[cfg(test)]
pub struct MockTimestampProvider {
    timestamp: core::cell::Cell<u64>,
}

#[cfg(test)]
impl MockTimestampProvider {
    pub fn new(initial_nanos: u64) -> Self {
        Self {
            timestamp: core::cell::Cell::new(initial_nanos),
        }
    }

    pub fn advance(&self, nanos: u64) {
        self.timestamp.set(self.timestamp.get() + nanos);
    }
}

#[cfg(test)]
impl TimestampProvider for MockTimestampProvider {
    fn now() -> BspResult<PrecisionTimestamp> {
        // This won't work for multiple instances - real implementation
        // would need thread-local storage or different approach
        Ok(PrecisionTimestamp::from_nanos(1_000_000_000_000)) // Fixed for testing
    }

    fn now_monotonic() -> BspResult<PrecisionTimestamp> {
        Self::now()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_creation() {
        let ts = PrecisionTimestamp::from_micros(1_000_000);
        assert_eq!(ts.as_micros(), 1_000_000);
        assert_eq!(ts.as_millis(), 1_000);
        assert_eq!(ts.as_secs(), 1);
    }

    #[test]
    fn test_timestamp_ordering() {
        let ts1 = PrecisionTimestamp::from_secs(100);
        let ts2 = PrecisionTimestamp::from_secs(200);
        assert!(ts1 < ts2);
    }

    #[test]
    fn test_duration_calculation() {
        let ts1 = PrecisionTimestamp::from_secs(100);
        let ts2 = PrecisionTimestamp::from_secs(105);
        let duration = ts2.duration_since(ts1).unwrap();
        assert_eq!(duration.as_secs(), 5);
    }

    #[test]
    fn test_timestamp_validation() {
        let valid_ts = PrecisionTimestamp::from_secs(1_600_000_000); // ~2020
        assert!(valid_ts.validate().is_ok());

        let invalid_ts = PrecisionTimestamp::from_secs(100); // 1970
        assert!(invalid_ts.validate().is_err());
    }

    #[test]
    fn test_duration_arithmetic() {
        let d1 = Duration::from_secs(5);
        let d2 = Duration::from_secs(3);
        let sum = d1 + d2;
        assert_eq!(sum.as_secs(), 8);

        let diff = d1 - d2;
        assert_eq!(diff.as_secs(), 2);
    }
}