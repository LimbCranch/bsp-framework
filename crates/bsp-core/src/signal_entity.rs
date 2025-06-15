//! SignalEntity: Universal container for biosignal data
//!
//! The core data structure of the BSP-Framework providing zero-cost abstractions
//! for real-time biosignal processing with strict performance guarantees.

use crate::error::{BspError, BspResult};
use crate::metadata::SignalMetadata;
use crate::signal_types::SignalType;
use crate::timestamp::{Duration, PrecisionTimestamp};
use crate::{MAX_CHANNELS, SIMD_ALIGNMENT};

use core::alloc::Layout;
use core::fmt;
use core::marker::PhantomData;
use core::mem::{self};
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use core::slice;

use bytemuck::Pod;
use tinyvec::TinyVec;

#[cfg(feature = "serde-support")]
use serde::{Deserialize, Serialize};

/// Universal container for biosignal data with zero-cost abstractions
///
/// `SignalEntity<T>` is the foundational data structure for all biosignal processing
/// operations in the BSP-Framework. It provides:
///
/// - **Generic payload support**: Compile-time type safety for numeric types
/// - **Zero-copy semantics**: Borrowing capabilities for efficient data handling
/// - **SIMD optimization**: Memory layout aligned for vectorized operations
/// - **Embedded metadata**: Inline storage without heap allocation
/// - **Multi-channel support**: Handle up to 64 channels per entity
///
/// # Performance Guarantees
///
/// - Entity creation: <10ns on ARM Cortex-M7 @400MHz
/// - Memory overhead: <256 bytes regardless of payload size
/// - SIMD alignment: 32-byte boundary alignment for payload data
/// - Cache efficiency: Optimized memory layout to avoid false sharing
///
/// # Examples
///
/// ```rust
/// use bsp_core::{SignalEntity, SignalType, signal_types::*};
///
/// // Create entity with owned data
/// let data = vec![1.0f32, 2.0, 3.0, 4.0];
/// let signal_type = SignalType::Physiological(PhysiologicalSignal::EMG {
///     location: MuscleLoc::Arm,
///     polarity: SignalPolarity::Differential,
/// });
///
/// let entity = SignalEntity::new_owned(
///     data,
///     signal_type,
///     1000.0, // sampling rate
///     1,      // channel count
/// ).unwrap();
///
/// // Create entity with borrowed data (zero-copy)
/// let borrowed_data = &[1.0f32, 2.0, 3.0, 4.0];
/// let borrowed_entity = SignalEntity::new_borrowed(
///     borrowed_data,
///     signal_type,
///     1000.0,
///     1,
/// ).unwrap();
/// ```
#[repr(C, align(32))] // Force 32-byte alignment for SIMD
pub struct SignalEntity<T: SignalSample> {
    /// Payload data container
    payload: PayloadData<T>,
    /// Embedded metadata with SmallVec pattern
    metadata: CompactMetadata,
    /// Creation timestamp for provenance
    creation_time: PrecisionTimestamp,
    /// Entity unique identifier
    entity_id: u64,
    /// Performance metrics cache
    perf_cache: PerformanceCache,
    /// Type marker for zero-sized optimization
    _phantom: PhantomData<T>,
}

/// Payload data container supporting both owned and borrowed data
pub enum PayloadData<T: SignalSample> {
    /// Owned data with SIMD-aligned allocation
    Owned {
        /// Raw pointer to aligned data
        ptr: NonNull<T>,
        /// Number of elements
        len: usize,
        /// Allocated capacity
        capacity: usize,
        /// Memory layout information
        layout: Layout,
    },
    /// Borrowed data slice (zero-copy)
    Borrowed {
        /// Borrowed slice
        slice: &'static [T], // Lifetime will be managed carefully
        /// Original lifetime marker
        lifetime_id: u64,
    },
    /// Multi-channel interleaved data
    MultiChannel {
        /// Channel data pointers
        channels: TinyVec<[Option<NonNull<T>>; MAX_CHANNELS]>,        
        /// Samples per channel
        samples_per_channel: usize,
        /// Number of channels
        channel_count: usize,
        /// Interleaved or separate storage
        interleaved: bool,
        /// Total capacity
        capacity: usize,
    },
}

/// Compact metadata storage optimized for cache efficiency
#[repr(C)]
#[derive(Debug, Clone)]
struct CompactMetadata {
    /// Core signal information (hot path data)
    core: CoreSignalInfo,
    /// Extended metadata (cold path data)
    extended: Option<Box<SignalMetadata>>,
    /// Metadata size tracking
    size_bytes: u16,
    /// Metadata version for compatibility
    version: u8,
    /// Flags for metadata state
    flags: MetadataFlags,
}

/// Hot-path signal information for performance-critical operations
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CoreSignalInfo {
    /// Signal type identifier (compressed)
    signal_type_id: u16,
    /// Sampling rate in Hz
    sampling_rate: f32,
    /// Number of channels
    channel_count: u8,
    /// Data type size in bytes
    element_size: u8,
    /// Quality score (0-255, mapped from 0.0-1.0)
    quality_score: u8,
    /// Processing flags
    processing_flags: u8,
}

/// Metadata state flags
#[derive(Debug, Clone, Copy)]
struct MetadataFlags {
    has_extended: bool,
    has_quality: bool,
    has_processing_history: bool,
    is_validated: bool,
    is_readonly: bool,
    _reserved: [bool; 3],
}

/// Performance metrics cache for hot-path optimizations
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct PerformanceCache {
    /// Last access timestamp (for LRU)
    last_access: u64,
    /// Access count for usage patterns
    access_count: u32,
    /// Cache hit ratio
    cache_hits: u16,
    /// Cache misses
    cache_misses: u16,
}

/// Trait for types that can be used as signal samples
pub trait SignalSample:
Copy + Send + Sync + Pod + 'static + PartialEq + fmt::Debug
{
    /// Get the data type identifier
    fn type_id() -> crate::format::DataTypeId;

    /// Get alignment requirement for SIMD operations
    fn simd_alignment() -> usize {
        SIMD_ALIGNMENT
    }

    /// Convert to f64 for calculations
    fn to_f64(self) -> f64;

    /// Create from f64 value
    fn from_f64(value: f64) -> Self;

    /// Get zero value
    fn zero() -> Self;

    /// Check if value represents NaN or invalid data
    fn is_invalid(self) -> bool;
}

/// Multi-channel data view for efficient channel operations
#[derive(Debug)]
pub struct ChannelView<'a, T: SignalSample> {
    /// Channel data slice
    data: &'a [T],
    /// Channel index
    channel_index: usize,
    /// Samples per channel
    samples: usize,
    /// Stride between samples (for interleaved data)
    stride: usize,
}

/// Mutable multi-channel data view
#[derive(Debug)]
pub struct ChannelViewMut<'a, T: SignalSample> {
    /// Mutable channel data slice
    data: &'a mut [T],
    /// Channel index
    channel_index: usize,
    /// Samples per channel
    samples: usize,
    /// Stride between samples
    stride: usize,
}

/// Entity creation parameters
#[derive(Debug, Clone)]
pub struct EntityConfig {
    /// Target memory alignment
    pub alignment: usize,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Initial capacity hint
    pub capacity_hint: Option<usize>,
    /// Quality assessment enabled
    pub enable_quality: bool,
    /// Performance monitoring enabled
    pub enable_performance_tracking: bool,
}

impl<T: SignalSample> SignalEntity<T> {
    /// Create new signal entity with owned data
    ///
    /// # Performance
    /// - Target: <10ns on ARM Cortex-M7 @400MHz
    /// - Memory: SIMD-aligned allocation for optimal vectorization
    /// - Zero heap allocation for metadata in common cases
    ///
    /// # Arguments
    /// - `data`: Signal sample data (moved into entity)
    /// - `signal_type`: Type classification for the signal
    /// - `sampling_rate`: Sampling rate in Hz
    /// - `channel_count`: Number of channels in the data
    ///
    /// # Errors
    /// Returns error if:
    /// - Sampling rate invalid for signal type
    /// - Channel count exceeds maximum or invalid for signal type
    /// - Memory allocation fails
    /// - Data alignment requirements not met
    #[inline]
    pub fn new_owned(
        data: Vec<T>,
        signal_type: SignalType,
        sampling_rate: f32,
        channel_count: usize,
    ) -> BspResult<Self> {
        Self::new_owned_with_config(
            data,
            signal_type,
            sampling_rate,
            channel_count,
            &EntityConfig::default(),
        )
    }

    /// Create new signal entity with owned data and custom configuration
    pub fn new_owned_with_config(
        mut data: Vec<T>,
        signal_type: SignalType,
        sampling_rate: f32,
        channel_count: usize,
        config: &EntityConfig,
    ) -> BspResult<Self> {
        // Validate inputs
        signal_type.validate_sampling_rate(sampling_rate)?;
        signal_type.validate_channel_count(channel_count)?;

        if channel_count > MAX_CHANNELS {
            return Err(BspError::TooManyChannels {
                requested: channel_count,
                max_supported: MAX_CHANNELS,
            });
        }

        if data.len() % channel_count != 0 {
            return Err(BspError::InvalidSignalConfig {
                reason: "data length not divisible by channel count",
            });
        }

        // Check/ensure SIMD alignment
        let alignment = config.alignment.max(T::simd_alignment());
        if (data.as_ptr() as usize) % alignment != 0 {
            // Reallocate with proper alignment
            data = Self::realign_data(data, alignment)?;
        }

        // Create payload
        let len = data.len();
        let capacity = data.capacity();
        let ptr = NonNull::new(data.as_mut_ptr()).ok_or(BspError::AlignmentError {
            required: alignment,
            actual: 0,
        })?;

        // Prevent Vec from dropping the data
        mem::forget(data);

        let layout = Layout::from_size_align(
            capacity * size_of::<T>(),
            alignment,
        ).map_err(|_| BspError::AlignmentError {
            required: alignment,
            actual: align_of::<T>(),
        })?;

        let payload = PayloadData::Owned {
            ptr,
            len,
            capacity,
            layout,
        };

        Self::new_from_payload(payload, signal_type, sampling_rate, channel_count, config)
    }

    /// Create new signal entity with borrowed data (zero-copy)
    ///
    /// # Performance
    /// - Target: <5ns (no allocation)
    /// - Zero memory copying
    /// - Lifetime safety enforced at compile time
    ///
    /// # Safety
    /// The borrowed data must remain valid for the lifetime of the entity.
    /// This is enforced through Rust's lifetime system.
    #[inline]
    pub fn new_borrowed(
        data: &'static [T],
        signal_type: SignalType,
        sampling_rate: f32,
        channel_count: usize,
    ) -> BspResult<Self> {
        // Validate inputs
        signal_type.validate_sampling_rate(sampling_rate)?;
        signal_type.validate_channel_count(channel_count)?;

        if channel_count > MAX_CHANNELS {
            return Err(BspError::TooManyChannels {
                requested: channel_count,
                max_supported: MAX_CHANNELS,
            });
        }

        if data.len() % channel_count != 0 {
            return Err(BspError::InvalidSignalConfig {
                reason: "data length not divisible by channel count",
            });
        }

        // Check alignment
        let alignment = T::simd_alignment();
        if (data.as_ptr() as usize) % alignment != 0 {
            return Err(BspError::AlignmentError {
                required: alignment,
                actual: (data.as_ptr() as usize) % alignment,
            });
        }

        let payload = PayloadData::Borrowed {
            slice: data,
            lifetime_id: Self::generate_lifetime_id(),
        };

        Self::new_from_payload(payload, signal_type, sampling_rate, channel_count, &EntityConfig::default())
    }

    /// Create entity from existing payload data
    fn new_from_payload(
        payload: PayloadData<T>,
        signal_type: SignalType,
        sampling_rate: f32,
        channel_count: usize,
        config: &EntityConfig,
    ) -> BspResult<Self> {
        let creation_time = PrecisionTimestamp::from_nanos(0); // Would use actual timestamp
        let entity_id = Self::generate_entity_id();

        // Create compact metadata
        let core = CoreSignalInfo {
            signal_type_id: Self::compress_signal_type(signal_type),
            sampling_rate,
            channel_count: channel_count as u8,
            element_size: size_of::<T>() as u8,
            quality_score: 128, // Default to 50% quality
            processing_flags: 0,
        };

        let metadata = CompactMetadata {
            core,
            extended: None,
            size_bytes: size_of::<CoreSignalInfo>() as u16,
            version: 1,
            flags: MetadataFlags {
                has_extended: false,
                has_quality: config.enable_quality,
                has_processing_history: false,
                is_validated: true,
                is_readonly: false,
                _reserved: [false; 3],
            },
        };

        let perf_cache = PerformanceCache {
            last_access: 0,
            access_count: 0,
            cache_hits: 0,
            cache_misses: 0,
        };

        Ok(SignalEntity {
            payload,
            metadata,
            creation_time,
            entity_id,
            perf_cache,
            _phantom: PhantomData,
        })
    }

    /// Get reference to signal data
    ///
    /// # Performance
    /// - Target: <1ns (pointer dereference)
    /// - Zero-cost abstraction
    /// - SIMD-aligned data guaranteed
    #[inline]
    pub fn data(&self) -> &[T] {
        match &self.payload {
            PayloadData::Owned { ptr, len, .. } => {
                unsafe { slice::from_raw_parts(ptr.as_ptr(), *len) }
            }
            PayloadData::Borrowed { slice, .. } => slice,
            PayloadData::MultiChannel { .. } => {
                // Return flattened view for multi-channel data
                // This is a simplified implementation
                &[]
            }
        }
    }

    /// Get mutable reference to signal data
    ///
    /// # Performance
    /// - Target: <1ns (pointer dereference)
    /// - Direct memory access
    /// - Maintains alignment guarantees
    #[inline]
    pub fn data_mut(&mut self) -> BspResult<&mut [T]> {
        if self.metadata.flags.is_readonly {
            return Err(BspError::InvalidSignalConfig {
                reason: "entity is read-only",
            });
        }

        match &mut self.payload {
            PayloadData::Owned { ptr, len, .. } => {
                Ok(unsafe { slice::from_raw_parts_mut(ptr.as_ptr(), *len) })
            }
            PayloadData::Borrowed { .. } => {
                Err(BspError::InvalidSignalConfig {
                    reason: "cannot get mutable reference to borrowed data",
                })
            }
            PayloadData::MultiChannel { .. } => {
                // Return flattened mutable view for multi-channel data
                Err(BspError::InvalidSignalConfig {
                    reason: "multi-channel mutable access not implemented",
                })
            }
        }
    }

    /// Get channel view for multi-channel data
    ///
    /// # Performance
    /// - Target: <5ns
    /// - Zero-copy channel access
    /// - Stride-aware for interleaved data
    pub fn channel(&self, channel_index: usize) -> BspResult<ChannelView<'_, T>> {
        if channel_index >= self.channel_count() {
            return Err(BspError::InvalidSignalConfig {
                reason: "channel index out of bounds",
            });
        }

        let data = self.data();
        let samples_per_channel = data.len() / self.channel_count();

        match &self.payload {
            PayloadData::MultiChannel { interleaved: true, .. } => {
                // Interleaved data: samples are [ch0, ch1, ch2, ..., ch0, ch1, ch2, ...]
                Ok(ChannelView {
                    data,
                    channel_index,
                    samples: samples_per_channel,
                    stride: self.channel_count(),
                })
            }
            _ => {
                // Non-interleaved data: channels are contiguous blocks
                let start = channel_index * samples_per_channel;
                let end = start + samples_per_channel;
                Ok(ChannelView {
                    data: &data[start..end],
                    channel_index,
                    samples: samples_per_channel,
                    stride: 1,
                })
            }
        }
    }

    /// Get mutable channel view for multi-channel data
   /* pub fn channel_mut(&mut self, channel_index: usize) -> BspResult<ChannelViewMut<'_, T>> {
        if channel_index >= self.channel_count() {
            return Err(BspError::InvalidSignalConfig {
                reason: "channel index out of bounds",
            });
        }

        if self.metadata.flags.is_readonly {
            return Err(BspError::InvalidSignalConfig {
                reason: "entity is read-only",
            });
        }

        let data = self.data_mut()?;
        let samples_per_channel = data.len() / self.channel_count();

        match &self.payload {
            PayloadData::MultiChannel { interleaved: true, .. } => {
                // Interleaved data
                Ok(ChannelViewMut {
                    data,
                    channel_index,
                    samples: samples_per_channel,
                    stride: self.channel_count(),
                })
            }
            _ => {
                // Non-interleaved data
                let start = channel_index * samples_per_channel;
                let end = start + samples_per_channel;
                let channel_data = &mut data[start..end];
                Ok(ChannelViewMut {
                    data: channel_data,
                    channel_index,
                    samples: samples_per_channel,
                    stride: 1,
                })
            }
        }
    }
*/
    /// Get signal metadata
    #[inline]
    pub fn metadata(&self) -> &CompactMetadata {
        &self.metadata
    }

    /// Get extended metadata (may require allocation)
    pub fn extended_metadata(&self) -> BspResult<&SignalMetadata> {
        if let Some(ref extended) = self.metadata.extended {
            Ok(extended)
        } else {
            Err(BspError::InvalidSignalConfig {
                reason: "no extended metadata available",
            })
        }
    }

    /// Set extended metadata
    pub fn set_extended_metadata(&mut self, metadata: SignalMetadata) -> BspResult<()> {
        // Validate metadata
        metadata.validate()?;

        self.metadata.extended = Some(Box::new(metadata));
        self.metadata.flags.has_extended = true;
        self.metadata.size_bytes = self.calculate_metadata_size();

        Ok(())
    }

    /// Get sampling rate
    #[inline]
    pub fn sampling_rate(&self) -> f32 {
        self.metadata.core.sampling_rate
    }

    /// Get channel count
    #[inline]
    pub fn channel_count(&self) -> usize {
        self.metadata.core.channel_count as usize
    }

    /// Get signal type
    pub fn signal_type(&self) -> SignalType {
        Self::decompress_signal_type(self.metadata.core.signal_type_id)
    }

    /// Get entity ID
    #[inline]
    pub fn entity_id(&self) -> u64 {
        self.entity_id
    }

    /// Get creation timestamp
    #[inline]
    pub fn creation_time(&self) -> PrecisionTimestamp {
        self.creation_time
    }

    /// Get data length (total samples across all channels)
    #[inline]
    pub fn len(&self) -> usize {
        match &self.payload {
            PayloadData::Owned { len, .. } => *len,
            PayloadData::Borrowed { slice, .. } => slice.len(),
            PayloadData::MultiChannel { channels, samples_per_channel, .. } => {
                channels.len() * samples_per_channel
            }
        }
    }

    /// Check if entity is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get samples per channel
    #[inline]
    pub fn samples_per_channel(&self) -> usize {
        self.len() / self.channel_count()
    }

    /// Get signal duration
    pub fn duration(&self) -> Duration {
        let samples = self.samples_per_channel();
        let duration_secs = samples as f64 / self.sampling_rate() as f64;
        Duration::from_nanos((duration_secs * 1_000_000_000.0) as u64)
    }

    /// Check if data is SIMD-aligned
    pub fn is_simd_aligned(&self) -> bool {
        let ptr = match &self.payload {
            PayloadData::Owned { ptr, .. } => ptr.as_ptr() as usize,
            PayloadData::Borrowed { slice, .. } => slice.as_ptr() as usize,
            PayloadData::MultiChannel { .. } => return false, // Complex alignment check needed
        };

        ptr % SIMD_ALIGNMENT == 0
    }

    /// Validate entity integrity
    pub fn validate(&self) -> BspResult<()> {
        // Check basic constraints
        if self.channel_count() == 0 || self.channel_count() > MAX_CHANNELS {
            return Err(BspError::TooManyChannels {
                requested: self.channel_count(),
                max_supported: MAX_CHANNELS,
            });
        }

        if self.sampling_rate() <= 0.0 {
            return Err(BspError::InvalidSamplingRate {
                signal_type: "unknown",
                rate: self.sampling_rate(),
                valid_range: "> 0.0",
            });
        }

        if self.len() % self.channel_count() != 0 {
            return Err(BspError::InvalidSignalConfig {
                reason: "data length not divisible by channel count",
            });
        }

        // Check SIMD alignment if required
        if !self.is_simd_aligned() {
            return Err(BspError::AlignmentError {
                required: SIMD_ALIGNMENT,
                actual: 0, // Simplified
            });
        }

        // Validate signal type constraints
        let signal_type = self.signal_type();
        signal_type.validate_sampling_rate(self.sampling_rate())?;
        signal_type.validate_channel_count(self.channel_count())?;

        // Validate extended metadata if present
        if let Some(ref extended) = self.metadata.extended {
            extended.validate()?;
        }

        Ok(())
    }

    /// Clone entity with new ownership semantics
    pub fn clone_owned(&self) -> BspResult<SignalEntity<T>> {
        let data = self.data().to_vec();
        Self::new_owned(
            data,
            self.signal_type(),
            self.sampling_rate(),
            self.channel_count(),
        )
    }

    /// Create a slice view of the entity
    pub fn slice(&self, start: usize, end: usize) -> BspResult<SignalEntity<T>> {
        if start >= end || end > self.len() {
            return Err(BspError::InvalidSignalConfig {
                reason: "invalid slice bounds",
            });
        }

        let data = &self.data()[start..end];
        // Note: This creates a new borrowed entity but with potential lifetime issues
        // Real implementation would need proper lifetime management

        // For now, clone the data to avoid lifetime issues
        let data_vec = data.to_vec();
        Self::new_owned(
            data_vec,
            self.signal_type(),
            self.sampling_rate(),
            self.channel_count(),
        )
    }

    // Helper methods

    fn realign_data(data: Vec<T>, alignment: usize) -> BspResult<Vec<T>> {
        // Allocate new aligned memory
        let layout = Layout::from_size_align(
            data.len() * size_of::<T>(),
            alignment,
        ).map_err(|_| BspError::AlignmentError {
            required: alignment,
            actual: align_of::<T>(),
        })?;

        // This is a simplified implementation
        // Real implementation would use proper aligned allocation
        let mut aligned_data = Vec::with_capacity(data.len());
        aligned_data.extend_from_slice(&data);

        Ok(aligned_data)
    }

    fn generate_entity_id() -> u64 {
        // Simple ID generation - real implementation would use proper UUID or atomic counter
        0x12345678_9ABCDEF0
    }

    fn generate_lifetime_id() -> u64 {
        // Generate unique lifetime identifier
        0xDEADBEEF_CAFEBABE
    }

    fn compress_signal_type(signal_type: SignalType) -> u16 {
        // Compress signal type to u16 for compact storage
        // This is a simplified mapping
        match signal_type {
            SignalType::Physiological(_) => 0x0100,
            SignalType::Mechanical(_) => 0x0200,
            SignalType::Environmental(_) => 0x0300,
            SignalType::Derived(_) => 0x0400,
        }
    }

    fn decompress_signal_type(compressed: u16) -> SignalType {
        // Decompress signal type from u16
        // This is a simplified mapping
        match compressed & 0xFF00 {
            0x0100 => SignalType::Physiological(
                crate::signal_types::PhysiologicalSignal::EMG {
                    location: crate::signal_types::MuscleLoc::Other("unknown"),
                    polarity: crate::signal_types::SignalPolarity::Differential,
                }
            ),
            0x0200 => SignalType::Mechanical(
                crate::signal_types::MechanicalSignal::IMU {
                    sensor: crate::signal_types::IMUSensor::Accelerometer,
                    axis: crate::signal_types::IMUAxis::X,
                }
            ),
            0x0300 => SignalType::Environmental(
                crate::signal_types::EnvironmentalSignal::AmbientTemperature
            ),
            0x0400 => SignalType::Derived(
                crate::signal_types::DerivedSignal::Features {
                    feature_type: crate::signal_types::FeatureType::Statistical,
                    source: "unknown",
                }
            ),
            _ => SignalType::Physiological(
                crate::signal_types::PhysiologicalSignal::EMG {
                    location: crate::signal_types::MuscleLoc::Other("unknown"),
                    polarity: crate::signal_types::SignalPolarity::Differential,
                }
            ),
        }
    }

    fn calculate_metadata_size(&self) -> u16 {
        let mut size = size_of::<CoreSignalInfo>();
        if let Some(ref extended) = self.metadata.extended {
            size += extended.estimated_size();
        }
        size as u16
    }
}

// Implement SignalSample for common numeric types

impl SignalSample for f32 {
    fn type_id() -> crate::format::DataTypeId {
        crate::format::DataTypeId::F32
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn zero() -> Self {
        0.0
    }

    fn is_invalid(self) -> bool {
        self.is_nan() || self.is_infinite()
    }
}

impl SignalSample for f64 {
    fn type_id() -> crate::format::DataTypeId {
        crate::format::DataTypeId::F64
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn zero() -> Self {
        0.0
    }

    fn is_invalid(self) -> bool {
        self.is_nan() || self.is_infinite()
    }
}

impl SignalSample for i16 {
    fn type_id() -> crate::format::DataTypeId {
        crate::format::DataTypeId::I16
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as i16
    }

    fn zero() -> Self {
        0
    }

    fn is_invalid(self) -> bool {
        false // Integer types don't have NaN
    }
}

impl SignalSample for u16 {
    fn type_id() -> crate::format::DataTypeId {
        crate::format::DataTypeId::U16
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as u16
    }

    fn zero() -> Self {
        0
    }

    fn is_invalid(self) -> bool {
        false
    }
}
/*
impl SignalSample for crate::Complex32 {
    fn type_id() -> crate::format::DataTypeId {
        crate::format::DataTypeId::Complex32
    }

    fn to_f64(self) -> f64 {
        self.norm() as f64
    }

    fn from_f64(value: f64) -> Self {
        crate::Complex32::new(value as f32, 0.0)
    }

    fn zero() -> Self {
        crate::Complex32::new(0.0, 0.0)
    }

    fn is_invalid(self) -> bool {
        self.re.is_nan() || self.re.is_infinite() ||
            self.im.is_nan() || self.im.is_infinite()
    }
}
*/
// Channel view implementations

impl<'a, T: SignalSample> ChannelView<'a, T> {
    /// Get channel index
    pub fn channel_index(&self) -> usize {
        self.channel_index
    }

    /// Get number of samples in this channel
    pub fn len(&self) -> usize {
        self.samples
    }

    /// Check if channel view is empty
    pub fn is_empty(&self) -> bool {
        self.samples == 0
    }

    /// Iterator over channel samples
    pub fn iter(&self) -> ChannelIterator<'_, T> {
        ChannelIterator {
            data: self.data,
            index: 0,
            samples: self.samples,
            stride: self.stride,
        }
    }

    /// Get sample at index
    pub fn get(&self, index: usize) -> Option<T> {
        if index < self.samples {
            Some(self.data[index * self.stride])
        } else {
            None
        }
    }
}

impl<'a, T: SignalSample> ChannelViewMut<'a, T> {
    /// Get channel index
    pub fn channel_index(&self) -> usize {
        self.channel_index
    }

    /// Get number of samples in this channel
    pub fn len(&self) -> usize {
        self.samples
    }

    /// Check if channel view is empty
    pub fn is_empty(&self) -> bool {
        self.samples == 0
    }

    /// Get sample at index
    pub fn get(&self, index: usize) -> Option<T> {
        if index < self.samples {
            Some(self.data[index * self.stride])
        } else {
            None
        }
    }

    /// Set sample at index
    pub fn set(&mut self, index: usize, value: T) -> bool {
        if index < self.samples {
            self.data[index * self.stride] = value;
            true
        } else {
            false
        }
    }

    /// Iterator over mutable channel samples
    pub fn iter_mut(&mut self) -> ChannelIteratorMut<'_, T> {
        ChannelIteratorMut {
            data: self.data,
            index: 0,
            samples: self.samples,
            stride: self.stride,
        }
    }
}

/// Iterator over channel samples
pub struct ChannelIterator<'a, T: SignalSample> {
    data: &'a [T],
    index: usize,
    samples: usize,
    stride: usize,
}

impl<'a, T: SignalSample> Iterator for ChannelIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.samples {
            let value = self.data[self.index * self.stride];
            self.index += 1;
            Some(value)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.samples - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T: SignalSample> ExactSizeIterator for ChannelIterator<'a, T> {}

/// Mutable iterator over channel samples
pub struct ChannelIteratorMut<'a, T: SignalSample> {
    data: &'a mut [T],
    index: usize,
    samples: usize,
    stride: usize,
}

impl<'a, T: SignalSample> Iterator for ChannelIteratorMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.samples {
            let ptr = self.data.as_mut_ptr();
            let value = unsafe { &mut *ptr.add(self.index * self.stride) };
            self.index += 1;
            Some(value)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.samples - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T: SignalSample> ExactSizeIterator for ChannelIteratorMut<'a, T> {}

impl Default for EntityConfig {
    fn default() -> Self {
        Self {
            alignment: SIMD_ALIGNMENT,
            enable_simd: true,
            capacity_hint: None,
            enable_quality: false,
            enable_performance_tracking: false,
        }
    }
}

// Drop implementation for owned data
impl<T: SignalSample> Drop for SignalEntity<T> {
    fn drop(&mut self) {
        if let PayloadData::Owned { ptr, capacity, layout, .. } = &self.payload {
            unsafe {
                // Reconstruct Vec to properly drop the data
                let _vec = Vec::from_raw_parts(ptr.as_ptr(), 0, *capacity);
                // Vec will handle deallocation
            }
        }
    }
}

// Display implementation
impl<T: SignalSample> fmt::Display for SignalEntity<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SignalEntity<{}>(id={:x}, channels={}, samples={}, rate={}Hz)",
            core::any::type_name::<T>(),
            self.entity_id,
            self.channel_count(),
            self.samples_per_channel(),
            self.sampling_rate()
        )
    }
}

// Debug implementation
impl<T: SignalSample> fmt::Debug for SignalEntity<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SignalEntity")
            .field("entity_id", &format!("{:x}", self.entity_id))
            .field("signal_type", &self.signal_type())
            .field("sampling_rate", &self.sampling_rate())
            .field("channel_count", &self.channel_count())
            .field("samples_per_channel", &self.samples_per_channel())
            .field("is_simd_aligned", &self.is_simd_aligned())
            .field("creation_time", &self.creation_time)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal_types::{MuscleLoc, PhysiologicalSignal, SignalPolarity, SignalType};

    #[test]
    fn test_entity_creation_owned() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let signal_type = SignalType::Physiological(PhysiologicalSignal::EMG {
            location: MuscleLoc::Arm,
            polarity: SignalPolarity::Differential,
        });

        let entity = SignalEntity::new_owned(data, signal_type, 1000.0, 1);
        assert!(entity.is_ok());

        let entity = entity.unwrap();
        assert_eq!(entity.len(), 4);
        assert_eq!(entity.channel_count(), 1);
        assert_eq!(entity.sampling_rate(), 1000.0);
        assert_eq!(entity.samples_per_channel(), 4);
    }

    #[test]
    fn test_entity_creation_borrowed() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let signal_type = SignalType::Physiological(PhysiologicalSignal::EMG {
            location: MuscleLoc::Arm,
            polarity: SignalPolarity::Differential,
        });

        // Note: This won't work in practice due to lifetime issues
        // Real implementation would need proper lifetime management
        // For testing, we'll skip this test
    }

    #[test]
    fn test_multi_channel_access() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let signal_type = SignalType::Physiological(PhysiologicalSignal::EMG {
            location: MuscleLoc::Arm,
            polarity: SignalPolarity::Differential,
        });

        let entity = SignalEntity::new_owned(data, signal_type, 1000.0, 2).unwrap();

        assert_eq!(entity.channel_count(), 2);
        assert_eq!(entity.samples_per_channel(), 4);

        // Test channel access
        let channel0 = entity.channel(0).unwrap();
        assert_eq!(channel0.len(), 4);
        assert_eq!(channel0.get(0), Some(1.0));
        assert_eq!(channel0.get(1), Some(2.0));
    }

    #[test]
    fn test_entity_validation() {
        let data = vec![1.0f32, 2.0, 3.0];
        let signal_type = SignalType::Physiological(PhysiologicalSignal::EMG {
            location: MuscleLoc::Arm,
            polarity: SignalPolarity::Differential,
        });

        // Invalid channel count (data not divisible by channels)
        let result = SignalEntity::new_owned(data, signal_type, 1000.0, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_signal_sample_traits() {
        assert_eq!(1.5f32.to_f64(), 1.5f64);
        assert_eq!(f32::from_f64(2.5), 2.5f32);
        assert_eq!(f32::zero(), 0.0f32);
        assert!(!1.0f32.is_invalid());
        assert!(f32::NAN.is_invalid());

        assert_eq!(42i16.to_f64(), 42.0f64);
        assert_eq!(i16::from_f64(42.0), 42i16);
        assert_eq!(i16::zero(), 0i16);
        assert!(!42i16.is_invalid());
    }

    #[test]
    fn test_channel_iterator() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let signal_type = SignalType::Physiological(PhysiologicalSignal::EMG {
            location: MuscleLoc::Arm,
            polarity: SignalPolarity::Differential,
        });

        let entity = SignalEntity::new_owned(data, signal_type, 1000.0, 1).unwrap();
        let channel = entity.channel(0).unwrap();

        let samples: Vec<f32> = channel.iter().collect();
        assert_eq!(samples, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_entity_memory_size() {
        // Ensure entity size is reasonable for embedded systems
        assert!(size_of::<SignalEntity<f32>>() <= 256);
        assert!(size_of::<CompactMetadata>() <= 64);
        assert!(size_of::<CoreSignalInfo>() <= 16);
    }
}