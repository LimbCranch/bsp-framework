//! Data format handling for biosignal data
//!
//! Supports multiple data formats including EDF+, XDF, and custom binary formats
//! with efficient serialization for real-time and embedded systems.

use crate::error::{BspError, BspResult};
use crate::metadata::{SignalMetadata, FormatVersion};
use crate::timestamp::PrecisionTimestamp;
use core::fmt;
use core::str::FromStr;
use heapless::{Vec as HVec, String};
use bytemuck::{Pod, Zeroable, cast_slice, try_cast_slice};

#[cfg(feature = "serde-support")]
use serde::{Deserialize, Serialize};


macro_rules! offset_of {
    ($type:ty, $field:ident) => {
        unsafe {
            let uninit = core::mem::MaybeUninit::<$type>::uninit();
            let ptr = uninit.as_ptr();
            core::ptr::addr_of!((*ptr).$field) as usize - ptr as usize
        }
    };
}

pub(crate) use offset_of;




/// Data format types supported by the framework
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum DataFormat {
    /// Native BSP binary format
    BSPNative,
    /// European Data Format Plus
    EDFPlus,
    /// Extensible Data Format
    XDF,
    /// Custom binary format
    CustomBinary,
    /// JSON format (requires std feature)
    #[cfg(feature = "json")]
    JSON,
}

/// Binary header for BSP native format
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct BinaryHeader {
    /// Magic bytes "BSP\0"
    pub magic: [u8; 4],
    /// Format version
    pub version: u32,
    /// Header size in bytes
    pub header_size: u32,
    /// Metadata size in bytes
    pub metadata_size: u32,
    /// Payload size in bytes
    pub payload_size: u64,
    /// Number of channels
    pub channel_count: u32,
    /// Samples per channel
    pub samples_per_channel: u64,
    /// Data type identifier
    pub data_type: u32,
    /// Sampling rate (Hz)
    pub sampling_rate: f32,
    /// Timestamp (nanoseconds since epoch)
    pub timestamp: u64,
    /// Checksum of header
    pub header_checksum: u32,
    /// Checksum of payload
    pub payload_checksum: u32,
}

unsafe impl Pod for BinaryHeader {}
unsafe impl Zeroable for BinaryHeader {}

/// Data type identifiers for serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DataTypeId {
    F32 = 1,
    F64 = 2,
    I16 = 3,
    I32 = 4,
    U16 = 5,
    U32 = 6,
    Complex32 = 7,
    Complex64 = 8,
}

/// Serialization options
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct SerializationOptions {
    /// Target format
    pub format: DataFormat,
    /// Include metadata
    pub include_metadata: bool,
    /// Compress data
    pub compress: bool,
    /// Byte order (true = little endian)
    pub little_endian: bool,
    /// Alignment requirement
    pub alignment: usize,
}

/// Deserialization options
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct DeserializationOptions {
    /// Expected format
    pub expected_format: Option<DataFormat>,
    /// Validate checksums
    pub validate_checksums: bool,
    /// Strict validation
    pub strict_validation: bool,
}

/// Format handler trait for different data formats
/// 
///

pub trait FormatHandler {
    /// Возвращает идентификатор формата
    fn format_id(&self) -> DataFormat;

    /// Проверяет метаданные сигнала
    fn validate(&self, metadata: &SignalMetadata) -> BspResult<()>;
}


/// Обобщённый трейт для сериализации/десериализации конкретного типа T
pub trait FormatHandlerGeneric<T: Pod> {
    fn serialize(
        &self,
        data: &[T],
        metadata: &SignalMetadata,
        options: &SerializationOptions,
    ) -> BspResult<HVec<u8, 4096>>;

    fn deserialize(
        &self,
        bytes: &[u8],
        options: &DeserializationOptions,
    ) -> BspResult<(HVec<T, 1024>, SignalMetadata)>;
}




/*pub trait FormatHandler {
    /// Serialize data to bytes
    fn serialize<T: Pod>(
        &self,
        data: &[T],
        metadata: &SignalMetadata,
        options: &SerializationOptions,
    ) -> BspResult<HVec<u8, 4096>>;

    /// Deserialize data from bytes
    fn deserialize<T: Pod>(
        &self,
        bytes: &[u8],
        options: &DeserializationOptions,
    ) -> BspResult<(HVec<T, 1024>, SignalMetadata)>;

    /// Get format identifier
    fn format_id(&self) -> DataFormat;

    /// Validate format-specific constraints
    fn validate(&self, metadata: &SignalMetadata) -> BspResult<()>;
}*/

/// BSP native format handler
pub struct BSPNativeHandler;

/// EDF+ format handler
pub struct EDFPlusHandler;

/// XDF format handler  
pub struct XDFHandler;






impl BSPNativeHandler {
    /// Create new BSP native format handler
    pub const fn new() -> Self {
        Self
    }

    /// Calculate checksum for data
    fn calculate_checksum(data: &[u8]) -> u32 {
        // Simple CRC32-like checksum
        let mut checksum = 0u32;
        for &byte in data {
            checksum = checksum.wrapping_mul(31).wrapping_add(byte as u32);
        }
        checksum
    }

    /// Create binary header
    fn create_header<T: Pod>(
        data: &[T],
        metadata: &SignalMetadata,
        metadata_bytes: &[u8],
    ) -> BspResult<BinaryHeader> {
        let data_type = Self::get_data_type_id::<T>()?;
        let payload_bytes = bytemuck::cast_slice(data);

        let header = BinaryHeader {
            magic: *b"BSP\0",
            version: Self::format_version_to_u32(&FormatVersion::current()),
            header_size: core::mem::size_of::<BinaryHeader>() as u32,
            metadata_size: metadata_bytes.len() as u32,
            payload_size: payload_bytes.len() as u64,
            channel_count: metadata.acquisition.channel_count as u32,
            samples_per_channel: (data.len() / metadata.acquisition.channel_count) as u64,
            data_type: data_type as u32,
            sampling_rate: metadata.acquisition.sampling_rate,
            timestamp: metadata.udf_header.timestamp.as_nanos(),
            header_checksum: 0, // Will be calculated
            payload_checksum: Self::calculate_checksum(payload_bytes),
        };

        // Calculate header checksum (excluding the checksum field itself)
        let header_bytes = bytemuck::bytes_of(&header);
        let checksum_offset = offset_of!(BinaryHeader, header_checksum);
        let pre_checksum = &header_bytes[..checksum_offset];
        let post_checksum = &header_bytes[checksum_offset + 4..];

        let mut checksum_data = HVec::<u8, 256>::new();
        checksum_data.extend_from_slice(pre_checksum).map_err(|_| {
            BspError::BufferOverflow { capacity: 256, requested: pre_checksum.len() }
        })?;
        checksum_data.extend_from_slice(post_checksum).map_err(|_| {
            BspError::BufferOverflow { capacity: 256, requested: post_checksum.len() }
        })?;

        let mut final_header = header;
        final_header.header_checksum = Self::calculate_checksum(&checksum_data);

        Ok(final_header)
    }

    /// Get data type identifier for type T
    fn get_data_type_id<T: Pod>() -> BspResult<DataTypeId> {
        match core::mem::size_of::<T>() {
            4 => {
                // Could be f32, i32, u32, or Complex32 (8 bytes actually)
                // This is simplified - real implementation would use type IDs
                Ok(DataTypeId::F32)
            }
            8 => Ok(DataTypeId::F64),
            2 => Ok(DataTypeId::I16),
            _ => Err(BspError::TypeConversionError {
                from: "unknown",
                to: "DataTypeId",
            }),
        }
    }

    /// Convert format version to u32
    fn format_version_to_u32(version: &FormatVersion) -> u32 {
        (version.major as u32) << 16 | (version.minor as u32) << 8 | (version.patch as u32)
    }

    /// Convert u32 to format version
    fn u32_to_format_version(value: u32) -> FormatVersion {
        FormatVersion {
            major: ((value >> 16) & 0xFFFF) as u16,
            minor: ((value >> 8) & 0xFF) as u16,
            patch: (value & 0xFF) as u16,
        }
    }

    /// Validate header magic and version
    fn validate_header(header: &BinaryHeader) -> BspResult<()> {
        if header.magic != *b"BSP\0" {
            return Err(BspError::FormatError {
                reason: "invalid magic bytes",
            });
        }

        let version = Self::u32_to_format_version(header.version);
        let current = FormatVersion::current();
        if !current.is_compatible(&version) {
            return Err(BspError::FormatError {
                reason: "incompatible format version",
            });
        }

        Ok(())
    }
}

impl FormatHandler for BSPNativeHandler {
    
    fn format_id(&self) -> DataFormat {
        DataFormat::BSPNative
    }

    fn validate(&self, metadata: &SignalMetadata) -> BspResult<()> {
        metadata.validate()
    }
}

impl<T: Pod> FormatHandlerGeneric<T> for BSPNativeHandler {
    fn serialize(
        &self,
        data: &[T],
        metadata: &SignalMetadata,
        options: &SerializationOptions,
    ) -> BspResult<HVec<u8, 4096>> {
        if options.format != DataFormat::BSPNative {
            return Err(BspError::FormatError {
                reason: "format mismatch",
            });
        }

        let mut result = HVec::new();

        // Serialize metadata if requested
        let metadata_bytes = if options.include_metadata {
            self.serialize_metadata(metadata)?
        } else {
            HVec::new()
        };

        // Create and serialize header
        let header = Self::create_header(data, metadata, &metadata_bytes)?;
        let header_bytes = bytemuck::bytes_of(&header);

        result.extend_from_slice(header_bytes).map_err(|_| {
            BspError::BufferOverflow {
                capacity: 4096,
                requested: header_bytes.len(),
            }
        })?;

        // Add metadata
        if options.include_metadata {
            result.extend_from_slice(&metadata_bytes).map_err(|_| {
                BspError::BufferOverflow {
                    capacity: 4096,
                    requested: metadata_bytes.len(),
                }
            })?;
        }

        // Add payload data
        let payload_bytes = bytemuck::cast_slice(data);
        result.extend_from_slice(payload_bytes).map_err(|_| {
            BspError::BufferOverflow {
                capacity: 4096,
                requested: payload_bytes.len(),
            }
        })?;

        Ok(result)
    }

    fn deserialize(
        &self,
        bytes: &[u8],
        options: &DeserializationOptions,
    ) -> BspResult<(HVec<T, 1024>, SignalMetadata)> {
        if bytes.len() < core::mem::size_of::<BinaryHeader>() {
            return Err(BspError::FormatError {
                reason: "insufficient data for header",
            });
        }

        // Parse header
        let header_bytes = &bytes[..core::mem::size_of::<BinaryHeader>()];
        let header: BinaryHeader = *bytemuck::from_bytes(header_bytes);

        // Validate header
        Self::validate_header(&header)?;

        if options.validate_checksums {
            // Validate header checksum
            let mut temp_header = header;
            temp_header.header_checksum = 0;
            let temp_bytes = bytemuck::bytes_of(&temp_header);
            let expected_checksum = Self::calculate_checksum(temp_bytes);
            if header.header_checksum != expected_checksum {
                return Err(BspError::FormatError {
                    reason: "header checksum mismatch",
                });
            }
        }

        let mut offset = header.header_size as usize;

        // Parse metadata if present
        let metadata = if header.metadata_size > 0 {
            let metadata_end = offset + header.metadata_size as usize;
            if bytes.len() < metadata_end {
                return Err(BspError::FormatError {
                    reason: "insufficient data for metadata",
                });
            }
            let metadata_bytes = &bytes[offset..metadata_end];
            offset = metadata_end;
            self.deserialize_metadata(metadata_bytes)?
        } else {
            // Create minimal metadata
            self.create_minimal_metadata(&header)?
        };

        // Parse payload
        let payload_end = offset + header.payload_size as usize;
        if bytes.len() < payload_end {
            return Err(BspError::FormatError {
                reason: "insufficient data for payload",
            });
        }

        let payload_bytes = &bytes[offset..payload_end];

        if options.validate_checksums {
            let actual_checksum = Self::calculate_checksum(payload_bytes);
            if header.payload_checksum != actual_checksum {
                return Err(BspError::FormatError {
                    reason: "payload checksum mismatch",
                });
            }
        }

        // Convert payload to typed data
        let typed_data = try_cast_slice::<u8, T>(payload_bytes).map_err(|_| {
            BspError::TypeConversionError {
                from: "bytes",
                to: "typed data",
            }
        })?;

        let mut result_data = HVec::new();
        result_data.extend_from_slice(typed_data).map_err(|_| {
            BspError::BufferOverflow {
                capacity: 1024,
                requested: typed_data.len(),
            }
        })?;

        Ok((result_data, metadata))
    }

}




impl BSPNativeHandler {
    /// Serialize metadata (simplified implementation)
    fn serialize_metadata(&self, metadata: &SignalMetadata) -> BspResult<HVec<u8, 1024>> {
        // In a real implementation, this would use a proper serialization format
        // For now, we'll create a minimal binary representation
        let mut result = HVec::new();

        // Add signal type info
        let signal_type_id = 1u32; // Simplified
        result.extend_from_slice(&signal_type_id.to_le_bytes()).map_err(|_| {
            BspError::BufferOverflow { capacity: 1024, requested: 4 }
        })?;

        // Add sampling rate
        result.extend_from_slice(&metadata.acquisition.sampling_rate.to_le_bytes()).map_err(|_| {
            BspError::BufferOverflow { capacity: 1024, requested: 4 }
        })?;

        // Add channel count
        let channel_count = metadata.acquisition.channel_count as u32;
        result.extend_from_slice(&channel_count.to_le_bytes()).map_err(|_| {
            BspError::BufferOverflow { capacity: 1024, requested: 4 }
        })?;

        Ok(result)
    }

    /// Deserialize metadata (simplified implementation)
    fn deserialize_metadata(&self, bytes: &[u8]) -> BspResult<SignalMetadata> {
        if bytes.len() < 12 {
            return Err(BspError::FormatError {
                reason: "insufficient metadata",
            });
        }

        // This is a simplified implementation
        // Real implementation would properly deserialize all metadata fields
        let sampling_rate = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let channel_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;

        // Create minimal metadata with default device info
        let device_info = crate::metadata::DeviceInfo {
            manufacturer: String::from_str("Unknown").unwrap(),
            model: String::from_str("Unknown").unwrap(),
            serial_number: String::from_str("Unknown").unwrap(),
            firmware_version: String::from_str("1.0").unwrap(),
            hardware_revision: String::from_str("A").unwrap(),
            calibration_date: None,
            device_config: crate::metadata::DeviceConfiguration {
                gain_settings: heapless::Vec::new(),
                offset_corrections: heapless::Vec::new(),
                impedances: heapless::Vec::new(),
                temperature: None,
                supply_voltage: None,
                additional_params: heapless::FnvIndexMap::new(),
            },
        };

        SignalMetadata::new(
            crate::signal_types::SignalType::Physiological(
                crate::signal_types::PhysiologicalSignal::EMG {
                    location: crate::signal_types::MuscleLoc::Other("unknown"),
                    polarity: crate::signal_types::SignalPolarity::Differential,
                }
            ),
            sampling_rate,
            channel_count,
            device_info,
        )
    }

    /// Create minimal metadata from header
    fn create_minimal_metadata(&self, header: &BinaryHeader) -> BspResult<SignalMetadata> {
        let device_info = crate::metadata::DeviceInfo {
            manufacturer: String::from_str("Unknown").unwrap(),
            model: String::from_str("Unknown").unwrap(),
            serial_number: String::from_str("Unknown").unwrap(),
            firmware_version: String::from_str("1.0").unwrap(),
            hardware_revision: String::from_str("A").unwrap(),
            calibration_date: None,
            device_config: crate::metadata::DeviceConfiguration {
                gain_settings: heapless::Vec::new(),
                offset_corrections: heapless::Vec::new(),
                impedances: heapless::Vec::new(),
                temperature: None,
                supply_voltage: None,
                additional_params: heapless::FnvIndexMap::new(),
            },
        };

        SignalMetadata::new(
            crate::signal_types::SignalType::Physiological(
                crate::signal_types::PhysiologicalSignal::EMG {
                    location: crate::signal_types::MuscleLoc::Other("unknown"),
                    polarity: crate::signal_types::SignalPolarity::Differential,
                }
            ),
            header.sampling_rate,
            header.channel_count as usize,
            device_info,
        )
    }
}

// Placeholder implementations for other format handlers
impl FormatHandler for EDFPlusHandler {
    /*fn serialize<T: Pod>(
        &self,
        _data: &[T],
        _metadata: &SignalMetadata,
        _options: &SerializationOptions,
    ) -> BspResult<HVec<u8, 4096>> {
        Err(BspError::FormatError {
            reason: "EDF+ serialization not implemented",
        })
    }

    fn deserialize<T: Pod>(
        &self,
        _bytes: &[u8],
        _options: &DeserializationOptions,
    ) -> BspResult<(HVec<T, 1024>, SignalMetadata)> {
        Err(BspError::FormatError {
            reason: "EDF+ deserialization not implemented",
        })
    }
*/
    fn format_id(&self) -> DataFormat {
        DataFormat::EDFPlus
    }

    fn validate(&self, _metadata: &SignalMetadata) -> BspResult<()> {
        Ok(())
    }
}

impl FormatHandler for XDFHandler {
   /* fn serialize<T: Pod>(
        &self,
        _data: &[T],
        _metadata: &SignalMetadata,
        _options: &SerializationOptions,
    ) -> BspResult<HVec<u8, 4096>> {
        Err(BspError::FormatError {
            reason: "XDF serialization not implemented",
        })
    }

    fn deserialize<T: Pod>(
        &self,
        _bytes: &[u8],
        _options: &DeserializationOptions,
    ) -> BspResult<(HVec<T, 1024>, SignalMetadata)> {
        Err(BspError::FormatError {
            reason: "XDF deserialization not implemented",
        })
    }*/

    fn format_id(&self) -> DataFormat {
        DataFormat::XDF
    }

    fn validate(&self, _metadata: &SignalMetadata) -> BspResult<()> {
        Ok(())
    }
}

impl Default for SerializationOptions {
    fn default() -> Self {
        Self {
            format: DataFormat::BSPNative,
            include_metadata: true,
            compress: false,
            little_endian: true,
            alignment: 32, // SIMD alignment
        }
    }
}

impl Default for DeserializationOptions {
    fn default() -> Self {
        Self {
            expected_format: None,
            validate_checksums: true,
            strict_validation: true,
        }
    }
}

/// Format registry for managing multiple format handlers
pub struct FormatRegistry {
    handlers: HVec<&'static dyn FormatHandler, 8>,
}

impl FormatRegistry {
    /// Create new format registry
    pub fn new() -> Self {
        Self {
            handlers: HVec::new(),
        }
    }

    /// Register a format handler
    pub fn register(&mut self, handler: &'static dyn FormatHandler) -> BspResult<()> {
        self.handlers.push(handler).map_err(|_| {
            BspError::BufferOverflow {
                capacity: 8,
                requested: self.handlers.len() + 1,
            }
        })
    }

    /// Get handler for format
    pub fn get_handler(&self, format: DataFormat) -> Option<&dyn FormatHandler> {
        self.handlers.iter()
            .find(|h| h.format_id() == format)
            .map(|h| *h)
    }

    /// Auto-detect format from data
    pub fn detect_format(&self, data: &[u8]) -> Option<DataFormat> {
        if data.len() >= 4 {
            // Check for BSP magic
            if &data[0..4] == b"BSP\0" {
                return Some(DataFormat::BSPNative);
            }

            // Add other format detection logic here
        }

        None
    }
}

impl Default for FormatRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Macro to simplify offset calculation for packed structs

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal_types::{SignalType, PhysiologicalSignal, MuscleLoc, SignalPolarity};
    use crate::metadata::DeviceInfo;

    #[test]
    fn test_bsp_native_serialization() {
        let data = [1.0f32, 2.0, 3.0, 4.0];

        let device_info = DeviceInfo {
            manufacturer: String::from_str("Test").unwrap(),
            model: String::from_str("Model").unwrap(),
            serial_number: String::from_str("123").unwrap(),
            firmware_version: String::from_str("1.0").unwrap(),
            hardware_revision: String::from_str("A").unwrap(),
            calibration_date: None,
            device_config: crate::metadata::DeviceConfiguration {
                gain_settings: heapless::Vec::new(),
                offset_corrections: heapless::Vec::new(),
                impedances: heapless::Vec::new(),
                temperature: None,
                supply_voltage: None,
                additional_params: heapless::FnvIndexMap::new(),
            },
        };

        let signal_type = SignalType::Physiological(PhysiologicalSignal::EMG {
            location: MuscleLoc::Arm,
            polarity: SignalPolarity::Differential,
        });

        let metadata = SignalMetadata::new(signal_type, 1000.0, 1, device_info).unwrap();
        let handler = BSPNativeHandler::new();
        let options = SerializationOptions::default();

        let serialized = handler.serialize(&data, &metadata, &options);
        assert!(serialized.is_ok());

        let bytes = serialized.unwrap();
        assert!(bytes.len() > core::mem::size_of::<BinaryHeader>());
    }

    #[test]
    fn test_format_detection() {
        let bsp_data = b"BSP\0test data";
        let registry = FormatRegistry::new();

        let detected = registry.detect_format(bsp_data);
        assert_eq!(detected, Some(DataFormat::BSPNative));

        let unknown_data = b"unknown format";
        let detected = registry.detect_format(unknown_data);
        assert_eq!(detected, None);
    }

    #[test]
    fn test_binary_header_size() {
        // Ensure header is reasonably sized
        assert!(core::mem::size_of::<BinaryHeader>() <= 64);
    }

    #[test]
    fn test_checksum_calculation() {
        let data = b"test data";
        let checksum1 = BSPNativeHandler::calculate_checksum(data);
        let checksum2 = BSPNativeHandler::calculate_checksum(data);
        assert_eq!(checksum1, checksum2);

        let different_data = b"different";
        let checksum3 = BSPNativeHandler::calculate_checksum(different_data);
        assert_ne!(checksum1, checksum3);
    }
}