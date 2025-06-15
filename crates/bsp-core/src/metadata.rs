//! Signal metadata structures and standards implementation
//!
//! Comprehensive metadata system supporting EDF+, XDF, IEEE 11073,
//! and ISO 14155 compliance with efficient no_std storage.

use crate::error::{BspError, BspResult};
use crate::signal_types::SignalType;
pub use crate::quality::QualityMetrics;
use crate::timestamp::PrecisionTimestamp;
use crate::{MAX_METADATA_SIZE, FRAMEWORK_VERSION};

use core::fmt;
use core::str::FromStr;
use tinyvec::{TinyVec, Array};
use heapless::{String, FnvIndexMap, Vec as HVec};
use uuid::Uuid;

#[cfg(feature = "serde-support")]
use serde::{Deserialize, Serialize};

/// Main signal metadata structure with standards compliance
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct SignalMetadata {
    /// Universal Data Format (UDF) header
    pub udf_header: UDFHeader,
    /// Device information and fingerprint
    pub device_info: DeviceInfo,
    /// Signal acquisition parameters
    pub acquisition: AcquisitionInfo,
    /// Quality assessment metrics
    pub quality: Option<QualityMetrics>,
    /// Processing history chain
    pub processing_history: ProcessingHistory,
    /// Clinical/medical metadata (ISO 14155)
    pub clinical: Option<ClinicalMetadata>,
    /// Custom extended attributes
    pub custom_attributes: CustomAttributes,
}

/// Universal Data Format header based on EDF+/XDF
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct UDFHeader {
    /// Format version for compatibility
    pub format_version: FormatVersion,
    /// Unique entity identifier
    pub entity_uuid: Uuid,
    /// High-precision creation timestamp
    pub timestamp: PrecisionTimestamp,
    /// Signal type classification
    pub signal_type: SignalType,
    /// Data integrity hash
    pub integrity_hash: Option<DataHash>,
    /// Digital signature for authentication
    #[cfg(feature = "crypto")]
    pub signature: Option<DigitalSignature>,
}

/// Device information and fingerprint for calibration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct DeviceInfo {
    /// Device manufacturer
    pub manufacturer: String<32>,
    /// Device model identifier
    pub model: String<32>,
    /// Serial number
    pub serial_number: String<32>,
    /// Firmware version
    pub firmware_version: String<16>,
    /// Hardware revision
    pub hardware_revision: String<16>,
    /// Calibration date
    pub calibration_date: Option<PrecisionTimestamp>,
    /// Device-specific configuration
    pub device_config: DeviceConfiguration,
}

/// Signal acquisition parameters
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct AcquisitionInfo {
    /// Sampling rate in Hz
    pub sampling_rate: f32,
    /// ADC resolution in bits
    pub resolution_bits: u8,
    /// Number of channels
    pub channel_count: usize,
    /// Channel layout and configuration
    pub channel_layout: ChannelLayout,
    /// Physical units
    pub units: String<16>,
    /// Scaling factor to physical units
    pub scaling_factor: f32,
    /// Anti-aliasing filter settings
    pub filter_config: Option<FilterConfig>,
    /// Acquisition duration
    pub duration: Option<crate::timestamp::Duration>,
}

/// Channel layout and configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct ChannelLayout {
    /// Channel names
    pub channel_names: HVec<String<16>, 64>,
    /// Channel types
    pub channel_types: HVec<ChannelType, 64>,
    /// Physical locations/positions
    pub positions: HVec<ChannelPosition, 64>,
    /// Reference configuration
    pub reference_config: ReferenceConfig,
}

/// Individual channel type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum ChannelType {
    /// Signal channel
    Signal,
    /// Reference channel
    Reference,
    /// Ground channel
    Ground,
    /// Trigger/event channel
    Trigger,
    /// Annotation channel
    Annotation,
}

/// Channel position/location
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct ChannelPosition {
    /// Position label (e.g., "C3", "Fp1", "EMG_LEFT_ARM")
    pub label: String<16>,
    /// 3D coordinates (x, y, z) if applicable
    pub coordinates: Option<[f32; 3]>,
    /// Anatomical region
    pub region: String<16>,
}

/// Reference configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum ReferenceConfig {
    /// Single-ended with common reference
    SingleEnded,
    /// Differential pairs
    Differential,
    /// Common average reference
    CommonAverage,
    /// Bipolar montage
    Bipolar,
    /// Custom reference scheme
    Custom,
}

/// Filter configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct FilterConfig {
    /// High-pass filter cutoff (Hz)
    pub highpass_cutoff: Option<f32>,
    /// Low-pass filter cutoff (Hz)
    pub lowpass_cutoff: Option<f32>,
    /// Notch filter frequency (Hz)
    pub notch_frequency: Option<f32>,
    /// Filter order
    pub filter_order: u8,
    /// Filter type
    pub filter_type: FilterType,
}

/// Filter type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum FilterType {
    Butterworth,
    Chebyshev,
    Elliptic,
    Bessel,
    FIR,
    IIR,
}

/// Device-specific configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct DeviceConfiguration {
    /// Gain settings per channel
    pub gain_settings: HVec<f32, 64>,
    /// Offset corrections per channel
    pub offset_corrections: HVec<f32, 64>,
    /// Impedance measurements (Ohms)
    pub impedances: HVec<f32, 64>,
    /// Temperature at acquisition (Celsius)
    pub temperature: Option<f32>,
    /// Supply voltage (Volts)
    pub supply_voltage: Option<f32>,
    /// Additional device parameters
    pub additional_params: FnvIndexMap<String<16>, f32, 16>,
}

/// Processing history for audit trail
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct ProcessingHistory {
    /// Processing steps
    pub steps: HVec<ProcessingStep, 32>,
    /// Framework version used
    pub framework_version: String<16>,
    /// Processing pipeline checksum
    pub pipeline_checksum: Option<u32>,
}

/// Individual processing step
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct ProcessingStep {
    /// Step identifier
    pub step_id: String<32>,
    /// Algorithm/method used
    pub algorithm: String<32>,
    /// Parameters used
    pub parameters: FnvIndexMap<String<16>, f32, 16>,
    /// Timestamp of processing
    pub timestamp: PrecisionTimestamp,
    /// Processing duration
    pub duration: crate::timestamp::Duration,
    /// Input data checksum
    pub input_checksum: Option<u32>,
    /// Output data checksum
    pub output_checksum: Option<u32>,
}

/// Clinical metadata for medical device compliance (ISO 14155)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct ClinicalMetadata {
    /// Study information
    pub study_info: StudyInfo,
    /// Subject/patient information (anonymized)
    pub subject_info: SubjectInfo,
    /// Clinical context
    pub clinical_context: ClinicalContext,
    /// Regulatory information
    pub regulatory: RegulatoryInfo,
}

/// Study information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct StudyInfo {
    /// Study identifier
    pub study_id: String<32>,
    /// Protocol version
    pub protocol_version: String<16>,
    /// Principal investigator
    pub principal_investigator: String<64>,
    /// Institution
    pub institution: String<64>,
    /// Ethics approval reference
    pub ethics_approval: Option<String<32>>,
}

/// Subject information (anonymized)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct SubjectInfo {
    /// Anonymous subject identifier
    pub subject_id: String<32>,
    /// Age group category
    pub age_group: AgeGroup,
    /// Biological sex
    pub sex: BiologicalSex,
    /// Relevant medical history flags
    pub medical_history: MedicalHistory,
    /// Medication information
    pub medications: HVec<String<32>, 16>,
}

/// Clinical context information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct ClinicalContext {
    /// Recording environment
    pub environment: Environment,
    /// Subject activity during recording
    pub activity: ActivityState,
    /// Clinical notes
    pub notes: String<256>,
    /// Adverse events
    pub adverse_events: HVec<String<64>, 8>,
}

/// Regulatory compliance information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct RegulatoryInfo {
    /// FDA 510(k) number
    pub fda_510k: Option<String<16>>,
    /// CE marking information
    pub ce_marking: Option<String<16>>,
    /// ISO 13485 certification
    pub iso13485_cert: Option<String<32>>,
    /// Data privacy compliance
    pub privacy_compliance: PrivacyCompliance,
}

/// Format version with semantic versioning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct FormatVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

/// Data integrity hash
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct DataHash {
    /// Hash algorithm used
    pub algorithm: HashAlgorithm,
    /// Hash value
    pub hash: HVec<u8, 64>,
}

/// Digital signature for authentication
#[cfg(feature = "crypto")]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct DigitalSignature {
    /// Signature algorithm
    pub algorithm: SignatureAlgorithm,
    /// Public key identifier
    pub key_id: String<32>,
    /// Signature bytes
    pub signature: HVec<u8, 128>,
    /// Signing timestamp
    pub timestamp: PrecisionTimestamp,
}

/// Custom attributes for extensibility
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct CustomAttributes {
    /// String attributes
    pub strings: FnvIndexMap<String<32>, String<64>, 16>,
    /// Numeric attributes
    pub numbers: FnvIndexMap<String<32>, f64, 16>,
    /// Boolean attributes
    pub booleans: FnvIndexMap<String<32>, bool, 16>,
}

// Supporting enums

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum AgeGroup {
    Infant,      // 0-2 years
    Child,       // 3-12 years
    Adolescent,  // 13-17 years
    YoungAdult,  // 18-39 years
    MiddleAged,  // 40-64 years
    Elderly,     // 65+ years
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum BiologicalSex {
    Male,
    Female,
    Intersex,
    Unknown,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct MedicalHistory {
    pub cardiovascular: bool,
    pub neurological: bool,
    pub psychiatric: bool,
    pub metabolic: bool,
    pub other: String<64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum Environment {
    Laboratory,
    Clinical,
    Home,
    Ambulatory,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum ActivityState {
    Resting,
    Active,
    Exercise,
    Sleep,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum PrivacyCompliance {
    HIPAA,
    GDPR,
    Both,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum HashAlgorithm {
    SHA256,
    SHA512,
    Blake3,
}

#[cfg(feature = "crypto")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum SignatureAlgorithm {
    Ed25519,
    ECDSA,
    RSA,
}

impl SignalMetadata {
    /// Create new signal metadata with required fields
    pub fn new(
        signal_type: SignalType,
        sampling_rate: f32,
        channel_count: usize,
        device_info: DeviceInfo,
    ) -> BspResult<Self> {
        // Validate inputs
        signal_type.validate_sampling_rate(sampling_rate)?;
        signal_type.validate_channel_count(channel_count)?;

        let entity_uuid = Uuid::new_v4();
        let timestamp = PrecisionTimestamp::from_nanos(0); // Would use actual timestamp

        let udf_header = UDFHeader {
            format_version: FormatVersion::current(),
            entity_uuid,
            timestamp,
            signal_type,
            integrity_hash: None,
            #[cfg(feature = "crypto")]
            signature: None,
        };

        let channel_layout = ChannelLayout::new(channel_count)?;

        let acquisition = AcquisitionInfo {
            sampling_rate,
            resolution_bits: 16, // Default
            channel_count,
            channel_layout,
            units: String::from_str("V").map_err(|_| BspError::FormatError {
                reason: "units string too long"
            })?,
            scaling_factor: 1.0,
            filter_config: None,
            duration: None,
        };

        let processing_history = ProcessingHistory {
            steps: HVec::new(),
            framework_version: String::from_str(FRAMEWORK_VERSION).map_err(|_| {
                BspError::FormatError { reason: "framework version string too long" }
            })?,
            pipeline_checksum: None,
        };

        let custom_attributes = CustomAttributes {
            strings: FnvIndexMap::new(),
            numbers: FnvIndexMap::new(),
            booleans: FnvIndexMap::new(),
        };

        Ok(SignalMetadata {
            udf_header,
            device_info,
            acquisition,
            quality: None,
            processing_history,
            clinical: None,
            custom_attributes,
        })
    }

    /// Add processing step to history
    pub fn add_processing_step(&mut self, step: ProcessingStep) -> BspResult<()> {
        self.processing_history.steps.push(step).map_err(|_| {
            BspError::BufferOverflow {
                capacity: 32,
                requested: self.processing_history.steps.len() + 1,
            }
        })
    }

    /// Set quality metrics
    pub fn set_quality_metrics(&mut self, quality: QualityMetrics) {
        self.quality = Some(quality);
    }

    /// Add custom string attribute
    pub fn add_custom_string(&mut self, key: &str, value: &str) -> BspResult<()> {
        let key_str = String::from_str(key).map_err(|_| BspError::FormatError {
            reason: "custom attribute key too long"
        })?;
        let value_str = String::from_str(value).map_err(|_| BspError::FormatError {
            reason: "custom attribute value too long"
        })?;

        self.custom_attributes.strings.insert(key_str, value_str).map_err(|_| {
            BspError::BufferOverflow {
                capacity: 16,
                requested: self.custom_attributes.strings.len() + 1,
            }
        })?;

        Ok(())
    }

    /// Add custom numeric attribute
    pub fn add_custom_number(&mut self, key: &str, value: f64) -> BspResult<()> {
        let key_str = String::from_str(key).map_err(|_| BspError::FormatError {
            reason: "custom attribute key too long"
        })?;

        self.custom_attributes.numbers.insert(key_str, value).map_err(|_| {
            BspError::BufferOverflow {
                capacity: 16,
                requested: self.custom_attributes.numbers.len() + 1,
            }
        })?;

        Ok(())
    }

    /// Validate metadata structure and constraints
    pub fn validate(&self) -> BspResult<()> {
        // Check metadata size constraint
        let estimated_size = self.estimated_size();
        if estimated_size > MAX_METADATA_SIZE {
            return Err(BspError::MetadataTooLarge {
                size: estimated_size,
                max_size: MAX_METADATA_SIZE,
            });
        }

        // Validate timestamp
        self.udf_header.timestamp.validate()?;

        // Validate signal type consistency
        self.udf_header.signal_type.validate_sampling_rate(self.acquisition.sampling_rate)?;
        self.udf_header.signal_type.validate_channel_count(self.acquisition.channel_count)?;

        // Validate channel layout consistency
        if self.acquisition.channel_layout.channel_names.len() != self.acquisition.channel_count {
            return Err(BspError::InvalidSignalConfig {
                reason: "channel count mismatch with channel names",
            });
        }

        Ok(())
    }

    /// Estimate metadata size in bytes
    pub fn estimated_size(&self) -> usize {
        // Rough estimation - real implementation would be more precise
        core::mem::size_of::<Self>() +
            self.processing_history.steps.len() * core::mem::size_of::<ProcessingStep>() +
            self.custom_attributes.strings.len() * 96 + // Estimated string storage
            self.custom_attributes.numbers.len() * 48 +
            self.custom_attributes.booleans.len() * 33
    }

    /// Check if metadata contains clinical data
    pub fn has_clinical_data(&self) -> bool {
        self.clinical.is_some()
    }

    /// Get signal specification
    pub fn signal_spec(&self) -> crate::signal_types::SignalSpec {
        self.udf_header.signal_type.spec()
    }
}

impl ChannelLayout {
    /// Create new channel layout with default names
    pub fn new(channel_count: usize) -> BspResult<Self> {
        if channel_count > 64 {
            return Err(BspError::TooManyChannels {
                requested: channel_count,
                max_supported: 64,
            });
        }

        let mut channel_names = HVec::new();
        let mut channel_types = HVec::new();
        let mut positions = HVec::new();

        for i in 0..channel_count {
            let name = String::from_str(&format!("CH{}", i + 1)).map_err(|_| {
                BspError::FormatError { reason: "channel name too long" }
            })?;

            let position = ChannelPosition {
                label: name.clone(),
                coordinates: None,
                region: String::from_str("unknown").map_err(|_| {
                    BspError::FormatError { reason: "region name too long" }
                })?,
            };

            channel_names.push(name).map_err(|_| {
                BspError::BufferOverflow { capacity: 64, requested: i + 1 }
            })?;

            channel_types.push(ChannelType::Signal).map_err(|_| {
                BspError::BufferOverflow { capacity: 64, requested: i + 1 }
            })?;

            positions.push(position).map_err(|_| {
                BspError::BufferOverflow { capacity: 64, requested: i + 1 }
            })?;
        }

        Ok(ChannelLayout {
            channel_names,
            channel_types,
            positions,
            reference_config: ReferenceConfig::SingleEnded,
        })
    }
}

impl FormatVersion {
    /// Get current format version
    pub const fn current() -> Self {
        Self {
            major: 1,
            minor: 0,
            patch: 0,
        }
    }

    /// Check if version is compatible
    pub fn is_compatible(&self, other: &FormatVersion) -> bool {
        self.major == other.major && self.minor >= other.minor
    }
}

impl ProcessingStep {
    /// Create new processing step
    pub fn new(
        step_id: &str,
        algorithm: &str,
        timestamp: PrecisionTimestamp,
        duration: crate::timestamp::Duration,
    ) -> BspResult<Self> {
        let step_id = String::from_str(step_id).map_err(|_| {
            BspError::FormatError { reason: "step ID too long" }
        })?;

        let algorithm = String::from_str(algorithm).map_err(|_| {
            BspError::FormatError { reason: "algorithm name too long" }
        })?;

        Ok(ProcessingStep {
            step_id,
            algorithm,
            parameters: FnvIndexMap::new(),
            timestamp,
            duration,
            input_checksum: None,
            output_checksum: None,
        })
    }

    /// Add parameter to processing step
    pub fn add_parameter(&mut self, key: &str, value: f32) -> BspResult<()> {
        let key_str = String::from_str(key).map_err(|_| {
            BspError::FormatError { reason: "parameter key too long" }
        })?;

        self.parameters.insert(key_str, value).map_err(|_| {
            BspError::BufferOverflow {
                capacity: 16,
                requested: self.parameters.len() + 1,
            }
        })?;

        Ok(())
    }
}

impl fmt::Display for FormatVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl Default for CustomAttributes {
    fn default() -> Self {
        Self {
            strings: FnvIndexMap::new(),
            numbers: FnvIndexMap::new(),
            booleans: FnvIndexMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal_types::{PhysiologicalSignal, MuscleLoc, SignalPolarity};

    #[test]
    fn test_metadata_creation() {
        let signal_type = SignalType::Physiological(PhysiologicalSignal::EMG {
            location: MuscleLoc::Arm,
            polarity: SignalPolarity::Differential,
        });

        let device_info = DeviceInfo {
            manufacturer: String::from_str("TestManufacturer").unwrap(),
            model: String::from_str("TestModel").unwrap(),
            serial_number: String::from_str("12345").unwrap(),
            firmware_version: String::from_str("1.0.0").unwrap(),
            hardware_revision: String::from_str("A1").unwrap(),
            calibration_date: None,
            device_config: DeviceConfiguration {
                gain_settings: HVec::new(),
                offset_corrections: HVec::new(),
                impedances: HVec::new(),
                temperature: Some(25.0),
                supply_voltage: Some(5.0),
                additional_params: FnvIndexMap::new(),
            },
        };

        let metadata = SignalMetadata::new(signal_type, 1000.0, 8, device_info);
        assert!(metadata.is_ok());

        let meta = metadata.unwrap();
        assert_eq!(meta.acquisition.sampling_rate, 1000.0);
        assert_eq!(meta.acquisition.channel_count, 8);
    }

    #[test]
    fn test_format_version_compatibility() {
        let v1_0_0 = FormatVersion { major: 1, minor: 0, patch: 0 };
        let v1_1_0 = FormatVersion { major: 1, minor: 1, patch: 0 };
        let v2_0_0 = FormatVersion { major: 2, minor: 0, patch: 0 };

        assert!(v1_1_0.is_compatible(&v1_0_0));
        assert!(!v1_0_0.is_compatible(&v1_1_0));
        assert!(!v2_0_0.is_compatible(&v1_0_0));
    }

    #[test]
    fn test_processing_step() {
        let timestamp = PrecisionTimestamp::from_secs(1000);
        let duration = crate::timestamp::Duration::from_millis(10);

        let mut step = ProcessingStep::new(
            "filter",
            "butterworth",
            timestamp,
            duration,
        ).unwrap();

        assert!(step.add_parameter("cutoff", 50.0).is_ok());
        assert!(step.add_parameter("order", 4.0).is_ok());

        assert_eq!(step.parameters.get(&String::from_str("cutoff").unwrap()), Some(&50.0));
    }

    #[test]
    fn test_metadata_validation() {
        let signal_type = SignalType::Physiological(PhysiologicalSignal::EMG {
            location: MuscleLoc::Arm,
            polarity: SignalPolarity::Differential,
        });

        let device_info = DeviceInfo {
            manufacturer: String::from_str("Test").unwrap(),
            model: String::from_str("Model").unwrap(),
            serial_number: String::from_str("123").unwrap(),
            firmware_version: String::from_str("1.0").unwrap(),
            hardware_revision: String::from_str("A").unwrap(),
            calibration_date: None,
            device_config: DeviceConfiguration {
                gain_settings: HVec::new(),
                offset_corrections: HVec::new(),
                impedances: HVec::new(),
                temperature: None,
                supply_voltage: None,
                additional_params: FnvIndexMap::new(),
            },
        };

        // Valid metadata
        let valid_meta = SignalMetadata::new(signal_type, 1000.0, 4, device_info.clone()).unwrap();
        assert!(valid_meta.validate().is_ok());

        // Invalid sampling rate
        let invalid_meta = SignalMetadata::new(signal_type, 100.0, 4, device_info);
        assert!(invalid_meta.is_err());
    }
}