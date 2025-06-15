//! Signal Type Ontology (STO) implementation
//!
//! Hierarchical signal classification system based on IEEE 11073-10404
//! with extensions for comprehensive biosignal taxonomy.

use crate::error::{BspError, BspResult};
use core::fmt;

#[cfg(feature = "serde-support")]
use serde::{Deserialize, Serialize};

/// Primary signal type classification
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum SignalType {
    /// Physiological biosignals from living tissue
    Physiological(PhysiologicalSignal),
    /// Mechanical and motion signals
    Mechanical(MechanicalSignal),
    /// Environmental sensor data
    Environmental(EnvironmentalSignal),
    /// Derived/computed signals from raw data
    Derived(DerivedSignal),
}

/// Physiological signal subtypes
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum PhysiologicalSignal {
    /// Electromyography - muscle electrical activity
    EMG {
        /// Muscle group location
        location: MuscleLoc,
        /// Signal polarity (differential, single-ended)
        polarity: SignalPolarity,
    },
    /// Electroencephalography - brain electrical activity
    EEG {
        /// Electrode placement system
        placement: EEGPlacement,
        /// Reference configuration
        reference: EEGReference,
    },
    /// Electrocardiography - heart electrical activity
    ECG {
        /// Lead configuration (I, II, III, aVR, aVL, aVF, V1-V6)
        lead: ECGLead,
    },
    /// Photoplethysmography - blood volume changes
    PPG {
        /// Measurement site
        site: PPGSite,
        /// Wavelength used
        wavelength: PPGWavelength,
    },
    /// Pulse oximetry - blood oxygen saturation
    SpO2 {
        /// Measurement confidence level
        confidence: f32,
    },
    /// Electrodermal activity - skin conductance
    EDA {
        /// Electrode configuration
        config: EDAConfig,
    },
    /// Body temperature
    Temperature {
        /// Measurement site
        site: TemperatureSite,
    },
    /// Blood pressure (systolic/diastolic)
    BloodPressure {
        /// Measurement method
        method: BPMethod,
    },
    /// Respiratory signals
    Respiration {
        /// Signal type (airflow, effort, etc.)
        signal_type: RespirationSignal,
    },
}

/// Mechanical signal subtypes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum MechanicalSignal {
    /// Inertial Measurement Unit data
    IMU {
        /// IMU sensor type
        sensor: IMUSensor,
        /// Axis configuration
        axis: IMUAxis,
    },
    /// Force/pressure measurements
    Force {
        /// Force direction
        direction: ForceDirection,
        /// Measurement range
        range: ForceRange,
    },
    /// Displacement/position tracking
    Position {
        /// Coordinate system
        coordinate_system: CoordinateSystem,
        /// Number of spatial dimensions
        dimensions: u8,
    },
    /// Velocity measurements
    Velocity {
        /// Velocity type
        velocity_type: VelocityType,
    },
}

/// Environmental signal subtypes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum EnvironmentalSignal {
    /// Ambient temperature
    AmbientTemperature,
    /// Relative humidity
    Humidity,
    /// Ambient light level
    AmbientLight {
        /// Light spectrum measured
        spectrum: LightSpectrum,
    },
    /// Atmospheric pressure
    Pressure,
    /// Air quality metrics
    AirQuality {
        /// Specific pollutant measured
        pollutant: AirPollutant,
    },
    /// Sound/noise level
    Sound {
        /// Frequency range
        frequency_range: SoundFrequencyRange,
    },
}

/// Derived signal subtypes
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum DerivedSignal {
    /// Feature extraction results
    Features {
        /// Feature type
        feature_type: FeatureType,
        /// Source signal type
        source: &'static str,
    },
    /// Machine learning model outputs
    MLOutput {
        /// Model type
        model_type: MLModelType,
        /// Confidence score
        confidence: f32,
    },
    /// Multi-modal sensor fusion results
    Fusion {
        /// Fusion algorithm used
        algorithm: FusionAlgorithm,
        /// Number of input signals
        input_count: u8,
    },
    /// Frequency domain analysis
    FrequencyDomain {
        /// Transform type (FFT, wavelet, etc.)
        transform: FrequencyTransform,
    },
    /// Time-frequency analysis
    TimeFrequency {
        /// Analysis method
        method: TimeFrequencyMethod,
    },
}

// Supporting enums for detailed classification

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum MuscleLoc {
    Facial, Arm, Leg, Torso, Hand, Foot, Other(&'static str)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum SignalPolarity {
    Differential, SingleEnded, Bipolar
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum EEGPlacement {
    International1010, International1020, Custom(&'static str)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum EEGReference {
    CommonAverage, Mastoid, Cz, Linked, Custom(&'static str)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum ECGLead {
    I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum PPGSite {
    Finger, Wrist, Forehead, Ear, Toe
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum PPGWavelength {
    Red(u16), Infrared(u16), Green(u16), Blue(u16)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum EDAConfig {
    TwoElectrode, ThreeElectrode
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum TemperatureSite {
    Core, Skin, Oral, Tympanic, Axillary
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum BPMethod {
    Oscillometric, Auscultatory, IntraArterial
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum RespirationSignal {
    Airflow, Effort, Rate, Volume
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum IMUSensor {
    Accelerometer, Gyroscope, Magnetometer
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum IMUAxis {
    X, Y, Z, Combined
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum ForceDirection {
    Normal, Tangential, Combined
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum ForceRange {
    MicroNewton, MilliNewton, Newton, KiloNewton
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum CoordinateSystem {
    Cartesian, Cylindrical, Spherical
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum VelocityType {
    Linear, Angular, Combined
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum LightSpectrum {
    Visible, Infrared, Ultraviolet, FullSpectrum
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum AirPollutant {
    PM25, PM10, NO2, O3, CO, SO2
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum SoundFrequencyRange {
    Subsonic, Audible, Ultrasonic, FullRange
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum FeatureType {
    Statistical, Spectral, Temporal, Morphological, Complexity
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum MLModelType {
    Classification, Regression, Clustering, DeepLearning
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum FusionAlgorithm {
    KalmanFilter, ParticleFilter, Bayesian, Weighted
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum FrequencyTransform {
    FFT, DCT, Wavelet, STFT
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum TimeFrequencyMethod {
    Spectrogram, Wavelet, HilbertHuang, WignerVille
}

/// Signal specification with validation rules
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct SignalSpec {
    /// Signal type
    pub signal_type: SignalType,
    /// Valid sampling rate range (Hz)
    pub sampling_rate_range: (f32, f32),
    /// Expected channel count range
    pub channel_count_range: (usize, usize),
    /// Physical units
    pub units: &'static str,
    /// Typical resolution bits
    pub resolution_bits: u8,
    /// Description
    pub description: &'static str,
}

impl SignalType {
    /// Get signal specification with validation rules
    pub fn spec(&self) -> SignalSpec {
        match self {
            SignalType::Physiological(PhysiologicalSignal::EMG { .. }) => SignalSpec {
                signal_type: *self,
                sampling_rate_range: (500.0, 4000.0),
                channel_count_range: (1, 16),
                units: "mV",
                resolution_bits: 16,
                description: "Electromyography - muscle electrical activity",
            },
            SignalType::Physiological(PhysiologicalSignal::EEG { .. }) => SignalSpec {
                signal_type: *self,
                sampling_rate_range: (250.0, 2000.0),
                channel_count_range: (8, 256),
                units: "µV",
                resolution_bits: 24,
                description: "Electroencephalography - brain electrical activity",
            },
            SignalType::Physiological(PhysiologicalSignal::ECG { .. }) => SignalSpec {
                signal_type: *self,
                sampling_rate_range: (250.0, 1000.0),
                channel_count_range: (1, 12),
                units: "mV",
                resolution_bits: 16,
                description: "Electrocardiography - heart electrical activity",
            },
            SignalType::Physiological(PhysiologicalSignal::PPG { .. }) => SignalSpec {
                signal_type: *self,
                sampling_rate_range: (25.0, 1000.0),
                channel_count_range: (1, 8),
                units: "au",
                resolution_bits: 16,
                description: "Photoplethysmography - blood volume changes",
            },
            SignalType::Mechanical(MechanicalSignal::IMU { .. }) => SignalSpec {
                signal_type: *self,
                sampling_rate_range: (50.0, 8000.0),
                channel_count_range: (3, 9),
                units: "varies",
                resolution_bits: 16,
                description: "Inertial measurement unit data",
            },
            // Add more specifications as needed
            _ => SignalSpec {
                signal_type: *self,
                sampling_rate_range: (1.0, 10000.0),
                channel_count_range: (1, 64),
                units: "au",
                resolution_bits: 16,
                description: "Generic signal type",
            },
        }
    }

    /// Validate sampling rate for this signal type
    pub fn validate_sampling_rate(&self, rate: f32) -> BspResult<()> {
        let spec = self.spec();
        if rate >= spec.sampling_rate_range.0 && rate <= spec.sampling_rate_range.1 {
            Ok(())
        } else {
            Err(BspError::InvalidSamplingRate {
                signal_type: spec.description,
                rate,
                valid_range: "see signal specification",
            })
        }
    }

    /// Validate channel count for this signal type
    pub fn validate_channel_count(&self, count: usize) -> BspResult<()> {
        let spec = self.spec();
        if count >= spec.channel_count_range.0 && count <= spec.channel_count_range.1 {
            Ok(())
        } else {
            Err(BspError::TooManyChannels {
                requested: count,
                max_supported: spec.channel_count_range.1,
            })
        }
    }

    /// Get signal category for grouping
    pub fn category(&self) -> &'static str {
        match self {
            SignalType::Physiological(_) => "physiological",
            SignalType::Mechanical(_) => "mechanical",
            SignalType::Environmental(_) => "environmental",
            SignalType::Derived(_) => "derived",
        }
    }

    /// Check if signal type requires high-frequency sampling
    pub fn is_high_frequency(&self) -> bool {
        self.spec().sampling_rate_range.1 > 1000.0
    }

    /// Check if signal type supports multiple channels
    pub fn is_multi_channel(&self) -> bool {
        self.spec().channel_count_range.1 > 1
    }
}

impl fmt::Display for SignalType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignalType::Physiological(p) => write!(f, "Physiological::{:?}", p),
            SignalType::Mechanical(m) => write!(f, "Mechanical::{:?}", m),
            SignalType::Environmental(e) => write!(f, "Environmental::{:?}", e),
            SignalType::Derived(d) => write!(f, "Derived::{:?}", d),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_type_validation() {
        let emg = SignalType::Physiological(PhysiologicalSignal::EMG {
            location: MuscleLoc::Arm,
            polarity: SignalPolarity::Differential,
        });

        // Valid sampling rate
        assert!(emg.validate_sampling_rate(1000.0).is_ok());

        // Invalid sampling rate
        assert!(emg.validate_sampling_rate(100.0).is_err());
        assert!(emg.validate_sampling_rate(5000.0).is_err());

        // Valid channel count
        assert!(emg.validate_channel_count(8).is_ok());

        // Invalid channel count
        assert!(emg.validate_channel_count(32).is_err());
    }

    #[test]
    fn test_signal_categorization() {
        let eeg = SignalType::Physiological(PhysiologicalSignal::EEG {
            placement: EEGPlacement::International1010,
            reference: EEGReference::CommonAverage,
        });

        assert_eq!(eeg.category(), "physiological");
        assert!(eeg.is_multi_channel());
        assert!(eeg.is_high_frequency());
    }

    #[test]
    fn test_signal_spec() {
        let ecg = SignalType::Physiological(PhysiologicalSignal::ECG {
            lead: ECGLead::II,
        });

        let spec = ecg.spec();
        assert_eq!(spec.units, "mV");
        assert_eq!(spec.resolution_bits, 16);
        assert!(spec.sampling_rate_range.0 > 0.0);
    }
}