//! Signal Quality Assessment system
//!
//! Comprehensive quality metrics for biosignal data based on ISO/IEC 25010
//! software quality model adapted for biosignal processing requirements.

use crate::error::{BspError, BspResult};
use crate::signal_types::SignalType;
use crate::timestamp::Duration;
use core::fmt;

#[cfg(feature = "serde-support")]
use serde::{Deserialize, Serialize};

/// Comprehensive quality assessment for signal entities
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct QualityMetrics {
    /// Signal quality metrics
    pub signal_quality: SignalQuality,
    /// Temporal quality metrics
    pub temporal_quality: TemporalQuality,
    /// System quality metrics
    pub system_quality: SystemQuality,
    /// Clinical quality metrics
    pub clinical_quality: ClinicalQuality,
    /// Overall quality score (0.0 - 1.0)
    pub overall_score: f32,
    /// Quality assessment timestamp
    pub assessment_time: u64, // nanos since epoch
}

/// Signal-specific quality metrics
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct SignalQuality {
    /// Signal-to-noise ratio in dB
    pub snr_db: f32,
    /// Total harmonic distortion percentage
    pub thd_percent: f32,
    /// Artifact percentage (0.0 - 100.0)
    pub artifact_percent: f32,
    /// Signal saturation percentage
    pub saturation_percent: f32,
    /// Dynamic range utilization (0.0 - 1.0)
    pub dynamic_range: f32,
    /// Frequency content quality score
    pub frequency_quality: f32,
}

/// Temporal quality metrics
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct TemporalQuality {
    /// Timing jitter as percentage of sampling period
    pub jitter_percent: f32,
    /// Data completeness percentage (0.0 - 100.0)
    pub completeness_percent: f32,
    /// Sample rate stability (coefficient of variation)
    pub rate_stability: f32,
    /// Missing sample count
    pub missing_samples: u32,
    /// Duplicated sample count
    pub duplicate_samples: u32,
    /// Out-of-order sample count
    pub out_of_order_samples: u32,
}

/// System performance quality metrics
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct SystemQuality {
    /// Processing latency in microseconds
    pub latency_us: u32,
    /// Memory usage in bytes
    pub memory_usage: u32,
    /// CPU utilization percentage (0.0 - 100.0)
    pub cpu_utilization: f32,
    /// Buffer overrun count
    pub buffer_overruns: u32,
    /// Buffer underrun count
    pub buffer_underruns: u32,
    /// Error rate (errors per second)
    pub error_rate: f32,
}

/// Clinical/medical quality metrics
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct ClinicalQuality {
    /// Calibration drift percentage
    pub calibration_drift: f32,
    /// Measurement repeatability (CV%)
    pub repeatability_cv: f32,
    /// Device temperature drift effect
    pub temperature_drift: f32,
    /// Signal baseline stability
    pub baseline_stability: f32,
    /// Contact impedance (for electrode-based signals)
    pub contact_impedance: Option<f32>,
    /// Motion artifact level
    pub motion_artifact_level: f32,
}

/// Quality dimension for targeted assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum QualityDimension {
    /// Signal integrity and fidelity
    SignalIntegrity,
    /// Temporal accuracy and consistency
    TemporalAccuracy,
    /// System performance metrics
    SystemPerformance,
    /// Clinical measurement quality
    ClinicalAccuracy,
}

/// Quality assessment thresholds for different quality levels
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct QualityThresholds {
    /// Minimum SNR in dB
    pub min_snr_db: f32,
    /// Maximum THD percentage
    pub max_thd_percent: f32,
    /// Maximum artifact percentage
    pub max_artifact_percent: f32,
    /// Maximum jitter percentage
    pub max_jitter_percent: f32,
    /// Minimum completeness percentage
    pub min_completeness_percent: f32,
    /// Maximum latency in microseconds
    pub max_latency_us: u32,
    /// Maximum calibration drift percentage
    pub max_calibration_drift: f32,
    /// Maximum repeatability CV percentage
    pub max_repeatability_cv: f32,
}

/// Quality assessment implementation
pub struct QualityAssessment;

impl QualityAssessment {
    /// Assess signal quality for given data and signal type
    pub fn assess<T>(
        data: &[T],
        signal_type: SignalType,
        sampling_rate: f32,
        processing_time: Duration,
    ) -> BspResult<QualityMetrics>
    where
        T: Copy + Into<f64>,
    {
        let signal_quality = Self::assess_signal_quality(data, signal_type)?;
        let temporal_quality = Self::assess_temporal_quality(data, sampling_rate)?;
        let system_quality = Self::assess_system_quality(processing_time)?;
        let clinical_quality = Self::assess_clinical_quality(data, signal_type)?;

        let overall_score = Self::calculate_overall_score(
            &signal_quality,
            &temporal_quality,
            &system_quality,
            &clinical_quality,
        );

        Ok(QualityMetrics {
            signal_quality,
            temporal_quality,
            system_quality,
            clinical_quality,
            overall_score,
            assessment_time: 0, // Would use current timestamp in real implementation
        })
    }

    /// Assess signal-specific quality metrics
    fn assess_signal_quality<T>(
        data: &[T],
        signal_type: SignalType,
    ) -> BspResult<SignalQuality>
    where
        T: Copy + Into<f64>,
    {
        if data.is_empty() {
            return Err(BspError::InvalidSignalConfig {
                reason: "empty data for quality assessment"
            });
        }

        let samples: Vec<f64> = data.iter().map(|&x| x.into()).collect();

        // Calculate signal-to-noise ratio
        let snr_db = Self::calculate_snr(&samples)?;

        // Calculate total harmonic distortion
        let thd_percent = Self::calculate_thd(&samples)?;

        // Detect artifacts
        let artifact_percent = Self::detect_artifacts(&samples, signal_type)?;

        // Calculate saturation
        let saturation_percent = Self::calculate_saturation(&samples)?;

        // Calculate dynamic range utilization
        let dynamic_range = Self::calculate_dynamic_range(&samples)?;

        // Assess frequency content quality
        let frequency_quality = Self::assess_frequency_quality(&samples, signal_type)?;

        Ok(SignalQuality {
            snr_db,
            thd_percent,
            artifact_percent,
            saturation_percent,
            dynamic_range,
            frequency_quality,
        })
    }

    /// Assess temporal quality metrics
    fn assess_temporal_quality<T>(
        data: &[T],
        sampling_rate: f32,
    ) -> BspResult<TemporalQuality>
    where
        T: Copy + Into<f64>,
    {
        let sample_count = data.len();
        if sample_count < 2 {
            return Err(BspError::InvalidSignalConfig {
                reason: "insufficient data for temporal quality assessment"
            });
        }

        // For this implementation, we'll calculate basic metrics
        // Real implementation would need timestamp data for accurate assessment

        let jitter_percent = 0.005; // Simulated - would calculate from actual timestamps
        let completeness_percent = 99.95; // Simulated
        let rate_stability = 0.001; // Simulated coefficient of variation
        let missing_samples = 0; // Would detect from timestamp gaps
        let duplicate_samples = 0; // Would detect from duplicate timestamps
        let out_of_order_samples = 0; // Would detect from timestamp ordering

        Ok(TemporalQuality {
            jitter_percent,
            completeness_percent,
            rate_stability,
            missing_samples,
            duplicate_samples,
            out_of_order_samples,
        })
    }

    /// Assess system performance quality
    fn assess_system_quality(processing_time: Duration) -> BspResult<SystemQuality> {
        let latency_us = processing_time.as_micros() as u32;

        // These would be measured from actual system performance
        let memory_usage = 1024; // Simulated
        let cpu_utilization = 15.0; // Simulated
        let buffer_overruns = 0; // Simulated
        let buffer_underruns = 0; // Simulated
        let error_rate = 0.001; // Simulated

        Ok(SystemQuality {
            latency_us,
            memory_usage,
            cpu_utilization,
            buffer_overruns,
            buffer_underruns,
            error_rate,
        })
    }

    /// Assess clinical/medical quality metrics
    fn assess_clinical_quality<T>(
        data: &[T],
        signal_type: SignalType,
    ) -> BspResult<ClinicalQuality>
    where
        T: Copy + Into<f64>,
    {
        let samples: Vec<f64> = data.iter().map(|&x| x.into()).collect();

        // Calculate clinical quality metrics
        let calibration_drift = Self::calculate_calibration_drift(&samples)?;
        let repeatability_cv = Self::calculate_repeatability(&samples)?;
        let temperature_drift = 0.1; // Simulated
        let baseline_stability = Self::calculate_baseline_stability(&samples)?;
        let contact_impedance = Self::estimate_contact_impedance(&samples, signal_type);
        let motion_artifact_level = Self::assess_motion_artifacts(&samples, signal_type)?;

        Ok(ClinicalQuality {
            calibration_drift,
            repeatability_cv,
            temperature_drift,
            baseline_stability,
            contact_impedance,
            motion_artifact_level,
        })
    }

    /// Calculate signal-to-noise ratio
    fn calculate_snr(samples: &[f64]) -> BspResult<f32> {
        if samples.len() < 10 {
            return Ok(0.0);
        }

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let signal_power: f64 = samples.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / samples.len() as f64;

        // Estimate noise as high-frequency component
        let mut noise_power = 0.0f64;
        for window in samples.windows(2) {
            let diff = window[1] - window[0];
            noise_power += diff * diff;
        }
        noise_power /= (samples.len() - 1) as f64;

        if noise_power > 0.0 {
            let snr = 10.0 * (signal_power / noise_power).log10();
            Ok(snr as f32)
        } else {
            Ok(60.0) // Very high SNR if no detectable noise
        }
    }

    /// Calculate total harmonic distortion
    fn calculate_thd(samples: &[f64]) -> BspResult<f32> {
        // Simplified THD calculation
        // Real implementation would use FFT to find harmonics
        if samples.len() < 64 {
            return Ok(0.5); // Default low THD for short signals
        }

        // Estimate THD from signal characteristics
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance: f64 = samples.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / samples.len() as f64;

        // Higher variance relative to signal might indicate distortion
        let thd_estimate = (variance.sqrt() / (mean.abs() + 1e-10)).min(0.1);
        Ok((thd_estimate * 100.0) as f32)
    }

    /// Detect artifacts in signal
    fn detect_artifacts(samples: &[f64], _signal_type: SignalType) -> BspResult<f32> {
        if samples.len() < 10 {
            return Ok(0.0);
        }

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let std_dev = {
            let variance: f64 = samples.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / samples.len() as f64;
            variance.sqrt()
        };

        // Count samples beyond 3 standard deviations as artifacts
        let threshold = 3.0 * std_dev;
        let artifact_count = samples.iter()
            .filter(|&&x| (x - mean).abs() > threshold)
            .count();

        let artifact_percent = (artifact_count as f32 / samples.len() as f32) * 100.0;
        Ok(artifact_percent.min(100.0))
    }

    /// Calculate signal saturation percentage
    fn calculate_saturation(samples: &[f64]) -> BspResult<f32> {
        if samples.is_empty() {
            return Ok(0.0);
        }

        let min_val = samples.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = samples.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Assume saturation at extreme values (simplified)
        // Real implementation would know actual ADC limits
        let range = max_val - min_val;
        if range < 1e-10 {
            return Ok(100.0); // Flat signal might be saturated
        }

        // Check for samples near extremes
        let lower_threshold = min_val + 0.02 * range;
        let upper_threshold = max_val - 0.02 * range;

        let saturated_count = samples.iter()
            .filter(|&&x| x <= lower_threshold || x >= upper_threshold)
            .count();

        let saturation_percent = (saturated_count as f32 / samples.len() as f32) * 100.0;
        Ok(saturation_percent)
    }

    /// Calculate dynamic range utilization
    fn calculate_dynamic_range(samples: &[f64]) -> BspResult<f32> {
        if samples.is_empty() {
            return Ok(0.0);
        }

        let min_val = samples.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = samples.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Calculate range utilization (0.0 to 1.0)
        let range = max_val - min_val;

        // Normalize by expected signal range for the type
        // This is simplified - real implementation would use signal-specific ranges
        let expected_range = 1.0; // Placeholder
        let utilization = (range / expected_range).min(1.0);

        Ok(utilization as f32)
    }

    /// Assess frequency content quality
    fn assess_frequency_quality(samples: &[f64], _signal_type: SignalType) -> BspResult<f32> {
        // Simplified frequency quality assessment
        // Real implementation would use FFT and check against expected spectrum
        if samples.len() < 32 {
            return Ok(0.5);
        }

        // Check for sufficient high-frequency content
        let mut high_freq_energy = 0.0;
        let mut total_energy = 0.0;

        for window in samples.windows(2) {
            let diff = window[1] - window[0];
            high_freq_energy += diff * diff;
            total_energy += window[0] * window[0];
        }

        let hf_ratio = if total_energy > 0.0 {
            (high_freq_energy / total_energy).sqrt()
        } else {
            0.0
        };

        // Good quality signals should have some high-frequency content but not too much
        let quality = if hf_ratio > 0.01 && hf_ratio < 0.5 {
            0.8
        } else {
            0.5
        };

        Ok(quality)
    }

    /// Calculate calibration drift
    fn calculate_calibration_drift(samples: &[f64]) -> BspResult<f32> {
        if samples.len() < 100 {
            return Ok(0.1); // Low drift for short signals
        }

        // Compare first and last segments for drift
        let segment_size = samples.len() / 10;
        let first_segment = &samples[0..segment_size];
        let last_segment = &samples[samples.len() - segment_size..];

        let first_mean = first_segment.iter().sum::<f64>() / first_segment.len() as f64;
        let last_mean = last_segment.iter().sum::<f64>() / last_segment.len() as f64;

        let drift_percent = ((last_mean - first_mean).abs() / (first_mean.abs() + 1e-10)) * 100.0;
        Ok(drift_percent as f32)
    }

    /// Calculate measurement repeatability
    fn calculate_repeatability(samples: &[f64]) -> BspResult<f32> {
        if samples.len() < 10 {
            return Ok(1.0);
        }

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance: f64 = samples.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (samples.len() - 1) as f64;

        let cv_percent = if mean.abs() > 1e-10 {
            (variance.sqrt() / mean.abs()) * 100.0
        } else {
            100.0
        };

        Ok(cv_percent as f32)
    }

    /// Calculate baseline stability
    fn calculate_baseline_stability(samples: &[f64]) -> BspResult<f32> {
        if samples.len() < 50 {
            return Ok(0.95);
        }

        // Calculate trend in signal baseline
        let window_size = samples.len() / 10;
        let mut baseline_trend = 0.0;

        for i in 0..(samples.len() - window_size) {
            let window_mean = samples[i..i + window_size].iter().sum::<f64>() / window_size as f64;
            if i > 0 {
                baseline_trend += window_mean.abs();
            }
        }

        // Stability score (0.0 to 1.0, higher is better)
        let stability = (1.0 / (1.0 + baseline_trend / samples.len() as f64)).min(1.0);
        Ok(stability as f32)
    }

    /// Estimate contact impedance for electrode-based signals
    fn estimate_contact_impedance(samples: &[f64], signal_type: SignalType) -> Option<f32> {
        match signal_type {
            SignalType::Physiological(_) => {
                // Simplified impedance estimation based on signal characteristics
                if samples.is_empty() {
                    return Some(1000.0); // High impedance for empty signal
                }

                let variance: f64 = {
                    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
                    samples.iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f64>() / samples.len() as f64
                };

                // High variance might indicate poor contact
                let impedance = (variance.sqrt() * 1000.0).min(50000.0);
                Some(impedance as f32)
            }
            _ => None, // Non-electrode signals don't have contact impedance
        }
    }

    /// Assess motion artifacts
    fn assess_motion_artifacts(samples: &[f64], _signal_type: SignalType) -> BspResult<f32> {
        if samples.len() < 10 {
            return Ok(0.1);
        }

        // Look for sudden changes that might indicate motion
        let mut motion_score = 0.0;
        for window in samples.windows(3) {
            let accel = window[2] - 2.0 * window[1] + window[0];
            motion_score += accel.abs();
        }

        motion_score /= (samples.len() - 2) as f64;
        Ok((motion_score * 10.0).min(1.0) as f32)
    }

    /// Calculate overall quality score
    fn calculate_overall_score(
        signal: &SignalQuality,
        temporal: &TemporalQuality,
        system: &SystemQuality,
        clinical: &ClinicalQuality,
    ) -> f32 {
        // Weighted combination of quality dimensions
        let signal_score = Self::signal_quality_score(signal);
        let temporal_score = Self::temporal_quality_score(temporal);
        let system_score = Self::system_quality_score(system);
        let clinical_score = Self::clinical_quality_score(clinical);

        // Weighted average (weights sum to 1.0)
        0.4 * signal_score + 0.3 * temporal_score + 0.15 * system_score + 0.15 * clinical_score
    }

    /// Convert signal quality to score (0.0 - 1.0)
    fn signal_quality_score(quality: &SignalQuality) -> f32 {
        let snr_score = (quality.snr_db / 60.0).min(1.0).max(0.0);
        let thd_score = (1.0 - quality.thd_percent / 10.0).max(0.0);
        let artifact_score = (1.0 - quality.artifact_percent / 20.0).max(0.0);
        let saturation_score = (1.0 - quality.saturation_percent / 10.0).max(0.0);

        (snr_score + thd_score + artifact_score + saturation_score + quality.dynamic_range + quality.frequency_quality) / 6.0
    }

    /// Convert temporal quality to score (0.0 - 1.0)
    fn temporal_quality_score(quality: &TemporalQuality) -> f32 {
        let jitter_score = (1.0 - quality.jitter_percent * 10.0).max(0.0);
        let completeness_score = quality.completeness_percent / 100.0;
        let stability_score = (1.0 - quality.rate_stability * 100.0).max(0.0);

        (jitter_score + completeness_score + stability_score) / 3.0
    }

    /// Convert system quality to score (0.0 - 1.0)
    fn system_quality_score(quality: &SystemQuality) -> f32 {
        let latency_score = (1.0 - (quality.latency_us as f32 / 100000.0)).max(0.0);
        let cpu_score = (1.0 - quality.cpu_utilization / 100.0).max(0.0);
        let error_score = (1.0 - quality.error_rate * 100.0).max(0.0);

        (latency_score + cpu_score + error_score) / 3.0
    }

    /// Convert clinical quality to score (0.0 - 1.0)
    fn clinical_quality_score(quality: &ClinicalQuality) -> f32 {
        let drift_score = (1.0 - quality.calibration_drift / 5.0).max(0.0);
        let repeatability_score = (1.0 - quality.repeatability_cv / 10.0).max(0.0);
        let baseline_score = quality.baseline_stability;
        let motion_score = 1.0 - quality.motion_artifact_level;

        (drift_score + repeatability_score + baseline_score + motion_score) / 4.0
    }

    /// Get default quality thresholds for medical-grade signals
    pub fn medical_grade_thresholds() -> QualityThresholds {
        QualityThresholds {
            min_snr_db: 20.0,
            max_thd_percent: 1.0,
            max_artifact_percent: 5.0,
            max_jitter_percent: 0.01,
            min_completeness_percent: 99.9,
            max_latency_us: 10000,
            max_calibration_drift: 1.0,
            max_repeatability_cv: 2.0,
        }
    }

    /// Validate quality against thresholds
    pub fn validate_quality(
        metrics: &QualityMetrics,
        thresholds: &QualityThresholds,
    ) -> BspResult<()> {
        if metrics.signal_quality.snr_db < thresholds.min_snr_db {
            return Err(BspError::QualityAssessmentFailed {
                dimension: "SNR",
                measured: metrics.signal_quality.snr_db,
                threshold: thresholds.min_snr_db,
            });
        }

        if metrics.signal_quality.thd_percent > thresholds.max_thd_percent {
            return Err(BspError::QualityAssessmentFailed {
                dimension: "THD",
                measured: metrics.signal_quality.thd_percent,
                threshold: thresholds.max_thd_percent,
            });
        }

        if metrics.signal_quality.artifact_percent > thresholds.max_artifact_percent {
            return Err(BspError::QualityAssessmentFailed {
                dimension: "artifacts",
                measured: metrics.signal_quality.artifact_percent,
                threshold: thresholds.max_artifact_percent,
            });
        }

        if metrics.temporal_quality.jitter_percent > thresholds.max_jitter_percent {
            return Err(BspError::QualityAssessmentFailed {
                dimension: "jitter",
                measured: metrics.temporal_quality.jitter_percent,
                threshold: thresholds.max_jitter_percent,
            });
        }

        if metrics.temporal_quality.completeness_percent < thresholds.min_completeness_percent {
            return Err(BspError::QualityAssessmentFailed {
                dimension: "completeness",
                measured: metrics.temporal_quality.completeness_percent,
                threshold: thresholds.min_completeness_percent,
            });
        }

        Ok(())
    }
}

impl fmt::Display for QualityDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QualityDimension::SignalIntegrity => write!(f, "Signal Integrity"),
            QualityDimension::TemporalAccuracy => write!(f, "Temporal Accuracy"),
            QualityDimension::SystemPerformance => write!(f, "System Performance"),
            QualityDimension::ClinicalAccuracy => write!(f, "Clinical Accuracy"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal_types::{PhysiologicalSignal, MuscleLoc, SignalPolarity};

    #[test]
    fn test_quality_assessment() {
        let data = vec![1.0f32, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98];
        let signal_type = SignalType::Physiological(PhysiologicalSignal::EMG {
            location: MuscleLoc::Arm,
            polarity: SignalPolarity::Differential,
        });

        let processing_time = Duration::from_micros(100);
        let metrics = QualityAssessment::assess(&data, signal_type, 1000.0, processing_time);

        assert!(metrics.is_ok());
        let quality = metrics.unwrap();
        assert!(quality.overall_score >= 0.0 && quality.overall_score <= 1.0);
    }

    #[test]
    fn test_snr_calculation() {
        let clean_signal = vec![1.0, 1.0, 1.0, 1.0, 1.0]; // No noise
        let snr = QualityAssessment::calculate_snr(&clean_signal).unwrap();
        assert!(snr > 40.0); // Should have high SNR

        let noisy_signal = vec![1.0, 0.5, 1.5, 0.8, 1.2]; // Noisy
        let snr_noisy = QualityAssessment::calculate_snr(&noisy_signal).unwrap();
        assert!(snr_noisy < snr); // Should have lower SNR
    }

    #[test]
    fn test_quality_thresholds() {
        let thresholds = QualityAssessment::medical_grade_thresholds();
        assert_eq!(thresholds.min_snr_db, 20.0);
        assert_eq!(thresholds.max_thd_percent, 1.0);
        assert_eq!(thresholds.max_artifact_percent, 5.0);
    }

    #[test]
    fn test_artifact_detection() {
        let normal_signal = vec![1.0, 1.1, 0.9, 1.0, 1.05, 0.95];
        let signal_type = SignalType::Physiological(PhysiologicalSignal::EMG {
            location: MuscleLoc::Arm,
            polarity: SignalPolarity::Differential,
        });

        let artifacts = QualityAssessment::detect_artifacts(&normal_signal, signal_type).unwrap();
        assert!(artifacts < 10.0); // Should have low artifact percentage

        let artifact_signal = vec![1.0, 1.0, 10.0, 1.0, 1.0, -10.0]; // Clear artifacts
        let artifacts_high = QualityAssessment::detect_artifacts(&artifact_signal, signal_type).unwrap();
        assert!(artifacts_high > artifacts); // Should detect more artifacts
    }
}