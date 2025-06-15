//! Basic usage examples for BSP-Framework
//!
//! This example demonstrates the fundamental operations of the BSP-Framework
//! including entity creation, data access, quality assessment, and serialization.

use std::str::FromStr;
use bsp_core::{
    SignalEntity, SignalType, signal_types::*, metadata::*, quality::*,
    timestamp::*, format::*, error::BspResult
};

fn main() -> BspResult<()> {
    println!("=== BSP-Framework Basic Usage Examples ===\n");

    // Example 1: Creating a signal entity with EMG data
    emg_signal_example()?;

    // Example 2: Multi-channel EEG data processing
    eeg_multichannel_example()?;

    // Example 3: Quality assessment and validation
    quality_assessment_example()?;

    // Example 4: Data serialization and deserialization
   // serialization_example()?;

    // Example 5: Real-time processing simulation
    realtime_processing_example()?;

    // Example 6: Complex signal types
    //complex_signal_example()?;

    println!("=== All examples completed successfully! ===");
    Ok(())
}

/// Example 1: Basic EMG signal processing
fn emg_signal_example() -> BspResult<()> {
    println!("1. EMG Signal Processing Example");
    println!("   Creating EMG signal entity from muscle activity data...");

    // Simulate EMG data: 1000 samples at 2kHz from arm muscle
    let emg_data: Vec<f32> = (0..1000)
        .map(|i| {
            let t = i as f32 / 2000.0; // Time in seconds
            // Simulate muscle activation with noise
            0.5 * (20.0 * t).sin() + 0.1 * (200.0 * t).sin() + 0.05 * rand_noise()
        })
        .collect();

    let signal_type = SignalType::Physiological(PhysiologicalSignal::EMG {
        location: MuscleLoc::Arm,
        polarity: SignalPolarity::Differential,
    });

    // Create signal entity
    let entity = SignalEntity::new_owned(
        emg_data,
        signal_type,
        2000.0, // 2 kHz sampling rate
        1,      // Single channel
    )?;

    println!("   ✓ Created EMG entity: {}", entity);
    println!("   ✓ Signal duration: {:.2}ms", entity.duration().as_millis());
    println!("   ✓ SIMD aligned: {}", entity.is_simd_aligned());

    // Access and analyze data
    let data = entity.data();
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let rms = (data.iter().map(|&x| x * x).sum::<f32>() / data.len() as f32).sqrt();

    println!("   ✓ Signal statistics: mean={:.3}, RMS={:.3}", mean, rms);

    Ok(())
}

/// Example 2: Multi-channel EEG processing
fn eeg_multichannel_example() -> BspResult<()> {
    println!("\n2. Multi-channel EEG Processing Example");
    println!("   Creating 8-channel EEG signal entity...");

    let channels = 8;
    let samples_per_channel = 512;
    let total_samples = channels * samples_per_channel;

    // Simulate EEG data with different frequency components per channel
    let eeg_data: Vec<f32> = (0..total_samples)
        .map(|i| {
            let channel = i % channels;
            let sample = i / channels;
            let t = sample as f32 / 256.0; // Time in seconds (256 Hz)

            // Different frequency content per channel
            match channel {
                0 => 0.1 * (8.0 * t).sin(),   // Alpha rhythm (8 Hz)
                1 => 0.15 * (10.0 * t).sin(), // Alpha rhythm (10 Hz)
                2 => 0.05 * (4.0 * t).sin(),  // Theta rhythm (4 Hz)
                3 => 0.08 * (12.0 * t).sin(), // Alpha rhythm (12 Hz)
                4 => 0.2 * (25.0 * t).sin(),  // Beta rhythm (25 Hz)
                5 => 0.12 * (6.0 * t).sin(),  // Theta rhythm (6 Hz)
                6 => 0.18 * (30.0 * t).sin(), // Beta rhythm (30 Hz)
                7 => 0.07 * (2.0 * t).sin(),  // Delta rhythm (2 Hz)
                _ => 0.0,
            }
        })
        .collect();

    let signal_type = SignalType::Physiological(PhysiologicalSignal::EEG {
        placement: EEGPlacement::International1010,
        reference: EEGReference::CommonAverage,
    });

    let entity = SignalEntity::new_owned(
        eeg_data,
        signal_type,
        256.0, // 256 Hz sampling rate
        channels,
    )?;

    println!("   ✓ Created EEG entity: {}", entity);

    // Analyze each channel
    for ch in 0..channels {
        let channel = entity.channel(ch)?;
        let power: f32 = channel.iter().map(|x| x * x).sum();
        println!("   ✓ Channel {}: power={:.4}", ch, power);
    }

    Ok(())
}

/// Example 3: Quality assessment and validation
fn quality_assessment_example() -> BspResult<()> {
    println!("\n3. Quality Assessment Example");
    println!("   Assessing signal quality metrics...");

    // Create a test signal with known characteristics
    let clean_signal: Vec<f32> = (0..1000)
        .map(|i| (i as f32 * 0.01).sin()) // Clean 1 Hz sine wave
        .collect();

    let noisy_signal: Vec<f32> = (0..1000)
        .map(|i| {
            let clean = (i as f32 * 0.01).sin();
            let noise = 0.1 * rand_noise();
            clean + noise
        })
        .collect();

    let signal_type = SignalType::Physiological(PhysiologicalSignal::EMG {
        location: MuscleLoc::Arm,
        polarity: SignalPolarity::Differential,
    });

    // Assess quality of clean signal
    let processing_time = Duration::from_micros(100);
    let clean_quality = QualityAssessment::assess(
        &clean_signal,
        signal_type,
        1000.0,
        processing_time,
    )?;

    // Assess quality of noisy signal
    let noisy_quality = QualityAssessment::assess(
        &noisy_signal,
        signal_type,
        1000.0,
        processing_time,
    )?;

    println!("   ✓ Clean signal quality: {:.2}", clean_quality.overall_score);
    println!("   ✓ Clean signal SNR: {:.1} dB", clean_quality.signal_quality.snr_db);

    println!("   ✓ Noisy signal quality: {:.2}", noisy_quality.overall_score);
    println!("   ✓ Noisy signal SNR: {:.1} dB", noisy_quality.signal_quality.snr_db);

    // Validate against medical-grade thresholds
    let thresholds = QualityAssessment::medical_grade_thresholds();

    match QualityAssessment::validate_quality(&clean_quality, &thresholds) {
        Ok(()) => println!("   ✓ Clean signal meets medical-grade standards"),
        Err(e) => println!("   ✗ Clean signal quality issue: {}", e),
    }

    match QualityAssessment::validate_quality(&noisy_quality, &thresholds) {
        Ok(()) => println!("   ✓ Noisy signal meets medical-grade standards"),
        Err(e) => println!("   ✗ Noisy signal quality issue: {}", e),
    }

    Ok(())
}

/// Example 4: Data serialization and format handling
fn serialization_example() -> BspResult<()> {
    println!("\n4. Serialization Example");
    println!("   Demonstrating data format handling...");

    // Create test data
    let data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
    let signal_type = SignalType::Physiological(PhysiologicalSignal::ECG {
        lead: ECGLead::II,
    });

    // Create device information
    let device_info = DeviceInfo {
        manufacturer: heapless::String::from_str("ExampleCorp").unwrap(),
        model: heapless::String::from_str("BioSensor-3000").unwrap(),
        serial_number: heapless::String::from_str("BSP-001").unwrap(),
        firmware_version: heapless::String::from_str("2.1.0").unwrap(),
        hardware_revision: heapless::String::from_str("B3").unwrap(),
        calibration_date: Some(PrecisionTimestamp::from_secs(1640995200)), // 2022-01-01
        device_config: DeviceConfiguration {
            gain_settings: heapless::Vec::new(),
            offset_corrections: heapless::Vec::new(),
            impedances: heapless::Vec::new(),
            temperature: Some(24.5),
            supply_voltage: Some(3.3),
            additional_params: heapless::FnvIndexMap::new(),
        },
    };

    // Create metadata
    let metadata = SignalMetadata::new(signal_type, 500.0, 1, device_info)?;

    // Serialize using BSP native format
    let handler = BSPNativeHandler::new();
    let options = SerializationOptions {
        format: DataFormat::BSPNative,
        include_metadata: true,
        compress: false,
        little_endian: true,
        alignment: 32,
    };

    let serialized = handler.serialize(&data, &metadata, &options)?;
    println!("   ✓ Serialized {} bytes", serialized.len());

    // Deserialize
    let deser_options = DeserializationOptions {
        expected_format: Some(DataFormat::BSPNative),
        validate_checksums: true,
        strict_validation: true,
    };

    let (deserialized_data, deserialized_metadata) = handler.deserialize(
        &serialized,
        &deser_options,
    )?;

    println!("   ✓ Deserialized {} samples", deserialized_data.len());
    println!("   ✓ Sampling rate: {} Hz", deserialized_metadata.acquisition.sampling_rate);

    // Verify data integrity
    let original_sum: f32 = data.iter().sum();
    let deserialized_sum: f32 = deserialized_data.iter().sum();
    let diff = (original_sum - deserialized_sum).abs();

    println!("   ✓ Data integrity check: diff={:.6}", diff);

    Ok(())
}

/// Example 5: Real-time processing simulation
fn realtime_processing_example() -> BspResult<()> {
    println!("\n5. Real-time Processing Simulation");
    println!("   Simulating streaming biosignal processing...");

    let chunk_size = 128; // Process in 128-sample chunks
    let total_chunks = 10;
    let sampling_rate = 1000.0;

    let signal_type = SignalType::Physiological(PhysiologicalSignal::PPG {
        site: PPGSite::Finger,
        wavelength: PPGWavelength::Red(660),
    });

    let mut processing_times = Vec::new();

    for chunk_idx in 0..total_chunks {
        let start_time = std::time::Instant::now();

        // Generate chunk of PPG data
        let chunk_data: Vec<f32> = (0..chunk_size)
            .map(|i| {
                let t = (chunk_idx * chunk_size + i) as f32 / sampling_rate;
                // Simulate PPG with heart rate ~60 BPM
                1.0 + 0.1 * (2.0 * std::f32::consts::PI * t).sin() + 0.02 * rand_noise()
            })
            .collect();

        // Create entity (simulates data acquisition)
        let entity = SignalEntity::new_owned(
            chunk_data,
            signal_type,
            sampling_rate,
            1,
        )?;

        // Process data (simple moving average filter)
        let data = entity.data();
        let filtered: Vec<f32> = data.windows(3)
            .map(|window| window.iter().sum::<f32>() / 3.0)
            .collect();

        // Quality assessment
        let processing_elapsed = start_time.elapsed();
        let _quality = QualityAssessment::assess(
            &filtered,
            signal_type,
            sampling_rate,
            Duration::from_nanos(processing_elapsed.as_nanos() as u64),
        )?;

        let total_elapsed = start_time.elapsed();
        processing_times.push(total_elapsed);

        println!("   ✓ Chunk {}: processed in {:.2}μs",
                 chunk_idx, total_elapsed.as_micros());
    }

    let avg_time: f64 = processing_times.iter()
        .map(|d| d.as_micros() as f64)
        .sum::<f64>() / processing_times.len() as f64;

    println!("   ✓ Average processing time: {:.2}μs per chunk", avg_time);

    // Check real-time performance
    let chunk_duration_us = (chunk_size as f64 / sampling_rate as f64) * 1_000_000.0;
    let realtime_ratio = avg_time / chunk_duration_us;

    println!("   ✓ Real-time ratio: {:.3} (< 1.0 = real-time capable)", realtime_ratio);

    Ok(())
}
/*
/// Example 6: Complex signal processing
fn complex_signal_example() -> BspResult<()> {
    println!("\n6. Complex Signal Processing Example");
    println!("   Working with complex-valued frequency domain data...");

    use bsp_core::Complex32;

    // Generate complex FFT-like data
    let fft_data: Vec<Complex32> = (0..512)
        .map(|i| {
            let freq = i as f32 / 512.0;
            let magnitude = if freq > 0.1 && freq < 0.3 { 1.0 } else { 0.1 };
            let phase = freq * 2.0 * std::f32::consts::PI;
            Complex32::new(magnitude * phase.cos(), magnitude * phase.sin())
        })
        .collect();

    let signal_type = SignalType::Derived(DerivedSignal::FrequencyDomain {
        transform: FrequencyTransform::FFT,
    });

    let entity = SignalEntity::new_owned(
        fft_data,
        signal_type,
        1000.0, // Original sampling rate before FFT
        1,
    )?;

    println!("   ✓ Created complex entity: {}", entity);

    // Analyze frequency content
    let data = entity.data();
    let total_power: f32 = data.iter().map(|c| c.norm_sqr()).sum();
    let peak_bin = data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let peak_frequency = peak_bin as f32 * 1000.0 / 512.0; // Convert bin to frequency

    println!("   ✓ Total power: {:.2}", total_power);
    println!("   ✓ Peak frequency: {:.1} Hz", peak_frequency);

    Ok(())
}
*/
/// Simple pseudo-random noise generator for examples
fn rand_noise() -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    let mut hasher = DefaultHasher::new();
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos().hash(&mut hasher);
    let hash = hasher.finish();

    // Convert to [-1, 1] range
    ((hash % 10000) as f32 / 5000.0) - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_examples() {
        assert!(emg_signal_example().is_ok());
        assert!(eeg_multichannel_example().is_ok());
        assert!(quality_assessment_example().is_ok());
        assert!(serialization_example().is_ok());
        assert!(realtime_processing_example().is_ok());
       // assert!(complex_signal_example().is_ok());
    }
}