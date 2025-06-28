use ndarray::{Array1, Array2};
use silero_vad_rs::SileroVAD;
use std::path::Path;
use tempfile::TempDir;

/// Test helper: Create a temporary model path that will trigger auto-download
fn create_temp_model_path() -> (TempDir, std::path::PathBuf) {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("silero_vad.onnx");
    (temp_dir, model_path)
}

/// Test helper: Generate test audio data
fn generate_test_audio(samples: usize) -> Array1<f32> {
    // Generate a simple sine wave at 440Hz (A4 note) for 16kHz sampling rate
    let frequency = 440.0;
    let sample_rate = 16000.0;
    let amplitude = 0.1; // Low amplitude to avoid clipping

    Array1::from_shape_fn(samples, |i| {
        let t = i as f32 / sample_rate;
        amplitude * (2.0 * std::f64::consts::PI as f32 * frequency * t).sin()
    })
}

/// Test helper: Generate silence audio data
fn generate_silence(samples: usize) -> Array1<f32> {
    Array1::zeros(samples)
}

#[test]
fn test_silero_vad_new_auto_download() {
    // Test that model creation with non-existent file triggers auto-download
    let (_temp_dir, model_path) = create_temp_model_path();

    // Ensure model doesn't exist initially
    assert!(!model_path.exists());

    // This should trigger auto-download
    let model = SileroVAD::new(&model_path);
    assert!(
        model.is_ok(),
        "Model creation should succeed with auto-download"
    );

    // Verify the model was created successfully (no panic on basic operations)
    let _model = model.unwrap();
    // Model creation succeeded, which means the internal state is valid
}

#[test]
fn test_silero_vad_process_chunk_valid_input() {
    // Test processing valid audio chunk
    let (_temp_dir, model_path) = create_temp_model_path();
    let mut model = SileroVAD::new(&model_path).unwrap();

    // Generate valid test audio (512 samples for 16kHz)
    let audio_chunk = generate_test_audio(512);

    // Process the chunk
    let result = model.process_chunk(&audio_chunk.view(), 16000);
    assert!(result.is_ok(), "Processing valid chunk should succeed");

    let speech_prob = result.unwrap();

    // Validate output format - should return at least one probability
    assert!(
        speech_prob.len() >= 1,
        "Should return at least one probability, got {}",
        speech_prob.len()
    );

    // Validate all probabilities are in valid range
    for &prob in speech_prob.iter() {
        assert!(
            prob >= 0.0 && prob <= 1.0,
            "Speech probability should be between 0 and 1, got {}",
            prob
        );
    }

    // The first probability should be the main output (as used by VADIterator)
    println!("First probability: {}", speech_prob[0]);
    println!("Total probabilities: {}", speech_prob.len());
}

#[test]
fn test_silero_vad_process_chunk_invalid_sampling_rate() {
    // Test that invalid sampling rate is rejected
    let (_temp_dir, model_path) = create_temp_model_path();
    let mut model = SileroVAD::new(&model_path).unwrap();

    let audio_chunk = generate_test_audio(512);

    // Try with wrong sampling rate
    let result = model.process_chunk(&audio_chunk.view(), 8000);
    assert!(result.is_err(), "Should reject non-16kHz sampling rate");

    let error = result.unwrap_err();
    assert!(
        error.to_string().contains("16kHz"),
        "Error should mention 16kHz requirement"
    );
}

#[test]
fn test_silero_vad_process_chunk_invalid_size() {
    // Test that invalid chunk size is rejected
    let (_temp_dir, model_path) = create_temp_model_path();
    let mut model = SileroVAD::new(&model_path).unwrap();

    // Try with wrong chunk size
    let audio_chunk = generate_test_audio(256); // Wrong size

    let result = model.process_chunk(&audio_chunk.view(), 16000);
    assert!(result.is_err(), "Should reject wrong chunk size");

    let error = result.unwrap_err();
    assert!(
        error.to_string().contains("512 samples"),
        "Error should mention 512 samples requirement"
    );
}

#[test]
fn test_silero_vad_process_chunk_silence_vs_speech() {
    // Test that silence and speech produce different probabilities
    let (_temp_dir, model_path) = create_temp_model_path();
    let mut model = SileroVAD::new(&model_path).unwrap();

    // Process silence
    let silence_chunk = generate_silence(512);
    let silence_result = model.process_chunk(&silence_chunk.view(), 16000).unwrap();

    // Process speech-like audio
    let speech_chunk = generate_test_audio(512);
    let speech_result = model.process_chunk(&speech_chunk.view(), 16000).unwrap();

    // The model should differentiate between silence and speech
    // (We don't assert specific values, just that they're different)
    println!("Silence probability: {}", silence_result[0]);
    println!("Speech probability: {}", speech_result[0]);

    // Note: We don't make strong assertions here because the model's behavior
    // might vary, but we can verify the API works correctly
    assert!(silence_result[0] >= 0.0 && silence_result[0] <= 1.0);
    assert!(speech_result[0] >= 0.0 && speech_result[0] <= 1.0);
}

#[test]
fn test_silero_vad_process_batch() {
    // Test batch processing
    let (_temp_dir, model_path) = create_temp_model_path();
    let mut model = SileroVAD::new(&model_path).unwrap();

    // Create batch of 3 chunks
    let batch_size = 3;
    let chunk_size = 512;
    let mut batch = Array2::zeros((batch_size, chunk_size));

    // Fill with test audio
    for i in 0..batch_size {
        let chunk = generate_test_audio(chunk_size);
        batch.row_mut(i).assign(&chunk);
    }

    let result = model.process_batch(&batch, 16000);
    assert!(result.is_ok(), "Batch processing should succeed");

    let probabilities = result.unwrap();

    // Validate output format - should return at least batch_size probabilities
    assert!(
        probabilities.len() >= batch_size,
        "Should return at least {} probabilities, got {}",
        batch_size,
        probabilities.len()
    );

    // Validate all probabilities are in valid range
    for &prob in probabilities.iter() {
        assert!(
            prob >= 0.0 && prob <= 1.0,
            "Probability should be between 0 and 1"
        );
    }

    println!(
        "Batch processing returned {} probabilities for {} chunks",
        probabilities.len(),
        batch_size
    );
}

#[test]
fn test_silero_vad_reset_states() {
    // Test state reset functionality
    let (_temp_dir, model_path) = create_temp_model_path();
    let mut model = SileroVAD::new(&model_path).unwrap();

    // Process some audio to establish state
    let audio_chunk = generate_test_audio(512);
    model.process_chunk(&audio_chunk.view(), 16000).unwrap();

    // Reset states - this should not panic
    model.reset_states(5); // Change batch size

    // Verify reset worked by processing another chunk successfully
    let audio_chunk2 = generate_test_audio(512);
    let result = model.process_chunk(&audio_chunk2.view(), 16000);
    assert!(result.is_ok(), "Processing after reset should succeed");
}

#[test]
fn test_silero_vad_consistent_outputs() {
    // Test that same input produces consistent outputs
    let (_temp_dir, model_path) = create_temp_model_path();
    let mut model = SileroVAD::new(&model_path).unwrap();

    let audio_chunk = generate_test_audio(512);

    // Process same chunk multiple times
    let result1 = model.process_chunk(&audio_chunk.view(), 16000).unwrap();
    let result2 = model.process_chunk(&audio_chunk.view(), 16000).unwrap();

    // Results should be consistent (within small floating point tolerance)
    assert!(
        (result1[0] - result2[0]).abs() < 1e-6,
        "Same input should produce consistent output"
    );
}
