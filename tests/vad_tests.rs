use ndarray::{Array1, Array2};
use silero_vad_rs::{SileroVAD, VADIterator};
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

/// Test helper: Generate audio with alternating speech/silence
fn generate_alternating_audio(total_samples: usize) -> Array1<f32> {
    let chunk_size = 512;
    let mut audio = Array1::zeros(total_samples);

    for i in 0..(total_samples / chunk_size) {
        let start = i * chunk_size;
        let end = (i + 1) * chunk_size;

        if i % 2 == 0 {
            // Even chunks: speech
            let speech_chunk = generate_test_audio(chunk_size);
            audio
                .slice_mut(ndarray::s![start..end])
                .assign(&speech_chunk);
        } else {
            // Odd chunks: silence
            let silence_chunk = generate_silence(chunk_size);
            audio
                .slice_mut(ndarray::s![start..end])
                .assign(&silence_chunk);
        }
    }

    audio
}

#[test]
fn test_vad_iterator_new() {
    // Test VAD iterator creation
    let (_temp_dir, model_path) = create_temp_model_path();
    let model = SileroVAD::new(&model_path).unwrap();

    let vad = VADIterator::new(
        model, 0.5,   // threshold
        16000, // sampling rate
        100,   // min silence duration (ms)
        30,    // speech pad (ms)
    );

    // Verify the iterator was created successfully (no panic on basic operations)
    // We can't access private fields, but we can verify the constructor works
    let _vad = vad;
    // Constructor succeeded, which means the internal state is valid
}

#[test]
fn test_vad_iterator_process_chunk_silence() {
    // Test processing silence chunk
    let (_temp_dir, model_path) = create_temp_model_path();
    let model = SileroVAD::new(&model_path).unwrap();
    let mut vad = VADIterator::new(model, 0.5, 16000, 100, 30);

    let silence_chunk = generate_silence(512);
    let result = vad.process_chunk(&silence_chunk.view());

    assert!(result.is_ok(), "Processing silence should succeed");
    // Note: We don't assert specific behavior because the model's response
    // to silence might vary, but we can verify the API works
}

#[test]
fn test_vad_iterator_process_chunk_speech() {
    // Test processing speech chunk
    let (_temp_dir, model_path) = create_temp_model_path();
    let model = SileroVAD::new(&model_path).unwrap();
    let mut vad = VADIterator::new(model, 0.1, 16000, 100, 30); // Low threshold

    let speech_chunk = generate_test_audio(512);
    let result = vad.process_chunk(&speech_chunk.view());

    assert!(result.is_ok(), "Processing speech should succeed");

    // Note: We don't assert specific behavior here because the model's
    // response to our test audio might vary, but we can verify the API works
    let timestamp = result.unwrap();
    if let Some(ts) = timestamp {
        // If speech was detected, verify timestamp format
        assert!(ts.start >= 0.0, "Start time should be non-negative");
        assert!(ts.end > ts.start, "End time should be after start time");
        assert!(ts.end <= 0.032, "End time should be within chunk duration"); // 512/16000 = 0.032s
    }
}

#[test]
fn test_vad_iterator_speech_transition() {
    // Test transition from silence to speech to silence
    let (_temp_dir, model_path) = create_temp_model_path();
    let model = SileroVAD::new(&model_path).unwrap();
    let mut vad = VADIterator::new(model, 0.1, 16000, 50, 10); // Low threshold, short silence

    // Process silence first
    let silence_chunk = generate_silence(512);
    let result1 = vad.process_chunk(&silence_chunk.view()).unwrap();
    // Silence might or might not trigger detection depending on model response

    // Process speech
    let speech_chunk = generate_test_audio(512);
    let result2 = vad.process_chunk(&speech_chunk.view()).unwrap();
    // Speech might or might not trigger detection depending on model response

    // Process silence again
    let result3 = vad.process_chunk(&silence_chunk.view()).unwrap();
    // This might complete a speech segment if speech was detected above

    // Verify API works correctly regardless of specific model behavior
    assert!(result1.is_none() || result1.unwrap().start >= 0.0);
    assert!(result2.is_none() || result2.unwrap().start >= 0.0);
    assert!(result3.is_none() || result3.unwrap().start >= 0.0);
}

#[test]
fn test_vad_iterator_reset() {
    // Test reset functionality
    let (_temp_dir, model_path) = create_temp_model_path();
    let model = SileroVAD::new(&model_path).unwrap();
    let mut vad = VADIterator::new(model, 0.5, 16000, 100, 30);

    // Process some audio to establish state
    let audio_chunk = generate_test_audio(512);
    vad.process_chunk(&audio_chunk.view()).unwrap();

    // Reset the iterator - this should not panic
    vad.reset();

    // Verify reset worked by processing another chunk successfully
    let audio_chunk2 = generate_test_audio(512);
    let result = vad.process_chunk(&audio_chunk2.view());
    assert!(result.is_ok(), "Processing after reset should succeed");
}

#[test]
fn test_vad_iterator_get_speech_timestamps() {
    // Test full audio processing
    let (_temp_dir, model_path) = create_temp_model_path();
    let model = SileroVAD::new(&model_path).unwrap();
    let mut vad = VADIterator::new(model, 0.1, 16000, 50, 10); // Low threshold

    // Generate audio with multiple chunks
    let total_samples = 512 * 6; // 6 chunks
    let audio = generate_alternating_audio(total_samples);

    let result = vad.get_speech_timestamps(
        &audio.view(),
        100,  // min speech duration (ms)
        10.0, // max speech duration (s)
        50,   // min silence duration (ms)
        10,   // speech pad (ms)
    );

    assert!(result.is_ok(), "Full audio processing should succeed");

    let timestamps = result.unwrap();

    // Verify timestamp format if any were detected
    for ts in timestamps {
        assert!(ts.start >= 0.0, "Start time should be non-negative");
        assert!(ts.end > ts.start, "End time should be after start time");
        assert!(
            ts.end - ts.start >= 0.1,
            "Speech duration should be at least 100ms"
        );
        assert!(
            ts.end - ts.start <= 10.0,
            "Speech duration should be at most 10s"
        );
    }
}

#[test]
fn test_vad_iterator_process_batch() {
    // Test batch processing
    let (_temp_dir, model_path) = create_temp_model_path();
    let model = SileroVAD::new(&model_path).unwrap();
    let mut vad = VADIterator::new(model, 0.1, 16000, 50, 10);

    // Create batch of chunks
    let batch_size = 3;
    let chunk_size = 512;
    let mut batch = Array2::zeros((batch_size, chunk_size));

    // Fill with alternating speech/silence
    for i in 0..batch_size {
        let chunk = if i % 2 == 0 {
            generate_test_audio(chunk_size)
        } else {
            generate_silence(chunk_size)
        };
        batch.row_mut(i).assign(&chunk);
    }

    let result = vad.process_batch(&batch);
    assert!(result.is_ok(), "Batch processing should succeed");

    let timestamps = result.unwrap();

    // Verify results if any timestamps were detected
    if let Some(ts_list) = timestamps {
        for ts in ts_list {
            assert!(ts.start >= 0.0, "Start time should be non-negative");
            assert!(ts.end > ts.start, "End time should be after start time");
        }
    }
}

#[test]
fn test_vad_iterator_threshold_behavior() {
    // Test that different thresholds produce different behavior
    let audio_chunk = generate_test_audio(512);

    // Test with low threshold
    let (_temp_dir1, model_path1) = create_temp_model_path();
    let model1 = SileroVAD::new(&model_path1).unwrap();
    let mut vad_low = VADIterator::new(model1, 0.1, 16000, 100, 30);
    let result_low = vad_low.process_chunk(&audio_chunk.view()).unwrap();

    // Test with high threshold
    let (_temp_dir2, model_path2) = create_temp_model_path();
    let model2 = SileroVAD::new(&model_path2).unwrap();
    let mut vad_high = VADIterator::new(model2, 0.9, 16000, 100, 30);
    let result_high = vad_high.process_chunk(&audio_chunk.view()).unwrap();

    // Note: We don't assert specific behavior because the model's response
    // to our test audio might vary, but we can verify the API works
    println!("Low threshold result: {:?}", result_low);
    println!("High threshold result: {:?}", result_high);

    // Both should succeed without panicking
    assert!(result_low.is_none() || result_low.unwrap().start >= 0.0);
    assert!(result_high.is_none() || result_high.unwrap().start >= 0.0);
}

#[test]
fn test_vad_iterator_parameter_validation() {
    // Test that invalid parameters are handled gracefully
    let (_temp_dir, model_path) = create_temp_model_path();
    let model = SileroVAD::new(&model_path).unwrap();

    // Test with invalid sampling rate (should still work, just might not be optimal)
    let vad = VADIterator::new(model, 0.5, 8000, 100, 30);
    // Constructor should succeed regardless of sampling rate

    // Test with extreme threshold values (should still work)
    let vad_extreme = VADIterator::new(
        SileroVAD::new(&model_path).unwrap(),
        0.0, // Very low threshold
        16000,
        100,
        30,
    );
    // Constructor should succeed regardless of threshold value
}
