//! Silero Voice Activity Detection (VAD) - Rust Implementation
//!
//! This crate provides a Rust implementation of the [Silero Voice Activity Detection (VAD) model](https://github.com/snakers4/silero-vad).
//! It uses the `ort` crate for efficient ONNX model inference and provides both streaming and batch processing capabilities.
//!
//! # Features
//!
//! - Voice Activity Detection using the Silero model
//! - Support for both 8kHz and 16kHz audio
//! - Streaming VAD with iterator interface and state management
//! - Batch processing for efficient handling of multiple audio chunks
//! - GPU acceleration support via ONNX Runtime with CUDA
//! - Audio file I/O utilities
//! - Automatic model downloading from Silero repository
//! - Multiple language support (English, Russian, German, Spanish)
//! - Comprehensive error handling
//!
//! # Example
//!
//! ```rust
//! use silero_vad_rs::{SileroVAD, VADIterator};
//! use ndarray::Array1;
//! use std::path::Path;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Load the model
//!     let model = SileroVAD::new(Path::new("path/to/silero_vad.onnx"))?;
//!     
//!     // Create a VAD iterator
//!     let mut vad = VADIterator::new(
//!         model,
//!         0.5,  // threshold
//!         16000, // sampling rate
//!         100,   // min silence duration (ms)
//!         30,    // speech pad (ms)
//!     );
//!
//!     // Example audio chunk (in practice, you'd read this from a file)
//!     let audio_chunk = Array1::zeros(512);
//!     
//!     // Process the chunk
//!     if let Some(ts) = vad.process_chunk(&audio_chunk.view())? {
//!         println!("Speech detected from {:.2}s to {:.2}s", ts.start, ts.end);
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod model;
pub mod utils;
pub mod vad;

pub use model::SileroVAD;
pub use vad::{SpeechTimestamps, VADIterator};

/// Supported languages for VAD
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    /// Russian language model
    Russian,
    /// English language model
    English,
    /// German language model
    German,
    /// Spanish language model
    Spanish,
}

/// Error types for the Silero VAD library
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Error occurred while loading the model
    #[error("Model loading error: {0}")]
    ModelLoad(String),
    /// Invalid input parameters or data
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    /// Error during audio processing
    #[error("Audio processing error: {0}")]
    AudioProcessing(String),
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// ONNX Runtime error
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),
}

/// Result type for the Silero VAD library
pub type Result<T> = std::result::Result<T, Error>;
