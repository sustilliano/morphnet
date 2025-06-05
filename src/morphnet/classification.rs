use super::*;
use crate::TensorData;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub label: String,
    pub confidence: f32,
}

pub fn classify(net: &MorphNet, input: &TensorData) -> Result<ClassificationResult> {
    // Simple brightness based classifier used as an example implementation.
    let sum: f32 = input.data.iter().copied().sum();
    let mean = if input.data.len() > 0 {
        sum / input.data.len() as f32
    } else { 0.0 };

    let threshold = net.brightness_threshold;
    let (label, confidence) = if mean >= threshold {
        ("light", (mean - threshold).min(1.0))
    } else {
        ("dark", (threshold - mean).min(1.0))
    };

    Ok(ClassificationResult { label: label.to_string(), confidence })
}
