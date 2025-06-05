use super::*;
use crate::TensorData;
use serde::{Serialize, Deserialize};
use linfa::prelude::*;
use ndarray::Array2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub label: String,
    pub confidence: f32,
}

pub fn classify(net: &MorphNet, input: &TensorData) -> Result<ClassificationResult> {
    if let Some(model) = &net.logistic_model {
        let flat = input.data.iter().map(|v| *v as f64).collect::<Vec<_>>();
        let arr = Array2::from_shape_vec((1, flat.len()), flat)
            .map_err(|e| MorphNetError::Classification(e.to_string()))?;
        let pred = model.predict(&arr);
        let probs = model.predict_probabilities(&arr);
        let label = pred[0].to_string();
        let confidence = probs[0] as f32;
        return Ok(ClassificationResult { label, confidence });
    }

    // Fallback simple brightness-based classifier
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
