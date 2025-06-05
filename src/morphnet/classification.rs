use super::*;
use crate::TensorData;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub label: String,
    pub confidence: f32,
}

pub fn classify(_net: &MorphNet, _input: &TensorData) -> Result<ClassificationResult> {
    // Placeholder classification routine
    Ok(ClassificationResult {
        label: "unknown".to_string(),
        confidence: 0.0,
    })
}
