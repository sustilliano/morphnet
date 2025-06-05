use super::*;
use crate::TensorData;
use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use ndarray::{Array1, Array2};

pub fn train(net: &mut MorphNet, data: &[TensorData]) -> Result<()> {
    // Extremely small example "training" that adjusts the brightness threshold
    // based on the mean brightness of the provided dataset.
    if data.is_empty() {
        return Ok(());
    }

    let mut total = 0.0;
    for t in data {
        let sum: f32 = t.data.iter().copied().sum();
        if t.data.len() > 0 {
            total += sum / t.data.len() as f32;
        }
    }
    net.brightness_threshold = total / data.len() as f32;
    Ok(())
}

/// Train a logistic regression classifier using labeled tensors.
pub fn train_logistic(net: &mut MorphNet, data: &[(TensorData, usize)]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }

    let n_samples = data.len();
    let n_features = data[0].0.data.len();
    let mut features = Array2::<f64>::zeros((n_samples, n_features));
    let mut labels = Array1::<usize>::zeros(n_samples);

    for (i, (tensor, label)) in data.iter().enumerate() {
        let slice = tensor.data.as_slice().ok_or_else(|| MorphNetError::Training("invalid tensor".into()))?;
        for (j, v) in slice.iter().enumerate() {
            features[[i, j]] = *v as f64;
        }
        labels[i] = *label;
    }

    let dataset = linfa::Dataset::new(features, labels);
    let model = LogisticRegression::default()
        .max_iterations(100)
        .fit(&dataset)
        .map_err(|e| MorphNetError::Training(e.to_string()))?;
    net.logistic_model = Some(model);
    Ok(())
}
