use super::*;
use crate::TensorData;

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
