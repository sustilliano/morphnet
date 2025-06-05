//! Analysis utilities for MorphNet

use super::mmx::EmbeddingData;
use serde::{Serialize, Deserialize};
use ndarray::{Array1, Array2, Axis};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingMethod {
    PCA,
    TSNE,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhylogeneticTree {
    pub nodes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphNetAnalyzer;

impl MorphNetAnalyzer {
    pub fn new() -> Self { Self }

    /// Compute the mean vector and covariance matrix of an embedding
    pub fn embedding_summary(&self, embedding: &EmbeddingData) -> (Array1<f32>, Array2<f32>) {
        let matrix = embedding.matrix.to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
        let mean = matrix.mean_axis(Axis(0)).unwrap();
        let centered = &matrix - &mean;
        let cov = centered.t().dot(&centered) / (matrix.nrows() as f32 - 1.0);
        (mean, cov)
    }
}
