//! Analysis utilities for MorphNet

use super::mmx::EmbeddingData;
use serde::{Serialize, Deserialize};

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
}
