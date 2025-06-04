//! Patch quilt and mesh refinement utilities

use super::mmx::MeshData;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Patch {
    pub vertices: Vec<[f32; 3]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchQuilt {
    pub patches: Vec<Patch>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshRefinement {
    pub mesh: MeshData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorData {
    pub readings: Vec<f32>,
}
