//! Patch quilt and mesh refinement utilities

use super::mmx::MeshData;
use serde::{Serialize, Deserialize};
use ndarray::{Array1, Array2, Array3};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use nalgebra::{Point3, Vector3};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RefinementConfig;

#[derive(Debug, Clone)]
pub struct Patch {
    pub id: Uuid,
    pub source_id: String,
    pub timestamp: DateTime<Utc>,
    pub position: Point3<f32>,
    pub normal: Vector3<f32>,
    pub texture: Array3<f32>,
    pub depth: Array2<f32>,
    pub normals: Array3<f32>,
    pub confidence: f32,
    pub world_size: (f32, f32),
    pub uv_coords: Option<(f32, f32)>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct SensorData {
    pub visual: Option<Array3<f32>>,        
    pub lidar: Option<Array2<f32>>,        
    pub accelerometer: Option<Array1<f32>>,        
    pub strain_gauges: Option<HashMap<String, f32>>,        
    pub temperature: Option<HashMap<String, f32>>,        
    pub pressure: Option<HashMap<String, f32>>,        
    pub custom: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct PatchQuilt {
    patches: Vec<Patch>,
}

impl PatchQuilt {
    pub fn new(_config: RefinementConfig) -> Self { Self { patches: Vec::new() } }

    /// Return IDs of all stored patches
    pub fn list_chunks(&self) -> Vec<String> {
        self.patches.iter().map(|p| p.id.to_string()).collect()
    }

    /// Append new patches associated with a subject
    pub fn update_patch_quilt(&mut self, _subject: String, patches: Vec<Patch>) {
        self.patches.extend(patches);
    }

    /// Find patches within the given radius of a point, optionally filtered by subject id
    pub fn find_patches_near(&self, position: Point3<f32>, radius: f32, subject: Option<&str>) -> Vec<&Patch> {
        self.patches
            .iter()
            .filter(|p| {
                let dist = (p.position - position).norm();
                dist <= radius && subject.map_or(true, |s| p.source_id == s)
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct MeshRefinement {
    pub mesh: MeshData,
}
