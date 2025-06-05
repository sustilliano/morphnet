//! MMX chunk implementations for different data types

use super::MMXError;
use ndarray::{ArrayD, IxDyn};
use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Generic tensor data container
#[derive(Debug, Clone)]
pub struct TensorData {
    pub data: ArrayD<f32>,
    pub dtype: String,
    pub shape: Vec<usize>,
}

impl TensorData {
    pub fn new(data: ArrayD<f32>) -> Self {
        let shape = data.shape().to_vec();
        Self {
            data,
            dtype: "float32".to_string(),
            shape,
        }
    }
    
    pub fn from_image(image: &image::RgbImage) -> Self {
        let (width, height) = image.dimensions();
        let mut data = ArrayD::zeros(IxDyn(&[height as usize, width as usize, 3]));
        
        for (x, y, pixel) in image.enumerate_pixels() {
            data[[y as usize, x as usize, 0]] = pixel[0] as f32 / 255.0;
            data[[y as usize, x as usize, 1]] = pixel[1] as f32 / 255.0;
            data[[y as usize, x as usize, 2]] = pixel[2] as f32 / 255.0;
        }
        
        Self::new(data)
    }
    
    pub fn to_bytes(&self) -> Vec<u8> {
        // Convert tensor to bytes (simplified - would need proper serialization)
        bincode::serialize(&(self.shape.clone(), self.data.as_slice().unwrap())).unwrap()
    }
    
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MMXError> {
        let (shape, data_vec): (Vec<usize>, Vec<f32>) = bincode::deserialize(bytes)
            .map_err(|e| MMXError::DataIntegrity(e.to_string()))?;
        
        let data = ArrayD::from_shape_vec(IxDyn(&shape), data_vec)
            .map_err(|e| MMXError::DataIntegrity(e.to_string()))?;
        
        Ok(Self::new(data))
    }
}

/// Sequence data for video or time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceData {
    pub frames: Vec<String>, // References to tensor chunks
    pub fps: Option<f32>,
    pub duration: Option<f32>,
    pub metadata: HashMap<String, String>,
}

/// Text data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextData {
    pub content: String,
    pub encoding: String,
    pub language: Option<String>,
    pub tokens: Option<Vec<u32>>, // Simplified token IDs
}

/// 3D mesh data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshData {
    pub vertices: Vec<Point3<f32>>,
    pub faces: Vec<[u32; 3]>, // Triangle faces
    pub normals: Option<Vec<Vector3<f32>>>,
    pub uvs: Option<Vec<[f32; 2]>>,
    pub materials: Option<HashMap<String, String>>,
}

impl MeshData {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            faces: Vec::new(),
            normals: None,
            uvs: None,
            materials: None,
        }
    }
    
    pub fn add_vertex(&mut self, vertex: Point3<f32>) -> u32 {
        self.vertices.push(vertex);
        (self.vertices.len() - 1) as u32
    }
    
    pub fn add_face(&mut self, face: [u32; 3]) {
        self.faces.push(face);
    }
    
    pub fn compute_normals(&mut self) {
        let mut normals = vec![Vector3::zeros(); self.vertices.len()];
        
        for face in &self.faces {
            let v0 = self.vertices[face[0] as usize];
            let v1 = self.vertices[face[1] as usize];
            let v2 = self.vertices[face[2] as usize];
            
            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let normal = edge1.cross(&edge2).normalize();
            
            for &idx in face {
                normals[idx as usize] += normal;
            }
        }
        
        for normal in &mut normals {
            *normal = normal.normalize();
        }
        
        self.normals = Some(normals);
    }
}

/// Embedding matrix data
#[derive(Debug, Clone)]
pub struct EmbeddingData {
    pub matrix: ArrayD<f32>, // N x D matrix
    pub method: String,
    pub n_components: usize,
    pub labels: Option<Vec<String>>,
}

impl EmbeddingData {
    pub fn new(matrix: ArrayD<f32>, method: String) -> Self {
        let n_components = matrix.shape()[1];
        Self {
            matrix,
            method,
            n_components,
            labels: None,
        }
    }
}

/// Generic metadata container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaData {
    pub data: HashMap<String, serde_json::Value>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub version: String,
}

impl MetaData {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            timestamp: chrono::Utc::now(),
            version: crate::VERSION.to_string(),
        }
    }
    
    pub fn insert<T: Serialize>(&mut self, key: String, value: T) {
        self.data.insert(key, serde_json::to_value(value).unwrap());
    }
    
    pub fn get<T: for<'a> Deserialize<'a>>(&self, key: &str) -> Option<T> {
        self.data.get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }
}
