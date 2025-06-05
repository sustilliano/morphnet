//! MorphNet: Geometric Template Learning for Structural Understanding

pub mod model;
pub mod classification;
pub mod training;

pub use model::*;
pub use classification::*;
pub use training::*;

use ndarray::{Array1, Array2, Array3};
use nalgebra::{Point2, Point3, Vector3};
use std::collections::HashMap;

/// Core MorphNet framework error types
#[derive(thiserror::Error, Debug)]
pub enum MorphNetError {
    #[error("Template error: {0}")]
    Template(String),
    
    #[error("Classification error: {0}")]
    Classification(String),
    
    #[error("Training error: {0}")]
    Training(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, MorphNetError>;

/// Utility functions frequently used across the framework
pub fn point_distance(a: Point2<f32>, b: Point2<f32>) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

pub fn vector_magnitude(v: Vector3<f32>) -> f32 {
    (v.x.powi(2) + v.y.powi(2) + v.z.powi(2)).sqrt()
}

pub fn stack_tensors(tensors: &[Array3<f32>]) -> Array3<f32> {
    let (_d0, d1, d2) = tensors[0].dim();
    let mut stacked = Array3::zeros((tensors.len(), d1, d2));
    for (i, t) in tensors.iter().enumerate() {
        stacked
            .slice_mut(ndarray::s![i, .., ..])
            .assign(&t.slice(ndarray::s![0, .., ..]));
    }
    stacked
}

pub fn stack_vectors(vectors: &[Array1<f32>]) -> Array2<f32> {
    let len = vectors[0].len();
    let mut out = Array2::zeros((vectors.len(), len));
    for (i, v) in vectors.iter().enumerate() {
        out.slice_mut(ndarray::s![i, ..]).assign(v);
    }
    out
}

pub fn group_points(points: Vec<Point3<f32>>) -> HashMap<String, Point3<f32>> {
    points
        .into_iter()
        .enumerate()
        .map(|(i, p)| (format!("p{}", i), p))
        .collect()
}
