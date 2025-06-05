//! MorphNet: Geometric Template Learning for Structural Understanding

// Re-export placeholder functionality for experimental extensions
pub use crate::morphnet_placeholder::*;

pub mod model;
pub mod templates;
pub mod classification;
pub mod training;

pub use model::*;
pub use templates::*;
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
