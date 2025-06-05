//! MorphNet-GTL: Geometric Template Learning and Spatial Intelligence Framework
//! 
//! A next-generation deep learning system for structural understanding of morphology
//! and spatial pre-awareness, combining neural classification, geometric template
//! inference, and patch-based mesh refinement.

pub mod mmx;
pub mod morphnet;
pub mod morphnet_placeholder;
pub mod patch_quilt;
pub mod analysis;
pub mod spatial;

pub use mmx::{MMXFile, MMXError, ChunkType, TensorData};
pub use morphnet::{MorphNet, BodyPlan, GeometricTemplate, ClassificationResult};
pub use patch_quilt::{PatchQuilt, Patch, MeshRefinement};
pub use analysis::{MorphNetAnalyzer, EmbeddingMethod, PhylogeneticTree};
pub use spatial::{SpatialAwareness, AccountabilityPredictor};

/// Core error types for the framework
#[derive(thiserror::Error, Debug)]
pub enum MorphNetError {
    #[error("MMX format error: {0}")]
    MMX(#[from] MMXError),
    
    #[error("Model error: {0}")]
    Model(String),
    
    #[error("Data processing error: {0}")]
    DataProcessing(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
}

pub type Result<T> = std::result::Result<T, MorphNetError>;

/// Framework version and metadata
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const MAGIC_BYTES: &[u8; 4] = b"MMX\x00";

/// Re-export commonly used types
pub mod prelude {
    pub use crate::{
        MMXFile, MorphNet, PatchQuilt, MorphNetAnalyzer, SpatialAwareness,
        Result, MorphNetError, BodyPlan, GeometricTemplate
    };
    pub use ndarray::{Array, Array1, Array2, Array3, ArrayD};
    pub use nalgebra::{Point3, Vector3, Matrix3, Matrix4};
}
