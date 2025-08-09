//! MorphNet-GTL: Geometric Template Learning and Spatial Intelligence Framework
//!
//! A next-generation deep learning system for structural understanding of morphology
//! and spatial pre-awareness, combining neural classification, geometric template
//! inference, and patch-based mesh refinement.

pub mod analysis;
pub mod mmx;
pub mod morphnet;
pub mod patch_quilt;
#[cfg(feature = "thingino")]
pub mod quilt;
pub mod spatial;

// Re-export main types for convenience
pub use analysis::{EmbeddingMethod, MorphNetAnalyzer, PhylogeneticTree};
pub use mmx::{
    ChunkType, ExtendedBodyPlan, GeometricParameters, GeometricTemplateData, MMXBuilder, MMXError,
    MMXFile, MMXMode, TensorData,
};
pub use morphnet::{
    BodyPlan, ClassificationResult, Connection, GeometricTemplate, Keypoint, MorphNet,
    MorphNetBuilder, MorphNetConfig, TemplateFactory,
};
pub use patch_quilt::{Patch, PatchQuilt, RefinementConfig};
pub use spatial::{SpatialAwareness, SpatialConfig, SpatialEvent};

/// Core error types for the framework
#[derive(thiserror::Error, Debug)]
pub enum MorphNetError {
    #[error("MMX format error: {0}")]
    MMX(#[from] MMXError),

    #[error("MorphNet error: {0}")]
    MorphNet(#[from] morphnet::MorphNetError),

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
    pub use crate::morphnet::{train, train_logistic};
    pub use crate::{
        BodyPlan, ExtendedBodyPlan, GeometricParameters, GeometricTemplate, GeometricTemplateData,
        MMXBuilder, MMXFile, MMXMode, MorphNet, MorphNetAnalyzer, MorphNetBuilder, MorphNetError,
        PatchQuilt, Result, SpatialAwareness, SpatialConfig, TemplateFactory,
    };
    pub use nalgebra::{Matrix3, Matrix4, Point3, Vector3};
    pub use ndarray::{Array, Array1, Array2, Array3, ArrayD};
}
