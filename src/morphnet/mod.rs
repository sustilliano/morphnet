//! MorphNet: Geometric Template Learning for Structural Understanding

pub mod model;
pub mod templates;
pub mod classification;
pub mod training;

pub use model::*;
pub use templates::*;
pub use classification::*;
pub use training::*;

use ndarray::{Array1, Array2, Array3};
use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};
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
    #[error("Model error: {0}")]
    Model(String),
    #[error("Data processing error: {0}")]
    DataProcessing(String),
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result type for MorphNet operations
pub type Result<T> = std::result::Result<T, MorphNetError>;

/// Supported body plan types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum BodyPlan {
    Quadruped,
    Biped,
    Bird,
    Snake,
    Fish,
    Insect,
    Spider,
    Custom(String),
}

impl std::fmt::Display for BodyPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BodyPlan::Quadruped => write!(f, "quadruped"),
            BodyPlan::Biped => write!(f, "biped"),
            BodyPlan::Bird => write!(f, "bird"),
            BodyPlan::Snake => write!(f, "snake"),
            BodyPlan::Fish => write!(f, "fish"),
            BodyPlan::Insect => write!(f, "insect"),
            BodyPlan::Spider => write!(f, "spider"),
            BodyPlan::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Types of anatomical landmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnatomicalType {
    Head,
    Neck,
    Torso,
    Limb,
    Joint,
    Extremity,
    Tail,
    Wing,
    Fin,
    Custom(String),
}

/// Connection types between keypoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionType {
    Bone,
    Joint,
    Muscle,
    Surface,
}

/// Geometric keypoint in 3D space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Keypoint {
    pub name: String,
    pub position: Point3<f32>,
    pub confidence: f32,
    pub visibility: bool,
    pub anatomical_type: AnatomicalType,
}

/// Connection between keypoints forming structural relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub from: String,
    pub to: String,
    pub connection_type: ConnectionType,
    pub strength: f32,
}

/// Structural constraints for template validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StructuralConstraint {
    Distance {
        from: String,
        to: String,
        min_distance: f32,
        max_distance: f32,
    },
    Angle {
        vertex: String,
        arm1: String,
        arm2: String,
        min_angle: f32,
        max_angle: f32,
    },
    Ordering {
        points: Vec<String>,
        axis: Vector3<f32>,
    },
}

/// Symmetry properties of templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymmetryType {
    None,
    Bilateral,
    Radial { n_fold: u32 },
    Helical,
}

/// Learnable parameters for template adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateParameters {
    pub scale_factors: HashMap<String, f32>,
    pub proportions: HashMap<String, f32>,
    pub flexibility: HashMap<String, f32>,
    pub adaptations: HashMap<String, f32>,
}

impl Default for TemplateParameters {
    fn default() -> Self {
        Self {
            scale_factors: HashMap::new(),
            proportions: HashMap::new(),
            flexibility: HashMap::new(),
            adaptations: HashMap::new(),
        }
    }
}

/// Core geometric template defining a body plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricTemplate {
    pub name: String,
    pub body_plan: BodyPlan,
    pub keypoints: HashMap<String, Keypoint>,
    pub connections: Vec<Connection>,
    pub constraints: Vec<StructuralConstraint>,
    pub symmetry: SymmetryType,
    pub scale_invariant: bool,
    pub parameters: TemplateParameters,
}

impl GeometricTemplate {
    /// Create a new geometric template
    pub fn new(name: String, body_plan: BodyPlan) -> Self {
        Self {
            name,
            body_plan,
            keypoints: HashMap::new(),
            connections: Vec::new(),
            constraints: Vec::new(),
            symmetry: SymmetryType::None,
            scale_invariant: true,
            parameters: TemplateParameters::default(),
        }
    }

    /// Add a keypoint to the template
    pub fn add_keypoint(&mut self, keypoint: Keypoint) {
        self.keypoints.insert(keypoint.name.clone(), keypoint);
    }

    /// Add a connection between keypoints
    pub fn add_connection(&mut self, connection: Connection) -> Result<()> {
        if !self.keypoints.contains_key(&connection.from) {
            return Err(MorphNetError::Template(format!(
                "Keypoint '{}' not found",
                connection.from
            )));
        }
        if !self.keypoints.contains_key(&connection.to) {
            return Err(MorphNetError::Template(format!(
                "Keypoint '{}' not found",
                connection.to
            )));
        }

        self.connections.push(connection);
        Ok(())
    }

    /// Validate template structure against constraints
    pub fn validate(&self) -> Result<()> {
        for constraint in &self.constraints {
            match constraint {
                StructuralConstraint::Distance { from, to, min_distance, max_distance } => {
                    let p1 = self.keypoints.get(from).ok_or_else(|| {
                        MorphNetError::Template(format!("Keypoint '{}' not found", from))
                    })?;
                    let p2 = self.keypoints.get(to).ok_or_else(|| {
                        MorphNetError::Template(format!("Keypoint '{}' not found", to))
                    })?;

                    let distance = (p1.position - p2.position).norm();
                    if distance < *min_distance || distance > *max_distance {
                        return Err(MorphNetError::Validation(format!(
                            "Distance constraint violated between {} and {}: {} not in [{}, {}]",
                            from, to, distance, min_distance, max_distance
                        )));
                    }
                }
                StructuralConstraint::Angle { vertex, arm1, arm2, min_angle, max_angle } => {
                    let v = self.keypoints.get(vertex).ok_or_else(|| {
                        MorphNetError::Template(format!("Vertex '{}' not found", vertex))
                    })?;
                    let a1 = self.keypoints.get(arm1).ok_or_else(|| {
                        MorphNetError::Template(format!("Arm1 '{}' not found", arm1))
                    })?;
                    let a2 = self.keypoints.get(arm2).ok_or_else(|| {
                        MorphNetError::Template(format!("Arm2 '{}' not found", arm2))
                    })?;

                    let vec1 = (a1.position - v.position).normalize();
                    let vec2 = (a2.position - v.position).normalize();
                    let angle = vec1.dot(&vec2).acos();

                    if angle < *min_angle || angle > *max_angle {
                        return Err(MorphNetError::Validation(format!(
                            "Angle constraint violated at {}: {} not in [{}, {}]",
                            vertex, angle, min_angle, max_angle
                        )));
                    }
                }
                StructuralConstraint::Ordering { points, axis } => {
                    let mut projections: Vec<(String, f32)> = points
                        .iter()
                        .map(|name| {
                            let point = self.keypoints.get(name).unwrap();
                            let projection = point.position.coords.dot(axis);
                            (name.clone(), projection)
                        })
                        .collect();

                    projections.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                    let ordered_names: Vec<String> =
                        projections.into_iter().map(|(name, _)| name).collect();
                    if ordered_names != *points {
                        return Err(MorphNetError::Validation(format!(
                            "Ordering constraint violated: expected {:?}, got {:?}",
                            points, ordered_names
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Apply learned parameters to adapt template
    pub fn apply_parameters(&mut self, params: &TemplateParameters) {
        self.parameters = params.clone();

        for (name, keypoint) in &mut self.keypoints {
            if let Some(&scale) = params.scale_factors.get(name) {
                keypoint.position = keypoint.position * scale;
            }
        }
    }

    /// Generate parameter vector for ML training
    pub fn to_parameter_vector(&self) -> Array1<f32> {
        let mut params = Vec::new();

        for keypoint in self.keypoints.values() {
            params.extend_from_slice(keypoint.position.coords.as_slice());
        }

        for value in self.parameters.scale_factors.values() {
            params.push(*value);
        }
        for value in self.parameters.proportions.values() {
            params.push(*value);
        }
        for value in self.parameters.flexibility.values() {
            params.push(*value);
        }

        Array1::from_vec(params)
    }

    /// Reconstruct template from parameter vector
    pub fn from_parameter_vector(&mut self, params: &Array1<f32>) -> Result<()> {
        let mut idx = 0;

        for keypoint in self.keypoints.values_mut() {
            if idx + 3 > params.len() {
                return Err(MorphNetError::DataProcessing(
                    "Parameter vector too short".to_string(),
                ));
            }
            keypoint.position = Point3::new(params[idx], params[idx + 1], params[idx + 2]);
            idx += 3;
        }

        Ok(())
    }

    /// Get the number of keypoints
    pub fn keypoint_count(&self) -> usize {
        self.keypoints.len()
    }

    /// Get the number of connections
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Check if template has a specific keypoint
    pub fn has_keypoint(&self, name: &str) -> bool {
        self.keypoints.contains_key(name)
    }

    /// Get keypoint by name
    pub fn get_keypoint(&self, name: &str) -> Option<&Keypoint> {
        self.keypoints.get(name)
    }

    /// Get all keypoint names
    pub fn keypoint_names(&self) -> Vec<String> {
        self.keypoints.keys().cloned().collect()
    }

    /// Calculate bounding box of all keypoints
    pub fn bounding_box(&self) -> Option<(Point3<f32>, Point3<f32>)> {
        if self.keypoints.is_empty() {
            return None;
        }

        let positions: Vec<Point3<f32>> = self.keypoints.values().map(|k| k.position).collect();

        let min_x = positions.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
        let min_y = positions.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let min_z = positions.iter().map(|p| p.z).fold(f32::INFINITY, f32::min);

        let max_x = positions.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
        let max_y = positions.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
        let max_z = positions.iter().map(|p| p.z).fold(f32::NEG_INFINITY, f32::max);

        Some((
            Point3::new(min_x, min_y, min_z),
            Point3::new(max_x, max_y, max_z),
        ))
    }

    /// Calculate centroid of all keypoints
    pub fn centroid(&self) -> Option<Point3<f32>> {
        if self.keypoints.is_empty() {
            return None;
        }

        let sum = self
            .keypoints
            .values()
            .map(|k| k.position.coords)
            .fold(Vector3::zeros(), |acc, pos| acc + pos);

        let count = self.keypoints.len() as f32;
        Some(Point3::from(sum / count))
    }
}

/// Classification results from MorphNet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub species_probs: HashMap<String, f32>,
    pub body_plan_probs: HashMap<BodyPlan, f32>,
    pub predicted_species: String,
    pub predicted_body_plan: BodyPlan,
    pub species_confidence: f32,
    pub body_plan_confidence: f32,
    pub template_parameters: Array1<f32>,
    pub template: GeometricTemplate,
}

/// Configuration for MorphNet model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphNetConfig {
    pub num_species: usize,
    pub feature_dim: usize,
    pub template_param_dim: usize,
    pub learning_rate: f64,
    pub dropout_rate: f64,
    pub input_size: (usize, usize),
}

impl Default for MorphNetConfig {
    fn default() -> Self {
        Self {
            num_species: 100,
            feature_dim: 2048,
            template_param_dim: 256,
            learning_rate: 1e-4,
            dropout_rate: 0.5,
            input_size: (224, 224),
        }
    }
}

/// Training metrics for MorphNet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub species_loss: f32,
    pub body_plan_loss: f32,
    pub template_loss: f32,
    pub total_loss: f32,
    pub species_accuracy: f32,
    pub body_plan_accuracy: f32,
    pub template_error: f32,
}

/// Validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub metrics: TrainingMetrics,
    pub confusion_matrix: HashMap<String, HashMap<String, usize>>,
    pub per_class_accuracy: HashMap<String, f32>,
    pub template_alignment_error: f32,
}


