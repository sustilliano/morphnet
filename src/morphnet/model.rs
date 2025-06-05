use super::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use ndarray::Array3;
use crate::TensorData;
use linfa_logistic::FittedLogisticRegression;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu,
}

impl Default for Device {
    fn default() -> Self { Device::Cpu }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MorphNetConfig {
    pub num_species: usize,
    pub device: Device,
    /// Threshold used by the simple brightness based classifier
    pub brightness_threshold: f32,
}

pub struct MorphNetBuilder {
    config: MorphNetConfig,
}

impl MorphNetBuilder {
    pub fn new() -> Self { Self { config: MorphNetConfig::default() } }
    pub fn with_device(mut self, device: Device) -> Self { self.config.device = device; self }
    pub fn with_num_species(mut self, n: usize) -> Self { self.config.num_species = n; self }
    pub fn with_brightness_threshold(mut self, t: f32) -> Self { self.config.brightness_threshold = t; self }
    pub fn build(self) -> Result<MorphNet> {
        Ok(MorphNet {
            body_plan: BodyPlanModel { templates: Vec::new() },
            brightness_threshold: self.config.brightness_threshold,
            logistic_model: None,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Keypoint {
    pub name: String,
    pub position: Point3<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BodyPlan {
    Quadruped,
    Bird,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricTemplate {
    pub body_plan: BodyPlan,
    pub keypoints: HashMap<String, Keypoint>,
    pub connections: Vec<Connection>,
}

impl GeometricTemplate {
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.keypoints.is_empty() { Err("no keypoints".into()) } else { Ok(()) }
    }

    pub fn to_parameter_vector(&self) -> Vec<f32> {
        self.keypoints
            .values()
            .flat_map(|k| [k.position.x, k.position.y, k.position.z])
            .collect()
    }
}

pub struct TemplateFactory;

impl TemplateFactory {
    pub fn create_quadruped() -> GeometricTemplate {
        let mut keypoints = HashMap::new();
        keypoints.insert(
            "hip".to_string(),
            Keypoint { name: "hip".to_string(), position: Point3::new(0.0, 0.0, 0.0) },
        );
        keypoints.insert(
            "shoulder".to_string(),
            Keypoint { name: "shoulder".to_string(), position: Point3::new(1.0, 0.0, 0.0) },
        );
        let connections = vec![Connection { from: "hip".to_string(), to: "shoulder".to_string() }];
        GeometricTemplate { body_plan: BodyPlan::Quadruped, keypoints, connections }
    }

    pub fn create_bird() -> GeometricTemplate {
        let mut keypoints = HashMap::new();
        keypoints.insert(
            "body".to_string(),
            Keypoint { name: "body".to_string(), position: Point3::new(0.0, 0.0, 0.0) },
        );
        keypoints.insert(
            "wing".to_string(),
            Keypoint { name: "wing".to_string(), position: Point3::new(0.5, 0.0, 0.0) },
        );
        let connections = vec![Connection { from: "body".to_string(), to: "wing".to_string() }];
        GeometricTemplate { body_plan: BodyPlan::Bird, keypoints, connections }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyPlanModel {
    pub templates: Vec<GeometricTemplate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphNet {
    pub body_plan: BodyPlanModel,
    /// Threshold used for basic brightness classification
    pub brightness_threshold: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub logistic_model: Option<FittedLogisticRegression<f64, usize>>,
}

impl MorphNet {
    pub fn new() -> Self {
        Self {
            body_plan: BodyPlanModel { templates: Vec::new() },
            brightness_threshold: 0.5,
            logistic_model: None,
        }
    }

    pub fn classify(&self, input: &Array3<f32>) -> Result<ClassificationResult> {
        let tensor = TensorData::new(input.clone().into_dyn());
        classify(self, &tensor)
    }
}
