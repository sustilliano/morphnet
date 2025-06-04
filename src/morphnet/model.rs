use super::*;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricTemplate {
    pub id: usize,
    pub points: Vec<Point3<f32>>, // simple representation
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyPlan {
    pub templates: Vec<GeometricTemplate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphNet {
    pub body_plan: BodyPlan,
}

impl MorphNet {
    pub fn new() -> Self {
        Self {
            body_plan: BodyPlan { templates: Vec::new() },
        }
    }
}
