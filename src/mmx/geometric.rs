use serde::{Serialize, Deserialize};
use crate::mmx::TensorData;
use ndarray::{ArrayD, IxDyn};

/// Extended body plans for geometric templates
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExtendedBodyPlan {
    QuadrupedSmall,
    QuadrupedMedium,
    QuadrupedLarge,
    BipedFlying,
    BipedGround,
}

/// Parameters describing a geometric template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricParameters {
    pub body_length: f32,
    pub body_width: f32,
    pub body_height: f32,
    pub leg_length: f32,
    pub leg_thickness: f32,
    pub num_legs: u32,
    pub head_length: f32,
    pub head_width: f32,
    pub neck_length: f32,
    pub tail_length: f32,
    pub wing_span: f32,
    pub stride_length: f32,
    pub turning_radius: f32,
    pub jump_height: f32,
    pub max_speed: f32,
}

impl GeometricParameters {
    /// Convert parameters into a tensor representation
    pub fn to_tensor(&self) -> TensorData {
        let values = vec![
            self.body_length,
            self.body_width,
            self.body_height,
            self.leg_length,
            self.leg_thickness,
            self.num_legs as f32,
            self.head_length,
            self.head_width,
            self.neck_length,
            self.tail_length,
            self.wing_span,
            self.stride_length,
            self.turning_radius,
            self.jump_height,
            self.max_speed,
        ];
        let array = ArrayD::from_shape_vec(IxDyn(&[values.len()]), values).unwrap();
        TensorData::new(array)
    }
}

/// Combined data used for MMX storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricTemplateData {
    pub parameters: GeometricParameters,
    pub body_plan: ExtendedBodyPlan,
}

impl GeometricTemplateData {
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("failed to serialize GeometricTemplateData")
    }

    pub fn from_bytes(bytes: &[u8]) -> bincode::Result<Self> {
        bincode::deserialize(bytes)
    }
}

