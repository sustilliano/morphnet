// src/morphnet/templates.rs
//! Template Factory and Management

use super::*;

/// Template factory for creating standard body plans
pub struct TemplateFactory;

impl TemplateFactory {
    /// Create a quadruped template
    pub fn create_quadruped() -> GeometricTemplate {
        let mut template = GeometricTemplate::new("quadruped".to_string(), BodyPlan::Quadruped);
        
        // Add keypoints
        template.add_keypoint(Keypoint {
            name: "nose".to_string(),
            position: Point3::new(0.0, 0.0, 1.0),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Head,
        });
        
        template.add_keypoint(Keypoint {
            name: "head".to_string(),
            position: Point3::new(0.0, 0.0, 0.8),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Head,
        });
        
        template.add_keypoint(Keypoint {
            name: "neck".to_string(),
            position: Point3::new(0.0, 0.0, 0.6),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Neck,
        });
        
        template.add_keypoint(Keypoint {
            name: "shoulder_left".to_string(),
            position: Point3::new(-0.3, 0.0, 0.5),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Joint,
        });
        
        template.add_keypoint(Keypoint {
            name: "shoulder_right".to_string(),
            position: Point3::new(0.3, 0.0, 0.5),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Joint,
        });
        
        template.add_keypoint(Keypoint {
            name: "spine".to_string(),
            position: Point3::new(0.0, 0.0, 0.0),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Torso,
        });
        
        template.add_keypoint(Keypoint {
            name: "hip_left".to_string(),
            position: Point3::new(-0.25, 0.0, -0.3),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Joint,
        });
        
        template.add_keypoint(Keypoint {
            name: "hip_right".to_string(),
            position: Point3::new(0.25, 0.0, -0.3),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Joint,
        });
        
        template.add_keypoint(Keypoint {
            name: "tail".to_string(),
            position: Point3::new(0.0, 0.0, -0.7),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Tail,
        });
        
        // Add paws
        for side in ["left", "right"] {
            for position in ["front", "back"] {
                let x = if side == "left" { -0.3 } else { 0.3 };
                let z = if position == "front" { 0.2 } else { -0.5 };
                
                template.add_keypoint(Keypoint {
                    name: format!("paw_{}_{}", position, side),
                    position: Point3::new(x, -0.6, z),
                    confidence: 1.0,
                    visibility: true,
                    anatomical_type: AnatomicalType::Extremity,
                });
            }
        }
        
        // Add connections
        let connections = vec![
            ("head", "nose"),
            ("neck", "head"),
            ("shoulder_left", "neck"),
            ("shoulder_right", "neck"),
            ("spine", "neck"),
            ("hip_left", "spine"),
            ("hip_right", "spine"),
            ("tail", "spine"),
            ("shoulder_left", "paw_front_left"),
            ("shoulder_right", "paw_front_right"),
            ("hip_left", "paw_back_left"),
            ("hip_right", "paw_back_right"),
        ];
        
        for (from, to) in connections {
            template.add_connection(Connection {
                from: from.to_string(),
                to: to.to_string(),
                connection_type: ConnectionType::Bone,
                strength: 1.0,
            }).unwrap();
        }
        
        template.symmetry = SymmetryType::Bilateral;
        template
    }
    
    /// Create a bird template
    pub fn create_bird() -> GeometricTemplate {
        let mut template = GeometricTemplate::new("bird".to_string(), BodyPlan::Bird);
        
        // Add keypoints
        template.add_keypoint(Keypoint {
            name: "beak".to_string(),
            position: Point3::new(0.0, 0.0, 0.8),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Head,
        });
        
        template.add_keypoint(Keypoint {
            name: "head".to_string(),
            position: Point3::new(0.0, 0.0, 0.6),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Head,
        });
        
        template.add_keypoint(Keypoint {
            name: "neck".to_string(),
            position: Point3::new(0.0, 0.0, 0.3),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Neck,
        });
        
        template.add_keypoint(Keypoint {
            name: "breast".to_string(),
            position: Point3::new(0.0, 0.0, 0.0),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Torso,
        });
        
        template.add_keypoint(Keypoint {
            name: "wing_left".to_string(),
            position: Point3::new(-0.5, 0.0, 0.0),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Wing,
        });
        
        template.add_keypoint(Keypoint {
            name: "wing_right".to_string(),
            position: Point3::new(0.5, 0.0, 0.0),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Wing,
        });
        
        template.add_keypoint(Keypoint {
            name: "body".to_string(),
            position: Point3::new(0.0, 0.0, -0.3),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Torso,
        });
        
        template.add_keypoint(Keypoint {
            name: "tail".to_string(),
            position: Point3::new(0.0, 0.0, -0.8),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Tail,
        });
        
        template.add_keypoint(Keypoint {
            name: "leg_left".to_string(),
            position: Point3::new(-0.15, -0.4, -0.2),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Limb,
        });
        
        template.add_keypoint(Keypoint {
            name: "leg_right".to_string(),
            position: Point3::new(0.15, -0.4, -0.2),
            confidence: 1.0,
            visibility: true,
            anatomical_type: AnatomicalType::Limb,
        });
        
        // Add connections
        let connections = vec![
            ("head", "beak"),
            ("neck", "head"),
            ("breast", "neck"),
            ("wing_left", "breast"),
            ("wing_right", "breast"),
            ("body", "breast"),
            ("tail", "body"),
            ("leg_left", "body"),
            ("leg_right", "body"),
        ];
        
        for (from, to) in connections {
            template.add_connection(Connection {
                from: from.to_string(),
                to: to.to_string(),
                connection_type: ConnectionType::Bone,
                strength: 1.0,
            }).unwrap();
        }
        
        template.symmetry = SymmetryType::Bilateral;
        template
    }
    
    /// Get all available templates
    pub fn get_all_templates() -> HashMap<BodyPlan, GeometricTemplate> {
        let mut templates = HashMap::new();
        templates.insert(BodyPlan::Quadruped, Self::create_quadruped());
        templates.insert(BodyPlan::Bird, Self::create_bird());
        templates
    }
    
    /// Create a template for a specific body plan
    pub fn create_template(body_plan: &BodyPlan) -> Option<GeometricTemplate> {
        match body_plan {
            BodyPlan::Quadruped => Some(Self::create_quadruped()),
            BodyPlan::Bird => Some(Self::create_bird()),
            _ => None, // Not implemented yet
        }
    }
}