//! Spatial awareness and monitoring utilities

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialAwareness;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialConfig {
    pub prediction_horizon: f32,
    pub alert_threshold: f32,
    pub update_frequency: f32,
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self { prediction_horizon: 1.0, alert_threshold: 0.5, update_frequency: 1.0 }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    Low,
    Moderate,
    High,
    Critical,
    Extreme,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialEvent {
    pub severity: EventSeverity,
    pub risk: RiskLevel,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountabilityPredictor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralMonitor;
