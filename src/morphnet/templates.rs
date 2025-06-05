use super::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateLibrary {
    pub templates: HashMap<String, GeometricTemplate>,
}

impl TemplateLibrary {
    pub fn new() -> Self {
        Self { templates: HashMap::new() }
    }
}
