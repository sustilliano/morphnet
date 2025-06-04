//! Classification utilities and metrics

use super::*;

/// Classification metrics
#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub confusion_matrix: Array2<usize>,
}

impl ClassificationMetrics {
    /// Calculate metrics from predictions and ground truth
    pub fn calculate(
        predictions: &[usize],
        ground_truth: &[usize],
        num_classes: usize,
    ) -> Self {
        let mut confusion_matrix = Array2::zeros((num_classes, num_classes));

        for (&pred, &gt) in predictions.iter().zip(ground_truth.iter()) {
            if pred < num_classes && gt < num_classes {
                confusion_matrix[[gt, pred]] += 1;
            }
        }

        let matched_len = predictions.len().min(ground_truth.len());

        if matched_len == 0 {
            return Self {
                accuracy: 0.0,
                precision: 0.0,
                recall: 0.0,
                f1_score: 0.0,
                confusion_matrix,
            };
        }

        let correct = predictions
            .iter()
            .zip(ground_truth.iter())
            .filter(|(&pred, &gt)| pred == gt)
            .count();
        let accuracy = correct as f32 / matched_len as f32;

        let mut total_precision = 0.0;
        let mut total_recall = 0.0;
        let mut valid_classes = 0;

        for class in 0..num_classes {
            let tp = confusion_matrix[[class, class]] as f32;
            let fp: f32 = (0..num_classes)
                .filter(|&i| i != class)
                .map(|i| confusion_matrix[[i, class]] as f32)
                .sum();
            let fn_: f32 = (0..num_classes)
                .filter(|&i| i != class)
                .map(|i| confusion_matrix[[class, i]] as f32)
                .sum();

            if tp + fp > 0.0 {
                total_precision += tp / (tp + fp);
                valid_classes += 1;
            }
            if tp + fn_ > 0.0 {
                total_recall += tp / (tp + fn_);
            }
        }

        let precision = if valid_classes > 0 {
            total_precision / valid_classes as f32
        } else {
            0.0
        };
        let recall = if valid_classes > 0 {
            total_recall / valid_classes as f32
        } else {
            0.0
        };
        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Self {
            accuracy,
            precision,
            recall,
            f1_score,
            confusion_matrix,
        }
    }
}

/// Species classifier utility
pub struct SpeciesClassifier {
    species_map: HashMap<String, usize>,
    reverse_map: HashMap<usize, String>,
}

impl SpeciesClassifier {
    pub fn new(species_list: Vec<String>) -> Self {
        let mut species_map = HashMap::new();
        let mut reverse_map = HashMap::new();

        for (idx, species) in species_list.into_iter().enumerate() {
            species_map.insert(species.clone(), idx);
            reverse_map.insert(idx, species);
        }

        Self {
            species_map,
            reverse_map,
        }
    }

    pub fn species_to_id(&self, species: &str) -> Option<usize> {
        self.species_map.get(species).copied()
    }

    pub fn id_to_species(&self, id: usize) -> Option<&String> {
        self.reverse_map.get(&id)
    }

    pub fn num_species(&self) -> usize {
        self.species_map.len()
    }
}

/// Body plan classifier utility
pub struct BodyPlanClassifier;

impl BodyPlanClassifier {
    pub fn body_plan_to_id(body_plan: &BodyPlan) -> usize {
        match body_plan {
            BodyPlan::Quadruped => 0,
            BodyPlan::Biped => 1,
            BodyPlan::Bird => 2,
            BodyPlan::Snake => 3,
            BodyPlan::Fish => 4,
            BodyPlan::Insect => 5,
            BodyPlan::Spider => 6,
            BodyPlan::Custom(_) => 7,
        }
    }

    pub fn id_to_body_plan(id: usize) -> Option<BodyPlan> {
        match id {
            0 => Some(BodyPlan::Quadruped),
            1 => Some(BodyPlan::Biped),
            2 => Some(BodyPlan::Bird),
            3 => Some(BodyPlan::Snake),
            4 => Some(BodyPlan::Fish),
            5 => Some(BodyPlan::Insect),
            6 => Some(BodyPlan::Spider),
            _ => None,
        }
    }

    pub fn num_body_plans() -> usize {
        8
    }
}

