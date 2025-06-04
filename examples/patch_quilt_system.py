import os
import json
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from scipy.spatial import KDTree
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== Data Structures ==================

class BodyPlan(Enum):
    QUADRUPED_SMALL = "quadruped_small"
    QUADRUPED_MEDIUM = "quadruped_medium"
    QUADRUPED_LARGE = "quadruped_large"
    BIPED_FLYING = "biped_flying"
    BIPED_GROUND = "biped_ground"
    ELONGATED = "elongated"

@dataclass
class GeometricParameters:
    """Parameters defining 3D template geometry"""
    body_length: float
    body_width: float
    body_height: float
    leg_length: float = 0.0
    leg_thickness: float = 0.0
    num_legs: int = 0
    head_length: float = 0.0
    head_width: float = 0.0
    neck_length: float = 0.0
    tail_length: float = 0.0
    wing_span: float = 0.0
    stride_length: float = 0.0
    turning_radius: float = 0.0
    jump_height: float = 0.0
    max_speed: float = 0.0

@dataclass
class PatchData:
    """Enhanced patch data with adaptive sizing and uncertainty"""
    patch_id: str
    surface_type: str
    texture_features: np.ndarray
    depth_estimate: float
    normal_vector: np.ndarray
    confidence: float
    pixel_coords: Tuple[int, int]
    mesh_coords: Tuple[float, float, float]
    view_angle: float
    lighting_conditions: Dict
    timestamp: float
    patch_size: int = 64
    uncertainty: float = 0.0
    view_confidence: float = 1.0

@dataclass
class MeshPatch:
    """3D mesh patch with reality data and uncertainty"""
    patch_id: str
    vertices: np.ndarray
    faces: np.ndarray
    texture_map: np.ndarray
    normal_map: np.ndarray
    reality_patches: List[PatchData]
    confidence_map: np.ndarray
    last_updated: float
    uncertainty: 'MeshUncertainty' = None

@dataclass
class MeshUncertainty:
    """Uncertainty quantification for mesh regions"""
    positional_variance: np.ndarray
    normal_variance: np.ndarray
    texture_entropy: float
    observation_count: int

@dataclass
class HierarchicalPatch:
    """Multi-resolution patch structure"""
    coarse: List[PatchData] = field(default_factory=list)
    medium: List[PatchData] = field(default_factory=list)
    fine: List[PatchData] = field(default_factory=list)

@dataclass
class ViewSuggestion:
    """Suggested view for active learning"""
    region_id: str
    azimuth: float
    elevation: float
    priority: float
    uncertainty_score: float

# ================== Neural Networks ==================

class PatchExtractor(nn.Module):
    """Enhanced patch extractor with adaptive features"""

    def __init__(self, patch_sizes=[32, 64, 128]):
        super().__init__()
        self.extractors = nn.ModuleDict()
        for size in patch_sizes:
            self.extractors[str(size)] = self._create_extractor(size)
        self.fusion = nn.Linear(512 * len(patch_sizes), 512)
        self.surface_classifier = nn.Linear(512, 4)
        self.depth_estimator = nn.Linear(512, 1)
        self.normal_predictor = nn.Linear(512, 3)
        self.confidence_estimator = nn.Linear(512, 1)
        self.uncertainty_estimator = nn.Linear(512, 1)

    def _create_extractor(self, _size):
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(256 * 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, patches_dict):
        features_list = []
        for size, patches in patches_dict.items():
            if size in self.extractors:
                features_list.append(self.extractors[size](patches))
        if len(features_list) > 1:
            features = self.fusion(torch.cat(features_list, dim=1))
        else:
            features = features_list[0]
        return {
            'features': features,
            'surface_type': self.surface_classifier(features),
            'depth': torch.sigmoid(self.depth_estimator(features)),
            'normals': F.normalize(self.normal_predictor(features), dim=1),
            'confidence': torch.sigmoid(self.confidence_estimator(features)),
            'uncertainty': torch.sigmoid(self.uncertainty_estimator(features))
        }

class BodyPlanClassifier(nn.Module):
    """Neural network to classify animal body plans"""

    def __init__(self, input_features=2048, num_body_plans=6):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_body_plans)
        )

    def forward(self, x):
        return self.classifier(x)

class GeometryPredictor(nn.Module):
    """Neural network to predict geometric parameters"""

    def __init__(self, input_features=2048, num_params=8):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_params),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.predictor(x)

# ================== Template Library ==================

class AnimalTemplateLibrary:
    """Enhanced template library with uncertainty tracking"""

    def __init__(self):
        self.templates = self._initialize_templates()
        self.template_uncertainty = defaultdict(lambda: 0.5)

    def _initialize_templates(self) -> Dict[BodyPlan, GeometricParameters]:
        return {
            BodyPlan.QUADRUPED_SMALL: GeometricParameters(
                body_length=1.0, body_width=0.3, body_height=0.4,
                leg_length=0.4, leg_thickness=0.08, num_legs=4,
                head_length=0.25, head_width=0.2, neck_length=0.15,
                tail_length=0.8, stride_length=0.6, turning_radius=0.5,
                jump_height=1.0, max_speed=8.0
            ),
            BodyPlan.QUADRUPED_MEDIUM: GeometricParameters(
                body_length=1.2, body_width=0.4, body_height=0.5,
                leg_length=0.6, leg_thickness=0.12, num_legs=4,
                head_length=0.3, head_width=0.25, neck_length=0.2,
                tail_length=0.9, stride_length=1.0, turning_radius=1.0,
                jump_height=0.8, max_speed=12.0
            ),
            BodyPlan.QUADRUPED_LARGE: GeometricParameters(
                body_length=2.0, body_width=0.8, body_height=1.2,
                leg_length=1.0, leg_thickness=0.2, num_legs=4,
                head_length=0.5, head_width=0.4, neck_length=0.6,
                tail_length=0.7, stride_length=1.8, turning_radius=2.5,
                jump_height=0.3, max_speed=15.0
            ),
            BodyPlan.BIPED_FLYING: GeometricParameters(
                body_length=0.4, body_width=0.15, body_height=0.3,
                leg_length=0.2, leg_thickness=0.03, num_legs=2,
                head_length=0.12, head_width=0.1, neck_length=0.08,
                wing_span=1.2, stride_length=0.3, turning_radius=0.2,
                jump_height=0.5, max_speed=20.0
            ),
            BodyPlan.BIPED_GROUND: GeometricParameters(
                body_length=0.5, body_width=0.2, body_height=0.4,
                leg_length=0.3, leg_thickness=0.05, num_legs=2,
                head_length=0.15, head_width=0.12, neck_length=0.1,
                wing_span=0.6, stride_length=0.4, turning_radius=0.3,
                jump_height=0.3, max_speed=5.0
            )
        }

    def get_template(self, body_plan: BodyPlan) -> GeometricParameters:
        return self.templates[body_plan]

    def customize_template(self, body_plan: BodyPlan, scale_factors: Dict[str, float]) -> GeometricParameters:
        base = self.get_template(body_plan)
        return GeometricParameters(
            body_length=base.body_length * scale_factors.get('body_length_scale', 1.0),
            body_width=base.body_width * scale_factors.get('body_width_scale', 1.0),
            body_height=base.body_height * scale_factors.get('body_height_scale', 1.0),
            leg_length=base.leg_length * scale_factors.get('leg_scale', 1.0),
            leg_thickness=base.leg_thickness * scale_factors.get('leg_scale', 1.0),
            num_legs=base.num_legs,
            head_length=base.head_length * scale_factors.get('head_scale', 1.0),
            head_width=base.head_width * scale_factors.get('head_scale', 1.0),
            neck_length=base.neck_length * scale_factors.get('neck_scale', 1.0),
            tail_length=base.tail_length * scale_factors.get('tail_scale', 1.0),
            wing_span=base.wing_span * scale_factors.get('wing_scale', 1.0),
            stride_length=base.stride_length * scale_factors.get('leg_scale', 1.0),
            turning_radius=base.turning_radius * scale_factors.get('body_length_scale', 1.0),
            jump_height=base.jump_height * scale_factors.get('leg_scale', 1.0),
            max_speed=base.max_speed
        )

# ================== Material Processors ==================

class MaterialProcessor:
    def process_patch(self, patch: PatchData) -> PatchData:
        raise NotImplementedError

class FurProcessor(MaterialProcessor):
    def process_patch(self, patch: PatchData) -> PatchData:
        patch.normal_vector = self._estimate_fur_direction(patch)
        return patch
    def _estimate_fur_direction(self, patch):
        base_normal = patch.normal_vector
        perturbation = np.random.normal(0, 0.1, 3)
        return base_normal + perturbation

class SkinProcessor(MaterialProcessor):
    def process_patch(self, patch: PatchData) -> PatchData:
        if patch.lighting_conditions.get('brightness', 100) < 50:
            patch.confidence *= 0.8
        return patch

class FeatherProcessor(MaterialProcessor):
    def process_patch(self, patch: PatchData) -> PatchData:
        patch.uncertainty *= 1.2
        return patch

class ScaleProcessor(MaterialProcessor):
    def process_patch(self, patch: PatchData) -> PatchData:
        patch.confidence *= 1.1
        return patch

# ================== Enhanced Patch Quilt System ==================

class PatchQuiltSystem:
    def __init__(self, device='cpu'):
        self.device = device
        self.patch_extractor = PatchExtractor().to(device)
        self.body_plan_classifier = BodyPlanClassifier().to(device)
        self.geometry_predictor = GeometryPredictor().to(device)
        self.template_library = AnimalTemplateLibrary()
        self.active_patches: Dict[str, List[PatchData]] = {}
        self.mesh_patches: Dict[str, MeshPatch] = {}
        self.patch_history: Dict[str, List[PatchData]] = {}
        self.hierarchical_patches: Dict[str, HierarchicalPatch] = {}
        self.spatial_indices: Dict[str, KDTree] = {}
        self.material_processors = {
            'fur': FurProcessor(),
            'skin': SkinProcessor(),
            'feather': FeatherProcessor(),
            'scale': ScaleProcessor()
        }
        self.max_patches_per_subject = 1000
        self.patch_overlap_threshold = 0.3
        self.confidence_threshold = 0.7
        self.temporal_decay = 0.95
        self.uncertainty_threshold = 0.5
        self.class_to_body_plan = self._create_class_mapping()

    def _create_class_mapping(self) -> Dict[str, BodyPlan]:
        return {
            'cat': BodyPlan.QUADRUPED_SMALL,
            'dog': BodyPlan.QUADRUPED_MEDIUM,
            'horse': BodyPlan.QUADRUPED_LARGE,
            'cow': BodyPlan.QUADRUPED_LARGE,
            'sheep': BodyPlan.QUADRUPED_MEDIUM,
            'pig': BodyPlan.QUADRUPED_MEDIUM,
            'chicken': BodyPlan.BIPED_GROUND,
            'duck': BodyPlan.BIPED_GROUND,
            'bird': BodyPlan.BIPED_FLYING,
        }

    def process_image(self, image, subject_id: str, template_mesh=None, camera_params=None):
        patches = self.extract_patches_from_image(image, subject_id, template_mesh, camera_params)
        self.update_patch_quilt(subject_id, patches)
        refined_mesh = self.get_refined_mesh(subject_id, template_mesh)
        suggestions = self.suggest_next_views(subject_id)
        return {
            'refined_mesh': refined_mesh,
            'view_suggestions': suggestions,
            'confidence': self._calculate_overall_confidence(subject_id)
        }

    def extract_patches_from_image(self, image, subject_id: str, template_mesh, camera_params=None):
        patches = []
        patch_regions = self._segment_image_patches_adaptive(image)
        patches_by_size = defaultdict(list)
        coords_by_size = defaultdict(list)
        for region in patch_regions:
            size = region['size']
            patches_by_size[size].append(region['patch'])
            coords_by_size[size].append(region['coords'])
        for size, patch_list in patches_by_size.items():
            if not patch_list:
                continue
            patch_tensor = torch.stack([torch.FloatTensor(p).permute(2, 0, 1) for p in patch_list]).to(self.device)
            with torch.no_grad():
                results = self.patch_extractor({str(size): patch_tensor})
            for i, (patch_img, coords) in enumerate(zip(patch_list, coords_by_size[size])):
                view_angle = self._calculate_view_angle(coords, template_mesh, camera_params)
                surface_normal = results['normals'][i].cpu().numpy()
                view_confidence = self._calculate_view_confidence(
                    results['confidence'][i].item(), view_angle, surface_normal
                )
                pd = PatchData(
                    patch_id=f"{subject_id}_{len(patches)}_{hash(str(coords))}",
                    surface_type=self._decode_surface_type(results['surface_type'][i]),
                    texture_features=results['features'][i].cpu().numpy(),
                    depth_estimate=results['depth'][i].item(),
                    normal_vector=surface_normal,
                    confidence=results['confidence'][i].item(),
                    pixel_coords=coords,
                    mesh_coords=self._project_to_mesh(coords, template_mesh, camera_params),
                    view_angle=view_angle,
                    lighting_conditions=self._analyze_lighting(patch_img),
                    timestamp=time.time(),
                    patch_size=size,
                    uncertainty=results['uncertainty'][i].item(),
                    view_confidence=view_confidence
                )
                if pd.surface_type in self.material_processors:
                    pd = self.material_processors[pd.surface_type].process_patch(pd)
                patches.append(pd)
        return patches

    def _segment_image_patches_adaptive(self, image):
        patches = []
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        detail_map = self._compute_detail_map(gray)
        sizes = [32, 64, 128]
        for size in sizes:
            stride = size // 2
            for y in range(0, h - size, stride):
                for x in range(0, w - size, stride):
                    patch = image[y:y+size, x:x+size]
                    detail_score = detail_map[y:y+size, x:x+size].mean()
                    if self._is_appropriate_patch_size(size, detail_score):
                        if self._is_informative_patch(patch):
                            patches.append({
                                'patch': cv2.resize(patch, (64, 64)),
                                'coords': (x + size//2, y + size//2),
                                'size': size,
                                'detail_score': detail_score
                            })
        return patches

    def _compute_detail_map(self, gray_image):
        kernel_size = 5
        mean = cv2.blur(gray_image.astype(float), (kernel_size, kernel_size))
        sqr_mean = cv2.blur(gray_image.astype(float)**2, (kernel_size, kernel_size))
        variance = sqr_mean - mean**2
        return np.sqrt(np.maximum(variance, 0))

    def _is_appropriate_patch_size(self, size, detail_score):
        if detail_score < 10:
            return size >= 128
        elif detail_score > 50:
            return size <= 32
        else:
            return size == 64

    def _is_informative_patch(self, patch):
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var > 100

    def _calculate_view_confidence(self, base_confidence, view_angle, surface_normal):
        angle_penalty = max(0.3, np.cos(np.radians(view_angle)))
        view_direction = np.array([0, 0, 1])
        grazing_penalty = max(0.3, abs(np.dot(surface_normal, view_direction)))
        return base_confidence * angle_penalty * grazing_penalty

    def update_patch_quilt(self, subject_id: str, new_patches: List[PatchData]):
        if subject_id not in self.active_patches:
            self.active_patches[subject_id] = []
            self.patch_history[subject_id] = []
            self.hierarchical_patches[subject_id] = HierarchicalPatch()
        for patch in new_patches:
            if patch.confidence > self.confidence_threshold:
                self.active_patches[subject_id].append(patch)
                self.patch_history[subject_id].append(patch)
                if patch.patch_size <= 32:
                    self.hierarchical_patches[subject_id].fine.append(patch)
                elif patch.patch_size <= 64:
                    self.hierarchical_patches[subject_id].medium.append(patch)
                else:
                    self.hierarchical_patches[subject_id].coarse.append(patch)
        self._update_spatial_index(subject_id)
        self._prune_patches(subject_id)
        self._cluster_similar_patches_enhanced(subject_id)
        self._update_mesh_patches(subject_id)

    def _update_spatial_index(self, subject_id: str):
        patches = self.active_patches[subject_id]
        if patches:
            coords = np.array([p.mesh_coords for p in patches])
            self.spatial_indices[subject_id] = KDTree(coords)

    def _cluster_similar_patches_enhanced(self, subject_id: str):
        patches = self.active_patches[subject_id]
        if len(patches) < 2:
            return
        if subject_id in self.spatial_indices:
            tree = self.spatial_indices[subject_id]
            merged_indices = set()
            new_patches = []
            for i, patch in enumerate(patches):
                if i in merged_indices:
                    continue
                indices = tree.query_ball_point(patch.mesh_coords, r=0.1)
                neighbor_patches = [patches[j] for j in indices if j != i]
                if neighbor_patches:
                    similar_patches = [patch]
                    for neighbor in neighbor_patches:
                        if self._patches_similar(patch, neighbor):
                            similar_patches.append(neighbor)
                            merged_indices.add(patches.index(neighbor))
                    if len(similar_patches) > 1:
                        merged_patch = self._merge_patches_poisson(similar_patches)
                        new_patches.append(merged_patch)
                        merged_indices.add(i)
                    else:
                        new_patches.append(patch)
                else:
                    new_patches.append(patch)
            self.active_patches[subject_id] = new_patches

    def _patches_similar(self, patch1: PatchData, patch2: PatchData) -> bool:
        spatial_dist = np.linalg.norm(np.array(patch1.mesh_coords) - np.array(patch2.mesh_coords))
        feature_sim = np.dot(patch1.texture_features, patch2.texture_features) / (
            np.linalg.norm(patch1.texture_features) * np.linalg.norm(patch2.texture_features)
        )
        type_match = patch1.surface_type == patch2.surface_type
        return spatial_dist < 0.1 and feature_sim > 0.8 and type_match

    def _merge_patches_poisson(self, patches: List[PatchData]) -> PatchData:
        patches.sort(key=lambda p: p.confidence, reverse=True)
        base_patch = patches[0]
        total_weight = sum(p.confidence for p in patches)
        merged_features = np.zeros_like(base_patch.texture_features)
        merged_normal = np.zeros(3)
        merged_depth = 0
        for patch in patches:
            weight = patch.confidence / total_weight
            merged_features += weight * patch.texture_features
            merged_normal += weight * patch.normal_vector
            merged_depth += weight * patch.depth_estimate
        merged_normal = merged_normal / np.linalg.norm(merged_normal)
        position_variance = np.var([p.mesh_coords for p in patches], axis=0)
        merged_uncertainty = np.mean(position_variance) + np.mean([p.uncertainty for p in patches])
        return PatchData(
            patch_id=f"merged_{base_patch.patch_id}",
            surface_type=base_patch.surface_type,
            texture_features=merged_features,
            depth_estimate=merged_depth,
            normal_vector=merged_normal,
            confidence=min(total_weight, 1.0),
            pixel_coords=base_patch.pixel_coords,
            mesh_coords=tuple(np.mean([p.mesh_coords for p in patches], axis=0)),
            view_angle=np.mean([p.view_angle for p in patches]),
            lighting_conditions=base_patch.lighting_conditions,
            timestamp=max(p.timestamp for p in patches),
            patch_size=int(np.mean([p.patch_size for p in patches])),
            uncertainty=merged_uncertainty,
            view_confidence=np.mean([p.view_confidence for p in patches])
        )

    def _calculate_mesh_uncertainty(self, patches: List[PatchData]) -> MeshUncertainty:
        if not patches:
            return None
        positions = np.array([p.mesh_coords for p in patches])
        normals = np.array([p.normal_vector for p in patches])
        pos_var = np.var(positions, axis=0) if len(positions) > 1 else np.zeros(3)
        normal_var = np.var(normals, axis=0) if len(normals) > 1 else np.zeros(3)
        features = np.array([p.texture_features[:10] for p in patches])
        texture_entropy = -np.sum(features * np.log(features + 1e-10)) / len(patches)
        return MeshUncertainty(
            positional_variance=pos_var,
            normal_variance=normal_var,
            texture_entropy=texture_entropy,
            observation_count=len(patches)
        )

    def suggest_next_views(self, subject_id: str) -> List[ViewSuggestion]:
        if subject_id not in self.mesh_patches:
            return []
        suggestions = []
        for region_id, mesh_patch in self.mesh_patches.items():
            if mesh_patch.uncertainty is None:
                continue
            uncertainty_score = (
                np.mean(mesh_patch.uncertainty.positional_variance) +
                np.mean(mesh_patch.uncertainty.normal_variance) +
                mesh_patch.uncertainty.texture_entropy
            ) / 3.0
            if uncertainty_score > self.uncertainty_threshold:
                center = np.mean(mesh_patch.vertices, axis=0)
                for angle_offset in [0, 45, -45]:
                    suggestions.append(ViewSuggestion(
                        region_id=region_id,
                        azimuth=np.arctan2(center[1], center[0]) + np.radians(angle_offset),
                        elevation=np.arctan2(center[2], np.sqrt(center[0]**2 + center[1]**2)),
                        priority=uncertainty_score,
                        uncertainty_score=uncertainty_score
                    ))
        suggestions.sort(key=lambda s: s.priority, reverse=True)
        return suggestions[:5]

    def _decode_surface_type(self, surface_logits):
        surface_types = ['fur', 'skin', 'feather', 'scale']
        idx = torch.argmax(surface_logits).item()
        return surface_types[idx]

    def _project_to_mesh(self, pixel_coords, template_mesh, camera_params):
        x, y = pixel_coords
        mesh_x = (x / 224.0 - 0.5) * 2.0
        mesh_y = (y / 224.0 - 0.5) * 2.0
        mesh_z = 0.0
        return (mesh_x, mesh_y, mesh_z)

    def _calculate_view_angle(self, pixel_coords, template_mesh, camera_params):
        center_x, center_y = 112, 112
        x, y = pixel_coords
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        return (dist / max_dist) * 90

    def _analyze_lighting(self, patch_img):
        hsv = cv2.cvtColor(patch_img, cv2.COLOR_RGB2HSV)
        return {
            'brightness': np.mean(hsv[:, :, 2]),
            'saturation': np.mean(hsv[:, :, 1]),
            'hue_dominant': np.argmax(np.bincount(hsv[:, :, 0].flatten())),
            'contrast': np.std(cv2.cvtColor(patch_img, cv2.COLOR_RGB2GRAY))
        }

    def _prune_patches(self, subject_id):
        current_time = time.time()
        patches = self.active_patches[subject_id]
        for patch in patches:
            age = current_time - patch.timestamp
            decay_factor = self.temporal_decay ** (age / 3600)
            patch.confidence *= decay_factor
        patches = [p for p in patches if p.confidence > self.confidence_threshold]
        if len(patches) > self.max_patches_per_subject:
            patches.sort(key=lambda p: p.confidence * p.view_confidence, reverse=True)
            patches = patches[:self.max_patches_per_subject]
        self.active_patches[subject_id] = patches

    def _update_mesh_patches(self, subject_id):
        patches = self.active_patches[subject_id]
        mesh_regions = self._group_patches_by_mesh_region(patches)
        for region_id, region_patches in mesh_regions.items():
            full_id = f"{subject_id}_{region_id}"
            if full_id not in self.mesh_patches:
                self.mesh_patches[full_id] = self._create_mesh_patch(region_patches)
            else:
                self._update_existing_mesh_patch(full_id, region_patches)
            self.mesh_patches[full_id].uncertainty = self._calculate_mesh_uncertainty(region_patches)

    def _group_patches_by_mesh_region(self, patches):
        regions = {}
        region_size = 0.2
        for patch in patches:
            x, y, z = patch.mesh_coords
            region_key = (int(x / region_size), int(y / region_size), int(z / region_size))
            regions.setdefault(region_key, []).append(patch)
        return regions

    def _create_mesh_patch(self, patches):
        vertices = np.array([p.mesh_coords for p in patches])
        faces = self._triangulate_points(vertices) if len(vertices) >= 3 else np.array([])
        texture_map = self._create_texture_map(patches)
        normal_map = self._create_normal_map(patches)
        confidence_map = np.array([p.confidence * p.view_confidence for p in patches])
        return MeshPatch(
            patch_id=f"mesh_{patches[0].patch_id}",
            vertices=vertices,
            faces=faces,
            texture_map=texture_map,
            normal_map=normal_map,
            reality_patches=patches,
            confidence_map=confidence_map,
            last_updated=max(p.timestamp for p in patches)
        )

    def _update_existing_mesh_patch(self, patch_id, new_patches):
        existing = self.mesh_patches[patch_id]
        all_patches = existing.reality_patches + new_patches
        self.mesh_patches[patch_id] = self._create_mesh_patch(all_patches)

    def _triangulate_points(self, vertices):
        n = len(vertices)
        if n < 3:
            return np.array([])
        faces = []
        for i in range(n - 2):
            faces.append([0, i + 1, i + 2])
        return np.array(faces)

    def _create_texture_map(self, patches):
        texture_features = np.array([p.texture_features for p in patches])
        return np.mean(texture_features, axis=0)

    def _create_normal_map(self, patches):
        normals = np.array([p.normal_vector for p in patches])
        return np.mean(normals, axis=0)

    def get_refined_mesh(self, subject_id: str, base_template):
        if subject_id not in self.active_patches:
            return base_template
        return {
            'base_template': base_template,
            'reality_patches': self.active_patches[subject_id],
            'mesh_patches': {k: v for k, v in self.mesh_patches.items() if k.startswith(subject_id)},
            'hierarchical_patches': self.hierarchical_patches.get(subject_id),
            'confidence_score': self._calculate_overall_confidence(subject_id),
            'last_updated': max((p.timestamp for p in self.active_patches[subject_id]), default=0)
        }

    def _calculate_overall_confidence(self, subject_id: str):
        patches = self.active_patches.get(subject_id, [])
        if not patches:
            return 0.0
        total_confidence = sum(p.confidence * p.view_confidence for p in patches)
        coverage_bonus = min(len(patches) / 100, 1.0)
        angle_diversity = len(set(int(p.view_angle / 10) for p in patches)) / 9.0
        return min((total_confidence / len(patches)) * (1 + coverage_bonus) * (1 + angle_diversity), 1.0)

    def save_system(self, path: str):
        torch.save({
            'patch_extractor': self.patch_extractor.state_dict(),
            'body_plan_classifier': self.body_plan_classifier.state_dict(),
            'geometry_predictor': self.geometry_predictor.state_dict(),
            'active_patches': self.active_patches,
            'mesh_patches': self.mesh_patches,
            'patch_history': self.patch_history,
            'hierarchical_patches': self.hierarchical_patches
        }, path)
        logger.info(f"System saved to {path}")

    def load_system(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.patch_extractor.load_state_dict(checkpoint['patch_extractor'])
        self.body_plan_classifier.load_state_dict(checkpoint['body_plan_classifier'])
        self.geometry_predictor.load_state_dict(checkpoint['geometry_predictor'])
        self.active_patches = checkpoint['active_patches']
        self.mesh_patches = checkpoint['mesh_patches']
        self.patch_history = checkpoint['patch_history']
        self.hierarchical_patches = checkpoint['hierarchical_patches']
        logger.info(f"System loaded from {path}")

# ================== Training System ==================

class FixedUnifiedTrainer:
    def __init__(self, system: PatchQuiltSystem):
        self.system = system
        self.device = system.device

    def train(self, train_loader, val_loader, epochs=100, lr=0.001):
        optimizer_pe = torch.optim.Adam(self.system.patch_extractor.parameters(), lr=lr)
        optimizer_bp = torch.optim.Adam(self.system.body_plan_classifier.parameters(), lr=lr)
        optimizer_gp = torch.optim.Adam(self.system.geometry_predictor.parameters(), lr=lr)
        surface_criterion = nn.CrossEntropyLoss()
        regression_criterion = nn.MSELoss()
        bp_criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.system.patch_extractor.train()
            self.system.body_plan_classifier.train()
            self.system.geometry_predictor.train()
            train_loss = 0
            num_batches = 0
            for batch in train_loader:
                patches_dict = {k: v.to(self.device) for k, v in batch['patches'].items()}
                optimizer_pe.zero_grad()
                patch_outputs = self.system.patch_extractor(patches_dict)
                surface_labels = batch['surface_labels'].to(self.device)
                depth_labels = batch['depth_labels'].to(self.device)
                normal_labels = batch['normal_labels'].to(self.device)
                confidence_labels = batch['confidence_labels'].to(self.device)
                uncertainty_labels = batch['uncertainty_labels'].to(self.device)
                surface_loss = surface_criterion(patch_outputs['surface_type'], surface_labels)
                depth_loss = regression_criterion(patch_outputs['depth'], depth_labels)
                normal_loss = regression_criterion(patch_outputs['normals'], normal_labels)
                confidence_loss = regression_criterion(patch_outputs['confidence'], confidence_labels)
                uncertainty_loss = regression_criterion(patch_outputs['uncertainty'], uncertainty_labels)
                patch_loss = surface_loss + depth_loss + normal_loss + confidence_loss + uncertainty_loss
                patch_loss.backward()
                optimizer_pe.step()

                optimizer_bp.zero_grad()
                image_features = torch.randn(batch['images'].shape[0], 2048).to(self.device)
                bp_outputs = self.system.body_plan_classifier(image_features)
                bp_labels = batch['body_plan_labels'].to(self.device)
                bp_loss = bp_criterion(bp_outputs, bp_labels)
                bp_loss.backward()
                optimizer_bp.step()

                optimizer_gp.zero_grad()
                gp_outputs = self.system.geometry_predictor(image_features)
                geometry_labels = batch['geometry_labels'].to(self.device)
                gp_loss = regression_criterion(gp_outputs, geometry_labels)
                gp_loss.backward()
                optimizer_gp.step()

                train_loss += patch_loss.item() + bp_loss.item() + gp_loss.item()
                num_batches += 1
            val_loss = self.validate(val_loader)
            avg_train_loss = train_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if (epoch + 1) % 5 == 0:
                self.system.save_system(f"checkpoint_epoch_{epoch+1}.pth")

    def validate(self, val_loader):
        self.system.patch_extractor.eval()
        self.system.body_plan_classifier.eval()
        self.system.geometry_predictor.eval()
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                patches_dict = {k: v.to(self.device) for k, v in batch['patches'].items()}
                patch_outputs = self.system.patch_extractor(patches_dict)
                surface_labels = batch['surface_labels'].to(self.device)
                surface_loss = F.cross_entropy(patch_outputs['surface_type'], surface_labels)
                total_loss += surface_loss.item()
                num_batches += 1
        return total_loss / max(num_batches, 1)

# ================== Dataset and Data Loading ==================

class AnimalPatchDataset(Dataset):
    def __init__(self, root_dir, label_map, transform=None):
        self.samples = []
        self.transform = transform
        self.label_map = label_map
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.png')):
                    self.samples.append((os.path.join(class_dir, fname), class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image_tensor = self.transform(image)
        np_image = np.array(image)
        patches_data = self._extract_patches_simple(np_image)
        body_plan_idx = self._get_body_plan_index(label_name)
        return {
            'image': image_tensor,
            'patches_data': patches_data,
            'label_name': label_name,
            'body_plan_label': body_plan_idx,
            'animal_class_label': self.label_map.get(label_name, 0),
            'image_path': img_path
        }

    def _get_body_plan_index(self, label_name):
        class_to_body_plan = {
            'cat': 0, 'rabbit': 0, 'squirrel': 0, 'hamster': 0, 'rat': 0,
            'dog': 1, 'sheep': 1, 'pig': 1, 'deer': 1, 'antelope': 1, 'goat': 1,
            'horse': 2, 'cow': 2, 'buffalo': 2, 'elephant': 2, 'rhinoceros': 2,
            'hippopotamus': 2, 'giraffe': 2, 'zebra': 2,
            'eagle': 3, 'sparrow': 3, 'parrot': 3, 'owl': 3, 'hawk': 3,
            'hummingbird': 3, 'pigeon': 3, 'crow': 3,
            'chicken': 4, 'duck': 4, 'goose': 4, 'ostrich': 4, 'peacock': 4,
            'turkey': 4, 'penguin': 4,
            'snake': 5, 'lizard': 5, 'crocodile': 5, 'alligator': 5
        }
        return class_to_body_plan.get(label_name.lower(), 1)

    def _extract_patches_simple(self, image):
        h, w = image.shape[:2]
        patches = {'32': [], '64': [], '128': []}
        for size in [32, 64, 128]:
            stride = size // 2
            count = 0
            max_patches = 5
            for y in range(0, h - size, stride):
                for x in range(0, w - size, stride):
                    if count >= max_patches:
                        break
                    patch = image[y:y+size, x:x+size]
                    patch_resized = cv2.resize(patch, (64, 64))
                    patch_tensor = torch.FloatTensor(patch_resized).permute(2, 0, 1) / 255.0
                    patches[str(size)].append(patch_tensor)
                    count += 1
                if count >= max_patches:
                    break
        return patches


def fixed_collate_fn(batch):
    collated = {'patches': defaultdict(list), 'body_plan_labels': [], 'animal_class_labels': [], 'images': []}
    batch_size = len(batch)
    max_patches_per_size = 5
    for size in ['32', '64', '128']:
        collated['patches'][size] = torch.zeros(batch_size * max_patches_per_size, 3, 64, 64)
    for i, sample in enumerate(batch):
        collated['images'].append(sample['image'])
        collated['body_plan_labels'].append(sample['body_plan_label'])
        collated['animal_class_labels'].append(sample.get('animal_class_label', 0))
        for size in ['32', '64', '128']:
            patches = sample['patches_data'][size]
            start_idx = i * max_patches_per_size
            for j, patch in enumerate(patches):
                if j < max_patches_per_size:
                    collated['patches'][size][start_idx + j] = patch
    collated['images'] = torch.stack(collated['images'])
    collated['body_plan_labels'] = torch.tensor(collated['body_plan_labels'], dtype=torch.long)
    collated['animal_class_labels'] = torch.tensor(collated['animal_class_labels'], dtype=torch.long)
    num_total_patches = batch_size * max_patches_per_size
    collated['surface_labels'] = torch.randint(0, 4, (num_total_patches,))
    collated['depth_labels'] = torch.rand(num_total_patches, 1)
    collated['normal_labels'] = F.normalize(torch.randn(num_total_patches, 3), dim=1)
    collated['confidence_labels'] = torch.rand(num_total_patches, 1)
    collated['uncertainty_labels'] = torch.rand(num_total_patches, 1)
    collated['body_plan_labels_patches'] = collated['body_plan_labels'].repeat_interleave(max_patches_per_size)
    collated['geometry_labels'] = torch.rand(batch_size, 8)
    return collated


def create_fixed_animal_dataloaders(root_dir, batch_size=4):
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    label_map = {cls: i for i, cls in enumerate(classes)}
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = AnimalPatchDataset(root_dir, label_map, transform)
    val_split = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [val_split, len(dataset) - val_split])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=fixed_collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=fixed_collate_fn, num_workers=0)
    return train_loader, val_loader


def main():
    dataset_path = os.getenv('ANIMAL_DATASET_PATH', './animals')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    system = PatchQuiltSystem(device=device)
    trainer = FixedUnifiedTrainer(system)
    if os.path.exists(dataset_path):
        train_loader, val_loader = create_fixed_animal_dataloaders(dataset_path, batch_size=4)
        logger.info('Starting training...')
        trainer.train(train_loader, val_loader, epochs=1, lr=0.0005)
        system.save_system('AnimalTuned_model.pth')
    else:
        logger.warning('Dataset path not found, skipping training')
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = system.process_image(dummy_image, 'test_subject')
    logger.info(f"Confidence: {result['confidence']:.3f}")
    logger.info(f"Suggested views: {len(result['view_suggestions'])}")
    export_data = {
        'refined_mesh': result['refined_mesh'],
        'confidence': result['confidence'],
        'timestamp': time.time()
    }
    with open('unity_export.json', 'w') as f:
        json.dump(export_data, f, default=str, indent=2)
    logger.info('Unity export complete!')


if __name__ == '__main__':
    main()
