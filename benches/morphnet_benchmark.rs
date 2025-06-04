use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use morphnet_gtl::{MorphNetBuilder, TemplateFactory, BodyPlan};
use morphnet_gtl::patch_quilt::{PatchQuilt, RefinementConfig};
use morphnet_gtl::spatial::SpatialConfig;
use ndarray::{Array3, Array4, Array1, Array2};

fn bench_morphnet_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("morphnet_classification");

    // Setup MorphNet model (CPU only for benchmarking stability)
    let morphnet = MorphNetBuilder::new()
        .with_device(Device::Cpu)
        .with_num_species(10)
        .build()
        .expect("Failed to create MorphNet model");

    let image_sizes = [(224, 224), (512, 512), (1024, 1024)];

    for (height, width) in image_sizes.iter() {
        let image = create_test_image(*height, *width);
        group.bench_with_input(
            BenchmarkId::new("classify_image", format!("{}x{}", height, width)),
            &image,
            |b, img| {
                b.iter(|| {
                    black_box(morphnet.classify(img).unwrap());
                });
            },
        );
    }

    group.finish();
}

fn bench_template_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("template_operations");

    let body_plans = [BodyPlan::Quadruped, BodyPlan::Bird];

    for body_plan in body_plans.iter() {
        let template = match body_plan {
            BodyPlan::Quadruped => TemplateFactory::create_quadruped(),
            BodyPlan::Bird => TemplateFactory::create_bird(),
            _ => continue,
        };

        group.bench_with_input(
            BenchmarkId::new("template_validation", format!("{:?}", body_plan)),
            &template,
            |b, template| {
                b.iter(|| {
                    black_box(template.validate().unwrap());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parameter_vector_conversion", format!("{:?}", body_plan)),
            &template,
            |b, template| {
                b.iter(|| {
                    let params = black_box(template.to_parameter_vector());
                    black_box(params);
                });
            },
        );
    }

    group.finish();
}

fn bench_patch_quilt_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("patch_quilt");

    let mut patch_quilt = PatchQuilt::new(RefinementConfig::default());
    let subject_id = "benchmark_subject";

    // Create base mesh
    let template = TemplateFactory::create_quadruped();
    let base_mesh = template_to_mesh(&template);

    let patch_counts = [10, 50, 100, 500];

    for patch_count in patch_counts.iter() {
        let patches = create_test_patches(*patch_count);
        group.bench_with_input(
            BenchmarkId::new("update_patch_quilt", patch_count),
            &patches,
            |b, patches| {
                b.iter(|| {
                    let mut pq = PatchQuilt::new(RefinementConfig::default());
                    black_box(pq.update_patch_quilt(subject_id.to_string(), patches.clone()));
                });
            },
        );
    }

    // Benchmark spatial queries
    let patches = create_test_patches(100);
    patch_quilt.update_patch_quilt(subject_id.to_string(), patches);

    group.bench_function("spatial_query", |b| {
        b.iter(|| {
            let position = nalgebra::Point3::new(0.0, 0.0, 0.0);
            black_box(patch_quilt.find_patches_near(position, 1.0, Some(subject_id)));
        });
    });

    group.finish();
}

fn bench_spatial_awareness(c: &mut Criterion) {
    let mut group = c.benchmark_group("spatial_awareness");

    // Setup spatial awareness system
    let spatial_config = SpatialConfig::default();

    let sensor_data_variants = [
        ("simple", create_simple_sensor_data()),
        ("complex", create_complex_sensor_data()),
    ];

    for (variant_name, sensor_data) in sensor_data_variants.iter() {
        group.bench_with_input(
            BenchmarkId::new("process_sensor_data", variant_name),
            sensor_data,
            |b, data| {
                b.iter(|| {
                    black_box(data.clone());
                });
            },
        );
    }

    group.finish();
}

fn bench_geometric_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("geometric_operations");

    let mesh_sizes = [100, 1000, 10000];

    for size in mesh_sizes.iter() {
        let mut mesh = create_test_mesh_data(*size);
        group.bench_with_input(
            BenchmarkId::new("compute_normals", size),
            &mesh,
            |b, mesh| {
                b.iter(|| {
                    let mut mesh_copy = mesh.clone();
                    black_box(mesh_copy.compute_normals());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("add_vertices", size),
            size,
            |b, &vertex_count| {
                b.iter(|| {
                    let mut mesh = morphnet_gtl::mmx::MeshData::new();
                    for i in 0..vertex_count {
                        let vertex = nalgebra::Point3::new(i as f32 * 0.1, (i % 100) as f32 * 0.1, 0.0);
                        black_box(mesh.add_vertex(vertex));
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_embedding_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_operations");

    let embedding_sizes = [(100, 64), (1000, 128), (10000, 256)];

    for (n_samples, n_features) in embedding_sizes.iter() {
        let embedding_data = create_test_embedding(*n_samples, *n_features);
        group.bench_with_input(
            BenchmarkId::new("create_embedding", format!("{}x{}", n_samples, n_features)),
            &(*n_samples, *n_features),
            |b, &(n_samples, n_features)| {
                b.iter(|| {
                    black_box(create_test_embedding(n_samples, n_features));
                });
            },
        );
    }

    group.finish();
}

fn create_test_image(height: usize, width: usize) -> Array3<f32> {
    Array3::zeros((height, width, 3))
}

fn create_test_patches(count: usize) -> Vec<morphnet_gtl::patch_quilt::Patch> {
    use morphnet_gtl::patch_quilt::Patch;
    use uuid::Uuid;
    use chrono::Utc;
    use nalgebra::{Point3, Vector3};
    use std::collections::HashMap;
    (0..count)
        .map(|i| Patch {
            id: Uuid::new_v4(),
            source_id: format!("source_{}", i),
            timestamp: Utc::now(),
            position: Point3::new(i as f32 * 0.1, 0.0, 0.0),
            normal: Vector3::new(0.0, 0.0, 1.0),
            texture: Array3::zeros((32, 32, 3)),
            depth: Array2::zeros((32, 32)),
            normals: Array3::zeros((32, 32, 3)),
            confidence: 0.8,
            world_size: (0.1, 0.1),
            uv_coords: Some((i as f32 * 0.01, 0.0)),
            metadata: HashMap::new(),
        })
        .collect()
}

fn create_simple_sensor_data() -> morphnet_gtl::patch_quilt::SensorData {
    use morphnet_gtl::patch_quilt::SensorData;
    use std::collections::HashMap;
    SensorData {
        visual: Some(Array3::zeros((240, 320, 3))),
        lidar: None,
        accelerometer: Some(Array1::from_vec(vec![0.1, 0.05, 9.81])),
        strain_gauges: None,
        temperature: None,
        pressure: None,
        custom: HashMap::new(),
    }
}

fn create_complex_sensor_data() -> morphnet_gtl::patch_quilt::SensorData {
    use morphnet_gtl::patch_quilt::SensorData;
    use std::collections::HashMap;
    let mut strain_gauges = HashMap::new();
    strain_gauges.insert("gauge_1".to_string(), 0.001);
    strain_gauges.insert("gauge_2".to_string(), 0.002);
    let mut temperature = HashMap::new();
    temperature.insert("sensor_1".to_string(), 25.0);
    temperature.insert("sensor_2".to_string(), 23.5);
    SensorData {
        visual: Some(Array3::zeros((480, 640, 3))),
        lidar: Some(Array2::zeros((1000, 3))),
        accelerometer: Some(Array1::from_vec(vec![0.1, 0.05, 9.81])),
        strain_gauges: Some(strain_gauges),
        temperature: Some(temperature),
        pressure: Some({
            let mut pressure = HashMap::new();
            pressure.insert("barometer".to_string(), 1013.25);
            pressure
        }),
        custom: HashMap::new(),
    }
}

fn template_to_mesh(template: &morphnet_gtl::GeometricTemplate) -> morphnet_gtl::mmx::MeshData {
    let mut mesh = morphnet_gtl::mmx::MeshData::new();
    for keypoint in template.keypoints.values() {
        mesh.add_vertex(keypoint.position);
    }
    let vertex_count = mesh.vertices.len();
    if vertex_count >= 3 {
        for i in 0..(vertex_count - 2) {
            mesh.add_face([0, (i + 1) as u32, (i + 2) as u32]);
        }
    }
    mesh.compute_normals();
    mesh
}

fn create_test_mesh_data(vertex_count: usize) -> morphnet_gtl::mmx::MeshData {
    let mut mesh = morphnet_gtl::mmx::MeshData::new();
    let grid_size = (vertex_count as f32).sqrt() as usize;
    for i in 0..grid_size {
        for j in 0..grid_size {
            if mesh.vertices.len() >= vertex_count {
                break;
            }
            mesh.add_vertex(nalgebra::Point3::new(i as f32 * 0.1, j as f32 * 0.1, 0.0));
        }
        if mesh.vertices.len() >= vertex_count {
            break;
        }
    }
    let max_faces = std::cmp::min(vertex_count / 3, (grid_size - 1) * (grid_size - 1) * 2);
    for i in 0..max_faces {
        let v0 = (i * 3) % mesh.vertices.len();
        let v1 = (i * 3 + 1) % mesh.vertices.len();
        let v2 = (i * 3 + 2) % mesh.vertices.len();
        mesh.add_face([v0 as u32, v1 as u32, v2 as u32]);
    }
    mesh
}

fn create_test_embedding(n_samples: usize, n_features: usize) -> morphnet_gtl::mmx::EmbeddingData {
    use ndarray::{ArrayD, IxDyn};
    let matrix = ArrayD::zeros(IxDyn(&[n_samples, n_features]));
    morphnet_gtl::mmx::EmbeddingData::new(matrix, "test".to_string())
}

criterion_group!(
    benches,
    bench_morphnet_classification,
    bench_template_operations,
    bench_patch_quilt_operations,
    bench_spatial_awareness,
    bench_geometric_operations,
    bench_embedding_operations
);
criterion_main!(benches);
