use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use morphnet_gtl::mmx::chunks::MetaData;
use morphnet_gtl::mmx::{CompressionType, MMXBuilder, MeshData, TensorData};
use nalgebra::Point3;
use ndarray::{Array3, ArrayD, IxDyn};
use tempfile::tempdir;

fn bench_mmx_tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("mmx_tensor");

    // Test different tensor sizes
    let sizes = [(64, 64, 3), (256, 256, 3), (512, 512, 3), (1024, 1024, 3)];

    for (h, w, c_channels) in sizes.iter() {
        let size = h * w * c_channels;
        let tensor_data = create_test_tensor(*h, *w, *c_channels);

        group.bench_with_input(
            BenchmarkId::new("write_tensor", size),
            &tensor_data,
            |b, tensor| {
                b.iter(|| {
                    let dir = tempdir().unwrap();
                    let path = dir.path().join("test.mmx");
                    let mut mmx_file = MMXBuilder::new("benchmark".to_string())
                        .create(&path)
                        .unwrap();

                    black_box(
                        mmx_file
                            .write_tensor("/test_tensor", tensor.clone())
                            .unwrap(),
                    );
                });
            },
        );

        // Benchmark reading
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_read.mmx");
        let mut mmx_file = MMXBuilder::new("benchmark".to_string())
            .create(&path)
            .unwrap();
        mmx_file
            .write_tensor("/test_tensor", tensor_data.clone())
            .unwrap();
        drop(mmx_file);

        group.bench_with_input(BenchmarkId::new("read_tensor", size), &path, |b, path| {
            b.iter(|| {
                let mut mmx_file =
                    morphnet_gtl::mmx::MMXFile::open(path, morphnet_gtl::mmx::MMXMode::Read)
                        .unwrap();
                black_box(mmx_file.read_tensor("/test_tensor").unwrap());
            });
        });
    }

    group.finish();
}

fn bench_mmx_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("mmx_compression");

    let tensor = create_test_tensor(256, 256, 3);
    let data = tensor.to_bytes();

    let compression_types = [
        CompressionType::None,
        CompressionType::Lz4,
        CompressionType::Zlib,
    ];

    for compression in compression_types.iter() {
        group.bench_with_input(
            BenchmarkId::new("compress", format!("{:?}", compression)),
            &(data.clone(), *compression),
            |b, (data, compression)| {
                b.iter(|| {
                    black_box(
                        morphnet_gtl::mmx::format::compress_data(data, *compression).unwrap(),
                    );
                });
            },
        );

        // Benchmark decompression
        let compressed = morphnet_gtl::mmx::format::compress_data(&data, *compression).unwrap();
        group.bench_with_input(
            BenchmarkId::new("decompress", format!("{:?}", compression)),
            &(compressed, *compression),
            |b, (compressed, compression)| {
                b.iter(|| {
                    black_box(
                        morphnet_gtl::mmx::format::decompress_data(compressed, *compression)
                            .unwrap(),
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_mmx_mesh_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("mmx_mesh");

    let mesh_sizes = [100, 1000, 10000, 50000];

    for vertex_count in mesh_sizes.iter() {
        let mesh = create_test_mesh(*vertex_count);

        group.bench_with_input(
            BenchmarkId::new("write_mesh", vertex_count),
            &mesh,
            |b, mesh| {
                b.iter(|| {
                    let dir = tempdir().unwrap();
                    let path = dir.path().join("test.mmx");
                    let mut mmx_file = MMXBuilder::new("benchmark".to_string())
                        .create(&path)
                        .unwrap();

                    black_box(mmx_file.write_mesh("/test_mesh", mesh.clone()).unwrap());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("compute_normals", vertex_count),
            &mesh,
            |b, mesh| {
                b.iter(|| {
                    let mut mesh_copy = mesh.clone();
                    black_box(mesh_copy.compute_normals());
                });
            },
        );
    }

    group.finish();
}

fn bench_mmx_concurrent_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("mmx_concurrent");

    // Setup test file with multiple chunks
    let dir = tempdir().unwrap();
    let path = dir.path().join("concurrent_test.mmx");
    let mut mmx_file = MMXBuilder::new("benchmark".to_string())
        .create(&path)
        .unwrap();

    // Write multiple tensors
    for i in 0..10 {
        let tensor = create_test_tensor(128, 128, 3);
        mmx_file
            .write_tensor(&format!("/tensor_{}", i), tensor)
            .unwrap();
    }
    drop(mmx_file);

    group.bench_function("concurrent_read", |b| {
        b.iter(|| {
            use std::sync::Arc;
            use std::thread;

            let path = Arc::new(path.clone());
            let handles: Vec<_> = (0..4)
                .map(|thread_id| {
                    let path = Arc::clone(&path);
                    thread::spawn(move || {
                        let mut mmx_file = morphnet_gtl::mmx::MMXFile::open(
                            &*path,
                            morphnet_gtl::mmx::MMXMode::Read,
                        )
                        .unwrap();

                        for i in 0..3 {
                            let tensor_name = format!("/tensor_{}", (thread_id * 2 + i) % 10);
                            black_box(mmx_file.read_tensor(&tensor_name).unwrap());
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        });
    });

    group.finish();
}

// Helper functions
fn create_test_tensor(height: usize, width: usize, channels: usize) -> TensorData {
    let data = ArrayD::zeros(IxDyn(&[height, width, channels]));
    TensorData::new(data)
}

fn create_test_mesh(vertex_count: usize) -> MeshData {
    let mut mesh = MeshData::new();

    // Create vertices in a grid pattern
    let grid_size = (vertex_count as f32).sqrt() as usize;
    for i in 0..grid_size {
        for j in 0..grid_size {
            if mesh.vertices.len() >= vertex_count {
                break;
            }
            mesh.add_vertex(Point3::new(i as f32, j as f32, 0.0));
        }
        if mesh.vertices.len() >= vertex_count {
            break;
        }
    }

    // Create triangular faces
    let faces_to_create = std::cmp::min(vertex_count / 3, (grid_size - 1) * (grid_size - 1) * 2);
    for i in 0..faces_to_create {
        let v0 = (i * 3) % mesh.vertices.len();
        let v1 = (i * 3 + 1) % mesh.vertices.len();
        let v2 = (i * 3 + 2) % mesh.vertices.len();
        mesh.add_face([v0 as u32, v1 as u32, v2 as u32]);
    }

    mesh
}

criterion_group!(
    benches,
    bench_mmx_tensor_operations,
    bench_mmx_compression,
    bench_mmx_mesh_operations,
    bench_mmx_concurrent_access
);
criterion_main!(benches);
