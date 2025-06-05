use morphnet_gtl::prelude::*;
use morphnet_gtl::mmx::{MMXBuilder, TensorData, MeshData, CompressionType};
use morphnet_gtl::mmx::{GeometricParameters, GeometricTemplateData, ExtendedBodyPlan};
use morphnet_gtl::morphnet::TemplateFactory;
use tempfile::tempdir;
use ndarray::{Array3, ArrayD, IxDyn};
use nalgebra::Point3;
use morphnet_gtl::morphnet::stack_tensors;

#[test]
fn test_mmx_basic_operations() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.mmx");

    // Create MMX file
    let mut mmx_file = MMXBuilder::new("test_creator".to_string())
        .create(&path)
        .expect("Failed to create MMX file");

    // Create test tensor
    let data = ArrayD::zeros(IxDyn(&[10, 10, 3]));
    let tensor = TensorData::new(data);

    // Write tensor
    mmx_file
        .write_tensor("/test_tensor", tensor.clone())
        .expect("Failed to write tensor");

    // Read tensor back
    let read_tensor = mmx_file
        .read_tensor("/test_tensor")
        .expect("Failed to read tensor");

    assert_eq!(tensor.shape, read_tensor.shape);
}

#[test]
fn test_mmx_mesh_operations() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mesh_test.mmx");

    let mut mmx_file = MMXBuilder::new("test_creator".to_string())
        .create(&path)
        .expect("Failed to create MMX file");

    // Create test mesh
    let mut mesh = MeshData::new();
    mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
    mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
    mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));
    mesh.add_face([0, 1, 2]);
    mesh.compute_normals();

    // Write mesh
    mmx_file
        .write_mesh("/test_mesh", mesh.clone())
        .expect("Failed to write mesh");

    // Read mesh back
    let read_mesh = mmx_file.read_mesh("/test_mesh").expect("Failed to read mesh");

    assert_eq!(mesh.vertices.len(), read_mesh.vertices.len());
    assert_eq!(mesh.faces.len(), read_mesh.faces.len());
}

#[test]
fn test_template_creation() {
    let quadruped = TemplateFactory::create_quadruped();
    assert_eq!(quadruped.body_plan, BodyPlan::Quadruped);
    assert!(!quadruped.keypoints.is_empty());
    assert!(!quadruped.connections.is_empty());

    let bird = TemplateFactory::create_bird();
    assert_eq!(bird.body_plan, BodyPlan::Bird);
    assert!(!bird.keypoints.is_empty());
    assert!(!bird.connections.is_empty());
}

#[test]
fn test_template_validation() {
    let quadruped = TemplateFactory::create_quadruped();

    // Template should be valid by default
    assert!(quadruped.validate().is_ok());

    // Test parameter vector conversion
    let params = quadruped.to_parameter_vector();
    assert!(!params.is_empty());
}

#[test]
fn test_compression_roundtrip() {
    let test_data = b"Hello, MorphNet-GTL! This is test data for compression.";

    // Test LZ4 compression
    let compressed = morphnet_gtl::mmx::format::compress_data(test_data, CompressionType::Lz4)
        .expect("Failed to compress with LZ4");
    let decompressed = morphnet_gtl::mmx::format::decompress_data(&compressed, CompressionType::Lz4)
        .expect("Failed to decompress with LZ4");
    assert_eq!(test_data, decompressed.as_slice());

    // Test Zlib compression
    let compressed = morphnet_gtl::mmx::format::compress_data(test_data, CompressionType::Zlib)
        .expect("Failed to compress with Zlib");
    let decompressed = morphnet_gtl::mmx::format::decompress_data(&compressed, CompressionType::Zlib)
        .expect("Failed to decompress with Zlib");
    assert_eq!(test_data, decompressed.as_slice());

    // Test no compression
    let compressed = morphnet_gtl::mmx::format::compress_data(test_data, CompressionType::None)
        .expect("Failed to compress with None");
    let decompressed = morphnet_gtl::mmx::format::decompress_data(&compressed, CompressionType::None)
        .expect("Failed to decompress with None");
    assert_eq!(test_data, decompressed.as_slice());
}

#[test]
fn test_patch_quilt_basic() {
    use morphnet_gtl::patch_quilt::{PatchQuilt, RefinementConfig};
    let patch_quilt = PatchQuilt::new(RefinementConfig::default());
    let chunks = patch_quilt.list_chunks();
    assert!(chunks.is_empty());
}

#[test]
fn test_tensor_data_from_image() {
    let img = image::RgbImage::new(64, 64);
    let tensor_data = TensorData::from_image(&img);
    assert_eq!(tensor_data.shape, vec![64, 64, 3]);
    assert_eq!(tensor_data.dtype, "float32");
}

#[test]
fn test_mesh_normal_computation() {
    let mut mesh = MeshData::new();
    mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
    mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
    mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));
    mesh.add_face([0, 1, 2]);
    assert!(mesh.normals.is_none());
    mesh.compute_normals();
    assert!(mesh.normals.is_some());
    let normals = mesh.normals.clone().unwrap();
    assert_eq!(normals.len(), 3);
}

#[test]
fn test_checksum_calculation() {
    let data1 = b"test data";
    let data2 = b"test data";
    let data3 = b"different data";

    let checksum1 = morphnet_gtl::mmx::format::calculate_checksum(data1);
    let checksum2 = morphnet_gtl::mmx::format::calculate_checksum(data2);
    let checksum3 = morphnet_gtl::mmx::format::calculate_checksum(data3);

    assert_eq!(checksum1, checksum2);
    assert_ne!(checksum1, checksum3);
}

#[test]
fn test_mmx_chunk_directory() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("directory_test.mmx");

    let mut mmx_file = MMXBuilder::new("test_creator".to_string())
        .create(&path)
        .expect("Failed to create MMX file");

    assert!(mmx_file.list_chunks().is_empty());

    let data = ArrayD::zeros(IxDyn(&[5, 5, 1]));
    let tensor = TensorData::new(data);
    mmx_file.write_tensor("/test", tensor).expect("Failed to write tensor");

    let chunks = mmx_file.list_chunks();
    assert_eq!(chunks.len(), 1);
    assert!(chunks.contains(&"/test".to_string()));

    let chunk_info = mmx_file.get_chunk_info("/test");
    assert!(chunk_info.is_some());
}

#[test]
fn test_geometric_template_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("geo.mmx");

    let mut mmx_file = MMXBuilder::new("creator".to_string())
        .create(&path)
        .expect("Failed to create MMX file");

    let params = GeometricParameters {
        body_length: 1.0,
        body_width: 0.5,
        body_height: 0.4,
        leg_length: 0.3,
        leg_thickness: 0.1,
        num_legs: 4,
        head_length: 0.2,
        head_width: 0.2,
        neck_length: 0.15,
        tail_length: 0.25,
        wing_span: 0.0,
        stride_length: 0.7,
        turning_radius: 1.0,
        jump_height: 0.5,
        max_speed: 5.0,
    };

    let template = GeometricTemplateData {
        parameters: params.clone(),
        body_plan: ExtendedBodyPlan::QuadrupedMedium,
    };

    mmx_file
        .write_geometric_template("/geo", template.clone())
        .expect("write geo template");

    let read = mmx_file
        .read_geometric_template("/geo")
        .expect("read geo template");

    assert_eq!(read.body_plan, template.body_plan);
    assert!((read.parameters.body_length - params.body_length).abs() < 1e-6);
}

#[cfg(test)]
mod spatial_tests {
    use morphnet_gtl::spatial::{SpatialConfig, EventSeverity, RiskLevel};

    #[test]
    fn test_spatial_config_defaults() {
        let config = SpatialConfig::default();
        assert!(config.prediction_horizon > 0.0);
        assert!(config.alert_threshold > 0.0 && config.alert_threshold <= 1.0);
        assert!(config.update_frequency > 0.0);
    }

    #[test]
    fn test_event_severity_ordering() {
        assert!(EventSeverity::Info < EventSeverity::Warning);
        assert!(EventSeverity::Warning < EventSeverity::Critical);
        assert!(EventSeverity::Critical < EventSeverity::Emergency);
    }

    #[test]
    fn test_risk_level_ordering() {
        assert!(RiskLevel::Low < RiskLevel::Moderate);
        assert!(RiskLevel::Moderate < RiskLevel::High);
        assert!(RiskLevel::High < RiskLevel::Critical);
        assert!(RiskLevel::Critical < RiskLevel::Extreme);
    }
}

#[test]
fn test_stack_tensors_helper() {
    let t1 = Array3::<f32>::zeros((1, 2, 2));
    let t2 = Array3::<f32>::ones((1, 2, 2));
    let stacked = stack_tensors(&[t1, t2]);
    assert_eq!(stacked.shape(), &[2, 2, 2]);
}
