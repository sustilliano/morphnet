use pyo3::prelude::*;

/// Minimal GUIDED mode stub for now (compiles without touching core crate).
/// Returns: (ellipse_params [cx, cy, a, b, theta], spline_points [(x,y)...], quality_score)
#[pyfunction]
fn guided_patch(
    depth_path: &str,
    flow_path: Option<&str>,
    max_iters: usize,
    aspect_ratio_max: f32,
    curvature_lambda: f32,
    area_prior: f32,
) -> PyResult<(Vec<f32>, Vec<(f32, f32)>, f32)> {
    // TODO: replace with real call into core crate once API is ready.
    let ellipse_params = vec![320.0, 180.0, 64.0, 48.0, 0.0];
    let spline_points = vec![(300.0, 180.0), (340.0, 180.0)];
    let quality_score = 0.0;
    Ok((ellipse_params, spline_points, quality_score))
}

#[pymodule]
fn morphnet(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(guided_patch, m)?)?;
    Ok(())
}
