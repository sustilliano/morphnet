use anyhow::{bail, Result};
use clap::Parser;
use serde::Deserialize;
use std::{path::PathBuf, time::Duration};

#[cfg(feature = "thingino")]
mod local {
    pub use morphnet::quilt::algo::tilt_triangulate::triangulate_scene;
    pub use morphnet::quilt::exporter::ply::write_ply_xyz;
    pub use morphnet::quilt::io::rtsp_capture::RtspCapture;
}

#[cfg(feature = "thingino")]
#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    rtsp: String,
    #[arg(long, default_value = "configs/cameras/d1_thingino.yaml")]
    intrinsics: PathBuf,
    #[arg(long, default_value_t = 12)]
    frames: usize,
    #[arg(long, default_value_t = 3)]
    step: usize,
    #[arg(long, default_value = "outputs/scenes/merged.ply")]
    output: PathBuf,
}

#[cfg(feature = "thingino")]
#[derive(Deserialize)]
struct Intr {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
}

#[cfg(feature = "thingino")]
fn main() -> Result<()> {
    let args = Args::parse();

    // load intrinsics
    let intr: Intr = serde_yaml::from_reader(std::fs::File::open(&args.intrinsics)?)?;

    // capture spaced frames during your tilt sweep
    let mut cap = local::RtspCapture::new(&args.rtsp)?;
    let mut frames = Vec::new();
    let mut grabbed = 0usize;
    std::thread::sleep(Duration::from_millis(300));

    while frames.len() < args.frames {
        if let Some((mat, _ts)) = cap.read()? {
            if grabbed % args.step == 0 {
                frames.push(mat);
            }
            grabbed += 1;
        } else {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    if frames.len() < 2 {
        bail!("Need ≥2 frames; got {}", frames.len());
    }

    let pts3d = local::triangulate_scene(&frames, (intr.fx, intr.fy, intr.cx, intr.cy))?;
    std::fs::create_dir_all(args.output.parent().unwrap_or(std::path::Path::new(".")))?;
    local::write_ply_xyz(&args.output, &pts3d)?;
    println!("Wrote {} points → {}", pts3d.len(), args.output.display());
    Ok(())
}

#[cfg(not(feature = "thingino"))]
fn main() {
    eprintln!("Enable the 'thingino' feature to run this example.");
}
