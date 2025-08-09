pub fn write_ply_xyz(path: impl AsRef<std::path::Path>, pts: &[[f32; 3]]) -> anyhow::Result<()> {
    use std::io::Write;
    let mut buf = Vec::with_capacity(128 + pts.len() * 36);
    writeln!(buf, "ply")?;
    writeln!(buf, "format ascii 1.0")?;
    writeln!(buf, "element vertex {}", pts.len())?;
    writeln!(buf, "property float x")?;
    writeln!(buf, "property float y")?;
    writeln!(buf, "property float z")?;
    writeln!(buf, "end_header")?;
    for p in pts {
        writeln!(buf, "{} {} {}", p[0], p[1], p[2])?;
    }
    std::fs::write(path, buf)?;
    Ok(())
}
