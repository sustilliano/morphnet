pub fn triangulate_scene(
    frames: &[opencv::core::Mat],
    intr: (f64, f64, f64, f64),
) -> anyhow::Result<Vec<[f32; 3]>> {
    use anyhow::bail;
    use opencv::{calib3d, core, features2d, imgproc, prelude::*, types};

    fn gray(src: &core::Mat) -> opencv::Result<core::Mat> {
        if src.channels()? == 1 {
            return Ok(src.clone());
        }
        let mut g = core::Mat::default();
        imgproc::cvt_color(src, &mut g, imgproc::COLOR_BGR2GRAY, 0)?;
        Ok(g)
    }
    fn detect(g: &core::Mat) -> opencv::Result<(types::VectorOfKeyPoint, core::Mat)> {
        let orb = features2d::ORB::create(
            2000,
            1.2,
            8,
            31,
            0,
            2,
            features2d::ORB_ScoreType::HARRIS_SCORE,
            31,
            20,
        )?;
        let mut kps = types::VectorOfKeyPoint::new();
        let mut d = core::Mat::default();
        orb.detect_and_compute(g, &core::no_array(), &mut kps, &mut d, false)?;
        Ok((kps, d))
    }
    fn match_bf(d1: &core::Mat, d2: &core::Mat) -> opencv::Result<types::VectorOfDMatch> {
        let mut bf = features2d::BFMatcher::create(features2d::NORM_HAMMING, true)?;
        let mut m = types::VectorOfDMatch::new();
        bf.match_(d1, d2, &mut m, &core::no_array())?;
        let mut v = m.to_vec();
        v.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        let keep = (v.len() as f32 * 0.5).max(30.0) as usize;
        let mut o = types::VectorOfDMatch::new();
        for x in v.into_iter().take(keep) {
            o.push(x);
        }
        Ok(o)
    }
    fn k(fx: f64, fy: f64, cx: f64, cy: f64) -> opencv::Result<core::Mat> {
        core::Mat::from_slice_2d::<f64>(&[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    }

    let (fx, fy, cx, cy) = intr;
    let k = k(fx, fy, cx, cy)?;
    let mut all = Vec::new();

    for w in frames.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        let (ga, gb) = (gray(a)?, gray(b)?);
        let ((k1, d1), (k2, d2)) = (detect(&ga)?, detect(&gb)?);
        if d1.empty() || d2.empty() {
            continue;
        }
        let m = match_bf(&d1, &d2)?;
        if m.len() < 8 {
            continue;
        }

        let mut p1 = types::VectorOfPoint2f::new();
        let mut p2 = types::VectorOfPoint2f::new();
        for mm in m {
            let a = k1.get(mm.query_idx as usize).unwrap();
            let b = k2.get(mm.train_idx as usize).unwrap();
            p1.push(core::Point2f::new(a.pt.x, a.pt.y));
            p2.push(core::Point2f::new(b.pt.x, b.pt.y));
        }
        let mut inl = core::Mat::default();
        let e = calib3d::find_essential_mat_1(&p1, &p2, &k, calib3d::RANSAC, 0.999, 1.0, &mut inl)?;
        if e.empty() {
            continue;
        }
        let (mut r, mut t) = (core::Mat::default(), core::Mat::default());
        let _ = calib3d::recover_pose_1(&e, &p1, &p2, &k, &mut r, &mut t, &mut inl)?;

        // P0 = K [I|0], P1 = K [R|t]
        let p0 = {
            let mut rt = core::Mat::zeros(3, 4, core::CV_64F)?.to_mat()?;
            core::Mat::eye(3, 3, core::CV_64F)?
                .to_mat()?
                .copy_to(&mut rt.roi(core::Rect::new(0, 0, 3, 3))?)?;
            let mut p = core::Mat::default();
            core::gemm(&k, &rt, 1.0, &core::Mat::default(), 0.0, &mut p, 0)?;
            p
        };
        let p1m = {
            let mut rt = core::Mat::zeros(3, 4, core::CV_64F)?.to_mat()?;
            r.copy_to(&mut rt.roi(core::Rect::new(0, 0, 3, 3))?)?;
            t.copy_to(&mut rt.roi(core::Rect::new(3, 0, 1, 3))?.t()?)?;
            let mut p = core::Mat::default();
            core::gemm(&k, &rt, 1.0, &core::Mat::default(), 0.0, &mut p, 0)?;
            p
        };
        let mut pf = core::Mat::default();
        let mut qf = core::Mat::default();
        p1.convert_to(&mut pf, core::CV_64F, 1.0, 0.0)?;
        p2.convert_to(&mut qf, core::CV_64F, 1.0, 0.0)?;
        let mut x4 = core::Mat::default();
        calib3d::triangulate_points(&p0, &p1m, &pf, &qf, &mut x4)?;
        for i in 0..x4.cols() {
            let x = *x4.at_2d::<f64>(0, i)?;
            let y = *x4.at_2d::<f64>(1, i)?;
            let z = *x4.at_2d::<f64>(2, i)?;
            let w = *x4.at_2d::<f64>(3, i)?;
            if w.abs() < 1e-6 {
                continue;
            }
            let (x, y, z) = ((x / w) as f32, (y / w) as f32, (z / w) as f32);
            if z > 0.0 && x.is_finite() && y.is_finite() {
                all.push([x, y, z]);
            }
        }
    }
    let step = (all.len() / 200_000).max(1);
    Ok(if step > 1 {
        all.into_iter().step_by(step).collect()
    } else {
        all
    })
}
