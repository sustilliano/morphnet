#!/usr/bin/env python3
import argparse, json, glob
from pathlib import Path

try:
    from morphnet import guided_patch
except Exception as e:
    raise SystemExit(f"Import error: install morphnet wheel first. {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vipe_results", required=True, help="Path to vipe_results/")
    ap.add_argument("--out", required=True, help="Output directory for morphnet artifacts")
    ap.add_argument("--max_iters", type=int, default=2)
    ap.add_argument("--aspect_ratio_max", type=float, default=4.0)
    ap.add_argument("--curvature_lambda", type=float, default=0.1)
    ap.add_argument("--area_prior", type=float, default=0.0)
    args = ap.parse_args()

    vipe = Path(args.vipe_results)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    depth_files = sorted(glob.glob(str(vipe / "depth" / "*.npy")))
    flow_files = sorted(glob.glob(str(vipe / "flow" / "*.npy")))

    for i, dpth in enumerate(depth_files):
        flow = flow_files[i] if i < len(flow_files) else None
        ellipse, spline, quality = guided_patch(
            dpth, flow, args.max_iters,
            args.aspect_ratio_max, args.curvature_lambda, args.area_prior
        )
        payload = {
            "frame_index": i,
            "ellipse_params": {"cx": ellipse[0], "cy": ellipse[1], "a": ellipse[2], "b": ellipse[3], "theta": ellipse[4]},
            "spline": [{"x": p[0], "y": p[1]} for p in spline],
            "quality": quality,
        }
        (out / f"morphnet_{i:06d}.json").write_text(json.dumps(payload))

    print(f"[morphnet] wrote {len(depth_files)} json files to {out}")

if __name__ == "__main__":
    main()
