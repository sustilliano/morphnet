# morphnet (Python wrapper)

Build (dev):
```bash
pip install maturin
cd bindings/python
maturin develop --release

API:

from morphnet import guided_patch
ellipse, spline, quality = guided_patch("path/to/depth.npy", None, 2, 4.0, 0.1, 0.0)
