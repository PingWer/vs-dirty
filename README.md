# vs-dirty

A collection of VapourSynth wrappers and utility functions focused on advanced denoising, masking, and edge fixing.

Follow the docstring for more information about the parameters.

## Installation

You can install `vsdirty` via pip:

```bash
pip install vsdirty
```

If you want to enable GPU acceleration for neural networks, install the package with the appropriate extra based on your hardware:

* **NVIDIA (TensorRT):**
  ```bash
  pip install vsdirty[nvidia]
  ```
* **AMD (DirectML / ORT):**
  ```bash
  pip install vsdirty[amd]
  ```
* **OpenCL (Generic):**
  ```bash
  pip install vsdirty[cl]
  ```

Or build from source:

```bash
git clone https://github.com/PingWer/vs-dirty
cd vs-dirty
pip install .[nvidia] # or .[amd], .[cl]
```

## Dependencies

This package relies on several external VapourSynth plugins. Most of them are automatically downloaded via pip, but ensure they are available in your VapourSynth environment:

| Plugin | Used For |
| :--- | :--- |
| **vapoursynth-edgemasks** | High-performance convolution masks (`Kroon`, `Sobel`, `Kirsch`, `Prewitt`) |
| **vapoursynth-cas** | Contrast Adaptive Sharpening |
| **vapoursynth-mvtools** | Motion interpolation and analysis |
| **vapoursynth-vszipcu** | Primary NLM and BM3D implementation (CUDA) |
| **vapoursynth-vszipcl** | Primary NLM and BM3D implementation (OpenCL) |
| **vapoursynth-bm3d** | BM3D implementation fallback (CUDA / HIP) |
| **vapoursynth-bm3dcpu** | BM3D implementation fallback (CPU) |
| **vapoursynth-nlm-ispc** | NLM implementation fallback (CPU) |
| **vapoursynth-akarin** | Fast mathematical expression evaluation (`Expr`) |
| **libbore** | Fix frame edges (`dirtyfixer.bore`) |

> **Note:** Currently, **libbore** is not available via pip. You will need to download and install it manually from its [GitHub repository](https://github.com/OpusGang/bore).

## License

MIT License
