# vs-dirty

A collection of VapourSynth wrappers and utility functions focused on advanced denoising, masking, and edge fixing.

## Installation

You can install `vsdirty` via pip:

```bash
pip install vsdirty
```

Or build from source:

```bash
git clone https://github.com/r74mi/vs-dirty.git
cd vs-dirty
pip install .
```

## Dependencies

This package relies on several external VapourSynth plugins. Ensure these are installed and available in your VapourSynth plugins folder.

| Plugin | URL |
| :--- | :--- |
| **fmtc** | [GitLab](https://gitlab.com/EleonoreMizo/fmtconv/) |
| **akarin** | [GitHub](https://github.com/AkarinVS/vapoursynth-plugin) |
| **cas** | [GitHub](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-CAS) |
| **bore** | [GitHub](https://github.com/opusgang/libbore) |
| **mvtools** | [GitHub](https://github.com/dubhater/vapoursynth-mvtools) |
| **BM3DCuda** | [GitHub](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA) |

## Key Functions

### `adenoise`

Designed specifically for film content, this intensive adaptive denoiser removes noise while carefully preserving the underlying grain structure and fine textures, ensuring the image retains its organic look without becoming "plastic" or over-smoothed. It combines `mc_degrain` (luma), `NLMeans`/`CBM3D` (chroma), and `BM3D` (luma), all modulated by adaptive masking.

**Presets:**
- `scan65mm` (Light)
- `scan35mm` (Medium)
- `scan16mm` (Heavy)
- `scan8mm` (Very Heavy)
- `digital` (Optimized for digital sources)
- `default` (Generic)

**Usage:**

```python
import vapoursynth as vs
from vsdirty import adenoise

core = vs.core
clip = ... # your video source

# Apply default adaptive denoising
denoised = adenoise.scan16mm(clip)
```

### `bore`

A powerful edge cleaner (dirty line fixer) that processes borders at their native resolution to avoid scaling artifacts.

```python
from vsdirty import bore

# Fix dirty lines: Top=1px, Bottom=1px, Left=2px, Right=2px
clean = bore(clip, ythickness=[1, 1, 2, 2])
```

### `msaa2x`

An antialiaser based on ArtCNN that targets specific edges to reduce aliasing without blurring textures.

```python
from vsdirty import adfunc

# Apply antialiasing
aa_clip = adfunc.msaa2x(clip)
```

### `advanced_edgemask`

Generates a high-quality edge mask by combining Retinex preprocessing with multiple edge detectors (Kirsch, Sobel) to capture faint and complex edges.

```python
from vsdirty import admask

emask = admask.advanced_edgemask(clip, luma_scaling=10)
```

### `hd_flatmask`

A specialized mask for flat areas, useful for protecting textures or targeting specific flat regions for filtering.

```python
from vsdirty import admask

flat_mask = admask.hd_flatmask(clip)
```

### `diff_and_swap`

A utility to repair damaged frames in a "base" clip using a "correction" clip. It compares frames and swaps them if the difference exceeds a threshold.

```python
from vsdirty import adutils

# automated patching
repaired, _ = adutils.diff_and_swap(correction_clip, base_clip, thr=30000)
```

## License

MIT License
