# vs-dirty

A collection of VapourSynth wrappers and utility functions focused on advanced denoising, masking, and edge fixing.

Follow the docstring for more information about the parameters.

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
| **bore** | [GitHub](https://github.com/OpusGang/bore) |
| **mvtools** | [GitHub](https://github.com/dubhater/vapoursynth-mvtools) |
| **BM3DCuda** | [GitHub](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA) |
| **nlm-cuda** | [GitHub](https://github.com/AmusementClub/vs-nlm-cuda) |
| **vsmlrt** | [GitHub](https://github.com/AmusementClub/vs-mlrt) |

## `adenoise`

Designed specifically for film content, this intensive adaptive denoiser removes noise while carefully preserving fine textures and small elements. It combines `mc_degrain` (luma), `NLMeans`/`CBM3D` (chroma), and `BM3D` (luma), all modulated by adaptive masking.

**Presets:**
- `scan65mm` (Light)
- `scan35mm` (Medium)
- `scan16mm` (Heavy)
- `scan8mm` (Very Heavy)
- `digital` (Optimized for digital sources)
- `default` (Generic)

**Parameters:**

```python
adenoise.default(
  clip: vs.VideoNode, 
  thsad: int = 500, 
  tr: int = 2, 
  sigma: float = 6, 
  luma_mask_weaken: float = 0.75, 
  luma_mask_thr: float = 0.196, 
  chroma_denoise: float | str | tuple[float, str] = [1.0, "nlm"], 
  precision: bool = True, 
  chroma_masking: bool = False, 
  luma_over_texture: float = 0.4, 
  kwargs_flatmask: dict = {},
  show_mask: int = 0
)
```

- `clip`: Clip to process (YUV or GRAY).
- `thsad`: Luma and chroma ref denoise strength.
  Recommended values: 300-800
- `tr`: Temporal radius for temporal consistency.
  Recommended values: 2-3 (1 means no temporal denoise).
- `sigma`: Sigma for BM3D (luma denoise strength).
  Recommended values: 2-10. 
- `luma_mask_weaken`: Controls how much dark spots should be denoised. Lower values mean stronger overall denoise.
  Recommended values: 0.6-0.9
- `luma_mask_thr`: Threshold that determines what is considered bright and what is dark in the luma mask.
  Recommended values: 0.15-0.25
- `chroma_denoise`: Denoiser strength and type for chroma. NLMeans/CBM3D/ArtCNN.
  Reccomended strength values: 0.5-2.
  Accepted denoiser types: "nlm", "cbm3d", "artcnn".
- `precision`: If True, a flat mask is created to enhance the denoise strenght avoiding textured area.
- `chroma_masking`: If True, enables specific chroma masking for U/V planes.
- `luma_over_texture`: Lower value means more importance to textured areas, higher value means more importance to luma levels.
  Accepted values: 0.0-1.0
- `kwargs_flatmask`: Additional arguments. dict values (check `hd_flatmask`'s docstring for more info):
  - `sigma1`: This value should be decided based on the details level of the clip and how much grain and noise is present. Recommended values: 1-5
  - `texture_strength`: Texture strength for mask (0-inf). Values above 1 decrease the strength of the texture in the mask.
  - `edges_strength`: Edges strength for mask (0-1). Basic multiplier for edges strength.
- `show_mask`: 1 = First luma mask, 2 = Textured luma mask, 3 = Complete luma mask, 4 = Show the Chroma U Plane mask (if chroma_masking = True), 5 = Show the Chroma V Plane mask (if chroma_masking = True).

- `return`: 16bit denoised clip. If show_mask is set, returns a tuple (denoised_clip, mask).

**Usage:**

```python
from vsdirty import adenoise

denoised = adenoise.scan16mm(clip)
```

## `bore`

A powerful edge cleaner (dirty line fixer) that processes borders at their native resolution to avoid scaling artifacts.

**Parameters:**

```python
def bore(
  clip : vs.VideoNode,
  ythickness: List[int] = None,
  uthickness: List[int] = None,
  vthickness: List[int] = None,
  planes: PlanesT = 0,
  singlePlane = True
)
```

- `clip`: Input clip (YUV or GRAY).
- `ythickness`: Luma border thicknesses to process. [top, bottom, left, right].
- `uthickness`: Chroma U border thicknesses to process. [top, bottom, left, right].
- `vthickness`: Chroma V border thicknesses to process. [top, bottom, left, right]. If None, uses ythickness or uthickness.
- `planes`: Plane(s) to process.
- `singlePlane`: If True uses bore.SinglePlane, otherwise bore.MultiPlane. MultiPlane works only on 444 clips.
- `return`: Processed clip with corrected borders, same format as input.

**Usage:**

```python
from vsdirty import bore

# Fix dirty lines: Top=1px, Bottom=1px, Left=2px, Right=2px
clean = bore(clip, ythickness=[1, 1, 2, 2])
```

## `msaa2x`

An antialiaser based on ArtCNN that targets specific edges to reduce aliasing without blurring textures.

```python
from vsdirty import adfunc

# Apply antialiasing
aa_clip = adfunc.msaa2x(clip)
```

## `advanced_edgemask`

Generates a high-quality edge mask by combining Retinex preprocessing with multiple edge detectors (Kirsch, Sobel) to capture faint and complex edges.

```python
from vsdirty import admask

emask = admask.advanced_edgemask(clip, luma_scaling=10)
```

## `hd_flatmask`

A specialized mask for flat areas, useful for protecting textures or targeting specific flat regions for filtering.

```python
from vsdirty import admask

flat_mask = admask.hd_flatmask(clip)
```

## `diff_and_swap`

A utility to repair damaged frames in a "base" clip using a "correction" clip. It compares frames and swaps them if the difference exceeds a threshold.

```python
from vsdirty import adutils

# automated patching
repaired, _ = adutils.diff_and_swap(correction_clip, base_clip, thr=30000)
```

## License

MIT License
