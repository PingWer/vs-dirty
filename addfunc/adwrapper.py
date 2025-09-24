import vapoursynth as vs
core = vs.core
from typing import Optional
from vstools import depth, PlanesT, get_y, get_v, get_u, get_r, get_g, get_b

def _bm3d (
    clip: vs.VideoNode,
    accel: Optional[str],
    **kwargs
) -> vs.VideoNode:
    accel_u = None if accel is None else (accel.upper() if isinstance(accel, str) else str(accel).upper())

    if accel is None or accel_u == "CUDA_RTC":
        try:
            return core.bm3dcuda_rtc.BM3Dv2(clip, **kwargs, fast=False)
        except Exception:
            try:
                return core.bm3dhip.BM3Dv2(clip, **kwargs, fast=False)
            except Exception:
                return core.bm3dcpu.BM3Dv2(clip, **kwargs)
    elif accel_u == "CUDA":
        return core.bm3dcuda.BM3Dv2(clip, **kwargs, fast=False)
    elif accel_u == "HIP":
        return core.bm3dhip.BM3Dv2(clip, **kwargs, fast=False)
    elif accel_u == "CPU":
        return core.bm3dcpu.BM3Dv2(clip, **kwargs)
    else:
        # fallback
        return core.bm3dcpu.BM3Dv2(clip, **kwargs)


def mini_BM3D(
    clip: vs.VideoNode, 
    profile: str = "LC", 
    accel: Optional[str] = None,
    ref_gen: bool = False,
    planes: PlanesT = None,
    **kwargs
) -> vs.VideoNode:
    """
    BM3D mini wrapper.

    :param clip:            Clip to process. Must be 32 bit float format.
    :param profile:         Precision. Accepted values: "FAST", "LC", "HIGH".
    :param accel:           Choose the hardware acceleration. Accepted values: "cuda_rtc", "cuda", "hip", "cpu".
    :param ref_gen:         Generate a reference clip for block-matching, you can also pass ref with kwargs.
                            If true while you passed a ref clip it will be used for the new ref.
    :param planes:          Which planes to process. Defaults to all planes.
    :return:                Denoised clip.
    """
    if clip.format.bits_per_sample != 32:
        clipS = depth(clip, 32)
    else:
        clipS = clip

    profile_u = profile.upper() if isinstance(profile, str) else str(profile).upper()
    if profile_u == "FAST":
        block_step = [8,7,8,7]
        bm_range = [9,9,7,7]
        ps_range = [4,5]
    elif profile_u == "LC":
        block_step = [6,5,6,5]
        bm_range = [9,9,9,9]
        ps_range = [4,5]
    elif profile_u == "HIGH":
        block_step = [3,2,3,2]
        bm_range = [16,16,16,16]
        ps_range = [7,8]
    else:
        raise ValueError("mini_BM3D: Profile not recognized.")

    kwargs = dict(kwargs, block_step=block_step, bm_range=bm_range, ps_range=ps_range)

    num_planes = clip.format.num_planes
    if clip.format.color_family == vs.GRAY:
        return depth(_bm3d(clipS, accel, **kwargs), clip.format.bits_per_sample)

    if planes is None:
        return depth(_bm3d(clipS, accel, **kwargs), clip.format.bits_per_sample)

    if isinstance(planes, int):
        planes = [planes]
    planes = list(dict.fromkeys(int(p) for p in planes))

    if planes == [0,1,2]:
        return depth(_bm3d(clipS, accel, **kwargs), clip.format.bits_per_sample)
    
    if clip.format.color_family == vs.RGB:
        get_plane = [get_r, get_g, get_b]
    elif clip.format.color_family == vs.YUV:
        get_plane = [get_y, get_u, get_v]
    else:
        raise ValueError("mini_BM3D: Unsupported color family.")

    filtered_planes = [
        _bm3d(get_plane[i](clipS), accel, **kwargs) if i in planes and 0 <= i < num_planes else get_plane[i](clipS)
        for i in range(num_planes)
    ]
    return depth(core.std.ShufflePlanes(filtered_planes, planes=[0, 0, 0], colorfamily=clip.format.color_family), clip.format.bits_per_sample)
