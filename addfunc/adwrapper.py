import vapoursynth as vs
core = vs.core
from typing import Optional
from vstools import depth, PlanesT, get_y, get_v, get_u, get_r, get_g, get_b

def mini_BM3D(
    clip: vs.VideoNode, 
    profile: str = "LC", 
    accel: Optional[str] = None,
    planes: PlanesT = None,
    **kwargs
) -> vs.VideoNode:
    """
    BM3D mini wrapper.

    :param clip:            Clip to process. Must be 32 bit float format.
    :param profile:         Precision. Accepted values: "FAST", "LC", "HIGH".
    :param accel:           Choose the hardware acceleration. Accepted values: "cuda_rtc", "cuda", "hip", "cpu", "auto".
    :param planes:          Which planes to process. Defaults to all planes.
    :param kwargs:          Accepts BM3DCUDA arguments, https://github.com/WolframRhodium/VapourSynth-BM3DCUDA.
    :return:                Denoised clip.
    """

    def _bm3d (
        clip: vs.VideoNode,
        accel: Optional[str] = "AUTO",
        **kwargs
    ) -> vs.VideoNode:
        accel_u = accel.upper() if accel is not None else "AUTO"

        if accel_u not in ("AUTO", "CUDA_RTC", "CUDA", "HIP", "CPU"):
            raise ValueError(f"Accel unknown: {accel}")

        if accel_u in ("AUTO", "CUDA_RTC"):
            try:
                return core.bm3dcuda_rtc.BM3Dv2(clip, **kwargs)
            except Exception:
                try:
                    return core.bm3dhip.BM3Dv2(clip, **kwargs)
                except Exception:
                    kwargs.pop("fast", None)
                    return core.bm3dcpu.BM3Dv2(clip, **kwargs)
        elif accel_u == "CUDA":
            return core.bm3dcuda.BM3Dv2(clip, **kwargs)
        elif accel_u == "HIP":
            return core.bm3dhip.BM3Dv2(clip, **kwargs)
        elif accel_u == "CPU":
            kwargs.pop("fast", None)
            return core.bm3dcpu.BM3Dv2(clip, **kwargs)
    
    if clip.format.bits_per_sample != 32:
        clipS = depth(clip, 32)
    else:
        clipS = clip

    profiles = {
        "FAST": {
            "block_step": [8, 7, 8, 7],
            "bm_range": [9, 9, 7, 7],
            "ps_range": [4, 5],
        },
        "LC": {
            "block_step": [6, 5, 6, 5],
            "bm_range": [9, 9, 9, 9],
            "ps_range": [4, 5],
        },
        "HIGH": {
            "block_step": [3, 2, 3, 2],
            "bm_range": [16, 16, 16, 16],
            "ps_range": [7, 8],
        },
    }

    profile_u = profile.upper() if isinstance(profile, str) else str(profile).upper()

    if profile_u not in profiles:
        raise ValueError(f"mini_BM3D: Profile '{profile}' not recognized.")

    params = profiles[profile_u]

    kwargs = dict(
        kwargs,
        **params,
        fast=False
    )

    num_planes = clip.format.num_planes
    if clip.format.color_family == vs.GRAY:
        return depth(_bm3d(clipS, accel, **kwargs), clip.format.bits_per_sample)

    if planes is None:
        planes = [0, 1, 2]
    if isinstance(planes, int):
        planes = [planes]
    planes = list(dict.fromkeys(int(p) for p in planes))

    if clip.format.color_family == vs.RGB:
        get_plane = [get_r, get_g, get_b]
        filtered_planes = [
            _bm3d(get_plane[i](clipS), accel, **kwargs) if i in planes and 0 <= i < num_planes else get_plane[i](clipS)
            for i in range(num_planes)
        ]
        dclip = core.std.ShufflePlanes(filtered_planes, planes=[0, 0, 0], colorfamily=clip.format.color_family)

    elif clip.format.color_family == vs.YUV:
        y = get_y(clipS)
        u = get_u(clipS)
        v = get_v(clipS)

        yd = _bm3d(y, accel, **kwargs) if 0 in planes else y

        if 1 in planes or 2 in planes:
            y_resized = y.resize.Bicubic(u.width, u.height, filter_param_a=0, filter_param_b=0)
            clip444 = core.std.ShufflePlanes([y_resized, u, v], planes=[0, 0, 0], colorfamily=clip.format.color_family)
            clip444 = _bm3d(clip444, accel, chroma=True, **kwargs)
            if 1 in planes:
                u = get_u(clip444)
            if 2 in planes:
                v = get_v(clip444)

        dclip = core.std.ShufflePlanes([yd, u, v], planes=[0, 0, 0], colorfamily=clip.format.color_family)

    else:
        raise ValueError("mini_BM3D: Unsupported color family.")
    
    return depth(dclip, clip.format.bits_per_sample, dither_type="sierra_2_4a")


