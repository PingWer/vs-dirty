import vapoursynth as vs
from typing import Optional
from vstools import PlanesT

core = vs.core

def mini_BM3D(
    clip: vs.VideoNode,
    planes: PlanesT = [0, 1, 2],
    profile: str = "LC",
    accel: Optional[str] = None,
    ref: Optional[vs.VideoNode] = None,
    dither: Optional[str] = "error_diffusion",
    fast_fused: Optional[bool] = True,
    **kwargs,
) -> vs.VideoNode:
    """
    BM3D mini wrapper.

    :param clip:            Clip to process (32bit, if not will be internally converted in 32bit).
    :param planes:          Which planes to process. Defaults to all planes.
    :param profile:         Precision. Accepted values: "FAST", "LC", "HIGH".
    :param accel:           Choose the hardware acceleration. Accepted values: "cuda", "hip", "cpu", "auto".
    :param ref:             Reference clip for BM3D (32bit, if not will be internally converted in 32bit).
    :param dither:          Dithering method for the output clip. If None, no dithering is applied.
    :param fast_fused:      Runs the collaborative filter and the temporal aggregation as one kernel chain, increases VRAM usage.
    :param kwargs:          Accepts vszipcu arguments, https://github.com/dnjulek/vapoursynth-zipcu.
    :return:                Denoised clip.
    """
    from vstools import depth
    from .adutils import plane

    def _conversion_to_444(clip: vs.VideoNode) -> vs.VideoNode:
        if "444" not in str(clip.format.name):
            y = plane(clip, 0)
            u = plane(clip, 1)
            v = plane(clip, 2)
            y_downscaled = y.resize.Bilinear(u.width, u.height)
            return core.std.ShufflePlanes(
                [y_downscaled, u, v], planes=[0, 0, 0], colorfamily=vs.YUV
            )
        else:
            return clip

    def _bm3d(
        clip: vs.VideoNode,
        accel: Optional[str] = "AUTO",
        ref: Optional[vs.VideoNode] = None,
        **kwargs,
    ) -> vs.VideoNode:
        accel_u = accel.upper() if accel is not None else "AUTO"

        if accel_u not in ("AUTO", "CL", "CUDA", "HIP", "CPU"):
            raise ValueError(f"Accel unknown: {accel}")

        if accel_u in ("AUTO", "CUDA"):
            try:
                if profile in ["HIGH", "LC"]:
                    kwargs["fast_fused"] = False
                return core.vszipcu.BM3Dv2(clip, ref, **kwargs)
            except Exception:
                kwargs.pop("fast_fused", None)
                try:
                    return core.bm3dhip.BM3Dv2(clip, ref, **kwargs)
                except Exception:
                    kwargs.pop("ps_range", kwargs.get("ps_range")[0])
                    try:
                        return core.bm3dcpu.BM3Dv2(clip, ref, **kwargs)
                    except Exception:
                        return core.bm3dneon.BM3Dv2(clip, ref, **kwargs)
        elif accel_u == "CL":
            kwargs.pop("fast_fused", None)
            return core.vszipcl.BM3Dv2(clip, ref, **kwargs)
        elif accel_u == "HIP":
            kwargs.pop("fast_fused", None)
            return core.bm3dhip.BM3Dv2(clip, ref, **kwargs)
        elif accel_u == "CPU":
            kwargs.pop("fast_fused", None)
            kwargs.pop("ps_range", kwargs.get("ps_range")[0])
            try:
                return core.bm3dcpu.BM3Dv2(clip, ref, **kwargs)
            except Exception:
                return core.bm3dneon.BM3Dv2(clip, ref, **kwargs)

    clipS = depth(clip, 32, dither_type="none")

    if ref is not None:
        refS = depth(ref, 32, dither_type="none")
    else:
        refS = None

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
        "FLATMASK": {
            "block_step": [3, 2, 3, 2],
            "bm_range": [3, 3, 1, 1],
            "ps_num": 1,
            "ps_range": [1, 2],
        },
    }

    profile_u = str(profile).upper()

    if profile_u not in profiles:
        raise ValueError(f"mini_BM3D: Profile '{profile}' not recognized.")

    params = profiles[profile_u]

    kwargs = dict(kwargs, fast_fused=fast_fused, **params)

    if clip.format.color_family == vs.GRAY:
        return (
            depth(_bm3d(clipS, accel, refS, **kwargs), clip.format.bits_per_sample)
            if refS is not None
            else depth(_bm3d(clipS, accel, **kwargs), clip.format.bits_per_sample)
        )

    if isinstance(planes, int):
        planes = [planes]
    planes = list(dict.fromkeys(int(p) for p in planes))

    if clip.format.color_family == vs.RGB:
        clipOPP = core.fmtc.matrix(
            clipS,
            fulls=True,
            fulld=True,
            coef=[1 / 3, 1 / 3, 1 / 3, 0, 1 / 2, 0, -1 / 2, 0, 1 / 4, -1 / 2, 1 / 4, 0],
            col_fam=vs.YUV,
        )
        if refS is not None:
            refOPP = core.fmtc.matrix(
                refS,
                fulls=True,
                fulld=True,
                coef=[
                    1 / 3,
                    1 / 3,
                    1 / 3,
                    0,
                    1 / 2,
                    0,
                    -1 / 2,
                    0,
                    1 / 4,
                    -1 / 2,
                    1 / 4,
                    0,
                ],
                col_fam=vs.YUV,
            )
        else:
            refOPP = None

        dclip = (
            _bm3d(clipOPP, accel, refOPP, **kwargs)
            if refOPP is not None
            else _bm3d(clipOPP, accel, **kwargs)
        )
        dclip = core.fmtc.matrix(
            dclip,
            fulls=True,
            fulld=True,
            coef=[1, 1, 2 / 3, 0, 1, 0, -4 / 3, 0, 1, -1, 2 / 3, 0],
            col_fam=vs.RGB,
        )
        dclip = core.std.ShufflePlanes(
            [
                dclip if 0 in planes else clipS,
                dclip if 1 in planes else clipS,
                dclip if 2 in planes else clipS,
            ],
            planes=[0, 1, 2],
            colorfamily=vs.RGB,
        )

    elif clip.format.color_family == vs.YUV:
        y = plane(clipS, 0)
        u = plane(clipS, 1)
        v = plane(clipS, 2)

        y_ref = None
        if refS is not None:
            if refS.format.num_planes not in (1, 3):
                raise ValueError(
                    "mini_BM3D: When providing a reference clip for YUV, it must have 1 or 3 planes."
                )
            y_ref = plane(refS, 0)

        y_denoised = _bm3d(y, accel, ref=y_ref, **kwargs) if 0 in planes else y

        if 1 in planes or 2 in planes:
            if refS is not None and refS.format.num_planes == 3:
                ref_444 = _conversion_to_444(refS)
            elif refS is not None and refS.format.num_planes == 1:
                ref_444 = y_ref
            else:
                ref_444 = None

            clip_444 = _conversion_to_444(clipS)
            clip_444_denoised = _bm3d(
                clip_444, accel, ref=ref_444, chroma=True, **kwargs
            )

            if 1 in planes:
                u = plane(clip_444_denoised, 1)
            if 2 in planes:
                v = plane(clip_444_denoised, 2)

        dclip = core.std.ShufflePlanes(
            [y_denoised, u, v], planes=[0, 0, 0], colorfamily=clip.format.color_family
        )

    else:
        raise ValueError("mini_BM3D: Unsupported color family.")

    return depth(
        dclip,
        clip.format.bits_per_sample,
        dither_type=dither if dither is not None else "none",
    )
