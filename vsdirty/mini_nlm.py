import vapoursynth as vs
from typing import Optional
from vstools import PlanesT

core = vs.core


def mini_NLM(
    clip: vs.VideoNode,
    planes: PlanesT = [0, 1, 2],
    tr: int = 1,
    accel: Optional[str] = None,
    ref: Optional[vs.VideoNode] = None,
    dither: Optional[str] = "error_diffusion",
    **kwargs,
) -> vs.VideoNode:
    """
    NLM mini wrapper.

    :param clip:            Clip to process (32bit, if not will be internally converted in 32bit).
    :param planes:          Which planes to process. Defaults to all planes.
    :param tr:              Temporal radius same as d in vszip.
    :param accel:           Choose the acceleration. Accepted values: "cuda", "cl", "cpu", "auto".
    :param ref:             Reference clip for NLM (32bit, if not will be internally converted in 32bit).
    :param dither:          Dithering method for the output clip. If None, no dithering is applied.
    :param kwargs:          Accepts vszip arguments. Some accepted arguments are:
                            - a (int): Spatial search radius (Default: 2).
                            - s (int): Patch radius (similarity window) (Default: 4).
                            - h (float): Filtering strength. Higher removes more noise and detail (Default: 1.2).
                            - wmode (int): Weight function applied to the patch distance (Default: 0).
                            - wref (float): Weight of the reference pixel in the average (Default: 1.0).
    :return:                Denoised clip.
    """
    from vstools import depth
    from .adutils import plane

    if isinstance(planes, int):
        planes = [planes]
    planes = list(dict.fromkeys(int(p) for p in planes))

    clipS = depth(clip, 32, dither_type="none")

    if ref is not None:
        refS = depth(ref, 32, dither_type="none")
    else:
        refS = None

    def _nlm(
        clip: vs.VideoNode,
        accel: Optional[str] = "AUTO",
        rclip: Optional[vs.VideoNode] = None,
        **kwargs,
    ) -> vs.VideoNode:
        accel_u = accel.upper() if accel is not None else "AUTO"

        if accel_u not in ("AUTO", "CL", "CUDA", "CPU"):
            raise ValueError(f"Accel unknown: {accel}")

        if accel_u in ("AUTO", "CUDA"):
            try:
                clip = core.vszipcu.NLMeans(clip, d=tr, rclip=rclip, **kwargs)
            except Exception:
                print("mini_NLM: FALLBACK TO CL ACCELERATION")
                try:
                    clip = core.vszipcl.NLMeans(clip, d=tr, rclip=rclip, **kwargs)
                except Exception:
                    print("mini_NLM: FALLBACK TO CPU")
                    clip = core.nlm_ispc.NLMeans(clip, d=tr, rclip=rclip, **kwargs)
        elif accel_u == "CL":
            try:
                clip = core.vszipcl.NLMeans(clip, d=tr, rclip=rclip, **kwargs)
            except Exception:
                print("mini_NLM: FALLBACK TO CPU")
                clip = core.nlm_ispc.NLMeans(clip, d=tr, rclip=rclip, **kwargs)
        elif accel_u == "CPU":
            clip = core.nlm_ispc.NLMeans(clip, d=tr, rclip=rclip, **kwargs)

        return clip

    is_444 = clip.format.subsampling_w == 0 and clip.format.subsampling_h == 0
    process_all = planes == list(range(clip.format.num_planes))

    if process_all and (
        clip.format.color_family in (vs.GRAY, vs.RGB)
        or (clip.format.color_family == vs.YUV and is_444)
    ):
        dclip = _nlm(clipS, accel, rclip=refS, **kwargs)
    else:
        dclip = core.std.ShufflePlanes(
            [
                _nlm(
                    plane(clipS, p),
                    accel,
                    rclip=plane(refS, p) if refS is not None else None,
                    **kwargs,
                )
                if p in planes
                else plane(clipS, p)
                for p in range(clip.format.num_planes)
            ],
            planes=[0] * clip.format.num_planes,
            colorfamily=clip.format.color_family,
        )

    return depth(
        dclip,
        clip.format.bits_per_sample,
        dither_type=dither if dither is not None else "none",
    )
