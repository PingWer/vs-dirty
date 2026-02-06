import vapoursynth as vs

from typing import Optional
from vstools import PlanesT

core = vs.core

if not (hasattr(core, 'fmtc') or hasattr(core, 'akarin')):
    raise ImportError("'fmtc' and 'akarin' are mandatory. Make sure the DLLs are present in the plugins folder.")

def mini_BM3D(
    clip: vs.VideoNode,
    planes: PlanesT = [0, 1, 2],
    profile: str = "LC", 
    accel: Optional[str] = None,
    ref: Optional[vs.VideoNode] = None,
    dither: Optional[str] = "error_diffusion",
    fast: Optional[bool] = False,
    **kwargs
) -> vs.VideoNode:
    """
    BM3D mini wrapper.

    :param clip:            Clip to process (32bit, if not will be internally converted in 32bit).
    :param planes:          Which planes to process. Defaults to all planes.
    :param profile:         Precision. Accepted values: "FAST", "LC", "HIGH".
    :param accel:           Choose the hardware acceleration. Accepted values: "cuda_rtc", "cuda", "hip", "cpu", "auto".
    :param ref:             Reference clip for BM3D (32bit, if not will be internally converted in 32bit).
    :param dither:          Dithering method for the output clip. If None, no dithering is applied.
    :param fast:            Use CPU+GPU, adds overhead.
    :param kwargs:          Accepts BM3DCUDA arguments, https://github.com/WolframRhodium/VapourSynth-BM3DCUDA.
    :return:                Denoised clip.
    """
    from vstools import depth
    from .adutils import plane
    
    def _bm3d (
        clip: vs.VideoNode,
        accel: Optional[str] = "AUTO",
        ref: Optional[vs.VideoNode] = None,
        **kwargs
    ) -> vs.VideoNode:
        accel_u = accel.upper() if accel is not None else "AUTO"

        if accel_u not in ("AUTO", "CUDA_RTC", "CUDA", "HIP", "CPU"):
            raise ValueError(f"Accel unknown: {accel}")
        
        if accel_u in ("AUTO", "CUDA_RTC"):
            try:
                return core.bm3dcuda_rtc.BM3Dv2(clip, ref, **kwargs)
            except Exception:
                try:
                    return core.bm3dhip.BM3Dv2(clip, ref, **kwargs)
                except Exception:
                    kwargs.pop("fast", None)
                    kwargs.pop("ps_range", kwargs.get("ps_range")[0])
                    try:
                        return core.bm3dcpu.BM3Dv2(clip, ref, **kwargs)
                    except Exception:
                        return core.bm3dneon.BM3Dv2(clip, ref, **kwargs)
        elif accel_u == "CUDA":
            return core.bm3dcuda.BM3Dv2(clip, ref, **kwargs)
        elif accel_u == "HIP":
            return core.bm3dhip.BM3Dv2(clip, ref, **kwargs)
        elif accel_u == "CPU":
            kwargs.pop("fast", None)
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
    }

    profile_u = str(profile).upper()

    if profile_u not in profiles:
        raise ValueError(f"mini_BM3D: Profile '{profile}' not recognized.")

    params = profiles[profile_u]

    kwargs = dict(
        kwargs,
        fast=fast,
        **params
    )

    num_planes = clip.format.num_planes
    if clip.format.color_family == vs.GRAY:
        return depth(_bm3d(clipS, accel, refS, **kwargs), clip.format.bits_per_sample) if refS is None else depth(_bm3d(clipS, accel, **kwargs), clip.format.bits_per_sample)

    if isinstance(planes, int):
        planes = [planes]
    planes = list(dict.fromkeys(int(p) for p in planes))

    if clip.format.color_family == vs.RGB:
        filtered_planes = [
            _bm3d(plane(clipS, p), accel, **kwargs) if p in planes else plane(clipS, p)
            for p in range(num_planes)
        ]
        dclip = core.std.ShufflePlanes(filtered_planes, planes=[0, 0, 0], colorfamily=clip.format.color_family)

    elif clip.format.color_family == vs.YUV:
        y = plane(clipS, 0)
        u = plane(clipS, 1)
        v = plane(clipS, 2)

        y_ref = None
        if refS is not None:
            if refS.format.num_planes not in (1, 3):
                raise ValueError("mini_BM3D: When providing a reference clip for YUV, it must have 1 or 3 planes.")
            y_ref = plane(refS, 0)

        y_denoised = _bm3d(y, accel, ref=y_ref, **kwargs) if 0 in planes else y

        if 1 in planes or 2 in planes:
            y_downscaled = y.resize.Spline36(u.width, u.height)

            ref_444 = None
            if refS is not None and refS.format.num_planes == 3:
                u_ref = plane(refS, 1); v_ref = plane(refS, 2)
                y_ref_downscaled = y_ref.resize.Spline36(u.width, u.height)
                ref_444 = core.std.ShufflePlanes([y_ref_downscaled, u_ref, v_ref], planes=[0, 0, 0], colorfamily=clip.format.color_family)
            elif refS is not None and refS.format.num_planes == 1:
                ref_444 = y_ref

            clip_444 = core.std.ShufflePlanes([y_downscaled, u, v], planes=[0, 0, 0], colorfamily=clip.format.color_family)
            clip_444 = _bm3d(clip_444, accel, ref=ref_444, chroma=True, **kwargs) if ref_444 is not None else _bm3d(clip_444, accel, chroma=True, **kwargs)

            if 1 in planes:
                u = plane(clip_444, 1)
            if 2 in planes:
                v = plane(clip_444, 2)

        dclip = core.std.ShufflePlanes([y_denoised, u, v], planes=[0, 0, 0], colorfamily=clip.format.color_family)

    else:
        raise ValueError("mini_BM3D: Unsupported color family.")

    return depth(dclip, clip.format.bits_per_sample, dither_type=dither if dither is not None else "none")

class adenoise:
    """ 
    Preset class for _adaptive_denoiser.

    Intensive Adaptive Denoise.

    Three denoisers are applied: mc_degrain (luma), NLMeans/CBM3D (chroma), and BM3D (luma).
    NLMeans/CBM3D uses mc_degrain as reference to remove dirt spots and scanner noise from the clip,
    while mc_degrain affects only the luma, which is then passed to BM3D for a second denoising pass.
    If precision = True, a series of masks are created to enhance the denoise strength on flat areas avoiding textured area.

    Luma masks ensure that denoising is applied mostly to the brighter areas of the frame, preserving details in darker regions while cleaning them as much as possible.
    Note: Luma masks are more sensitive to variations than the sigma value for the final result.

    :param clip:                Clip to process (YUV or GRAY 16bit, if not will be internally converted in 16bit).
    :param thsad:               Thsad for mc_degrain (luma denoise strength and chroma ref).
                                Recommended values: 300-800
    :param tr:                  Temporal radius for temporal consistency across al the filter involved.
                                Recommended values: 2-3 (1 means no temporal denoise).
    :param sigma:               Sigma for BM3D (luma denoise strength).
                                Recommended values: 1-5. 
    :param luma_mask_weaken:    Controls how much dark spots should be denoised. Lower values mean stronger overall denoise.
                                Recommended values: 0.6-0.9
    :param luma_mask_thr:       Threshold that determines what is considered bright and what is dark in the luma mask.
                                Recommended values: 0.15-0.25
    :param chroma_denoise:      Denoiser strength and type for chroma. NLMeans/CBM3D/ArtCNN.
                                Reccomended strength values: 0.5-2. If not given, 1.0 is used (or none for ArtCNN).
                                Accepted denoiser types: "nlm", "cbm3d", "artcnn". If not given, nlm is used.
    :param precision:           If True, a flat mask is created to enhance the denoise strenght on flat areas avoiding textured area (95% accuracy).
    :param chroma_masking:      If True, enables specific chroma masking for U/V planes.
    :param luma_over_texture:   Multiplier for the luma mask in precision mode. Lower value means more importance to textured areas, higher value means more importance to luma levels.
                                Accepted values: 0.0-1.0
    :param kwargs_flatmask:     Additional arguments for flatmask creation.
                                dict values (check hd_flatmask for more info):
                                sigma1: This value should be decided based on the details level of the clip and how much grain and noise is present. Usually 1 for really textured clip, 2-3 for a normal clip, 4-5 for a clip with strong noise or grain.
                                texture_strength: Texture strength for mask (0-inf). Values above 1 decrese the strength of the texture in the mask, lower values increase it. The max value is theoretical infinite, but there is no gain after some point.
                                edges_strength: Edges strength for mask (0-1). Basic multiplier for edges strength.
    :param show_mask:           1 = Show the first luma mask, 2 = Show the textured luma mask, 3 = Show the complete luma mask, 4 = Show the Chroma U Plane mask (if chroma_masking = True), 5 = Show the Chroma V Plane mask (if chroma_masking = True). Any other value returns the denoised clip.

    :return:                    16bit denoised clip. If show_mask is 1, 2, 3, 4 or 5, returns a tuple (denoised_clip, mask).
    """

    @classmethod
    def _adaptive_denoiser(
        cls,
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
        kwargs_flatmask: Optional[dict] = {},
        **kwargs
    ) -> vs.VideoNode:
        
        from vstools import depth
        from vsdenoise import Prefilter, mc_degrain, nl_means, MVTools, SearchMode, MotionMode, SADMode, MVTools, SADMode, MotionMode
        from .admask import luma_mask_ping, luma_mask_man, hd_flatmask
        from .adutils import plane

        core = vs.core

        if clip.format.color_family not in {vs.YUV, vs.GRAY}:
            raise ValueError('adaptive_denoiser: only YUV and GRAY formats are supported')

        clip = depth(clip, 16, dither_type="none")

        lumamask = luma_mask_ping(clip, thr=luma_mask_thr)
        darken_luma_mask = core.akarin.Expr([lumamask], f"x {luma_mask_weaken} *")

        # Degrain
        if "is_digital" not in kwargs:
            # Mvtool initialization
            mvtools = MVTools(clip)
            vectors = mvtools.analyze(blksize=16, tr=tr, overlap=8, lsad=300, search=SearchMode.UMH, truemotion=MotionMode.SAD, dct=SADMode.MIXED_SATD_DCT)
            mfilter = mini_BM3D(clip, sigma=sigma*1.25, radius=tr, profile="LC", planes=0)
            degrain = mc_degrain(clip, prefilter=Prefilter.DFTTEST, blksize=8, mfilter=mfilter, thsad=thsad, vectors=vectors, tr=tr, limit=1)
        else:
            degrain = clip
            
        if precision:
            flatmask_defaults = {
                "sigma1": 3,
                "texture_strength": 2,
                "edges_strength": 0.05
            }
            flatmask = hd_flatmask(degrain, **(flatmask_defaults | kwargs_flatmask))

            if luma_over_texture > 1.0:
                raise ValueError("luma_over_texture must be less than 1")
            elif luma_over_texture < 0.0:
                raise ValueError("luma_over_texture must be greater than 0")
            elif luma_over_texture == 1:
                raise ValueError("don't use precision mode if luma_over_texture is 1")
            
            final_mask = core.akarin.Expr([darken_luma_mask, flatmask], f"x {luma_over_texture} * y {abs(luma_over_texture-1)} * +")
        else:
            final_mask = darken_luma_mask

        denoised = mini_BM3D(plane(degrain, 0), sigma=sigma, radius=tr, profile="HIGH")
        y_denoised = core.std.MaskedMerge(denoised, plane(clip, 0), final_mask) #denoise applied to darker areas

        if clip.format.color_family == vs.GRAY:
            return y_denoised
        
        # Chroma denoise
        if isinstance(chroma_denoise, str):
            chroma_denoise = [1.0, chroma_denoise]
        if (isinstance(chroma_denoise, float) and chroma_denoise <= 0) or (isinstance(chroma_denoise, tuple) and chroma_denoise[0] <= 0):
            chroma_denoised = clip
        else:
            if isinstance(chroma_denoise, float):
                chroma_denoised = nl_means(clip, h=chroma_denoise, tr=tr, ref=degrain, planes=[1,2])
            elif "nlm" in chroma_denoise:
                chroma_denoised = nl_means(clip, h=chroma_denoise[0], tr=tr, ref=degrain, planes=[1,2])
            elif "cbm3d" in chroma_denoise:
                chroma_denoised = mini_BM3D(clip, sigma=chroma_denoise[0], radius=tr, ref=degrain, planes=[1,2])
            elif "artcnn" in chroma_denoise:
                from vsscale import ArtCNN
                chroma_denoised = ArtCNN.R8F64_JPEG420().scale(clip)

            if chroma_masking:
                u=plane(clip, 1)
                u_mask= luma_mask_man(u, t=1.5, s=2, a=0)
                u_denoised = core.std.MaskedMerge(u, plane(chroma_denoised, 1), u_mask)
                v=plane(clip, 2)
                v_mask= luma_mask_man(v, t=1.5, s=2, a=0)
                v_denoised = core.std.MaskedMerge(v, plane(chroma_denoised, 2), v_mask)
                return core.std.ShufflePlanes(clips=[chroma_denoised, u_denoised, v_denoised], planes=[0,0,0], colorfamily=vs.YUV)
        
        return core.std.ShufflePlanes(clips=[y_denoised, chroma_denoised, chroma_denoised], planes=[0,1,2], colorfamily=vs.YUV)
    
    @classmethod
    def _adaptive_denoiser_tuple(
        cls,
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
        kwargs_flatmask: Optional[dict] = {},
        show_mask: int = 0,
        **kwargs
    ) -> tuple[vs.VideoNode, vs.VideoNode]:
        
        from vstools import depth
        from vsdenoise import Prefilter, mc_degrain, nl_means, MVTools, SearchMode, MotionMode, SADMode, MVTools, SADMode, MotionMode
        from .admask import luma_mask_ping, luma_mask_man, hd_flatmask
        from .adutils import plane

        core = vs.core
        
        selected_mask = None

        if clip.format.color_family not in {vs.YUV, vs.GRAY}:
            raise ValueError('adaptive_denoiser: only YUV and GRAY formats are supported')

        clip = depth(clip, 16, dither_type="none")

        lumamask = luma_mask_ping(clip, thr=luma_mask_thr)
        darken_luma_mask = core.akarin.Expr([lumamask], f"x {luma_mask_weaken} *")

        if show_mask == 1:
            selected_mask = darken_luma_mask

        #Degrain
        if "is_digital" not in kwargs:
            mvtools = MVTools(clip)
            vectors = mvtools.analyze(blksize=16, tr=tr, overlap=8, lsad=300, search=SearchMode.UMH, truemotion=MotionMode.SAD, dct=SADMode.MIXED_SATD_DCT)
            mfilter = mini_BM3D(clip, sigma=sigma*1.25, radius=tr, profile="LC", planes=0)
            degrain = mc_degrain(clip, prefilter=Prefilter.DFTTEST, blksize=8, mfilter=mfilter, thsad=thsad, vectors=vectors, tr=tr, limit=1)
        else:
            degrain = clip

        if precision:
            flatmask_defaults = {
                "sigma1": 3,
                "texture_strength": 2,
                "edges_strength": 0.05
            }
            flatmask = hd_flatmask(degrain, **(flatmask_defaults | kwargs_flatmask))
            if show_mask == 2:
                selected_mask = flatmask
            
            if luma_over_texture > 1.0:
                raise ValueError("luma_over_texture must be less than 1")
            elif luma_over_texture < 0.0:
                raise ValueError("luma_over_texture must be greater than 0")
            elif luma_over_texture == 1:
                raise ValueError("don't use precision mode if luma_over_texture is 1")
            final_mask = core.akarin.Expr([darken_luma_mask, flatmask], f"x {luma_over_texture} * y {abs(luma_over_texture-1)} * +")
        else:
            final_mask = darken_luma_mask
        
        if show_mask == 3:
            selected_mask = final_mask
        
        denoised = mini_BM3D(plane(degrain, 0), sigma=sigma, radius=tr, profile="HIGH")
        y_denoised = core.std.MaskedMerge(denoised, plane(clip, 0), final_mask) #denoise applied to darker areas

        #Chroma denoise
        if chroma_denoise[0] <= 0:
            chroma_denoised = clip
        else:
            if chroma_denoise[1] == "nlm":
                chroma_denoised = nl_means(clip, h=chroma_denoise[0], tr=tr, ref=degrain, planes=[1,2])
            if chroma_denoise[1] == "cbm3d":
                chroma_denoised = mini_BM3D(clip, sigma=chroma_denoise[0], radius=tr, ref=degrain, planes=[1,2])
            if chroma_denoise[1] == "artcnn":
                from vsscale import ArtCNN
                chroma_denoised = ArtCNN.R8F64_JPEG420().scale(clip)

        if (chroma_masking and chroma_denoise[0]>0) and clip.format.color_family == vs.YUV:
            u=plane(clip, 1)
            u_mask= luma_mask_man(u, t=1.5, s=2, a=0)
            u_masked = core.std.MaskedMerge(u, plane(chroma_denoised, 1), u_mask)
            v=plane(clip, 2)
            v_mask= luma_mask_man(v, t=1.5, s=2, a=0)
            v_masked = core.std.MaskedMerge(v, plane(chroma_denoised, 2), v_mask)
            final = core.std.ShufflePlanes(clips=[chroma_denoised, u_masked, v_masked], planes=[0,0,0], colorfamily=vs.YUV)

            if show_mask == 4:
                selected_mask = v_mask
            elif show_mask == 5:
                selected_mask = u_mask
        
        if clip.format.color_family == vs.GRAY:
            final = y_denoised
        else:
            final = core.std.ShufflePlanes(clips=[y_denoised, chroma_denoised, chroma_denoised], planes=[0,1,2], colorfamily=vs.YUV)
        return final, selected_mask

    # Presets
    @staticmethod
    def scan65mm (clip: vs.VideoNode, thsad: int = 200, tr: int = 2, sigma: float = 2, luma_mask_weaken: float = 0.9, luma_mask_thr: float = 0.196, chroma_denoise: float | tuple[float, str] = [0.5, "nlm"], precision: bool = True, chroma_masking: bool = False, show_mask: int = 0, luma_over_texture: float = 0.4, kwargs_flatmask: dict = {})->vs.VideoNode:
        """ changes: thsad=200, sigma=2, luma_mask_weaken=0.9, chroma_strength=0.5 """
        if show_mask in [1, 2, 3, 4, 5]:
            return adenoise._adaptive_denoiser_tuple(clip, thsad, tr, sigma, luma_mask_weaken, luma_mask_thr, chroma_denoise, precision, chroma_masking, show_mask, luma_over_texture, kwargs_flatmask)
        return adenoise._adaptive_denoiser(clip, thsad, tr, sigma, luma_mask_weaken, luma_mask_thr, chroma_denoise, precision, chroma_masking, luma_over_texture, kwargs_flatmask)
    @staticmethod
    def scan35mm (clip: vs.VideoNode, thsad: int = 400, tr: int = 2, sigma: float = 4, luma_mask_weaken: float = 0.8, luma_mask_thr: float = 0.196, chroma_denoise: float | tuple[float, str] = [0.7, "nlm"], precision: bool = True, chroma_masking: bool = False, show_mask: int = 0, luma_over_texture: float = 0.4, kwargs_flatmask: dict = {})->vs.VideoNode:
        """ changes: thsad=400, sigma=4, luma_mask_weaken=0.8, chroma_strength=0.7 """
        if show_mask in [1, 2, 3, 4, 5]:
            return adenoise._adaptive_denoiser_tuple(clip, thsad, tr, sigma, luma_mask_weaken, luma_mask_thr, chroma_denoise, precision, chroma_masking, show_mask, luma_over_texture, kwargs_flatmask)
        return adenoise._adaptive_denoiser(clip, thsad, tr, sigma, luma_mask_weaken, luma_mask_thr, chroma_denoise, precision, chroma_masking, luma_over_texture, kwargs_flatmask)
    
    @staticmethod
    def scan16mm (clip: vs.VideoNode, thsad: int = 600, tr: int = 2, sigma: float = 8, luma_mask_weaken: float = 0.75, luma_mask_thr: float = 0.196, chroma_denoise: float | tuple[float, str] = [1.0, "nlm"], precision: bool = True, chroma_masking: bool = False, show_mask: int = 0, luma_over_texture: float = 0.4, kwargs_flatmask: dict = {})->vs.VideoNode:
        """ changes: thsad=600, sigma=8 """
        if show_mask in [1, 2, 3, 4, 5]:
            return adenoise._adaptive_denoiser_tuple(clip, thsad, tr, sigma, luma_mask_weaken, luma_mask_thr, chroma_denoise, precision, chroma_masking, show_mask, luma_over_texture, kwargs_flatmask)
        return adenoise._adaptive_denoiser(clip, thsad, tr, sigma, luma_mask_weaken, luma_mask_thr, chroma_denoise, precision, chroma_masking, luma_over_texture, kwargs_flatmask)
    
    @staticmethod
    def scan8mm (clip: vs.VideoNode, thsad: int = 800, tr: int = 2, sigma: float = 12, luma_mask_weaken: float = 0.75, luma_mask_thr: float = 0.196, chroma_denoise: float | tuple[float, str] = [1.5, "nlm"], precision: bool = True, chroma_masking: bool = False, show_mask: int = 0, luma_over_texture: float = 0.4, kwargs_flatmask: dict = {})->vs.VideoNode:
        """ changes: thsad=800, sigma=12, chroma_strength=1.5 """
        if show_mask in [1, 2, 3, 4, 5]:
            return adenoise._adaptive_denoiser_tuple(clip, thsad, tr, sigma, luma_mask_weaken, luma_mask_thr, chroma_denoise, precision, chroma_masking, show_mask, luma_over_texture, kwargs_flatmask)
        return adenoise._adaptive_denoiser(clip, thsad, tr, sigma, luma_mask_weaken, luma_mask_thr, chroma_denoise, precision, chroma_masking, luma_over_texture, kwargs_flatmask)
    
    @staticmethod
    def digital (clip: vs.VideoNode, thsad: int = 300, tr: int = 2, sigma: float = 3, luma_mask_weaken: float = 0.75, luma_mask_thr: float = 0.196, chroma_denoise: float | tuple[float, str] = [1.0, "nlm"], precision: bool = True, chroma_masking: bool = False, show_mask: int = 0, luma_over_texture: float = 0.0, kwargs_flatmask: dict = {})->vs.VideoNode:
        """ changes: thsad=300, sigma=3, luma_over_texture=0 """
        if show_mask in [1, 2, 3, 4, 5]:
            return adenoise._adaptive_denoiser_tuple(clip, thsad, tr, sigma, luma_mask_weaken, luma_mask_thr, chroma_denoise, precision, chroma_masking, show_mask, luma_over_texture, kwargs_flatmask, is_digital=True)
        return adenoise._adaptive_denoiser(clip, thsad, tr, sigma, luma_mask_weaken, luma_mask_thr, chroma_denoise, precision, chroma_masking, luma_over_texture, kwargs_flatmask, is_digital=True)
    
    @staticmethod
    def default (clip: vs.VideoNode, thsad: int = 500, tr: int = 2, sigma: float = 6, luma_mask_weaken: float = 0.75, luma_mask_thr: float = 0.196, chroma_denoise: float | tuple[float, str] = [1.0, "nlm"], precision: bool = True, chroma_masking: bool = False, show_mask: int = 0, luma_over_texture: float = 0.4, kwargs_flatmask: dict = {})->vs.VideoNode:
        """ default profile """
        if show_mask in [1, 2, 3, 4, 5]:
            return adenoise._adaptive_denoiser_tuple(clip, thsad, tr, sigma, luma_mask_weaken, luma_mask_thr, chroma_denoise, precision, chroma_masking, show_mask, luma_over_texture, kwargs_flatmask)
        return adenoise._adaptive_denoiser(clip, thsad, tr, sigma, luma_mask_weaken, luma_mask_thr, chroma_denoise, precision, chroma_masking, luma_over_texture, kwargs_flatmask)

#Ported from fvsfunc 
def auto_deblock(
    clip: vs.VideoNode,
    planes: PlanesT = [0, 1, 2],
    sigma: int = 15,
    tbsize: int = 1,
    luma_mask_strength: float = 0.9,
    pre: bool = False,
    mask_type: int = 0,
) -> vs.VideoNode:
    """
    Deblocker 8x8 and other.

    :param clip:                Clip to process (YUV 16bit, if not will be internally converted in 16bit).
    :param planes:              Which planes to process. Defaults to all planes.
    :param sigma:               Sigma value for dfttest deblock.
    :param tbsize:              Length of the temporal dimension (i.e. number of frames).
    :param luma_mask_strength:  Mask strength multiplier. Lower values mean stronger overall deblock.
    :param pre:                 If True, applies a preliminary deblocking with vsdenoise.deblock_qed.
    :param mask_type:           Mask type to use.
    """

    from .admask import luma_mask_ping, luma_mask_man, luma_mask
    from vsdenoise import deblock_qed
    from vstools import depth
    from dfttest2 import DFTTest

    if clip.format.color_family not in [vs.YUV]:
        raise TypeError("AutoDeblock: clip must be YUV color family!")

    clip = depth(clip, 16, dither_type="none")
    
    if pre:
        clip = deblock_qed(clip, planes=planes)

    deblock = DFTTest(clip, sigma=sigma, tbsize=tbsize, planes=planes)
    
    if (mask_type == 0):
        lumamask = luma_mask(clip)
    elif (mask_type == 1):
        lumamask = luma_mask_man(clip)
    else:
        lumamask = luma_mask_ping(clip)
    darken_luma_mask = core.std.Expr([lumamask], f"x {luma_mask_strength} *")
    final = core.std.MaskedMerge(deblock, clip, darken_luma_mask, planes=planes)

    return final

def msaa2x(
    clip: vs.VideoNode,
    planes: PlanesT = 0,
    ref: Optional[vs.VideoNode] = None,
    mask: bool = False,
    sigma: float = 2,
    thr: float = None,
    **kwargs
) -> vs.VideoNode:
    """
    Upscales only the edges with AI (ArtCNN DN) and downscales them.

    :param clip:            Clip to process (YUV or Grayscale).
    :param planes:          Which planes to process. Defaults to Y.
    :param ref:             Reference clip used to create the edgemask (should be the original not filtered clip). If None, clip will be used and will be denoised with adenoise.digital to prevent edge detail loss, but remove grain and noise.
    :param mask:            If True will return the mask used.
    :param sigma:           Sigma used for edge fixing during antialiasing (remove dirty spots and blocking) only if ref is None.
    :param thr:             Threshold used for Binarize the clip, only 0-1 value area allowed. If None, no Binarize will be applied.
    :param kwargs:          Accepts advanced_edgemask arguments.
    """
    from vsscale import ArtCNN
    from vstools import depth
    from .admask import advanced_edgemask
    from .adutils import scale_binary_value, plane

    if isinstance(planes, int):
        planes = [planes]
    if clip.format.color_family == vs.GRAY:
        planes = [0]
    
    if clip.format.color_family == vs.RGB:
        raise ValueError("msaa2x: clip must be YUV or Gray color family!")
    
    clip = depth(clip, 16, dither_type="none")

    if ref is None:
        ref = adenoise.digital(clip, sigma=sigma, precision=False, chroma_denoise=[(0 if (1 in planes or 2 in planes) else 1), "cbm3d"])
            
    if len(planes) == 1:
        edgemask = advanced_edgemask(plane(ref, planes[0]), **kwargs)
    else:
        masks = [
            advanced_edgemask(plane(ref, p), **kwargs) if p in planes else plane(ref, p).std.BlankClip()
            for p in range(3)
        ]
        edgemask = core.std.ShufflePlanes(masks, planes=[0, 0, 0], colorfamily=ref.format.color_family)
    
    if thr is not None and thr != 0:
        edgemask = edgemask.std.Binarize(threshold=scale_binary_value(edgemask, thr, return_int=True))
    if mask:
        return edgemask

    upscaled = ArtCNN.C4F32_DN().scale(clip, clip.width*2, clip.height*2)
    downscaled = core.resize.Bicubic(upscaled, clip.width, clip.height)
    aa = core.std.MaskedMerge(clip, downscaled, edgemask, planes=0)

    if 1 in planes or 2 in planes:
        lefted = aa.resize.Spline36(src_left=-0.5)
        aa = core.std.ShufflePlanes([aa, lefted, lefted], planes=[0,1,2], colorfamily=clip.format.color_family)
        aa = ArtCNN.R8F64_Chroma().scale(aa)
        chroma_downscaled = core.resize.Bicubic(aa, clip.width/2, clip.height/2)
        u = plane(chroma_downscaled, 1)
        v = plane(chroma_downscaled, 2)
        if 0 not in planes:
            downscaled = clip
        all_downscaled = core.std.ShufflePlanes([downscaled, u, v], planes=[0,0,0], colorfamily=clip.format.color_family)
        aa = core.std.MaskedMerge(clip, all_downscaled, edgemask, planes=planes)

    return aa

