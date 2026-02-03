import vapoursynth as vs

from typing import Optional
from vstools import PlanesT

core = vs.core

if not (hasattr(core, 'dfttest') or hasattr(core, 'fmtc') or hasattr(core, 'akarin')):
    raise ImportError("'dfttest', 'fmtc' and 'akarin' are mandatory. Make sure the DLLs are present in the plugins folder.")

def mini_BM3D(
    clip: vs.VideoNode, 
    profile: str = "LC", 
    accel: Optional[str] = None,
    planes: PlanesT = [0, 1, 2],
    ref: Optional[vs.VideoNode] = None,
    dither: Optional[str] = "error_diffusion",
    fast: Optional[bool] = False,
    **kwargs
) -> vs.VideoNode:
    """
    BM3D mini wrapper.

    :param clip:            Clip to process (32bit, if not will be internally converted in 32bit).
    :param profile:         Precision. Accepted values: "FAST", "LC", "HIGH".
    :param accel:           Choose the hardware acceleration. Accepted values: "cuda_rtc", "cuda", "hip", "cpu", "auto".
    :param planes:          Which planes to process. Defaults to all planes.
    :param kwargs:          Accepts BM3DCUDA arguments, https://github.com/WolframRhodium/VapourSynth-BM3DCUDA.
    :param kwargs:          Accepts DitherType class names.
    :param fast:            Use CPU+GPU, adds overhead.
    :return:                Denoised clip.
    """
    from vstools import depth, plane, get_y, get_u, get_v
    
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
                    return core.bm3dcpu.BM3Dv2(clip, ref, **kwargs)
        elif accel_u == "CUDA":
            return core.bm3dcuda.BM3Dv2(clip, ref, **kwargs)
        elif accel_u == "HIP":
            return core.bm3dhip.BM3Dv2(clip, ref, **kwargs)
        elif accel_u == "CPU":
            kwargs.pop("fast", None)
            kwargs.pop("ps_range", kwargs.get("ps_range")[0])
            return core.bm3dcpu.BM3Dv2(clip, ref, **kwargs)

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
            _bm3d(plane(clipS, p), accel, **kwargs) if p in planes and 0 <= p < num_planes else plane(clipS, p)
            for p in range(num_planes)
        ]
        dclip = core.std.ShufflePlanes(filtered_planes, planes=[0, 0, 0], colorfamily=clip.format.color_family)

    elif clip.format.color_family == vs.YUV:
        y = get_y(clipS)
        u = get_u(clipS)
        v = get_v(clipS)

        y_ref = None
        if refS is not None:
            if refS.format.num_planes not in (1, 3):
                raise ValueError("mini_BM3D: When providing a reference clip for YUV, it must have 1 or 3 planes.")
            y_ref = get_y(refS)

        y_denoised = _bm3d(y, accel, ref=y_ref, **kwargs) if 0 in planes else y

        if 1 in planes or 2 in planes:
            y_downscaled = y.resize.Spline36(u.width, u.height)

            ref_444 = None
            if refS is not None and refS.format.num_planes == 3:
                u_ref = get_u(refS); v_ref = get_v(refS)
                y_ref_downscaled = y_ref.resize.Spline36(u.width, u.height)
                ref_444 = core.std.ShufflePlanes([y_ref_downscaled, u_ref, v_ref], planes=[0, 0, 0], colorfamily=clip.format.color_family)
            elif refS is not None and refS.format.num_planes == 1:
                ref_444 = y_ref

            clip_444 = core.std.ShufflePlanes([y_downscaled, u, v], planes=[0, 0, 0], colorfamily=clip.format.color_family)
            clip_444 = _bm3d(clip_444, accel, ref=ref_444, chroma=True, **kwargs) if ref_444 is not None else _bm3d(clip_444, accel, chroma=True, **kwargs)

            if 1 in planes:
                u = get_u(clip_444)
            if 2 in planes:
                v = get_v(clip_444)

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

    :param clip:                Clip to process (YUV 16bit, if not will be internally converted in 16bit).
    :param thsad:               Thsad for mc_degrain (luma denoise strength and chroma ref).
                                Recommended values: 300-800
    :param tr:                  Temporal radius for temporal consistency across al the filter involved.
                                Recommended values: 2-3 (1 means no temporal denoise).
    :param sigma:               Sigma for BM3D (luma denoise strength).
                                Recommended values: 1-5. 
    :param sigma_mask:          Sigma for flat mask denoising.
                                This value should be decided based on the details level of the clip and how much grain and noise is present.
                                Usally 1 for really textured clip, 2-3 for a normal clip, 4-5 for a clip with strong noise or grain.
    :param luma_mask_weaken:    Controls how much dark spots should be denoised. Lower values mean stronger overall denoise.
                                Recommended values: 0.6-0.9
    :param luma_mask_thr:       Threshold that determines what is considered bright and what is dark in the luma mask.
                                Recommended values: 0.15-0.25
    :param chroma_strength:     Strength for NLMeans/CM3DBM (chroma denoise strength).
                                Recommended values: 0.5-2
    :param chroma_denoise:      Denoiser for chroma.
                                Accepted values: "nlm", "cbm3d"
    :param precision:           If True, a flat mask is created to enhance the denoise strenght on flat areas avoiding textured area (90% accuracy).
    :param chroma_masking:      If True, enables specific chroma masking for U/V planes.
    :param show_mask:           1 = Show the first luma mask, 2 = Show the textured luma mask, 3 = Show the complete luma mask, 4 = Show the Chroma U Plane mask (if chroma_masking = True), 5 = Show the Chroma V Plane mask (if chroma_masking = True). Any other value returns the denoised clip.
    :param luma_penalty:        Multiplier for the luma mask in precision mode. This value is combined with texture_penalty and their sum must be always 1. Lower value means more importance to textured areas.
    :param texture_penalty:     Multiplier for the texture mask in precision mode. This value is combined with luma_penalty and their sum must be always 1. Lower value means more importance to overall denoise.
    :param texture_strength:    Texture strength for mask (0-inf). Values above 1 decrese the strength of the texture in the mask, lower values increase it. The max value is theoretical infinite, but there is no gain after some point.
    :param edges_strength:      Edges strength for mask (0-1). Basic multiplier for edges strength.

    :return:                    16bit denoised clip. If show_mask is 1, 2, 3, 4 or 5, returns a tuple (denoised_clip, mask).
    """

    @classmethod
    def _adaptive_denoiser(
        cls,
        clip: vs.VideoNode,
        thsad: int = 500,
        tr: int = 2,
        sigma: float = 6,
        sigma_mask: float = 3,
        luma_mask_weaken: float = 0.75,
        luma_mask_thr: float = 0.196,
        chroma_strength: float = 1.0,
        chroma_denoise: str = "nlm", 
        precision: bool = True,
        chroma_masking: bool = False,
        luma_penalty: float = 0.4,
        texture_penalty: float = 0.6,
        texture_strength: float = 2,
        edges_strength: float = 0.05,
        **kwargs
    ) -> vs.VideoNode:
        
        from vstools import get_y, get_u, get_v, depth
        from vsdenoise import Prefilter, mc_degrain, nl_means, MVTools, SearchMode, MotionMode, SADMode, MVTools, SADMode, MotionMode
        from .admask import luma_mask_ping, luma_mask_man, godflatmask

        core = vs.core

        if clip.format.color_family not in {vs.YUV}:
            raise ValueError('adaptive_denoiser: only YUV formats are supported')

        depth(clip, 16, dither_type="none")

        lumamask = luma_mask_ping(clip, thr=luma_mask_thr)
        darken_luma_mask = core.akarin.Expr([lumamask], f"x {luma_mask_weaken} *")

        #Degrain
        if "is_digital" not in kwargs:
            mvtools = MVTools(clip)
            vectors = mvtools.analyze(blksize=16, tr=tr, overlap=8, lsad=300, search=SearchMode.UMH, truemotion=MotionMode.SAD, dct=SADMode.MIXED_SATD_DCT)
            mfilter = mini_BM3D(clip=get_y(clip), sigma=sigma*1.25, radius=tr, profile="LC", planes=0)
            mfilter = core.std.ShufflePlanes(clips=[mfilter, get_u(clip), get_v(clip)], planes=[0,0,0], colorfamily=vs.YUV)
            degrain = mc_degrain(clip, prefilter=Prefilter.DFTTEST, blksize=8, mfilter=mfilter, thsad=thsad, vectors=vectors, tr=tr, limit=1)
        else:
            degrain = clip

        if precision:
            flatmask = godflatmask(degrain, sigma1=sigma_mask, texture_strength=texture_strength, edges_strength=edges_strength)
            if luma_penalty + texture_penalty != 1.0:
                raise ValueError("luma_penalty + texture_penalty must be equal to 1")
            darken_luma_mask = core.akarin.Expr(
            [darken_luma_mask, flatmask],
            f"x {luma_penalty} * y {texture_penalty} * +")
        
        denoised = mini_BM3D(get_y(degrain), sigma=sigma, radius=tr, profile="HIGH", planes=0)
        luma = get_y(core.std.MaskedMerge(denoised, get_y(clip), darken_luma_mask, planes=0)) #denoise applied to darker areas

        #Chroma denoise
        
        if chroma_strength <= 0:
            chroma_denoised = clip
        else:
            if chroma_denoise == "nlm":
                chroma_denoised = nl_means(clip, h=chroma_strength, tr=tr, ref=degrain, planes=[1,2])
            if chroma_denoise == "cbm3d":
                chroma_denoised = mini_BM3D(clip, sigma=chroma_strength, radius=tr, ref=degrain, planes=[1,2])
        
        #TODO 
        #chroma mask fine tuning
        v_mask = None #Per evitare UnboundLocalError
        u_mask = None

        if chroma_masking and chroma_strength>0:
            v=get_v(clip)
            v_mask= luma_mask_man(v, t=1.5, s=2, a=0)
            v_masked = core.std.MaskedMerge(v, get_v(chroma_denoised), v_mask)
            u=get_u(clip)
            u_mask= luma_mask_man(u, t=1.5, s=2, a=0)
            u_masked = core.std.MaskedMerge(u, get_u(chroma_denoised), u_mask)
            chroma_denoised = core.std.ShufflePlanes(clips=[chroma_denoised, u_masked, v_masked], planes=[0,0,0], colorfamily=vs.YUV)
        
        final = core.std.ShufflePlanes(clips=[luma, get_u(chroma_denoised), get_v(chroma_denoised)], planes=[0,0,0], colorfamily=vs.YUV)
        return final
    
    @classmethod
    def _adaptive_denoiser_tuple(
        cls,
        clip: vs.VideoNode,
        thsad: int = 500,
        tr: int = 2,
        sigma: float = 6,
        sigma_mask: float = 3,
        luma_mask_weaken: float = 0.75,
        luma_mask_thr: float = 0.196,
        chroma_strength: float = 1.0,
        chroma_denoise: str = "nlm", 
        precision: bool = True,
        chroma_masking: bool = False,
        show_mask: int = 0,
        luma_penalty: float = 0.4,
        texture_penalty: float = 0.6,
        texture_strength: float = 2,
        edges_strength: float = 0.05,
        **kwargs
    ) -> tuple[vs.VideoNode, vs.VideoNode]:
        
        from vstools import get_y, get_u, get_v, depth
        from vsdenoise import Prefilter, mc_degrain, nl_means, MVTools, SearchMode, MotionMode, SADMode, MVTools, SADMode, MotionMode
        from .admask import luma_mask_ping, luma_mask_man, godflatmask

        core = vs.core
        
        selected_mask = None

        if clip.format.color_family not in {vs.YUV}:
            raise ValueError('adaptive_denoiser: only YUV formats are supported')

        clip = depth(clip, 16, dither_type="none")

        lumamask = luma_mask_ping(clip, thr=luma_mask_thr)
        darken_luma_mask = core.akarin.Expr([lumamask], f"x {luma_mask_weaken} *")

        if show_mask == 1:
            selected_mask = darken_luma_mask

        #Degrain
        if "is_digital" not in kwargs:
            mvtools = MVTools(clip)
            vectors = mvtools.analyze(blksize=16, tr=tr, overlap=8, lsad=300, search=SearchMode.UMH, truemotion=MotionMode.SAD, dct=SADMode.MIXED_SATD_DCT)
            mfilter = mini_BM3D(clip=get_y(clip), sigma=sigma*1.25, radius=tr, profile="LC", planes=0)
            mfilter = core.std.ShufflePlanes(clips=[mfilter, get_u(clip), get_v(clip)], planes=[0,0,0], colorfamily=vs.YUV)
            degrain = mc_degrain(clip, prefilter=Prefilter.DFTTEST, blksize=8, mfilter=mfilter, thsad=thsad, vectors=vectors, tr=tr, limit=1)
        else:
            degrain = clip

        if precision:
            flatmask = godflatmask(degrain, sigma1=sigma_mask, texture_strength=texture_strength, edges_strength=edges_strength)
            if show_mask == 2:
                selected_mask = flatmask
            if luma_penalty + texture_penalty != 1.0:
                raise ValueError("luma_penalty + texture_penalty must be equal to 1")
            darken_luma_mask = core.akarin.Expr(
            [darken_luma_mask, flatmask],
            f"x {luma_penalty} * y {texture_penalty} * +")
            
        if show_mask == 3:
            selected_mask = darken_luma_mask
        
        denoised = mini_BM3D(get_y(degrain), sigma=sigma, radius=tr, profile="HIGH", planes=0)
        luma = get_y(core.std.MaskedMerge(denoised, get_y(clip), darken_luma_mask, planes=0)) #denoise applied to darker areas

        #Chroma denoise
        
        if chroma_strength <= 0:
            chroma_denoised = clip
        else:
            if chroma_denoise == "nlm":
                chroma_denoised = nl_means(clip, h=chroma_strength, tr=tr, ref=degrain, planes=[1,2])
            if chroma_denoise == "cbm3d":
                chroma_denoised = mini_BM3D(clip, sigma=chroma_strength, radius=tr, ref=degrain, planes=[1,2])
        
        #TODO 
        #chroma mask fine tuning
        v_mask = None #Per evitare UnboundLocalError
        u_mask = None

        if chroma_masking or chroma_strength<=0:
            v=get_v(clip)
            v_mask= luma_mask_man(v, t=1.5, s=2, a=0)
            v_masked = core.std.MaskedMerge(v, get_v(chroma_denoised), v_mask)
            u=get_u(clip)
            u_mask= luma_mask_man(u, t=1.5, s=2, a=0)
            u_masked = core.std.MaskedMerge(u, get_u(chroma_denoised), u_mask)
            chroma_denoised = core.std.ShufflePlanes(clips=[chroma_denoised, u_masked, v_masked], planes=[0,0,0], colorfamily=vs.YUV)
        
            if show_mask == 4:
                selected_mask = v_mask
            elif show_mask == 5:
                selected_mask = u_mask
        
        final = core.std.ShufflePlanes(clips=[luma, get_u(chroma_denoised), get_v(chroma_denoised)], planes=[0,0,0], colorfamily=vs.YUV)
        return final, selected_mask

    # Presets
    @staticmethod
    def scan65mm (clip:vs.VideoNode, thsad: int = 200, tr: int = 2, sigma: float = 2, sigma_mask: float = 2, luma_mask_weaken: float = 0.9, luma_mask_thr: float = 0.196, chroma_strength: float = 0.5, chroma_denoise: str = "nlm", precision: bool = True, chroma_masking: bool = False, show_mask: int = 0, luma_penalty: float = 0.4, texture_penalty: float = 0.6, texture_strength: float = 2, edges_strength: float = 0.05)->vs.VideoNode:
        """ changes: thsad=200, sigma=2, luma_mask_weaken=0.9, chroma_strength=0.5 """
        if show_mask in [1, 2, 3, 4, 5]:
            return adenoise._adaptive_denoiser_tuple(clip, thsad, tr, sigma, sigma_mask, luma_mask_weaken, luma_mask_thr, chroma_strength, chroma_denoise, precision, chroma_masking, show_mask, luma_penalty, texture_penalty, texture_strength, edges_strength)
        return adenoise._adaptive_denoiser(clip, thsad, tr, sigma, sigma_mask, luma_mask_weaken, luma_mask_thr, chroma_strength, chroma_denoise, precision, chroma_masking, luma_penalty, texture_penalty, texture_strength, edges_strength)
    @staticmethod
    def scan35mm (clip:vs.VideoNode, thsad: int = 400, tr: int = 2, sigma: float = 4, sigma_mask: float = 4, luma_mask_weaken: float = 0.8, luma_mask_thr: float = 0.196, chroma_strength: float = 0.7, chroma_denoise: str = "nlm", precision: bool = True, chroma_masking: bool = False, show_mask: int = 0, luma_penalty: float = 0.4, texture_penalty: float = 0.6, texture_strength: float = 2, edges_strength: float = 0.05)->vs.VideoNode:
        """ changes: thsad=400, sigma=4, luma_mask_weaken=0.8, chroma_strength=0.7 """
        if show_mask in [1, 2, 3, 4, 5]:
            return adenoise._adaptive_denoiser_tuple(clip, thsad, tr, sigma, sigma_mask, luma_mask_weaken, luma_mask_thr, chroma_strength, chroma_denoise, precision, chroma_masking, show_mask, luma_penalty, texture_penalty, texture_strength, edges_strength)
        return adenoise._adaptive_denoiser(clip, thsad, tr, sigma, sigma_mask, luma_mask_weaken, luma_mask_thr, chroma_strength, chroma_denoise, precision, chroma_masking, luma_penalty, texture_penalty, texture_strength, edges_strength)
    
    @staticmethod
    def scan16mm (clip:vs.VideoNode, thsad: int = 600, tr: int = 2, sigma: float = 8, sigma_mask: float = 8, luma_mask_weaken: float = 0.75, luma_mask_thr: float = 0.196, chroma_strength: float = 1.0, chroma_denoise: str = "nlm", precision: bool = True, chroma_masking: bool = False, show_mask: int = 0, luma_penalty: float = 0.4, texture_penalty: float = 0.6, texture_strength: float = 2, edges_strength: float = 0.05)->vs.VideoNode:
        """ changes: thsad=600, sigma=8 """
        if show_mask in [1, 2, 3, 4, 5]:
            return adenoise._adaptive_denoiser_tuple(clip, thsad, tr, sigma, sigma_mask, luma_mask_weaken, luma_mask_thr, chroma_strength, chroma_denoise, precision, chroma_masking, show_mask, luma_penalty, texture_penalty, texture_strength, edges_strength)
        return adenoise._adaptive_denoiser(clip, thsad, tr, sigma, sigma_mask, luma_mask_weaken, luma_mask_thr, chroma_strength, chroma_denoise, precision, chroma_masking, luma_penalty, texture_penalty, texture_strength, edges_strength)
    
    @staticmethod
    def scan8mm (clip:vs.VideoNode, thsad: int = 800, tr: int = 2, sigma: float = 12, sigma_mask: float = 12, luma_mask_weaken: float = 0.75, luma_mask_thr: float = 0.196, chroma_strength: float = 1.5, chroma_denoise: str = "nlm", precision: bool = True, chroma_masking: bool = False, show_mask: int = 0, luma_penalty: float = 0.4, texture_penalty: float = 0.6, texture_strength: float = 2, edges_strength: float = 0.05)->vs.VideoNode:
        """ changes: thsad=800, sigma=12, chroma_strength=1.5 """
        if show_mask in [1, 2, 3, 4, 5]:
            return adenoise._adaptive_denoiser_tuple(clip, thsad, tr, sigma, sigma_mask, luma_mask_weaken, luma_mask_thr, chroma_strength, chroma_denoise, precision, chroma_masking, show_mask, luma_penalty, texture_penalty, texture_strength, edges_strength)
        return adenoise._adaptive_denoiser(clip, thsad, tr, sigma, sigma_mask, luma_mask_weaken, luma_mask_thr, chroma_strength, chroma_denoise, precision, chroma_masking, luma_penalty, texture_penalty, texture_strength, edges_strength)
    
    @staticmethod
    def digital (clip:vs.VideoNode, thsad: int = 300, tr: int = 2, sigma: float = 3, sigma_mask: float = 3, luma_mask_weaken: float = 0.75, luma_mask_thr: float = 0.196, chroma_strength: float = 1.0, chroma_denoise: str = "nlm", precision: bool = True, chroma_masking: bool = False, show_mask: int = 0, luma_penalty: float = 0.4, texture_penalty: float = 0.6, texture_strength: float = 5, edges_strength: float = 0.2)->vs.VideoNode:
        """ changes: thsad=300, sigma=3, texture_penalty=1 """
        if show_mask in [1, 2, 3, 4, 5]:
            return adenoise._adaptive_denoiser_tuple(clip, thsad, tr, sigma, sigma_mask, luma_mask_weaken, luma_mask_thr, chroma_strength, chroma_denoise, precision, chroma_masking, show_mask, luma_penalty, texture_penalty, texture_strength, edges_strength, is_digital=True)
        return adenoise._adaptive_denoiser(clip, thsad, tr, sigma, sigma_mask, luma_mask_weaken, luma_mask_thr, chroma_strength, chroma_denoise, precision, chroma_masking, luma_penalty, texture_penalty, texture_strength, edges_strength, is_digital=True)
    
    @staticmethod
    def default (clip:vs.VideoNode, thsad: int = 500, tr: int = 2, sigma: float = 6, sigma_mask: float = 3, luma_mask_weaken: float = 0.75, luma_mask_thr: float = 0.196, chroma_strength: float = 1.0, chroma_denoise: str = "nlm", precision: bool = True, chroma_masking: bool = False, show_mask: int = 0, luma_penalty: float = 0.4, texture_penalty: float = 0.6, texture_strength: float = 2, edges_strength: float = 0.05)->vs.VideoNode:
        """ default profile """
        if show_mask in [1, 2, 3, 4, 5]:
            return adenoise._adaptive_denoiser_tuple(clip, thsad, tr, sigma, sigma_mask, luma_mask_weaken, luma_mask_thr, chroma_strength, chroma_denoise, precision, chroma_masking, show_mask, luma_penalty, texture_penalty, texture_strength, edges_strength)
        return adenoise._adaptive_denoiser(clip, thsad, tr, sigma, sigma_mask, luma_mask_weaken, luma_mask_thr, chroma_strength, chroma_denoise, precision, chroma_masking, luma_penalty, texture_penalty, texture_strength, edges_strength)

#Ported from fvsfunc 
def auto_deblock(
    clip: vs.VideoNode,
    sigma: int = 15,
    tbsize: int = 1,
    luma_mask_strength: float = 0.9,
    pre: bool = False,
    mask_type: int = 0,
    planes: PlanesT = None
) -> vs.VideoNode:
    """
    Deblocker 8x8 and other.

    :param clip:                Clip to process (YUV 16bit, if not will be internally converted in 16bit).
    :param sigma:               Sigma value for dfttest deblock.
    :param tbsize:              Length of the temporal dimension (i.e. number of frames).
    :param luma_mask_strength:  Mask strength multiplier. Lower values mean stronger overall deblock.
    :param pre:                 If True, applies a preliminary deblocking with vsdenoise.deblock_qed.
    :param mask_type:           Mask type to use.
    :param planes:              Which planes to process. Defaults to all planes.
    """

    from .admask import luma_mask_ping, luma_mask_man, luma_mask
    from vsdenoise import deblock_qed
    from vstools import depth

    if clip.format.color_family not in [vs.YUV]:
        raise TypeError("AutoDeblock: clip must be YUV color family!")

    clip = depth(clip, 16, dither_type="none")
    
    if pre:
        clip = deblock_qed(clip, planes=planes)

    deblock = core.dfttest.DFTTest(clip, sigma=sigma, tbsize=tbsize, planes=planes)
    
    if (mask_type == 0):
        lumamask = luma_mask(clip)
    elif (mask_type == 1):
        lumamask = luma_mask_man(clip)
    else:
        lumamask = luma_mask_ping(clip)
    darken_luma_mask = core.std.Expr([lumamask], f"x {luma_mask_strength} *")
    final = core.std.MaskedMerge(deblock, clip, darken_luma_mask, planes=planes)

    return final

#TODO: advanced_edgemask
def msaa2x(
    clip: vs.VideoNode,
    ref: Optional[vs.VideoNode] = None,
    mask: bool = False,
    sigma: float = 2,
    thr: float = None,
    planes: PlanesT = 0,
    **kwargs
) -> vs.VideoNode:
    """
    Upscales only the edges with AI (ArtCNN DN) and downscales them.

    :param clip:            Clip to process (YUV or Grayscale).
    :param ref:             Reference clip used to create the edgemask (should be the original not filtered clip). If None, clip will be used and will be denoised with adenoise.digital to prevent edge detail loss, but remove grain and noise.
    :param mask:            If True will return the mask used.
    :param sigma:           Sigma used for edge fixing during antialiasing (remove dirty spots and blocking) only if ref is None.
    :param thr:             Threshold used for Binarize the clip, only 0-1 value area allowed. If None, no Binarize will be applied.
    :param planes:          Which planes to process. Defaults to Y.
    :param kwargs:          Accepts advanced_edgemask arguments.
    """
    from vsscale import ArtCNN
    from vstools import get_y, get_u, get_v, plane
    from addfunc.admask import advanced_edgemask
    from addfunc.adutils import scale_binary_value

    if isinstance(planes, int):
        planes = [planes]
    if clip.format.color_family == vs.GRAY:
        planes = [0]
    
    if clip.format.color_family == vs.RGB:
        raise ValueError("msaa2x: clip must be YUV or Gray color family!")

    if ref is None:
        ref = adenoise.digital(clip, sigma=sigma, precision=False, chroma_denoise="cbm3d", chroma_strength=(0 if (1 in planes or 2 in planes) else 1))
    
    masks = [
        plane(ref, p) if p in planes and 0 <= p < 3 else None
        for p in range(3)
    ]
            
    if len(planes) == 1:
        edgemask = masks[planes[0]]
    else:
        if masks[0] is None: masks[0] = get_y(ref).std.BlankClip()
        if masks[1] is None: masks[1] = get_u(ref).std.BlankClip()
        if masks[2] is None: masks[2] = get_v(ref).std.BlankClip()
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
        u = get_u(chroma_downscaled)
        v = get_v(chroma_downscaled)
        if 0 not in planes:
            downscaled = clip
        all_downscaled = core.std.ShufflePlanes([downscaled, u, v], planes=[0,0,0], colorfamily=clip.format.color_family)
        aa = core.std.MaskedMerge(clip, all_downscaled, edgemask, planes=planes)

    return aa

