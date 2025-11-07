import vapoursynth as vs

from typing import Optional
from vsdenoise import Prefilter, mc_degrain, nl_means, MVTools, SearchMode, MotionMode, SADMode, MVTools, SADMode, MotionMode, deblock_qed
from vstools import get_y, get_u, get_v, PlanesT, depth, plane
from vsmasktools import Morpho

core = vs.core

if not (hasattr(core, 'dfttest') or hasattr(core, 'fmtc') or hasattr(core, 'akarin')):
    raise ImportError("'dfttest', 'fmtc' and 'akarin' are mandatory. Make sure the DLLs are present in the plugins folder.")

def mini_BM3D(
    clip: vs.VideoNode, 
    profile: str = "LC", 
    accel: Optional[str] = None,
    planes: PlanesT = None,
    ref: Optional[vs.VideoNode] = None,
    dither: Optional[str] = "error_diffusion",
    fast: Optional[bool] = False,
    **kwargs
) -> vs.VideoNode:
    """
    BM3D mini wrapper.

    :param clip:            Clip to process. Must be 32 bit float format.
    :param profile:         Precision. Accepted values: "FAST", "LC", "HIGH".
    :param accel:           Choose the hardware acceleration. Accepted values: "cuda_rtc", "cuda", "hip", "cpu", "auto".
    :param planes:          Which planes to process. Defaults to all planes.
    :param kwargs:          Accepts BM3DCUDA arguments, https://github.com/WolframRhodium/VapourSynth-BM3DCUDA.
    :param kwargs:          Accepts DitherType class names.
    :param fast:            Use CPU+GPU, adds overhead.
    :return:                Denoised clip.
    """

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
                    return core.bm3dcpu.BM3Dv2(clip, ref, **kwargs)
        elif accel_u == "CUDA":
            return core.bm3dcuda.BM3Dv2(clip, ref, **kwargs)
        elif accel_u == "HIP":
            return core.bm3dhip.BM3Dv2(clip, ref, **kwargs)
        elif accel_u == "CPU":
            kwargs.pop("fast", None)
            return core.bm3dcpu.BM3Dv2(clip, ref, **kwargs)
    
    if clip.format.bits_per_sample != 32:
        clipS = depth(clip, 32)
    else:
        clipS = clip
    
    if ref is not None:
        if ref.format.bits_per_sample != 32:
            refS = depth(ref, 32)
        else:
            refS = ref
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

    profile_u = profile.upper() if isinstance(profile, str) else str(profile).upper()

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

    if planes is None:
        planes = [0, 1, 2]
    if isinstance(planes, int):
        planes = [planes]
    planes = list(dict.fromkeys(int(p) for p in planes))

    if clip.format.color_family == vs.RGB:
        filtered_planes = [
            _bm3d(plane(clipS, [i]), accel, **kwargs) if i in planes and 0 <= i < num_planes else plane(clipS, [i])
            for i in range(num_planes)
        ]
        dclip = core.std.ShufflePlanes(filtered_planes, planes=[0, 0, 0], colorfamily=clip.format.color_family)

    elif clip.format.color_family == vs.YUV:
        y = get_y(clipS)
        u = get_u(clipS)
        v = get_v(clipS)

        yr = ur = vr = None
        if refS is not None:
            if refS.format.num_planes == 3:
                yr = get_y(refS); ur = get_u(refS); vr = get_v(refS)
            elif refS.format.num_planes == 1:
                yr = refS
            else:
                raise ValueError("mini_BM3D: When providing a reference clip for YUV, it must have 1 or 3 planes.")

        yd = _bm3d(y, accel, ref=yr, **kwargs) if 0 in planes else y

        if 1 in planes or 2 in planes:
            y_resized = y.resize.Spline36(u.width, u.height)

            clipr444 = None
            if refS is not None and refS.format.num_planes == 3:
                yr_resized = yr.resize.Spline36(u.width, u.height)
                clipr444 = core.std.ShufflePlanes([yr_resized, ur, vr], planes=[0, 0, 0], colorfamily=clip.format.color_family)
            elif refS is not None and refS.format.num_planes == 1:
                clipr444 = yr

            clip444 = core.std.ShufflePlanes([y_resized, u, v], planes=[0, 0, 0], colorfamily=clip.format.color_family)
            clip444 = _bm3d(clip444, accel, ref=clipr444, chroma=True, **kwargs) if clipr444 is not None else _bm3d(clip444, accel, chroma=True, **kwargs)

            if 1 in planes:
                u = get_u(clip444)
            if 2 in planes:
                v = get_v(clip444)

        dclip = core.std.ShufflePlanes([yd, u, v], planes=[0, 0, 0], colorfamily=clip.format.color_family)

    else:
        raise ValueError("mini_BM3D: Unsupported color family.")

    return depth(dclip, clip.format.bits_per_sample, dither_type=dither)

class adenoise:
    """Preset class for _adaptive_denoiser."""

    @classmethod
    def _adaptive_denoiser(
        cls,
        clip: vs.VideoNode,
        thsad: int = 500,
        tr: int = 2,
        tr2: int = 1,
        sigma: float = 6,
        luma_mask_weaken: float = 0.75,
        luma_mask_thr: float = 50,
        chroma_strength: float = 1.0,
        chroma_denoise: str = "nlm", 
        precision: bool = True,
        chroma_masking: bool = False,
        show_mask: int = 0,
        flat_penalty: float = 0.5,
        texture_penalty: float = 1.1,
        **kwargs
    ) -> vs.VideoNode:
        """
        Intensive Adaptive Denoise with default parameters for film scans (16mm).

        Three denoisers are applied: mc_degrain (luma), NLMeans (chroma), and BM3D (luma).
        NLMeans uses mc_degrain as reference to remove dirt spots and scanner noise from the clip,
        while mc_degrain affects only the luma, which is then passed to BM3D for a second denoising pass.
        If precision = True, BM3D receives a new mc_degrain reference based on the already cleaned clip (slower).

        Luma masks ensure that denoising is applied only to the brighter areas of the frame, preserving details in darker regions while cleaning them as much as possible.
        Note: Luma masks are more sensitive to variations than the sigma value for the final result.

        :param clip:                Clip to process (YUV 16bit, if not will be internally converted in 16bit with void dither).
        :param thsad:               Thsad for mc_degrain (luma denoise strength and chroma ref).
                                    Recommended values: 300-800
        :param tr:                  Temporal radius for mvtools and nlm.
                                    Recommended values: 2-3 (1 means no temporal denoise).
        :param tr2:                 Temporal radius for BM3D.
                                    Recommended values: 1-2 (0 means no temporal denoise).
        :param sigma:               Sigma for BM3D (luma denoise strength).
                                    Recommended values: 3-10. If precision = True, you can safely double he value used.
        :param luma_mask_weaken:    Controls how much dark spots should be denoised. Lower values mean stronger denoise.
                                    Recommended values: 0.6-0.9
        :param luma_mask_thr:       Mi chiamo ping e non metto le descrizioni.
        :param chroma_strength:     Strength for NLMeans (chroma denoise strength).
                                    Recommended values: 0.5-2
        :param chroma_denoise:      Denoiser for chroma.
                                    Accepted values: "nlm", "cbm3d"
        :param precision:           If True, a flat mask is created to enhance the denoise strenght on flat areas avoiding textured area (90% accuracy).
        :param mask_type:           0 = Standard Luma mask, 1 = Custom Luma mask (more linear) , 2 = Custom Luma mask (less linear).
        :param show_mask:           1 = Show the first luma mask, 2 = Show the Chroma V Plane mask (if chroma_masking = True), 3 = Show the Chroma U Plane mask (if chroma_masking = True), 4 = Show the flatmask.

        :return:                    16bit denoised clip or luma_mask if show_mask is 1, 2 or 3.
        """
        

        core = vs.core
        from .admask import flat_mask, luma_mask_ping, luma_mask_man, luma_mask

        if clip.format.color_family not in {vs.YUV}:
            raise ValueError('adaptive_denoiser: only YUV formats are supported')

        if clip.format.bits_per_sample != 16:
            clip = depth(clip, 16)

        lumamask = luma_mask_ping(clip, thr=luma_mask_thr)
        darken_luma_mask = core.std.Expr([lumamask], f"x {luma_mask_weaken} *")

        #Degrain
        if "is_digital" not in kwargs:
            mvtools = MVTools(clip)
            vectors = mvtools.analyze(blksize=16, tr=tr, overlap=8, lsad=300, search=SearchMode.UMH, truemotion=MotionMode.SAD, dct=SADMode.MIXED_SATD_DCT)
            mfilter = mini_BM3D(clip=get_y(clip), sigma=sigma*1.5, radius=1, profile="LC", planes=0)
            mfilter = core.std.ShufflePlanes(clips=[mfilter, get_u(clip), get_v(clip)], planes=[0,0,0], colorfamily=vs.YUV)
            degrain = mc_degrain(clip, prefilter=Prefilter.DFTTEST, blksize=8, mfilter=mfilter, thsad=thsad, vectors=vectors, tr=tr, limit=1)
        else:
            degrain = clip

        if precision:
            flatmask = flat_mask(degrain, sigma=sigma*1.5)
            if show_mask == 4:
                return flatmask
            darken_luma_mask = core.std.Expr(
            [darken_luma_mask, flatmask],
            f"y 65535 = x {flat_penalty} * x {texture_penalty} * ?")
            
            darken_luma_mask = Morpho.deflate(Morpho.inflate(darken_luma_mask)) # Inflate+Deflate for smoothing

        if show_mask == 1:
            return darken_luma_mask
        
        denoised = mini_BM3D(get_y(degrain), sigma=sigma, radius=tr2, profile="HIGH", planes=0)
        luma = get_y(core.std.MaskedMerge(denoised, get_y(clip), darken_luma_mask, planes=0)) ##denoise applied to darker areas

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
        
        if show_mask == 2:
            return v_mask
        elif show_mask == 3:
            return u_mask
        
        final = core.std.ShufflePlanes(clips=[luma, get_u(chroma_denoised), get_v(chroma_denoised)], planes=[0,0,0], colorfamily=vs.YUV)
        return final
    
    # Presets
    @staticmethod
    def scan65mm (clip:vs.VideoNode, thsad: int = 200, tr: int = 2, tr2: int = 1, sigma: float = 2, luma_mask_weaken: float = 0.9, luma_mask_thr: float = 50, chroma_strength: float = 0.5, chroma_denoise: str = "nlm", precision: bool = True, chroma_masking: bool = False, show_mask: int = 0, flat_penalty: float = 0.5, texture_penalty: float = 1.1)->vs.VideoNode:
        """ changes: thsad=200, sigma=2, luma_mask_weaken=0.9, chroma_strength=0.5 """
        return adenoise._adaptive_denoiser(clip, thsad, tr, tr2, sigma, luma_mask_weaken, luma_mask_thr, chroma_strength, chroma_denoise, precision, chroma_masking, show_mask, flat_penalty, texture_penalty)
    @staticmethod
    def scan35mm (clip:vs.VideoNode, thsad: int = 400, tr: int = 2, tr2: int = 1, sigma: float = 4, luma_mask_weaken: float = 0.8, luma_mask_thr: float = 50, chroma_strength: float = 0.7, chroma_denoise: str = "nlm", precision: bool = True, chroma_masking: bool = False, show_mask: int = 0, flat_penalty: float = 0.5, texture_penalty: float = 1.1)->vs.VideoNode:
        """ changes: thsad=400, sigma=4, luma_mask_weaken=0.8, chroma_strength=0.7 """
        return adenoise._adaptive_denoiser(clip, thsad, tr, tr2, sigma, luma_mask_weaken, luma_mask_thr, chroma_strength, chroma_denoise, precision, chroma_masking, show_mask, flat_penalty, texture_penalty)
    
    @staticmethod
    def scan16mm (clip:vs.VideoNode, thsad: int = 600, tr: int = 2, tr2: int = 1, sigma: float = 8, luma_mask_weaken: float = 0.75, luma_mask_thr: float = 50, chroma_strength: float = 1.0, chroma_denoise: str = "nlm", precision: bool = True, chroma_masking: bool = False, show_mask: int = 0, flat_penalty: float = 0.5, texture_penalty: float = 1.1)->vs.VideoNode:
        """ changes: thsad=600, sigma=8 """
        return adenoise._adaptive_denoiser(clip, thsad, tr, tr2, sigma, luma_mask_weaken, luma_mask_thr, chroma_strength, chroma_denoise, precision, chroma_masking, show_mask, flat_penalty, texture_penalty)
    
    @staticmethod
    def scan8mm (clip:vs.VideoNode, thsad: int = 800, tr: int = 2, tr2: int = 2, sigma: float = 12, luma_mask_weaken: float = 0.75, luma_mask_thr: float = 50, chroma_strength: float = 1.5, chroma_denoise: str = "nlm", precision: bool = True, chroma_masking: bool = False, show_mask: int = 0, flat_penalty: float = 0.5, texture_penalty: float = 1.1)->vs.VideoNode:
        """ changes: thsad=800, sigma=12, tr2=2, chroma_strength=1.5 """
        return adenoise._adaptive_denoiser(clip, thsad, tr, tr2, sigma, luma_mask_weaken, luma_mask_thr, chroma_strength, chroma_denoise, precision, chroma_masking, show_mask, flat_penalty, texture_penalty)
    
    @staticmethod
    def digital (clip:vs.VideoNode, thsad: int = 300, tr: int = 2, tr2: int = 1, sigma: float = 3, luma_mask_weaken: float = 0.75, luma_mask_thr: float = 50, chroma_strength: float = 1.0, chroma_denoise: str = "nlm", precision: bool = True, chroma_masking: bool = False, show_mask: int = 0, flat_penalty: float = 0.5, texture_penalty: float = 1)->vs.VideoNode:
        """ changes: thsad=300, sigma=3, texture_penalty=1 """
        return adenoise._adaptive_denoiser(clip, thsad, tr, tr2, sigma, luma_mask_weaken, luma_mask_thr, chroma_strength, chroma_denoise, precision, chroma_masking, show_mask, flat_penalty, texture_penalty, is_digital=True)
    
    @staticmethod
    def default (clip:vs.VideoNode, thsad: int = 500, tr: int = 2, tr2: int = 1, sigma: float = 6, luma_mask_weaken: float = 0.75, luma_mask_thr: float = 50, chroma_strength: float = 1.0, chroma_denoise: str = "nlm", precision: bool = True, chroma_masking: bool = False, show_mask: int = 0, flat_penalty: float = 0.5, texture_penalty: float = 1.1)->vs.VideoNode:
        """ default profile """
        return adenoise._adaptive_denoiser(clip, thsad, tr, tr2, sigma, luma_mask_weaken, luma_mask_thr, chroma_strength, chroma_denoise, precision, chroma_masking, show_mask, flat_penalty, texture_penalty)

#TODO
#Ported from fvsfunc 
def auto_deblock(
    clip: vs.VideoNode,
    sigma: int = 15,
    tbsize: int = 1,
    luma_mask_strength: float = 0.9,
    mask_type: int = 0,
    planes: PlanesT = None
) -> vs.VideoNode:
    """
    Deblocker 8x8 and other.
    """

    core=vs.core
    from .admask import flat_mask, luma_mask_ping, luma_mask_man, luma_mask
    
    try:
        from functools import partial
    except ImportError:
        raise ImportError('functools is required')

    if clip.format.color_family not in [vs.YUV]:
        raise TypeError("AutoDeblock: clip must be YUV color family!")

    if clip.format.bits_per_sample != 16:
        clip = depth(clip, 16)

    predeblock = deblock_qed(clip.rgvs.RemoveGrain(2).rgvs.RemoveGrain(2), planes=planes)

    deblock = core.dfttest.DFTTest(predeblock, sigma=sigma, tbsize=tbsize, planes=planes)
    
    if (mask_type == 0):
        lumamask = luma_mask(clip)
    elif (mask_type == 1):
        lumamask = luma_mask_man(clip)
    else:
        lumamask = luma_mask_ping(clip)
    darken_luma_mask = core.std.Expr([lumamask], f"x {luma_mask_strength} *")
    final = core.std.MaskedMerge(deblock, clip, darken_luma_mask, planes=planes)

    return final

#TODO
def increase_dynamic(
    clip: vs.VideoNode,
    t: float = 0.7,
    s: float = 50,
    a: float = 300,
)-> vs.VideoNode:
    """
    Makes dark areas more dark, and white areas more white.
    
    You can use it if you lost dynamic but it's very subjective, using this filter will raise many eyebrows.
    """
    
    core=vs.core
    from .admask import flat_mask, luma_mask_ping, luma_mask_man, luma_mask

    lumamask = luma_mask_man(clip, t=t, s=s, a=a)
    lumamask = lumamask.std.Invert()
    u = get_u(clip)
    v = get_v(clip)
    lumamask = core.std.ShufflePlanes(clips=[lumamask, u, v], planes=[0,0,0], colorfamily=vs.YUV)
    clip = core.std.Merge(clip, lumamask, weight=0.1)
    return clip

def deblock(
    clip: vs.VideoNode
)-> vs.VideoNode:
    
    core = vs.core

    # stringhe di resto già pronte per 4x4 e 8x8, esempio se voglio il valore del primo pixel nel blocco 8x8 sarà "{hor8x8} 0 ="
    hor8x8 = "X 8 %"
    hor4x4 = "X 4 %"
    ver8x8 = "Y 8 %"
    ver4x4 = "Y 4 %"

    # COSTRUZIONE DELLA MASK
    # due metodi per rilevare differenze di luma ai bordi, due clip di valori, una orizzontale l'altra verticale
    sumhor8_pxdiff = "x x[0,-1] - abs x[1,0] x[1,-1] - abs + x[2,0] x[2,-1] - abs x[3,0] x[3,-1] - abs + + x[4,0] x[4,-1] - abs x[5,0] x[5,-1] - abs + x[6,0] x[6,-1] - abs x[7,0] x[7,-1] - abs + + +"
    sumhor8_pxdiff2 = "x[0,1] x[0,-2] - abs x[1,1] x[1,-2] - abs + x[2,1] x[2,-2] - abs x[3,1] x[3,-2] - abs + + x[4,1] x[4,-2] - abs x[5,1] x[5,-2] - abs + x[6,1] x[6,-2] - abs x[7,1] x[7,-2] - abs + + +"
    sumhor8_diffpx = "x x[1,0] + x[2,0] + x[3,0] + x[4,0] + x[5,0] + x[6,0] + x[7,0] + x[0,-1] x[1,-1] + x[2,-1] + x[3,-1] + x[4,-1] + x[5,-1] + x[6,-1] + x[7,-1] + - abs"
    sumhor4_diffpx = "x x[1,0] + x[2,0] + x[3,0] + x[0,-1] x[1,-1] + x[2,-1] + x[3,-1] + - abs"
    sumver8_pxdiff = "x x[-1,0] - abs x[0,1] x[-1,1] - abs + x[0,2] x[-1,2] - abs x[0,3] x[-1,3] - abs + + x[0,4] x[-1,4] - abs x[0,5] x[-1,5] - abs + x[0,6] x[-1,6] - abs x[0,7] x[-1,7] - abs + + +"
    sumver8_pxdiff2 = "x[1,0] x[-2,0] - abs x[1,1] x[-2,1] - abs + x[1,2] x[-2,2] - abs x[1,3] x[-2,3] - abs + + x[1,4] x[-2,4] - abs x[1,5] x[-2,5] - abs + x[1,6] x[-2,6] - abs x[1,7] x[-2,7] - abs + + +"
    sumver8_diffpx = "x x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5] + x[0,6] + x[0,7] + x[-1,0] x[-1,1] + x[-1,2] + x[-1,3] + x[-1,4] + x[-1,5] + x[-1,6] + x[-1,7] + - abs"
    sumver4_diffpx = "x x[0,1] + x[0,2] + x[0,3] + x[-1,0] x[-1,1] + x[-1,2] + x[-1,3] + - abs"

    horblockvalue = core.akarin.Expr([clip], f"{hor8x8} 0 = {ver8x8} 0 = {sumhor8_pxdiff} {sumhor8_pxdiff2} 0.33 * + {sumhor8_diffpx} + 2 / 0 ? 0 ?")
    horblockvalue4 = core.akarin.Expr([clip], f"{hor4x4} 0 = {ver4x4} 0 = {sumhor4_diffpx} 0 ? 0 ?")
    verblockvalue = core.akarin.Expr([clip], f"{hor8x8} 0 = {ver8x8} 0 = {sumver8_pxdiff} {sumver8_pxdiff2} 0.33 * + {sumver8_diffpx} + 2 / 0 ? 0 ?")
    verblockvalue4 = core.akarin.Expr([clip], f"{hor4x4} 0 = {ver4x4} 0 = {sumver4_diffpx} 0 ? 0 ?")

    #allungamento dei valori verticalmente e orizzontalmente
    h1 = f"{hor8x8} 1 = x[-1,0] x ?"
    h2 = f"{hor8x8} 2 = x[-2,0] {h1} ?"
    h3 = f"{hor8x8} 3 = x[-3,0] {h2} ?"
    h4 = f"{hor8x8} 4 = x[-4,0] {h3} ?"
    h5 = f"{hor8x8} 5 = x[-5,0] {h4} ?"
    h6 = f"{hor8x8} 6 = x[-6,0] {h5} ?"
    h7 = f"{hor8x8} 7 = x[-7,0] {h6} ?"
    horblockmask = core.akarin.Expr([horblockvalue], f"{h7}")
    h1 = f"{hor4x4} 1 = x[-1,0] x ?"
    h2 = f"{hor4x4} 2 = x[-2,0] {h1} ?"
    h3 = f"{hor4x4} 3 = x[-3,0] {h2} ?"
    horblockmask4 = core.akarin.Expr([horblockvalue4], f"{h3}")
    v1 = f"{ver8x8} 1 = x[0,-1] x ?"
    v2 = f"{ver8x8} 2 = x[0,-2] {v1} ?"
    v3 = f"{ver8x8} 3 = x[0,-3] {v2} ?"
    v4 = f"{ver8x8} 4 = x[0,-4] {v3} ?"
    v5 = f"{ver8x8} 5 = x[0,-5] {v4} ?"
    v6 = f"{ver8x8} 6 = x[0,-6] {v5} ?"
    v7 = f"{ver8x8} 7 = x[0,-7] {v6} ?"
    verblockmask = core.akarin.Expr([verblockvalue], f"{v7}")
    v1 = f"{ver4x4} 1 = x[0,-1] x ?"
    v2 = f"{ver4x4} 2 = x[0,-2] {v1} ?"
    v3 = f"{ver4x4} 3 = x[0,-3] {v2} ?"
    verblockmask4 = core.akarin.Expr([verblockvalue4], f"{v3}")

    hormaskshift_top1 = horblockmask.resize.Point(clip.width, clip.height, src_top=1)
    hormaskshift_top2 = horblockmask.resize.Point(clip.width, clip.height, src_top=2)
    hormaskshift_bot1 = horblockmask.resize.Point(clip.width, clip.height, src_top=-1)
    vermaskshift_left1 = verblockmask.resize.Point(clip.width, clip.height, src_left=1)
    vermaskshift_left2 = verblockmask.resize.Point(clip.width, clip.height, src_left=2)
    vermaskshift_rig1 = verblockmask.resize.Point(clip.width, clip.height, src_left=-1)

    #fusione delle mask orizzontali e verticali, poi due shift per allargare la mask 4x4 da 1px a 2px
    blockmask4 = core.akarin.Expr([horblockmask4, verblockmask4], "x y max")
    blockmaskshift1_4 = blockmask4.resize.Point(blockmask4.width, blockmask4.height, src_left=1)
    blockmaskshift2_4 = blockmask4.resize.Point(blockmask4.width, blockmask4.height, src_top=1)
    blockmaskfull8 = core.akarin.Expr([horblockmask, verblockmask, hormaskshift_top1, hormaskshift_top2, hormaskshift_bot1, vermaskshift_left1, vermaskshift_left2, vermaskshift_rig1], "x y max z max a max b max c max d max e max")
    blockmaskfull4 = core.akarin.Expr([blockmask4, blockmaskshift1_4, blockmaskshift2_4], "x y max z max")

    #potenziamento della mask, potrebbe essere utile inserire un thr variabile prima di questo potenziamento
    blockmaskfull8 = core.std.Expr([blockmaskfull8], "x 0.4 pow 700 *")
    blockmaskfull4 = core.std.Expr([blockmaskfull4], "x 0.4 pow 350 *")

    # DEBLOCK 4X4

    # dalla funzione 12: D = ( 3(s[0] – s[–1]) – ( s[1] – s[–2] ) ) / 2
    deltahor_0 = f"{hor4x4} 0 = x x[-1,0] - 3 * x[1,0] x[-2,0] - - 2 / 0 ?"
    deltahor_n1 = f"{hor4x4} 3 = x x[1,0] - 3 * x[-1,0] x[2,0] - - 2 / {deltahor_0} ?"
    deltaver_0 = f"{ver4x4} 0 = x x[0,-1] - 3 * x[0,1] x[0,-2] - - 2 / 0 ?"
    deltaver_n1 = f"{ver4x4} 3 = x x[0,1] - 3 * x[0,-1] x[0,2] - - 2 / {deltaver_0} ?"
    
    # nella funzione 13 c'è una funzione di clipping molto importante, modificata in quanto non abbiamo qp
    thr = 2000
    # funzione 14: s'[i] = s[i] – D (N – i) / (2N + 1), per un blocco 4x4 è costante: s'[i] = s[i] – D 1/3
    cost4x4 = 1/3
    f14_hor = f"x {deltahor_n1} {cost4x4} * -"
    f14_ver = f"x {deltaver_n1} {cost4x4} * -"
    deltahor = core.akarin.Expr([clip], f"{deltahor_n1} abs {thr} < {f14_hor} {deltahor_n1} 0 > x {thr} - x {thr} + ? ?")
    deltahor = core.akarin.Expr([deltahor], f"{hor4x4} 3 = x {hor4x4} 0 = x 0 ? ?")
    deltaver = core.akarin.Expr([clip], f"{deltaver_n1} abs {thr} < {f14_ver} {deltaver_n1} 0 > x {thr} - x {thr} + ? ?")
    deltaver = core.akarin.Expr([deltaver], f"{ver4x4} 3 = x {ver4x4} 0 = x 0 ? ?")
    delta = core.akarin.Expr([deltahor, deltaver], "x y max")
    deblock4x4 = core.std.MaskedMerge(clipa=clip, clipb=delta, mask=blockmaskfull4)

    # DEBLOCK 8X8

    # dalla funzione 12: D = ( 3(s[0] – s[–1]) – ( s[1] – s[–2] ) ) / 2
    deltahor_0 = f"{hor8x8} 0 = x x[-1,0] - 3 * x[1,0] x[-2,0] - - 2 / 0 ?"
    deltahor_n1 = f"{hor8x8} 7 = x x[1,0] - 3 * x[-1,0] x[2,0] - - 2 / {deltahor_0} ?"
    deltahor_1 = f"{hor8x8} 1 = x x[-1,0] - 3 * x[1,0] x[-2,0] - - 2 / 0 ?"
    deltahor_n2 = f"{hor8x8} 6 = x x[1,0] - 3 * x[-1,0] x[2,0] - - 2 / {deltahor_1} ?"
    deltaver_0 = f"{ver8x8} 0 = x x[0,-1] - 3 * x[0,1] x[0,-2] - - 2 / 0 ?"
    deltaver_n1 = f"{ver8x8} 7 = x x[0,1] - 3 * x[0,-1] x[0,2] - - 2 / {deltaver_0} ?"
    deltaver_1 = f"{ver8x8} 1 = x x[0,-1] - 3 * x[0,1] x[0,-2] - - 2 / 0 ?"
    deltaver_n2 = f"{ver8x8} 6 = x x[0,1] - 3 * x[0,-1] x[0,2] - - 2 / {deltaver_1} ?"

    # nella funzione 13 c'è una funzione di clipping molto importante, modificata in quanto non abbiamo qp
    thr = 2000
    # funzione 14: s'[i] = s[i] – D (N – i) / (2N + 1)
    f14_hor0 = f"x {deltahor_n1} 0.4 * -"
    f14_hor1 = f"x {deltahor_n2} 0.2 * -"
    f14_ver0 = f"x {deltaver_n1} 0.4 * -"
    f14_ver1 = f"x {deltaver_n2} 0.2 * -"
    deltahor = core.akarin.Expr([clip], f"{deltahor_n1} abs {thr} < {f14_hor0} {deltahor_n1} 0 > x {thr} - x {thr} + ? ?")
    deltahor = core.akarin.Expr([deltahor], f"{deltahor_n2} abs {thr} < {f14_hor1} {deltahor_n2} 0 > x {thr} - x {thr} + ? ?")
    deltahor = core.akarin.Expr([deltahor], f"{hor8x8} 7 = x {hor8x8} 6 = x {hor8x8} 1 = x {hor8x8} 0 = x 0 ? ? ? ?")
    deltaver = core.akarin.Expr([clip], f"{deltaver_n1} abs {thr} < {f14_ver0} {deltaver_n1} 0 > x {thr} - x {thr} + ? ?")
    deltaver = core.akarin.Expr([deltaver], f"{deltaver_n2} abs {thr} < {f14_ver1} {deltaver_n2} 0 > x {thr} - x {thr} + ? ?")
    deltaver = core.akarin.Expr([deltaver], f"{ver8x8} 7 = x {ver8x8} 6 = x {ver8x8} 1 = x {ver8x8} 0 = x 0 ? ? ? ?")
    delta = core.akarin.Expr([deltahor, deltaver], "x y max")
    deblock8x8 = core.std.MaskedMerge(clipa=clip, clipb=delta, mask=blockmaskfull8)

    # unione cancellando il deblock 4x4 dove c'è il blocco 8x8
    deleted4 = core.akarin.Expr([deblock4x4], f"{hor8x8} 5 = x {hor8x8} 4 = x {hor8x8} 3 = x {hor8x8} 2 = x 0 ? ? ? ?")
    deleted4 = core.akarin.Expr([deleted4], f"{ver8x8} 5 = x {ver8x8} 4 = x {ver8x8} 3 = x {ver8x8} 2 = x 0 ? ? ? ?")
    deleted8 = core.akarin.Expr([deblock8x8], f"{hor8x8} 7 = x {hor8x8} 6 = x {hor8x8} 1 = x {hor8x8} 0 = x {ver8x8} 7 = x {ver8x8} 6 = x {ver8x8} 1 = x {ver8x8} 0 = x 0 ? ? ? ? ? ? ? ?")
    union = core.akarin.Expr([deleted8, deleted4], "x y max")

    return union

#TODO
def msaa2x(
    clip: vs.VideoNode
) -> vs.VideoNode:
    """
    Upscales only the edges with AI (ArtCNN DN) and downscales them.
    """
    from vsscale import ArtCNN
    from addfunc import admask

    denoise = adenoise.scan65mm(clip, precision=False)
    emask = admask.edgemask(denoise, sigma=50, blur_radius=2)
    upsc = ArtCNN().C4F32_DN().scale(denoise, clip.width*2, clip.height*2)
    aa = core.resize.Spline16(upsc, clip.width, clip.height)
    merged = core.std.MaskedMerge(clip, aa, emask)
    return merged

