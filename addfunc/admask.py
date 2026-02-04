import vapoursynth as vs
from typing import Optional

core = vs.core

if not (hasattr(vs.core, 'cas') or hasattr(vs.core, 'fmtc') or hasattr(vs.core, 'akarin')):
    raise ImportError("'cas', 'fmtc' and 'akarin' are mandatory. Make sure the DLLs are present in the plugins folder.")


def _get_stdev(avg: float, sq_avg: float) -> float:
    return abs(sq_avg - avg ** 2) ** 0.5

#fatta dalla IA e anche male, ma fa il suo la cambierò in futuro forse
def _soft_threshold(clip: vs.VideoNode, thr: float, steepness: float = 20.0) -> vs.VideoNode:
    """
    Applies a soft threshold to the clip.
    Values below thr accept a penalty that decays exponentially based on the distance from thr.
    formula: x < thr ? x * exp((x - thr) * steepness) : x
    """
    from .adutils import scale_binary_value
    
    core = vs.core
    
    thr_scaled = scale_binary_value(clip, thr, return_int=False)
    
    if clip.format.sample_type == vs.INTEGER:
        max_val = (1 << clip.format.bits_per_sample) - 1
        diff_expr = f"x {thr_scaled} - {max_val} /"
    else:
        diff_expr = f"x {thr_scaled} -"

    return core.akarin.Expr(
        [clip],
        f"x {thr_scaled} >= x "
        f"{diff_expr} {steepness} * exp x * "
        f"?"
    )

def auto_thr_high(stddev):
    if stddev >0.900:
        stdev_min, stdev_max = 0.800, 1.000
    else:
        stdev_min, stdev_max = 0.400, 1.000
    thr_high_min, thr_high_max = 0.005, 1.000
    norm = min(max((stddev - stdev_min) / (stdev_max - stdev_min), 0.000), 1.000)
    return (thr_high_max * ((thr_high_min / thr_high_max) ** norm))

def luma_mask (
        clip: vs.VideoNode,
        min_value: float = 17500,
        sthmax: float = 0.95,
        sthmin: float = 1.4,
)-> vs.VideoNode :
    from .adutils import plane
    
    luma = plane(clip, 0)
    lumamask = core.std.Expr(
        [luma],
        "x {0} < x {2} * x {0} - 0.0001 + log {1} * exp x + ?".format(min_value, sthmax, sthmin)
    )
    lumamask = lumamask.std.Invert()

    return lumamask

def luma_mask_man (
        clip: vs.VideoNode,
        t: float = 0.3,
        s: float = 5,
        a: float = 0.3,
)-> vs.VideoNode :
    """
    Custom luma mask that uses a different approach to calculate the mask (Made By Mhanz).
    This mask is sensitive to the brightness of the image producing a smooth transition between dark and bright areas of th clip based on brightness levels.
    The mask exalt bright areas and darkens dark areas, inverting them.
    
    Curve graph https://www.geogebra.org/calculator/cqnfnqyk
    
    :param clip:            Clip to process.
    :param s:               
    :param t:               Threshold that determines what is considered light and what is dark.
    :param a:               
    :return:                Luma mask.
    """
    from .adutils import plane
    
    luma = plane(clip, 0)
    f=1/3

    maxvalue = (1 << clip.format.bits_per_sample) - 1
    normx = f"x 2 * {maxvalue} / "

    lumamask = core.std.Expr(
        [luma],
        f"x "                  # Mettiamo x sullo stack per la moltiplicazione finale
        f"{normx} {t} < "            # x < b ?
        # - Ramo TRUE → f(x):
        f"{normx} {t} - {normx} {t} - 2 pow {s} * {a} + {f} pow / 1 + "
        # - Ramo FALSE → h(x):
        f"{normx} {t} - {normx} {t} - 2 pow {s} * 5 * {a} + {f} pow / 1 + "
        # - Operatore ternario:
        f"? "
        # - moltiplica per x:
        f"*"
    )

    lumamask = lumamask.std.Invert()

    return lumamask

def luma_mask_ping(
        clip: vs.VideoNode,
        low_amp: float = 0.8,
        thr: float = 0.196,
) -> vs.VideoNode:
    """
    Custom luma mask that uses a different approach to calculate the mask (Made By PingWer).
    This mask is sensitive to the brightness of the image, producing a constant dark mask for bright areas, 
    a constant white mask for very dark areas, and a exponential transition between these extremes based on brightness levels.

    Curve graph https://www.geogebra.org/calculator/fxbrx4s4

    :param clip:            Clip to process.
    :param low_amp:         General preamplification value, but more sensitive for values lower than thr.
    :param thr:             Threshold that determines what is considered bright and what is dark.
    :return:                Luma mask.
    """

    core = vs.core
    import math
    from .adutils import scale_binary_value, plane

    bit_depth = clip.format.bits_per_sample
    max_val = (1 << bit_depth) - 1

    thr_scaled = scale_binary_value(clip, value=thr, bit=bit_depth, return_int=False)

    high_amp = (math.exp(low_amp - 1) + low_amp * math.exp(low_amp)) / (math.exp(low_amp) - 1)

    expr = (
        f"x {max_val} / "
        f"dup {thr_scaled} < "
        f"{thr_scaled} 1 + - exp {low_amp} + "
        f"{high_amp} {high_amp} dup {thr_scaled} 1 - - log {high_amp} * exp {low_amp} + / - "
        f"? "
        f"x *"
    )

    cc = core.akarin.Expr([plane(clip, 0)], expr)

    cc = core.std.Invert(cc)

    return cc

def flat_mask(
        clip: vs.VideoNode, 
        blur_radius: int = 1,  
        sigma: Optional[float] = 2, 
        thr_high: Optional[float] = None, 
        thr_low: float = 0.03, 
        edge_thr: float = 0.015,
        debug: bool = False,
        ref: vs.VideoNode = None
        ) -> vs.VideoNode:
    """
    This custom flat mask (Made By PingWer) is more conservative then JET one, so when a white flat area exit is 99% a real flat area (only if you use reasonable parameters).
    With high values of edge_thr_high you can get a really good edge mask.
    The default values of sigma are inteded to be used with noisy and grany video. It's reccomended to pass a not denoised clip, or at least a really light denoised one in order to prevent detail loss. 

    :param clip:            Clip to process.
    :param blur_radius:     Blur radius for the box blur. Default is 1 (should be fine for most content, increse if it has a serious amount of blocking).
    :param thr_low:         Threshold for the low edge detection (This is a very sensible value, so it's better to leave it as default).
    :param sigma:           Sigma value for the BM3D denoiser. Higher values produce stronger denoising (this value should be higher then the standard, usually 1-2 for regular content, 4-5 for noisy).
    :param thr_high:        Threshold for the high edge detection. If None, a default value is calculated based on the standard deviation of the clip (suggested to leave None, except you are doing scene filtering).
    :param debug:           If True, prints the standard deviation and threshold values for each frame.
    :return:                Flat mask.
    """

    core = vs.core
    from vstools import depth
    from .adutils import scale_binary_value, plane

    def _add_stddev(n, f):
        core = vs.core
        avg    = float(f[0].props['PlaneStatsAverage'])
        avg_sq = float(f[1].props['PlaneStatsAverage'])
        stddev = _get_stdev(avg, avg_sq)
        return core.std.SetFrameProp(y, prop="std_dev", floatval=stddev)

    y = depth(plane(clip, 0), 16)

    # Add stats to the clip
    stats_avg = y.std.PlaneStats() 
    stats_sq = core.std.Expr([y], "x x *").std.PlaneStats() 

    y_std = core.std.FrameEval(y, _add_stddev, prop_src=[stats_avg, stats_sq], clip_src=[stats_avg, stats_sq])

    edges = edgemask(y, ref=ref, sigma=(sigma if sigma is not None else 0.0), blur_radius=blur_radius, thr=edge_thr, chroma=False)

    mask_fine = edges.std.Binarize(threshold=scale_binary_value(edges, thr_low, return_int=True))

    matrix3x3 = "x x[-1,-1] + x[0,-1] x[1,-1] + + x[-1,0] x[1,0] + x[-1,1] x[0,1] + + + x[1,1] +"
    matrix5x5_ext = " x[-2,-1] x[-2,0] + x[-2,1] + x[-1,-2] x[0,-2] + x[1,-2] + + x[-1,2] x[0,2] + x[1,2] + x[2,-1] x[2,0] + x[2,1] + + +"
    akarin_expr= f"{matrix3x3} {matrix5x5_ext} + 660000 > 65535 0 ?"

    def select_mask(n, f) -> vs.VideoNode:
        stdev = f.props['std_dev']
        thr = thr_high if thr_high is not None else auto_thr_high(stdev)
        mask_medium = edges.std.Binarize(scale_binary_value(edges, thr, return_int=True))
        mask = core.akarin.Expr([mask_fine, mask_medium], "x y min").std.Invert()
        mask = mask.std.Minimum().std.Maximum()
        if debug:
            print(f"Frame {n}: stdev={stdev}, sigma={sigma}, thr_high={thr}")
    
        final = mask.akarin.Expr(akarin_expr)
        return final
    return core.std.FrameEval(y, clip_src=[edges, mask_fine], prop_src=[y_std], eval=lambda n,f: select_mask(n,f))


def edgemask(
        clip: vs.VideoNode, 
        ref:Optional[vs.VideoNode] = None, 
        sigma: float = 10, 
        blur_radius: int = 1, 
        thr: float = 0.015,
        presharp : float = 0.5,
        postsharp : float = 0.6,
        chroma: bool = False
        ) -> vs.VideoNode:
    '''
    This is a custom edge mask (made by PingWer) based on well know operators with particular combination. Really effective as mask for dehalo or sharpening filtering.

    :param clip:        Clip to process.
    :param ref:         Reference clip for denoising (really usefull for grainy or noise content, to avoid details loss).
    :param sigma:       Sigma value for the BM3D denoiser. Higher values produce stronger denoising (this value should be higher then the standard, usually 1-2 for regular content, 4-5 for noisy).
    :param blur_radius: Blur radius for the box blur. Default is 1 (should be fine for most content, increse if it has a serious amount of blocking).
    :param thr:         Threshold for the edge detection. Value should be between 0-1. Lower values produce more edges (don't go lower then default).
    :param presharp:    Amount of sharpening to apply before denoising (never go higher then 0.8).
    :param postsharp:   Amount of sharpening to apply after denoising (never go higher then 0.8).
    :param chroma:      If True, returns a YUV clip with edge masks for all planes. If False (default), returns Luma edge mask (Gray).
    :return:            Edge mask.
    '''
    
    core=vs.core
    from vstools import depth, join, split
    from .adutils import scale_binary_value, plane
    from .adfunc import mini_BM3D
    from vsdenoise import nl_means
    from vsmasktools import Morpho

    if clip.format.color_family == vs.RGB:
        raise ValueError("edgemask: RGB clips are not supported, yet")

    if chroma and clip.format.color_family == vs.YUV:
        work_clip = clip
        work_ref = ref
        plane=[0,1,2]
    else:
        work_clip = plane(clip, 0)
        work_ref = plane(ref, 0) if ref is not None else None
        plane=[0]

    if work_clip.format.bits_per_sample != 16:
        work_clip = depth(work_clip, 16)
        if work_ref is not None and work_ref.format.bits_per_sample != 16:
             work_ref = depth(work_ref, 16)

    thr_scaled = scale_binary_value(work_clip, thr, return_int=True)

    if presharp != 0:
        work_clip = core.cas.CAS(work_clip, sharpness=presharp, opt=0, planes=plane)

    if work_ref is None:
        y_dn = mini_BM3D(work_clip, sigma=sigma, radius=1, profile="HIGH")
    else:
        y_dn = mini_BM3D(work_clip, sigma=sigma, ref=work_ref, radius=1, profile="HIGH")

    if postsharp != 0:
        y_dn = core.cas.CAS(y_dn, sharpness=postsharp, opt=0, planes=plane)

    y_dn = nl_means(y_dn, h=1, tr=1, a=2)
    
    blurred1 = core.std.BoxBlur(y_dn, hradius=blur_radius, vradius=blur_radius)
    blurred2 = core.std.BoxBlur(y_dn, hradius=blur_radius * 2, vradius=blur_radius * 2)

    edges = Morpho.inflate(core.std.Expr([
        y_dn.std.Sobel(),
        blurred1.std.Sobel(),
        blurred2.std.Prewitt()
    ], f"x {thr_scaled} > x 0 ?  y {thr_scaled} > y 0 ? + z {thr_scaled} > z 0 ? max"), iterations=2)

    if not chroma or edges.format.color_family == vs.GRAY:
        return edges

    planes = split(edges)
    y_mask = planes[0]
    u_mask_raw = planes[1]
    v_mask_raw = planes[2]

    y_mask_down = y_mask.resize.Bilinear(width=u_mask_raw.width, height=u_mask_raw.height)

    u_mask = core.std.Expr([u_mask_raw, y_mask_down], "x y max")
    v_mask = core.std.Expr([v_mask_raw, y_mask_down], "x y max")

    return join([y_mask, u_mask, v_mask], family=vs.YUV)


def unbloat_retinex(
    clip: vs.VideoNode,
    sigma: list[float] = [25, 80, 250],
    lower_thr: float = 0.001,
    upper_thr: float = 0.005,
    fast: bool = True
    ) -> vs.VideoNode:
    """
    Multi-Scale Retinex (MSR) optimized for edge enhancement and dynamic range compression.
    
    Uses Gaussian blur (via vsrgtools.gauss_blur) with optional downscaling optimization
    for large sigma values. The output is a normalized [0, 1] Float32 grayscale clip
    suitable for edge detection or as a preprocessing step for masking.
    
    :param clip:        Input clip. Must be Grayscale (GRAY format).
    :param sigma:       List of sigma values for multi-scale blur. Default: [25, 80, 250].
                        Larger values capture coarser illumination variations.
    :param lower_thr:   Quantile threshold for black level (0-1). Default: 0.001 (0.1%).
                        Used to ignore dark outliers during final normalization.
    :param upper_thr:   Quantile threshold for white level (0-1). Default: 0.001 (0.1%).
                        Used to ignore bright outliers during final normalization.
    :param fast:        Enable downscaling optimization for large sigmas. Default: True.
                        Significantly faster with minimal quality loss.
    :return:            Processed Float32 Grayscale clip with enhanced local contrast.
    
    :raises ValueError: If input clip is not Grayscale.
    """
    from vstools import depth
    from vsrgtools import gauss_blur
    
    if clip.format.color_family != vs.GRAY:
        raise ValueError("unbloat_retinex: Input must be a Grayscale clip.")
    
    if clip.format.sample_type != vs.FLOAT:
        luma = depth(clip, 32)
    else:
        luma = clip
        
    stats = luma.std.PlaneStats()
    luma_norm = core.akarin.Expr([luma, stats], "x y.PlaneStatsMin - y.PlaneStatsMax y.PlaneStatsMin - 0.000001 max /")

    sigmas_sorted = sorted(sigma)
    sigmas_to_blur = sigmas_sorted[:-1] if fast else sigmas_sorted
        
    blurs = []
    w, h = luma_norm.width, luma_norm.height
    
    for s in sigmas_to_blur:
        if fast and s > 6:
            ds_ratio = max(1, s / 3)
            ds_w, ds_h = max(1, int(w / ds_ratio)), max(1, int(h / ds_ratio))
            
            if ds_ratio > 2:
                down = luma_norm.resize.Bicubic(ds_w, ds_h)
                
                s_down = s / ds_ratio
                blurred_down = gauss_blur(down, sigma=s_down)
                
                blurred = blurred_down.resize.Bicubic(w, h)
            else:
                 blurred = gauss_blur(luma_norm, sigma=s)
        else:
             blurred = gauss_blur(luma_norm, sigma=s)
        blurs.append(blurred)
        
    inputs = [luma_norm] + blurs
    
    def get_char(i):
        if i == 0: return 'x'
        if i == 1: return 'y'
        if i == 2: return 'z'
        if i == 3: return 'a'
        return chr(ord('a') + (i - 3))

    terms = []
    for i in range(1, len(inputs)):
        c = get_char(i)
        terms.append(f"{c} 0 <= 1 x {c} / 1 + ?")
        
    if fast:
        terms.append("x 1 +")

    expr_code = " ".join(terms)
    if len(terms) > 1:
         expr_code += " " + " ".join(["+"] * (len(terms) - 1))
         
    slen = len(sigma)
    expr_code += f" log {slen} /"
    
    msr = core.akarin.Expr(inputs, expr_code)
    
    if hasattr(core, 'vszip') and (lower_thr > 0 or upper_thr > 0):
        msr_stats = core.vszip.PlaneMinMax(msr, lower_thr, upper_thr)
        min_key, max_key = 'psmMin', 'psmMax'
    else:
        msr_stats = msr.std.PlaneStats()
        min_key, max_key = 'PlaneStatsMin', 'PlaneStatsMax'

    balanced = core.akarin.Expr([msr, msr_stats], f"x y.{min_key} - y.{max_key} y.{min_key} - 0.000001 max /")
    
    return balanced


def advanced_edgemask(
    clip: vs.VideoNode,
    ref: Optional[vs.VideoNode] = None,
    sigma1: float = 3,
    retinex_sigma: list[float] = [50, 200, 350],
    sigma2: float = 1,
    sharpness: float = 0.8,
    kirsch_weight: float = 0.5,
    kirsch_thr: float = 0.35,
    edge_thr: float = 0.02,
    **kwargs
) -> vs.VideoNode:
    """
    Advanced edge mask combining Retinex preprocessing with multiple edge detectors.
    
    This mask uses BM3D denoising + Multi-Scale Retinex to enhance edges before detection,
    then combines Sobel, Prewitt, TCanny and Kirsch edge detectors for robust edge detection.
    
    :param clip:                Clip to process (YUV or Gray).
    :param ref:                 Optional reference clip for denoising.
    :param sigma1:              BM3D sigma for initial denoising. Default: 3.
    :param retinex_sigma:       Sigma values for Multi-Scale Retinex. Default: [50, 200, 350].
    :param sigma2:              Nlmeans strength for post-Retinex denoising. Default: 1.
    :param sharpness:           CAS sharpening amount (0-1). Default: 0.8.
    :param kirsch_weight:       Weight for Kirsch edges in final blend (0-1). Default: 0.7.
    :param kirsch_thr:          Kirsch threshold. Default: 0.25.
    :param edge_thr:            Threshold for edge combination logic (0-1). Default: 0.02.
    :param kwargs:              Additional arguments for Retinex.
    :return:                    Edge mask (Gray clip).
    """
    from vstools import get_y, depth
    from vsdenoise import nl_means
    from vsmasktools import Morpho, Kirsch
    from .adfunc import mini_BM3D
    from .adutils import scale_binary_value
    
    core = vs.core
    
    if clip.format.color_family == vs.RGB:
        raise ValueError("advanced_edgemask: RGB clips are not supported.")
    
    if clip.format.color_family == vs.GRAY:
        luma = clip
    else:
        luma = get_y(clip)
    
    luma = depth(luma, 16)

    if ref is not None:
        if ref.format.color_family == vs.RGB:
            raise ValueError("advanced_edgemask: RGB reference clips are not supported.")
        
        if ref.format.color_family == vs.GRAY:
            ref_y = ref
        else:
            ref_y = get_y(ref)
            
        if ref_y.format.bits_per_sample != 16:
            ref_y = depth(ref_y, 16)
        clipd = mini_BM3D(luma, sigma=sigma1, ref=ref_y, radius=1, profile="HIGH", planes=0)
    else:
        clipd = mini_BM3D(luma, sigma=sigma1, radius=1, profile="HIGH", planes=0)
    
    msrcpa = depth(unbloat_retinex(
        depth(clipd, 32), 
        sigma=retinex_sigma,
        fast=True,
        **kwargs
    ), 16, dither_type="none")
    
    msrcp = nl_means(msrcpa, h=sigma2, a=2)
    
    if sharpness > 0:
        msrcp = core.cas.CAS(msrcp, sharpness=sharpness, opt=0, planes=0)
        clipd = core.cas.CAS(clipd, sharpness=sharpness, opt=0, planes=0)
    
    preSobel = core.akarin.Expr([
        get_y(msrcp).std.Sobel(),
        get_y(clipd).std.Sobel(),
    ], "x y max")
    
    prePrewitt = core.akarin.Expr([
        get_y(msrcp).std.Prewitt(),
        get_y(clipd).std.Prewitt(),
    ], "x y max")
    
    edges = core.akarin.Expr([preSobel, prePrewitt], "x y +")
    
    tcanny = core.akarin.Expr([
        core.tcanny.TCanny(get_y(msrcp), mode=1, sigma=0),
        core.tcanny.TCanny(get_y(clipd), mode=1, sigma=0)
    ], "x y max")
    
    kirco = core.akarin.Expr([
        Kirsch.edgemask(get_y(msrcp), clamp=False, lthr=kirsch_thr),
        Kirsch.edgemask(get_y(clipd), clamp=False, lthr=kirsch_thr)
    ], "x y max")
    
    edge_thr_scaled = scale_binary_value(edges, edge_thr, return_int=True)
    mask = core.akarin.Expr(
        [edges, tcanny, kirco], 
        f"x y + {edge_thr_scaled} < x y + z {kirsch_weight} * + x y + ?"
    )

    return mask


def godflatmask(
    clip: vs.VideoNode,
    ref: Optional[vs.VideoNode] = None,
    sigma1: float = 3,
    retinex_sigma: list[float] = [50, 200, 350],
    sigma2: float = 1,
    sharpness: float = 0.8,
    edge_thr: float = 0.55,
    texture_strength: float = 2,
    edges_strength: float = 0.02,
    blur: float = 2,
    expand: int = 3,
    **kwargs
) -> vs.VideoNode:
    """
    Advanced edge mask combining Retinex preprocessing with multiple edge detectors.
    
    This mask uses BM3D denoising + Multi-Scale Retinex to enhance edges before detection,
    then combines Sobel, Prewitt, TCanny and Kirsch edge detectors for robust edge detection.
    
    :param clip:                Clip to process (YUV or Gray).
    :param ref:                 Optional reference clip for denoising.
    :param sigma1:              BM3D sigma for initial denoising. Default: 3.
    :param retinex_sigma:       Sigma values for Multi-Scale Retinex. Default: [50, 200, 350].
    :param sigma2:              Nlmeans strength for post-Retinex denoising. Default: 1.
    :param sharpness:           CAS sharpening amount (0-1). Default: 0.8.
    :param edge_thr:            Threshold for edge combination logic (0-1). This allows to separate edges from texture. Default: 0.55. 
    :param texture_strength:    Texture strength for mask (0-inf). Values above 1 decrese the strength of the texture in the mask, lower values increase it. The max value is theoretical infinite, but there is no gain after some point. Default: 0.8. 
    :param edges_strength:      Edges strength for mask (0-1). Basic multiplier for edges strength. Default: 0.03.
    :param blur:                Blur amount for mask (0-1). Default: 2.
    :param expand:              Expand amount for mask (0-1). Higher value increases the size of the texture in the mask. Default: 3.
    :param kwargs:              Additional arguments for Retinex.
    :return:                    Edge mask (Gray clip) where dark values are texture and edges, bright values are flat areas.
    """

    from vstools import get_y, depth
    from vsdenoise import nl_means
    from vsmasktools import Morpho, Kirsch, XxpandMode
    from .adfunc import mini_BM3D
    from .adutils import scale_binary_value
    from vsrgtools import gauss_blur
    
    core = vs.core
    
    if clip.format.color_family == vs.RGB:
        raise ValueError("advanced_edgemask: RGB clips are not supported.")
    
    if clip.format.color_family == vs.GRAY:
        luma = clip
    else:
        luma = get_y(clip)
    
    if luma.format.bits_per_sample != 16:
        luma = depth(luma, 16)

    if ref is not None:
        if ref.format.color_family == vs.RGB:
            raise ValueError("advanced_edgemask: RGB reference clips are not supported.")
        
        if ref.format.color_family == vs.GRAY:
            ref_y = ref
        else:
            ref_y = get_y(ref)
            
        if ref_y.format.bits_per_sample != 16:
            ref_y = depth(ref_y, 16)
        clipd = mini_BM3D(luma, sigma=sigma1, ref=ref_y, radius=1, profile="HIGH", planes=0)
    else:
        clipd = mini_BM3D(luma, sigma=sigma1, radius=1, profile="HIGH", planes=0)

    msrcpa = depth(unbloat_retinex(
        depth(clipd, 32), 
        sigma=retinex_sigma,
        fast=True,
        **kwargs
    ), 16, dither_type="none")
    
    msrcp = nl_means(msrcpa, h=sigma2, a=2)
    
    if sharpness > 0:
        msrcp = core.cas.CAS(msrcp, sharpness=sharpness, opt=0, planes=0)
        clipd = core.cas.CAS(clipd, sharpness=sharpness, opt=0, planes=0)
    
    edges = core.akarin.Expr([
        msrcp.std.Sobel(),
        clipd.std.Sobel(),
        msrcp.std.Prewitt(),
        clipd.std.Prewitt()
    ], "x y max z a max +")

    if edge_thr > 0:
        edges = _soft_threshold(edges, edge_thr, 10)
    
    tcanny = core.akarin.Expr([
        core.tcanny.TCanny(msrcp, mode=1, sigma=0),
        core.tcanny.TCanny(clipd, mode=1, sigma=0)
    ], "x y max")
    tcanny = core.std.Minimum(tcanny)

    edgescombo = Morpho.inflate(core.akarin.Expr([edges, tcanny], "x y +"), iterations=2)
    
    kirco = core.akarin.Expr([
        Kirsch.edgemask(msrcp),
        Kirsch.edgemask(clipd)
    ], "x y +")

    edges_expanded = Morpho.expand(edgescombo, mode=XxpandMode.ELLIPSE, sw=1, sh=1)
    kirco_diff = core.akarin.Expr([kirco, edges_expanded], "x y -")
    kirco_expanded = Morpho.expand(kirco_diff, mode=XxpandMode.ELLIPSE, sw=expand, sh=expand)

    edgescombo = core.akarin.Expr(edgescombo.std.Invert(), f"x {scale_binary_value(luma, edges_strength, return_int=True)} +")
    kirco_expanded = luma_mask_man(kirco_expanded, t=0.001, s=texture_strength, a=0.5)

    mask = core.akarin.Expr([edgescombo.std.Invert(), kirco_expanded.std.Invert()], "x y +")

    mask = gauss_blur(mask, blur)

    return mask
