import vapoursynth as vs
from typing import Optional

core = vs.core

if not (hasattr(vs.core, 'cas') or hasattr(vs.core, 'fmtc') or hasattr(vs.core, 'akarin')):
    raise ImportError("'cas', 'fmtc' and 'akarin' are mandatory. Make sure the DLLs are present in the plugins folder.")


def _get_stdev(avg: float, sq_avg: float) -> float:
    return abs(sq_avg - avg ** 2) ** 0.5

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
    from vstools import get_y
    
    luma = get_y(clip)
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
    from vstools import get_y
    
    luma = get_y(clip)
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
        thr: float = 0.2,
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
    from vstools import get_y
    import math
    from .adutils import scale_binary_value

    bit_depth = clip.format.bits_per_sample
    max_val = (1 << bit_depth) - 1

    thr_scaled = scale_binary_value(clip, value=thr, bit=bit_depth, return_int=False)

    high_amp = (math.exp(low_amp - 1) + low_amp * math.exp(low_amp)) / (math.exp(low_amp) - 1)

    expr = (
        f"x {max_val} / "  # x_n
        f"dup {thr_scaled} < "  # condizione
        # ramo TRUE
        f"{thr_scaled} 1 + - exp {low_amp} + "
        # ramo FALSE
        f"{high_amp} {high_amp} dup {thr_scaled} 1 - - log {high_amp} * exp {low_amp} + / - "
        # ternario
        f"? "
        # moltiplica per x (valore originale)
        f"x *"
    )

    cc = core.akarin.Expr([get_y(clip)], expr)

    # Inverti il risultato
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
    from vstools import get_y, depth
    from .adutils import scale_binary_value

    def _add_stddev(n, f):
        core = vs.core
        avg    = float(f[0].props['PlaneStatsAverage'])
        avg_sq = float(f[1].props['PlaneStatsAverage'])
        stddev = _get_stdev(avg, avg_sq)
        return core.std.SetFrameProp(y, prop="std_dev", floatval=stddev)

    y = get_y(clip)

    if y.format.bits_per_sample != 16:
        y = depth(y, 16)

    # Add stats to the clip
    stats_avg = y.std.PlaneStats() 
    stats_sq = core.std.Expr([y], "x x *").std.PlaneStats() 

    y_std = core.std.FrameEval(y, _add_stddev, prop_src=[stats_avg, stats_sq], clip_src=[stats_avg, stats_sq])

    edges = edgemask(y, ref=ref, sigma=(sigma if sigma is not None else 0.0), blur_radius=blur_radius, thr=edge_thr)

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
        plane: int = 0
        ) ->vs.VideoNode:
    '''
    This is a custom edge mask (made by PingWer) based on well know operators with particular combination. Really effective as mask for dehalo or sharpening filtering.
    :param clip:        Clip to process.
    :param ref:         Reference clip for denoising (really usefull for grainy or noise content, to avoid details loss).
    :param sigma:       Sigma value for the BM3D denoiser. Higher values produce stronger denoising (this value should be higher then the standard, usually 1-2 for regular content, 4-5 for noisy).
    :param blur_radius: Blur radius for the box blur. Default is 1 (should be fine for most content, increse if it has a serious amount of blocking).
    :param thr:         Threshold for the edge detection. Value should be between 0-1. Lower values produce more edges (don't go lower then default).
    :param presharp:    Amount of sharpening to apply before denoising (never go higher then 0.8).
    :param postsharp:   Amount of sharpening to apply after denoising (never go higher then 0.8).
    :param plane:       Plane to process (0 for Y, 1 for U, 2 for V).
    :return:            Edge mask.
    '''
    
    core=vs.core
    from vstools import get_y, depth, get_u, get_v
    from .adutils import scale_binary_value
    from .adfunc import mini_BM3D
    from vsdenoise import nl_means

    if plane == 0:
        y = get_y(clip)
    elif plane == 1:
        y = get_u(clip)
    elif plane == 2:
        y = get_v(clip)
    else:
        return ValueError("Invalid plane number")
        
    if y.format.bits_per_sample != 16:
        y = depth(y, 16)

    thr_scaled = scale_binary_value(clip, thr, return_int=True)

    if presharp!=0:
        y=core.cas.CAS(y, sharpness=presharp, opt=0)

    if ref is None:
        y_dn = mini_BM3D(y, sigma=sigma, radius=1, profile="HIGH")
    else:
        y_dn = mini_BM3D(y, sigma=sigma, ref=ref, radius=1, profile="HIGH")

    if postsharp !=0:
        y_dn=core.cas.CAS(y_dn, sharpness=postsharp, opt=0)

    y_dn=nl_means(y_dn, h=1, tr=1, a=2)
    
    blurred1 = core.std.BoxBlur(y_dn, hradius=blur_radius, vradius=blur_radius)
    blurred2 = core.std.BoxBlur(y_dn, hradius=blur_radius * 2, vradius=blur_radius * 2)

    edges = core.std.Expr([
        y_dn.std.Sobel(),
        blurred1.std.Sobel(),
        blurred2.std.Prewitt()
    ], f"x {thr_scaled} > x 0 ?  y {thr_scaled} > y 0 ? + z {thr_scaled} > z 0 ? max")

    return edges
