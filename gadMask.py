try:
    import vapoursynth as vs
except ImportError:
    raise ImportError('Vapoursynth R70> is required. Download it via: pip install vapoursynth')

try:
    from vsdenoise import BM3DCuda, nl_means, Profile
    from vstools import get_y
    import math
except ImportError:
    raise ImportError('vsdenoise, vstools are required. Download them via: pip install vsjetpack. Other depedencies can be found here: https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack' )



def luma_mask (
        clip: vs.VideoNode,
        min_value: float = 17500,
        sthmax: float = 0.95,
        sthmin: float = 1.4,
)-> vs.VideoNode :
    
    core = vs.core
    
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
    core = vs.core
    
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
        thr: float = 50,
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

    bit_depth = clip.format.bits_per_sample
    max_val = (1 << bit_depth) - 1

    scaled_thr = thr * (max_val / 255.0)
    thr_scaled = scaled_thr / max_val

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

    cc = core.std.Expr(get_y(clip), expr)

    # Inverti il risultato
    cc = core.std.Invert(cc)

    return cc

def flat_mask(
        clip: vs.VideoNode, 
        blur_radius: int = 1, 
        tr: int = 1, 
        edge_thr_low: float = 0.001, 
        bm3d_sigma: float = 25, 
        edge_thr_high: float = None,
        debug: bool = False
        )-> vs.VideoNode:
    """
    This custom flat mask (Made By PingWer) is more conservative then JET one, so when a white flat area exit is 99% a real flat area (only if you use reasonable parameters).
    With high values of edge_thr_high you can get a really good edge mask.
    The default value are inteded to be used with noisy and grany video. It's reccomended to pass a not denoised clip, or at least a really light denoised one. 

    :param clip:            Clip to process.
    :param blur_radius:     Blur radius for the box blur. Default is 1 (should be fine for most content).
    :param tr:              Temporal radius for the BM3D denoiser (temporal radius must be the same of whatever temporal denoiser you are using).
    :param edge_thr_low:    Threshold for the low edge detection (ideally the impact of the changes may very, so is better to leave it as default).
    :param bm3d_sigma:      Sigma value for the BM3D denoiser. If None, a default value of 25.0 is used, this should be fine for most noisy content as default, but is better to set other sigma parameter based on the contet. Should be fine to use the same sigma value for all the clip, but need to be tested. 
    :param edge_thr_high:   Threshold for the high edge detection. If None, a default value is calculated based on the standard deviation of the clip (suggested to leave None, except you are doing scene filtering).
    :param debug:           If True, prints the standard deviation and threshold values for each frame.
    :return:                Flat mask.
    """
    core = vs.core

    y = get_y(clip)

    if clip.format.bits_per_sample != 16:
        clip = clip.fmtc.bitdepth(bits=16)

    # Add stats to the clip
    stats_avg = y.std.PlaneStats()
    sq_clip = core.std.Expr([y], "x x *")
    stats_sq = sq_clip.std.PlaneStats()

    # BM3DCuda must be available and the temporal radius must be the same of whatever temporal denoiser you are using
    # y_dn = BM3DCuda.denoise(y, sigma=bm3d_sigma, tr=tr, planes=0, profile=Profile.HIGH)
    y_dn = y.std.Median().std.Median()

    # Edge detection e blurring. Remove grain and noise from flat areas (intentionally remove some details)
    blurred1 = core.std.BoxBlur(y_dn, hradius=blur_radius, vradius=blur_radius)
    blurred2 = core.std.BoxBlur(blurred1, hradius=blur_radius * 2, vradius=blur_radius * 2)
    edges = core.std.Expr([
        y_dn.std.Sobel(),
        blurred1.std.Sobel(),
        blurred2.std.Prewitt()
    ], "x y max z max")

    mask_fine = edges.std.Binarize(threshold=int(edge_thr_low * 65535))

    def get_stdev(avg: float, sq_avg: float) -> float:
        return (sq_avg - avg ** 2) ** 0.5

    def auto_thr_high(stddev):
        if stddev >0.8:
            stdev_min, stdev_max = 0.8, 1.0
        else:
            stdev_min, stdev_max = 0.1, 1.00
        thr_high_min, thr_high_max = 0.005, 1.0
        norm = min(max((stddev - stdev_min) / (stdev_max - stdev_min), 0.0), 1.0)
        return (thr_high_max * ((thr_high_min / thr_high_max) ** norm))
    
    #Mask and thr operations
    def select_mask(n: int) -> vs.VideoNode:
        f_avg = stats_avg.get_frame(n)
        f_sq = stats_sq.get_frame(n)
        avg = float(f_avg.props.PlaneStatsAverage)
        sq_avg = float(f_sq.props.PlaneStatsAverage)
        stdev = get_stdev(avg, sq_avg)
        thr = edge_thr_high if edge_thr_high is not None else auto_thr_high(stdev)
        mask_medium = edges.std.Binarize(threshold=int(thr * 65535))
        mask = core.std.Expr([mask_fine, mask_medium], "x y min")
        mask = mask.std.Invert().std.Minimum().std.Maximum()
        if debug:
            print(f"Frame {n}: stdev={stdev}, sigma={bm3d_sigma}, thr_high={thr}")
        mask = mask.std.Median().std.Median()
        return mask

    return core.std.FrameEval(clip=y, eval=select_mask)
