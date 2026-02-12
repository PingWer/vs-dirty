import vapoursynth as vs
from typing import Optional

core = vs.core

if not (
    hasattr(vs.core, "cas") and hasattr(vs.core, "fmtc") and hasattr(vs.core, "akarin")
):
    raise ImportError(
        "'cas', 'fmtc' and 'akarin' are mandatory. Make sure the DLLs are present in the plugins folder."
    )


def _soft_threshold(
    clip: vs.VideoNode, thr: float, steepness: float = 20.0
) -> vs.VideoNode:
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
        [clip], f"x {thr_scaled} >= x {diff_expr} {steepness} * exp x * ?"
    )


def luma_mask(
    clip: vs.VideoNode,
    min_value: float = 17500,
    sthmax: float = 0.95,
    sthmin: float = 1.4,
) -> vs.VideoNode:
    from .adutils import plane

    luma = plane(clip, 0)
    lumamask = core.std.Expr(
        [luma],
        "x {0} < x {2} * x {0} - 0.0001 + log {1} * exp x + ?".format(
            min_value, sthmax, sthmin
        ),
    )
    lumamask = lumamask.std.Invert()

    return lumamask


def luma_mask_man(
    clip: vs.VideoNode,
    t: float = 0.3,
    s: float = 5,
    a: float = 0.3,
) -> vs.VideoNode:
    """
    Custom luma mask that uses a different approach to calculate the mask (Made By Mhanz).
    This mask is sensitive to the brightness of the image producing a smooth transition between dark and bright areas of th clip based on brightness levels.
    The mask exalt bright areas and darkens dark areas, inverting them.

    Curve graph https://www.geogebra.org/calculator/cqnfnqyk

    :param clip:            Clip to process (only the first plane will be processed).
    :param s:
    :param t:               Threshold that determines what is considered light and what is dark.
    :param a:

    :return:                Luma mask.
    """
    from .adutils import plane

    luma = plane(clip, 0)
    f = 1 / 3

    maxvalue = (1 << clip.format.bits_per_sample) - 1
    normx = f"x 2 * {maxvalue} / "

    lumamask = core.std.Expr(
        [luma],
        f"x "
        f"{normx} {t} < "
        f"{normx} {t} - {normx} {t} - 2 pow {s} * {a} + {f} pow / 1 + "
        f"{normx} {t} - {normx} {t} - 2 pow {s} * 5 * {a} + {f} pow / 1 + "
        f"? "
        f"*",
    )

    lumamask = lumamask.std.Invert()

    return lumamask


def luma_mask_ping(
    clip: vs.VideoNode,
    low_amp: float = 0.8,
    thr: float = 0.196,
    linear: bool = False,
    blur: float = 2,
) -> vs.VideoNode:
    """
    Custom luma mask that uses a different approach to calculate the mask (Made By PingWer).
    This mask is sensitive to the brightness of the image, producing a constant dark mask for bright areas,
    a constant white mask for very dark areas, and a exponential transition between these extremes based on brightness levels.

    Curve graph https://www.geogebra.org/calculator/fxbrx4s4

    :param clip:            Clip to process (only the first plane will be processed).
    :param low_amp:         General preamplification value, but more sensitive for values lower than thr.
    :param thr:             Threshold that determines what is considered bright and what is dark.
    :param linear:          If True (default), uses a linear amplification method (Legacy).
                            If False, uses the original exponential formula which is mathematically more precise and more aggressive.
    :param blur:            Sigma value for Gaussian blur. Default: 2. Set to 0 to disable.

    :return:                Luma mask.
    """

    core = vs.core
    import math
    from .adutils import plane
    from vsrgtools import gauss_blur

    bit_depth = clip.format.bits_per_sample
    max_val = (1 << bit_depth) - 1

    high_amp = (math.exp(low_amp - 1) + low_amp * math.exp(low_amp)) / (
        math.exp(low_amp) - 1
    )

    # Legacy (often more useful)
    expr1 = f"x {low_amp} *"

    # Exponential expression
    expr2 = (
        f"x {max_val} / {thr} < "
        f"x {max_val} / {thr} 1 + - exp {low_amp} + "
        f"{high_amp} {high_amp} dup {thr} 1 - - log {high_amp} * exp {low_amp} + / - "
        f"? "
        f"x *"
    )

    cc = core.akarin.Expr([plane(clip, 0)], expr1 if linear else expr2)

    cc = core.std.Invert(cc)

    return gauss_blur(cc, sigma=blur) if blur != 0 else cc


def unbloat_retinex(
    clip: vs.VideoNode,
    sigma: list[float] = [25, 80, 250],
    lower_thr: float = 0.001,
    upper_thr: float = 0.005,
    fast: bool = True,
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
    luma_norm = core.akarin.Expr(
        [luma, stats],
        "x y.PlaneStatsMin - y.PlaneStatsMax y.PlaneStatsMin - 0.000001 max /",
    )

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
        if i == 0:
            return "x"
        if i == 1:
            return "y"
        if i == 2:
            return "z"
        if i == 3:
            return "a"
        return chr(ord("a") + (i - 3))

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

    if hasattr(core, "vszip") and (lower_thr > 0 or upper_thr > 0):
        msr_stats = core.vszip.PlaneMinMax(msr, lower_thr, upper_thr)
        min_key, max_key = "psmMin", "psmMax"
    else:
        msr_stats = msr.std.PlaneStats()
        min_key, max_key = "PlaneStatsMin", "PlaneStatsMax"

    balanced = core.akarin.Expr(
        [msr, msr_stats], f"x y.{min_key} - y.{max_key} y.{min_key} - 0.000001 max /"
    )

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
    expand: int = 0,
    **kwargs,
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
    :param expand:              Expand the mask by a given number of pixels. Default: 0.
    :param kwargs:              Additional arguments for Retinex.
    :return:                    Edge mask (Gray clip).
    """
    from vstools import depth
    from vsdenoise import nl_means
    from vsmasktools import Kirsch, Morpho, XxpandMode
    from .adfunc import mini_BM3D
    from .adutils import scale_binary_value, plane

    core = vs.core

    if clip.format.color_family == vs.RGB:
        raise ValueError("advanced_edgemask: RGB clips are not supported.")

    if clip.format.color_family == vs.GRAY:
        luma = clip
    else:
        luma = plane(clip, 0)

    luma = depth(luma, 16)

    if ref is not None:
        if ref.format.color_family == vs.RGB:
            raise ValueError(
                "advanced_edgemask: RGB reference clips are not supported."
            )

        if ref.format.color_family == vs.GRAY:
            ref_y = ref
        else:
            ref_y = plane(ref, 0)

        if ref_y.format.bits_per_sample != 16:
            ref_y = depth(ref_y, 16)
        clipd = mini_BM3D(
            luma, sigma=sigma1, ref=ref_y, radius=1, profile="HIGH", planes=0
        )
    else:
        clipd = mini_BM3D(luma, sigma=sigma1, radius=1, profile="HIGH", planes=0)

    msrcpa = depth(
        unbloat_retinex(depth(clipd, 32), sigma=retinex_sigma, fast=True, **kwargs),
        16,
        dither_type="none",
    )

    msrcp = nl_means(msrcpa, h=sigma2, a=2)

    if sharpness > 0:
        msrcp = core.cas.CAS(msrcp, sharpness=sharpness, opt=0, planes=0)
        clipd = core.cas.CAS(clipd, sharpness=sharpness, opt=0, planes=0)

    preSobel = core.akarin.Expr(
        [
            plane(msrcp, 0).std.Sobel(),
            plane(clipd, 0).std.Sobel(),
        ],
        "x y max",
    )

    prePrewitt = core.akarin.Expr(
        [
            plane(msrcp, 0).std.Prewitt(),
            plane(clipd, 0).std.Prewitt(),
        ],
        "x y max",
    )

    edges = core.akarin.Expr([preSobel, prePrewitt], "x y +")

    tcanny = core.akarin.Expr(
        [
            core.tcanny.TCanny(plane(msrcp, 0), mode=1, sigma=0),
            core.tcanny.TCanny(plane(clipd, 0), mode=1, sigma=0),
        ],
        "x y max",
    )

    kirco = core.akarin.Expr(
        [
            Kirsch.edgemask(plane(msrcp, 0), clamp=False, lthr=kirsch_thr),
            Kirsch.edgemask(plane(clipd, 0), clamp=False, lthr=kirsch_thr),
        ],
        "x y max",
    )

    edge_thr_scaled = scale_binary_value(edges, edge_thr, return_int=True)
    mask = core.akarin.Expr(
        [edges, tcanny, kirco],
        f"x y + {edge_thr_scaled} < x y + z {kirsch_weight} * + x y + ?",
    )

    return (
        mask
        if expand == 0
        else Morpho.expand(mask, mode=XxpandMode.ELLIPSE, sw=expand, sh=expand)
    )


def hd_flatmask(
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
    fast: bool = True,
    **kwargs,
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

    from vstools import depth
    from vsdenoise import nl_means
    from vsmasktools import Morpho, Kirsch, XxpandMode
    from .adfunc import mini_BM3D
    from .adutils import scale_binary_value, plane
    from vsrgtools import gauss_blur

    core = vs.core

    if clip.format.color_family == vs.RGB:
        raise ValueError("advanced_edgemask: RGB clips are not supported.")

    if clip.format.color_family == vs.GRAY:
        luma = clip
    else:
        luma = plane(clip, 0)

    luma = depth(luma, 16)

    if ref is not None:
        if ref.format.color_family == vs.RGB:
            raise ValueError(
                "advanced_edgemask: RGB reference clips are not supported."
            )

        if ref.format.color_family == vs.GRAY:
            ref_y = ref
        else:
            ref_y = plane(ref, 0)

        ref_y = depth(ref_y, 16)

        clipd = mini_BM3D(
            luma,
            sigma=sigma1,
            ref=ref_y,
            radius=1,
            profile="FLATMASK" if fast else "HIGH",
            planes=0,
        )
    else:
        clipd = mini_BM3D(
            luma,
            sigma=sigma1,
            radius=1,
            profile="FLATMASK" if fast else "HIGH",
            planes=0,
        )

    msrcpa = depth(
        unbloat_retinex(depth(clipd, 32), sigma=retinex_sigma, fast=True, **kwargs),
        16,
        dither_type="none",
    )

    msrcp = nl_means(msrcpa, h=sigma2, a=2)

    if sharpness > 0:
        msrcp = core.cas.CAS(msrcp, sharpness=sharpness, opt=0, planes=0)
        clipd = core.cas.CAS(clipd, sharpness=sharpness, opt=0, planes=0)

    edges = core.akarin.Expr(
        [
            msrcp.std.Sobel(),
            clipd.std.Sobel(),
            msrcp.std.Prewitt(),
            clipd.std.Prewitt(),
        ],
        "x y max z a max +",
    )

    if edge_thr > 0:
        edges = _soft_threshold(edges, edge_thr, 10)

    tcanny = core.akarin.Expr(
        [
            core.tcanny.TCanny(msrcp, mode=1, sigma=0),
            core.tcanny.TCanny(clipd, mode=1, sigma=0),
        ],
        "x y max",
    )
    tcanny = core.std.Minimum(tcanny)

    edgescombo = Morpho.inflate(
        core.akarin.Expr([edges, tcanny], "x y +"), iterations=2
    )

    kirco = core.akarin.Expr([Kirsch.edgemask(msrcp), Kirsch.edgemask(clipd)], "x y +")

    edges_expanded = Morpho.expand(edgescombo, mode=XxpandMode.ELLIPSE, sw=1, sh=1)
    kirco_diff = core.akarin.Expr([kirco, edges_expanded], "x y -")
    kirco_expanded = Morpho.expand(
        kirco_diff, mode=XxpandMode.ELLIPSE, sw=expand, sh=expand
    )

    edgescombo = core.akarin.Expr(
        edgescombo.std.Invert(),
        f"x {scale_binary_value(luma, edges_strength, return_int=True)} +",
    )
    kirco_expanded = luma_mask_man(kirco_expanded, t=0.001, s=texture_strength, a=0.5)

    mask = core.akarin.Expr(
        [edgescombo.std.Invert(), kirco_expanded.std.Invert()], "x y +"
    )

    mask = gauss_blur(mask, blur)

    return mask
