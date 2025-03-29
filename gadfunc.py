try:
    import vapoursynth as vs
except ImportError:
    raise ImportError('Vapoursynth R70> is required. Download it via: pip install vapoursynth')

try:
    from vsdenoise import MVToolsPresets, Prefilter, mc_degrain, BM3DCuda, nl_means, MVTools, MotionMode, SADMode, MVTools, SADMode, MotionMode, Profile, deblock_qed
    from vstools import get_y, get_u, get_v, PlanesT
    from vstools.enums import color
except ImportError:
    raise ImportError('vsdenoise, vstools, vsmasktools are required. Download them via: pip install vsjetpack. Other depedencies can be found here: https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack' )

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
    Curve graph https://www.geogebra.org/calculator/cqnfnqyk
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

def luma_mask_ping (
        clip: vs.VideoNode,
        low_amp: float = 1,
        thr: float = 30000,
        high_amp: float = 0.2,
)-> vs.VideoNode :
    """
    Custom luma mask that uses a different approach to calculate the mask (Made By PingWer).
    This mask is sensitive to the brightness of the image, producing a constant dark mask for bright areas, 
    a constant white mask for very dark areas, and a smooth transition between these extremes based on brightness levels.

    Curve graph https://www.geogebra.org/calculator/fxbrx4s4


    :param clip:            Clip to process.
    :param low_amp:         General preamplification value, but more sensitive for values lower than thr.
    :param thr:             Threshold that determines what is considered light and what is dark.
    :param high_ampc:       Amplification value for values higher than b. Recommended values range from 0.05 to 0.3.
    :return:                Luma mask.
    """
    import math

    core = vs.core

    if clip.format.bits_per_sample != 16:
        clip = clip.fmtc.bitdepth(bits=16)

    high_amp = (math.exp(low_amp - 1) + low_amp * math.exp(low_amp)) / (math.exp(low_amp) - 1)

    expr2 = (
        f"x "                  # Mettiamo x sullo stack per la moltiplicazione finale
        f"x {thr} < "            # x < b ?
        # - Ramo TRUE → f(x):
        f"x {thr} 1 + - exp {low_amp} + "
        # - Ramo FALSE → h(x):
        f"{high_amp} {high_amp} x {thr} 1 - - log {high_amp} * exp {low_amp} + / - "
        # - Operatore ternario:
        f"? "
        # - moltiplica per x:
        f"*"
    )

    cc = core.std.Expr(get_y(clip), expr2)
    cc = core.std.Invert(cc)

    return cc

def IntesiveAdaptiveDenoiser (
    clip: vs.VideoNode,
    thsad: int = 800,
    tr1: int = 3,
    tr2: int = 2,
    sigma: int = 12,
    luma_mask_weaken1: float = 0.85,
    luma_mask_weaken2: float | None = None,
    chroma_strength: float = 1.0,
    precision: bool = False,
    mask_type: int = 2,
    show_mask: int = 0
) -> vs.VideoNode:
    """
    Intensive Adaptive Denoise with default parameters for film scans (16mm).

    Three denoisers are applied: mc_degrain (luma), NLMeans (chroma), and BM3DCuda (luma).
    NLMeans uses mc_degrain as reference to remove dirt spots and scanner noise from the clip,
    while mc_degrain affects only the luma, which is then passed to BM3DCuda for a second denoising pass.
    If precision = True, BM3DCuda receives a new mc_degrain reference based on the already cleaned clip (slower).

    Luma masks ensure that denoising is applied only to the brighter areas of the frame, preserving details in darker regions while cleaning them as much as possible.
    Note: Luma masks are more sensitive to variations than the sigma value for the final result.

    :param clip:                Clip to process (YUV 16bit, if not will be internally converted in 16bit with fmtc).
    :param thsad:               Thsad for mc_degrain (luma denoise strength and chroma ref).
                                Recommended values: 300-800
    :param tr1:                 Temporal radius for the first mc_degrain and NLMeans. Recommended values: 2-4
    :param tr2:                 Temporal radius for BM3DCuda (always) and the second mc_degrain (if precision = True).
                                Recommended values: 2-3
    :param sigma:               Sigma for BM3DCuda (luma denoise strength). Recommended values: 3-10
    :param luma_mask_weaken1:   Controls how much dark spots should be denoised. Lower values mean stronger denoise.
                                Recommended values: 0.6-0.9
    :param luma_mask_weaken2:   Only used if precision = True. Controls how much dark spots should be denoised on BM3DCuda.
                                Lower values mean stronger denoise. Recommended values: 0.6-0.9
    :param chroma_strength:     Strength for NLMeans (chroma denoise strength). Recommended values: 0.5-2
    :param precision:           If True, a second reference and mask are created for BM3DCuda. Very slow.
    :param mask_type:           0 = Standard Luma mask, 1 = Custom Luma mask (more linear) , 2 = Custom Luma mask (less linear).
    :param show_mask:           1 = Show the first luma mask, 2 = Show the second luma mask (if precision = True).

    :return:                    16bit denoised clip or luma_mask if show_mask is 1 or 2.
    """
    
    if precision == True and luma_mask_weaken2 == None:
        luma_mask_weaken2 = luma_mask_weaken1

    core = vs.core

    if clip.format.color_family not in {vs.YUV}:
        raise ValueError('GAD: only YUV formats are supported')

    if clip.format.bits_per_sample != 16:
        clip = clip.fmtc.bitdepth(bits=16)

    if (mask_type == 0):
        lumamask = luma_mask(clip)
    elif (mask_type == 1):
        lumamask = luma_mask_man(clip)
    else:
        lumamask = luma_mask_ping(clip)
    
    darken_luma_mask = core.std.Expr([lumamask], f"x {luma_mask_weaken1} *")
    if show_mask == 1:
        return darken_luma_mask

    #Denoise
    mvtools = MVTools(clip)
    vectors = mvtools.analyze(blksize=16, overlap=8, lsad=300, truemotion=MotionMode.SAD, dct=SADMode.DCT)
    ref = mc_degrain(clip, prefilter=Prefilter.DFTTEST, preset=MVToolsPresets.HQ_SAD, thsad=thsad, vectors=vectors, tr=tr1)
    luma = get_y(core.std.MaskedMerge(ref, clip, darken_luma_mask, planes=0))

    #Chroma NLMeans
    chroma_denoised = nl_means(clip, tr=tr1, strength=chroma_strength, ref=ref, planes=[1,2])

    #Luma BM3D
    if precision:
        if (mask_type == 0):
            lumamask = luma_mask(luma)
        elif (mask_type == 1):
            lumamask = luma_mask_man(luma)
        else:
            lumamask = luma_mask_ping(luma)
        darken_luma_mask = core.std.Expr([lumamask], f"x {luma_mask_weaken2} *")
        if show_mask == 2:
            return darken_luma_mask
        mvtools = MVTools(luma)
        vectors = mvtools.analyze(blksize=16, overlap=8, lsad=300, truemotion=MotionMode.SAD, dct=SADMode.DCT)
        ref = mc_degrain(luma, prefilter=Prefilter.DFTTEST, preset=MVToolsPresets.HQ_SAD, thsad=thsad, vectors=vectors, tr=tr2)

    denoised = BM3DCuda.denoise(luma, sigma=sigma, tr=tr2, ref=ref, planes=0, matrix=color.Matrix.BT709, profile=Profile.HIGH)
    luma_final = core.std.MaskedMerge(denoised, luma, darken_luma_mask, planes=0)

    final = core.std.ShufflePlanes(clips=[luma_final, get_u(chroma_denoised), get_v(chroma_denoised)], planes=[0,0,0], colorfamily=vs.YUV)

    return final

def AdaptiveDenoiser (
    clip: vs.VideoNode,
    thsad: int = 800,
    tr1: int = 3,
    tr2: int = 2,
    sigma: int = 12,
    luma_mask_weaken1: float = 0.85,
    luma_mask_weaken2: float | None = None,
    precision: bool = False,
    mask_type: int = 2,
    show_mask: int = 0
) -> vs.VideoNode: 
    """
    Adaptive Denoise with default parameters for film scans (16mm).

    Two denoisers are applied: mc_degrain (luma) and BM3DCuda (luma).
    Mc_degrain affects only the luma and is used as reference for BM3DCuda for a second denoising pass.
    If precision = True, BM3DCuda receives a new mc_degrain reference based on the already cleaned clip (slower).

    Luma masks ensure that denoising is applied only to the brighter areas of the frame, preserving details in darker regions while cleaning them as much as possible.
    Note: Luma masks are more sensitive to variations than the sigma value for the final result.

    :param clip:                Clip to process (YUV 16bit, if not will be internally converted in 16bit with fmtc).
    :param thsad:               Thsad for mc_degrain (luma denoise strength and chroma ref).
                                Recommended values: 300-800
    :param tr1:                 Temporal radius for the first mc_degrain and NLMeans. Recommended values: 2-4
    :param tr2:                 Temporal radius for BM3DCuda (always) and the second mc_degrain (if precision = True).
                                Recommended values: 2-3
    :param sigma:               Sigma for BM3DCuda (luma denoise strength). Recommended values: 3-10
    :param luma_mask_weaken1:   Controls how much dark spots should be denoised. Lower values mean stronger denoise.
                                Recommended values: 0.6-0.9
    :param luma_mask_weaken2:   Only used if precision = True. Controls how much dark spots should be denoised on BM3DCuda.
                                Lower values mean stronger denoise. Recommended values: 0.6-0.9
    :param precision:           If True, a second reference and mask are created for BM3DCuda. Very slow.
    :param mask_type:           0 = Standard Luma mask, 1 = Custom Luma mask (more linear) , 2 = Custom Luma mask (less linear).
    :param show_mask:           1 = Show the first luma mask, 2 = Show the second luma mask (if precision = True).

    :return:                    16bit denoised clip or luma_mask if show_mask is 1 or 2.
    """
    
    core = vs.core

    if precision == True and luma_mask_weaken2 == None:
        luma_mask_weaken2 = luma_mask_weaken1

    core = vs.core

    if clip.format.color_family not in {vs.YUV}:
        raise ValueError('GAD: only YUV formats are supported')

    if clip.format.bits_per_sample != 16:
        clip = clip.fmtc.bitdepth(bits=16)

    if (mask_type == 0):
        lumamask = luma_mask(clip)
    elif (mask_type == 1):
        lumamask = luma_mask_man(clip)
    else:
        lumamask = luma_mask_ping(clip)
    
    darken_luma_mask = core.std.Expr([lumamask], f"x {luma_mask_weaken1} *")
    if show_mask == 1:
        return darken_luma_mask

    #Denoise
    mvtools = MVTools(clip)
    vectors = mvtools.analyze(blksize=16, overlap=8, lsad=300, truemotion=MotionMode.SAD, dct=SADMode.DCT)
    ref = mc_degrain(clip, prefilter=Prefilter.DFTTEST, preset=MVToolsPresets.HQ_SAD, thsad=thsad, vectors=vectors, tr=tr1)
    luma = get_y(core.std.MaskedMerge(ref, clip, darken_luma_mask, planes=0))

    #Luma BM3D
    if precision:
        if (mask_type == 0):
            lumamask = luma_mask(luma)
        elif (mask_type == 1):
            lumamask = luma_mask_man(luma)
        else:
            lumamask = luma_mask_ping(luma)
        darken_luma_mask = core.std.Expr([lumamask], f"x {luma_mask_weaken2} *")
        if show_mask == 2:
            return darken_luma_mask
        mvtools = MVTools(luma)
        vectors = mvtools.analyze(blksize=16, overlap=8, lsad=300, truemotion=MotionMode.SAD, dct=SADMode.DCT)
        ref = mc_degrain(luma, prefilter=Prefilter.DFTTEST, preset=MVToolsPresets.HQ_SAD, thsad=thsad, vectors=vectors, tr=tr2)

    denoised = BM3DCuda.denoise(luma, sigma=sigma, tr=tr2, ref=ref, planes=0, matrix=color.Matrix.BT709, profile=Profile.HIGH)
    luma_final = core.std.MaskedMerge(denoised, luma, darken_luma_mask, planes=0)

    final = core.std.ShufflePlanes(clips=[luma_final, get_u(clip), get_v(clip)], planes=[0,0,0], colorfamily=vs.YUV)

    return final   

# TODO
#Ported from fvsfunc 
def auto_deblock(
    clip: vs.VideoNode,
    # edgevalue: int = 24,
    sigma: int = 15,
    tbsize: int = 1,
    luma_mask_strength: float = 0.9,
    mask_type: int = 0,
    planes: PlanesT = None
) -> vs.VideoNode:
    """
    Funzione che si spera funzioni, dovrebbe fare deblock MPEG2 ma essendo portata da avisynth di eoni fa dubito lo faccia bene.
    """

    core=vs.core
    try:
        from functools import partial
    except ImportError:
        raise ImportError('functools is required')

    if clip.format.color_family not in [vs.YUV]:
        raise TypeError("AutoDeblock: clip must be YUV color family!")

    if clip.format.bits_per_sample != 16:
        clip = clip.fmtc.bitdepth(bits=16)

    # Scale values to handle high bit depths
    # shift = clip.format.bits_per_sample - 8
    # edgevalue = edgevalue << shift
    # maxvalue = (1 << clip.format.bits_per_sample) - 1

    # orig è una edgemask, che significa orig lo sa solo jesus
    # orig = core.std.Prewitt(clip)
    # Se x è maggiore o uguale di edgevalue (def:24) allora restituisci maxvalue altrimenti x
    # È quasi un binarize ma i valori sotto edgevalue rimangono uguali
    # orig = core.std.Expr(orig, f"x {edgevalue} >= {maxvalue} x ?")
    # Doppia Median sulla edgemask
    # orig_d = orig.std.Median().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])

    # Passa la clip con un po' meno grana a vsdenoise.deblock_qed
    predeblock = deblock_qed(clip.rgvs.RemoveGrain(2).rgvs.RemoveGrain(2), planes=planes)

    # Se questi vengono svolti anche se fast = True farebbe molto ridere, sigma regola la forza, ne vengono fatti diversi
    # Analogamente al nostro approccio su Adaptive Denoise si potrebbe usare un signolo DFT e una mask per la forza
    deblock = core.dfttest.DFTTest(predeblock, sigma=sigma, tbsize=tbsize, planes=planes)

    # Prende le differenze di statistiche tra edgemask prima e dopo la Median
    # difforig = core.std.PlaneStats(orig, orig_d, prop='Orig')
    # Prende le differenze di statistiche tra la clip e il frame successivo
    # diffnext = core.std.PlaneStats(clip, clip.std.DeleteFrames([0]), prop='YNext')
    # Frame eval che prende la clip, gli effettua eval_deblock_strength passandogli le statistiche

    # Da implementare per usare le informazioni per regolare la mask
    # autodeblock = core.std.FrameEval(clip, partial(eval_deblock_strength,
    #                                  clip=clip, deblock=deblock),
    #                                  prop_src=[difforig,diffnext])
    
    if (mask_type == 0):
        lumamask = luma_mask(clip)
    elif (mask_type == 1):
        lumamask = luma_mask_man(clip)
    else:
        lumamask = luma_mask_ping(clip)
    darken_luma_mask = core.std.Expr([lumamask], f"x {luma_mask_strength} *")
    final = core.std.MaskedMerge(deblock, clip, darken_luma_mask, planes=planes)

    return final