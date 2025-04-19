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
        thr: float = 55,
) -> vs.VideoNode:
    """
    Custom luma mask that uses a different approach to calculate the mask (Made By PingWer).
    This mask is sensitive to the brightness of the image, producing a constant dark mask for bright areas, 
    a constant white mask for very dark areas, and a exponential transition between these extremes based on brightness levels.

    Curve graph https://www.geogebra.org/calculator/fxbrx4s4

    :param clip:            Clip to process.
    :param low_amp:         General preamplification value, but more sensitive for values lower than thr.
    :param thr:             Threshold that determines what is considered light and what is dark.
    :return:                Luma mask.
    """
    import math
    import vapoursynth as vs

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

def intensive_adaptive_denoiser (
    clip: vs.VideoNode,
    thsad: int = 800,
    tr1: int = 3,
    tr2: int = 2,
    sigma: int = 12,
    luma_mask_weaken1: float = 0.85,
    luma_mask_weaken2: float | None = None,
    luma_mask_thr: float = 50,
    chroma_strength: float = 1.0,
    precision: bool = False,
    chroma_masking: bool = False,
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
    :param show_mask:           1 = Show the first luma mask, 2 = Show the second luma mask (if precision = True), 3 = Show the Chroma v mask (if chroma_masking = True).

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
        lumamask = luma_mask_ping(clip, thr=luma_mask_thr)
    
    darken_luma_mask = core.std.Expr([lumamask], f"x {luma_mask_weaken1} *")
    if show_mask == 1:
        return darken_luma_mask

    #Denoise
    mvtools = MVTools(clip)
    vectors = mvtools.analyze(blksize=16, overlap=8, lsad=300, truemotion=MotionMode.SAD, dct=SADMode.MIXED_SATD_DCT)
    ref = mc_degrain(clip, prefilter=Prefilter.DFTTEST, preset=MVToolsPresets.HQ_SAD, thsad=thsad, vectors=vectors, tr=tr1)
    luma = get_y(core.std.MaskedMerge(ref, clip, darken_luma_mask, planes=0))

    #Chroma NLMeans
    chroma_denoised = nl_means(clip, tr=tr1, strength=chroma_strength, ref=ref, planes=[1,2])
    #TODO
    #chroma mask fine tuning
    if chroma_masking:
        v=get_v(clip)
        v_mask= luma_mask_man(v,t=1.5,s=2, a=0)
        v_masked = core.std.MaskedMerge(get_v(chroma_denoised), v, core.std.Invert(v_mask))
        u=get_u(clip)
        u_mask= luma_mask_man(u,t=1.5,s=2, a=0)
        u_masked = core.std.MaskedMerge(get_u(chroma_denoised), u, core.std.Invert(u_mask))
        chroma_denoised = core.std.ShufflePlanes(clips=[chroma_denoised, u_masked, v_masked], planes=[0,0,0], colorfamily=vs.YUV)
    
    if show_mask == 3:
        return v_mask

    #Luma BM3D
    if precision:
        if (mask_type == 0):
            lumamask = luma_mask(luma)
        elif (mask_type == 1):
            lumamask = luma_mask_man(luma)
        else:
            lumamask = luma_mask_ping(luma, thr=luma_mask_thr)
        darken_luma_mask = core.std.Expr([lumamask], f"x {luma_mask_weaken2} *")
        if show_mask == 2:
            return darken_luma_mask
        mvtools = MVTools(luma)
        vectors = mvtools.analyze(blksize=16, overlap=8, lsad=300, truemotion=MotionMode.SAD, dct=SADMode.DCT)
        ref = mc_degrain(luma, prefilter=Prefilter.DFTTEST, preset=MVToolsPresets.HQ_SAD, thsad=thsad, vectors=vectors, tr=tr2)

    denoised = BM3DCuda.denoise(luma, sigma=sigma, tr=tr2, ref=ref, planes=0, profile=Profile.HIGH)
    luma_final = core.std.MaskedMerge(denoised, luma, darken_luma_mask, planes=0)

    final = core.std.ShufflePlanes(clips=[luma_final, get_u(chroma_denoised), get_v(chroma_denoised)], planes=[0,0,0], colorfamily=vs.YUV)

    return final

def adaptive_denoiser (
    clip: vs.VideoNode,
    thsad: int = 800,
    tr1: int = 3,
    tr2: int = 2,
    sigma: int = 12,
    luma_mask_weaken1: float = 0.85,
    luma_mask_thr: float = 50,
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

    if clip.format.color_family not in {vs.YUV}:
        raise ValueError('GAD: only YUV formats are supported')

    if clip.format.bits_per_sample != 16:
        clip = clip.fmtc.bitdepth(bits=16)

    if (mask_type == 0):
        lumamask = luma_mask(clip)
    elif (mask_type == 1):
        lumamask = luma_mask_man(clip)
    else:
        lumamask = luma_mask_ping(clip, thr=luma_mask_thr)
    
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
            lumamask = luma_mask_ping(luma, thr=luma_mask_thr)
        darken_luma_mask = core.std.Expr([lumamask], f"x {luma_mask_weaken2} *")
        if show_mask == 2:
            return darken_luma_mask
        mvtools = MVTools(luma)
        vectors = mvtools.analyze(blksize=16, overlap=8, lsad=300, truemotion=MotionMode.SAD, dct=SADMode.DCT)
        ref = mc_degrain(luma, prefilter=Prefilter.DFTTEST, preset=MVToolsPresets.HQ_SAD, thsad=thsad, vectors=vectors, tr=tr2)

    denoised = BM3DCuda.denoise(luma, sigma=sigma, tr=tr2, ref=ref, planes=0, profile=Profile.HIGH)
    luma_final = core.std.MaskedMerge(denoised, luma, darken_luma_mask, planes=0)

    final = core.std.ShufflePlanes(clips=[luma_final, get_u(clip), get_v(clip)], planes=[0,0,0], colorfamily=vs.YUV)

    return final   

#TODO
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

    lumamask = luma_mask_man(clip, t=t, s=s, a=a)
    lumamask = lumamask.std.Invert()
    u = get_u(clip)
    v = get_v(clip)
    lumamask = core.std.ShufflePlanes(clips=[lumamask, u, v], planes=[0,0,0], colorfamily=vs.YUV)
    clip = core.std.Merge(clip, lumamask, weight=0.1)
    return clip

def deblock_old(
    clip: vs.VideoNode,
    thrvalue: int = 22,
    show_mask: bool = False
)-> vs.VideoNode:
    """
    Sono stata curata
    """
        
    core=vs.core

    maxvalue = (1 << clip.format.bits_per_sample) - 1
    thr = (thrvalue / 255) * maxvalue

    y, u, v = clip.std.SplitPlanes()
    ymask = luma_mask_man(y, t=1, a=20, s=20)
    # umask = luma_mask_man(u, t=1, a=20, s=20)
    # vmask = luma_mask_man(v, t=1, a=20, s=20)
    umask = u
    vmask = v

    diag = "x x[1,1] - 2 / abs x x[1,-1] - 2 / abs + x x[-1,1] - 2 / abs x x[-1,-1] - 2 / abs + +"
    hor = "x x[1,0] - abs x x[-1,0] - abs + x x[2,0] - 2 / abs x x[-2,0] - 2 / abs + + x x[3,0] - 4 / abs x x[-3,0] - 4 / abs + +"
    ver = "x x[0,1] - abs x x[0,-1] - abs + x x[0,2] - 2 / abs x x[0,-2] - 2 / abs + + x x[0,3] - 4 / abs x x[0,-3] - 4 / abs + +"
    ymask1 = core.akarin.Expr([ymask], f"{hor} {diag} +").std.Binarize(thr)
    ymask2 = core.akarin.Expr([ymask], f"{ver} {diag} +").std.Binarize(thr)

    umask1 = core.akarin.Expr([umask], f"{hor} {diag} +").std.Binarize(thr)
    umask2 = core.akarin.Expr([umask], f"{ver} {diag} +").std.Binarize(thr)

    vmask1 = core.akarin.Expr([vmask], f"{hor} {diag} +").std.Binarize(thr)
    vmask2 = core.akarin.Expr([vmask], f"{ver} {diag} +").std.Binarize(thr)

    ymask = core.std.Expr([ymask1, ymask2], "x y max")
    umask = core.std.Expr([umask1, umask2], "x y max")
    vmask = core.std.Expr([vmask1, vmask2], "x y max")

    y1 = deblock_qed(y, quant_edge=30, quant_inner=32)
    y1 = core.dfttest.DFTTest(y1, sigma=25, tbsize=1, sosize=8)
    y1 = core.std.MaskedMerge(y, y1, ymask)
    u1 = deblock_qed(u, quant_edge=30, quant_inner=32)
    u1 = core.dfttest.DFTTest(u1, sigma=25, tbsize=1, sosize=8)
    u1 = core.std.MaskedMerge(u, u1, umask)
    v1 = deblock_qed(v, quant_edge=30, quant_inner=32)
    v1 = core.dfttest.DFTTest(v1, sigma=25, tbsize=1, sosize=8)
    v1 = core.std.MaskedMerge(v, v1, vmask)

    if show_mask is True:
        return core.std.ShufflePlanes(clips=[ymask, umask, vmask], planes=[0,0,0], colorfamily=vs.YUV)
    return core.std.ShufflePlanes(clips=[y1, u1, v1], planes=[0,0,0], colorfamily=vs.YUV)  

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
    sumhor8_1 = "x x[0,-1] - abs x[1,0] x[1,-1] - abs + x[2,0] x[2,-1] - abs x[3,0] x[3,-1] - abs + + x[4,0] x[4,-1] - abs x[5,0] x[5,-1] - abs + x[6,0] x[6,-1] - abs x[7,0] x[7,-1] - abs + + +"
    sumhor8_2 = "x x[1,0] + x[2,0] + x[3,0] + x[4,0] + x[5,0] + x[6,0] + x[7,0] + x[0,-1] x[1,-1] + x[2,-1] + x[3,-1] + x[4,-1] + x[5,-1] + x[6,-1] + x[7,-1] + - abs"
    sumhor4_1 = "x x[1,0] + x[2,0] + x[3,0] + x[0,-1] x[1,-1] + x[2,-1] + x[3,-1] + - abs"
    sumver8_1 = "x x[-1,0] - abs x[0,1] x[-1,1] - abs + x[0,2] x[-1,2] - abs x[0,3] x[-1,3] - abs + + x[0,4] x[-1,4] - abs x[0,5] x[-1,5] - abs + x[0,6] x[-1,6] - abs x[0,7] x[-1,7] - abs + + +"
    sumver8_2 = "x x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5] + x[0,6] + x[0,7] + x[-1,0] x[-1,1] + x[-1,2] + x[-1,3] + x[-1,4] + x[-1,5] + x[-1,6] + x[-1,7] + - abs"
    sumver4_1 = "x x[0,1] + x[0,2] + x[0,3] + x[-1,0] x[-1,1] + x[-1,2] + x[-1,3] + - abs"
    horblockvalue = core.akarin.Expr([clip], f"{hor8x8} 0 = {ver8x8} 0 = {sumhor8_1} {sumhor8_2} + 2 / 0 ? 0 ?")
    horblockvalue4 = core.akarin.Expr([clip], f"{hor4x4} 0 = {ver4x4} 0 = {sumhor4_1} 0 ? 0 ?")
    verblockvalue = core.akarin.Expr([clip], f"{hor8x8} 0 = {ver8x8} 0 = {sumver8_1} {sumver8_2} + 2 / 0 ? 0 ?")
    verblockvalue4 = core.akarin.Expr([clip], f"{hor4x4} 0 = {ver4x4} 0 = {sumver4_1} 0 ? 0 ?")

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
    blockmaskfull8.set_output(5)

    # unione cancellando il deblock 4x4 dove c'è il blocco 8x8
    deleted4 = core.akarin.Expr([deblock4x4], f"{hor8x8} 5 = x {hor8x8} 4 = x {hor8x8} 3 = x {hor8x8} 2 = x 0 ? ? ? ?")
    deleted4 = core.akarin.Expr([deleted4], f"{ver8x8} 5 = x {ver8x8} 4 = x {ver8x8} 3 = x {ver8x8} 2 = x 0 ? ? ? ?")
    deleted8 = core.akarin.Expr([deblock8x8], f"{hor8x8} 7 = x {hor8x8} 6 = x {hor8x8} 1 = x {hor8x8} 0 = x {ver8x8} 7 = x {ver8x8} 6 = x {ver8x8} 1 = x {ver8x8} 0 = x 0 ? ? ? ? ? ? ? ?")
    union = core.akarin.Expr([deleted8, deleted4], "x y max")

    return union