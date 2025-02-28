try:
    import vapoursynth as vs
except ImportError:
    raise ImportError('Vapoursynth R70> is required. Download it via: pip install vapoursynth')

try:
    from vsdenoise import MVToolsPresets, Prefilter, mc_degrain, BM3DCuda, nl_means, MVTools, MotionMode, SADMode, MVTools, SADMode, MotionMode, Profile, deblock_qed
    from vstools import get_y, get_u, get_v
    from vstools.enums import color
    from vsmasktools import adg_mask
except ImportError:
    raise ImportError('vsdenoise, vstools, vsmasktools are required. Download them via: pip install vsjetpack. Other depedencies can be found here: https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack' )

def AdaptiveDenoise (
    clip: vs.VideoNode,
    thsad: int = 800,
    tr1: int = 3,
    tr2: int = 2,
    sigma: int = 10,
    luma_mask_weaken1: float = 0.75,
    luma_mask_weaken2: float | None = None,
    chroma_strength: float = 1.0,
    precision: bool = False,
    show_mask: int = 0
) -> vs.VideoNode:
    """
    Adaptive denoise with default parameters for film scans (16mm).

    Three denoisers are applied: mc_degrain (luma), NLMeans (chroma), and BM3DCuda (luma).
    NLMeans uses mc_degrain as reference to remove dirt spots and scanner noise from the clip,
    while mc_degrain affects only the luma, which is then passed to BM3DCuda for a second denoising pass.
    If precision = True, BM3DCuda receives a new mc_degrain reference based on the already cleaned clip (slower).

    Luma masks ensure that denoising is applied only to the brighter areas of the frame, preserving details in darker regions.
    Note: Luma masks are more sensitive to variations than the sigma value for the final result.

    :param clip:                Clip to process.
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
    :param show_mask:           1 = Show the first luma mask, 2 = Show the second luma mask (if precision = True).

    :return:                    Denoised clip or luma_mask if show_mask is 1 or 2.
    """

    
    if precision == True and luma_mask_weaken2 == None:
        luma_mask_weaken2 = luma_mask_weaken1

    core = vs.core

    if clip.format.color_family not in {vs.YUV}:
        raise ValueError('AdaptiveDenoise: only YUV formats are supported')

    if clip.format.bits_per_sample != 16:
        clip = clip.fmtc.bitdepth(bits=16)

    lumamask = adg_mask(clip)
    darkenLumaMask = core.std.Expr([lumamask], f"x {luma_mask_weaken1} *")

    if show_mask == 1:
        return darkenLumaMask

    #Denoise
    mvtools = MVTools(clip)
    vectors = mvtools.analyze(blksize=16, overlap=8, lsad=300, truemotion=MotionMode.SAD, dct=SADMode.DCT)
    ref = mc_degrain(clip, prefilter=Prefilter.DFTTEST, preset=MVToolsPresets.HQ_SAD, thsad=thsad, vectors=vectors, tr=tr1)
    luma = get_y(core.std.MaskedMerge(ref, clip, darkenLumaMask, planes=0))

    #Chroma NLMeans
    chroma_denoised = nl_means(clip, tr=tr1, strength=chroma_strength, ref=ref, planes=[1,2])

    #Luma BM3D
    if precision:
        lumamask = adg_mask(luma)
        darkenLumaMask = core.std.Expr([lumamask], f"x {luma_mask_weaken2} *")
        if show_mask == 2:
            return darkenLumaMask
        mvtools = MVTools(luma)
        ref = mc_degrain(luma, prefilter=Prefilter.DFTTEST, preset=MVToolsPresets.HQ_SAD, thsad=thsad, vectors=vectors, tr=tr2)

    denoised = BM3DCuda.denoise(luma, sigma=sigma, tr=tr2, ref=ref, planes=0, matrix=color.Matrix.BT709, profile=Profile.HIGH)
    lumaFinal = core.std.MaskedMerge(denoised, luma, darkenLumaMask, planes=0)

    final = core.std.ShufflePlanes(clips=[lumaFinal, get_u(chroma_denoised), get_v(chroma_denoised)], planes=[0,0,0], colorfamily=vs.YUV)

    return final

#TODO
#Ported from fvsfunc 
def AutoDeblock(
    src,
    edgevalue=24,
    db1=1, db2=6, db3=15,
    deblocky=True,
    deblockuv=True,
    debug=False,
    redfix=False,
    fastdeblock=False,
    adb1=3, adb2=4, adb3=8,
    adb1d=2, adb2d=7, adb3d=11,
    planes=None
) -> vs.VideoNode:
    core=vs.core
    try:
        from functools import partial #da vedere se non è rotta 
    except ImportError:
        raise ImportError('functools is required')

    if src.format.color_family not in [vs.YUV]:
        raise TypeError("AutoDeblock: src must be YUV color family!")

    if src.format.bits_per_sample < 8 or src.format.bits_per_sample > 16 or src.format.sample_type != vs.INTEGER:
        raise TypeError("AutoDeblock: src must be between 8 and 16 bit integer format")

    # Scale values to handle high bit depths
    shift = src.format.bits_per_sample - 8
    edgevalue = edgevalue << shift
    maxvalue = (1 << src.format.bits_per_sample) - 1

    # Scales the output of PlaneStats (which is a float, 0-1) to 8 bit values.
    # We scale to 8 bit because all thresholds/parameters for this function are
    # specified in an 8-bit scale.
    # All processing still happens in the native bit depth of the input format.
    def to8bit(f):
        return f * 255

    def sub_props(src, f, name):

        OrigDiff_str = str(to8bit(f[0].props.OrigDiff))
        YNextDiff_str = str(to8bit(f[1].props.YNextDiff))
        return core.sub.Subtitle(src, name + f"\nOrigDiff: {OrigDiff_str}\nYNextDiff: {YNextDiff_str}")

    def eval_deblock_strength(n, f, fastdeblock, debug, unfiltered, fast, weakdeblock,
                              mediumdeblock, strongdeblock):
        unfiltered = sub_props(unfiltered, f, "unfiltered") if debug else unfiltered
        out = unfiltered
        if fastdeblock:
            if to8bit(f[0].props.OrigDiff) > adb1 and to8bit(f[1].props.YNextDiff) > adb1d:
                return sub_props(fast, f, "deblock") if debug else fast
            else:
                return unfiltered
        if to8bit(f[0].props.OrigDiff) > adb1 and to8bit(f[1].props.YNextDiff) > adb1d:
            out = sub_props(weakdeblock, f, "weakdeblock") if debug else weakdeblock
        if to8bit(f[0].props.OrigDiff) > adb2 and to8bit(f[1].props.YNextDiff) > adb2d:
            out = sub_props(mediumdeblock, f, "mediumdeblock") if debug else mediumdeblock
        if to8bit(f[0].props.OrigDiff) > adb3 and to8bit(f[1].props.YNextDiff) > adb3d:
            out = sub_props(strongdeblock, f, "strongdeblock") if debug else strongdeblock
        return out

    def fix_red(n, f, unfiltered, autodeblock):
        if (to8bit(f[0].props.YAverage) > 50 and to8bit(f[0].props.YAverage) < 130
                and to8bit(f[1].props.UAverage) > 95 and to8bit(f[1].props.UAverage) < 130
                and to8bit(f[2].props.VAverage) > 130 and to8bit(f[2].props.VAverage) < 155):
            return unfiltered
        return autodeblock

    if redfix and fastdeblock:
        raise ValueError('AutoDeblock: You cannot set both "redfix" and "fastdeblock" to True!')

    if planes is None:
        planes = []
        if deblocky: planes.append(0)
        if deblockuv: planes.extend([1,2])

    orig = core.std.Prewitt(src)
    orig = core.std.Expr(orig, f"x {edgevalue} >= {maxvalue} x ?")
    orig_d = orig.rgvs.RemoveGrain(4).rgvs.RemoveGrain(4)

    predeblock = deblock_qed(src.rgvs.RemoveGrain(2).rgvs.RemoveGrain(2))
    fast = core.dfttest.DFTTest(predeblock, tbsize=1)

    unfiltered = src
    weakdeblock = core.dfttest.DFTTest(predeblock, sigma=db1, tbsize=1, planes=planes)
    mediumdeblock = core.dfttest.DFTTest(predeblock, sigma=db2, tbsize=1, planes=planes)
    strongdeblock = core.dfttest.DFTTest(predeblock, sigma=db3, tbsize=1, planes=planes)

    difforig = core.std.PlaneStats(orig, orig_d, prop='Orig')
    diffnext = core.std.PlaneStats(src, src.std.DeleteFrames([0]), prop='YNext')
    autodeblock = core.std.FrameEval(unfiltered, partial(eval_deblock_strength, fastdeblock=fastdeblock,
                                     debug=debug, unfiltered=unfiltered, fast=fast, weakdeblock=weakdeblock,
                                     mediumdeblock=mediumdeblock, strongdeblock=strongdeblock),
                                     prop_src=[difforig,diffnext])

    if redfix:
        src = core.std.PlaneStats(src, prop='Y')
        src_u = core.std.PlaneStats(src, plane=1, prop='U')
        src_v = core.std.PlaneStats(src, plane=2, prop='V')
        autodeblock = core.std.FrameEval(unfiltered, partial(fix_red, unfiltered=unfiltered,
                                         autodeblock=autodeblock), prop_src=[src,src_u,src_v])

    return autodeblock