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
    clip: vs.VideoNode,
    edgevalue: int = 24,
    db1: int = 1, db2: int = 6, db3: int = 15,
    deblocky: bool = True,
    deblockuv: bool = True,
    # redfix: bool = False,
    fastdeblock: bool = False,
    adb1: int = 3, adb2: int = 4, adb3: int = 8,
    adb1d: int = 2, adb2d: int = 7, adb3d: int = 11,
    planes: list[int] = None
) -> vs.VideoNode:
    """
    Funzione che si spera funzioni, dovrebbe fare deblock MPEG2 ma essendo portata da avisynth di eoni fa dubito lo faccia bene.
    redfix momentaneamente rimosso perchè non so manco cosa fa
    debug momentaneamente rimosso perchè molto cringe

    """

    core=vs.core
    try:
        from functools import partial #da vedere se non è rotta 
    except ImportError:
        raise ImportError('functools is required')

    if clip.format.color_family not in [vs.YUV]:
        raise TypeError("AutoDeblock: clip must be YUV color family!")

    if clip.format.bits_per_sample < 8 or clip.format.bits_per_sample > 16 or clip.format.sample_type != vs.INTEGER:
        raise TypeError("AutoDeblock: clip must be between 8 and 16 bit integer format")

    # Scale values to handle high bit depths
    shift = clip.format.bits_per_sample - 8
    edgevalue = edgevalue << shift
    maxvalue = (1 << clip.format.bits_per_sample) - 1

    # Output PlaneStats (0-1) -> 8bit per coerenza con altri parametri (da migliorare sicuramente)
    def to8bit(
        f: int
    ) -> int:
        return f * 255

    def eval_deblock_strength(
        f, 
        fastdeblock: bool,
        clip: vs.VideoNode, 
        fast: vs.VideoNode, 
        weakdeblock: vs.VideoNode, mediumdeblock: vs.VideoNode, strongdeblock: vs.VideoNode
    ) -> vs.VideoNode:
        out = clip
        # Applica o scarta il deblock (sigma 8) in base ai valori
        if fastdeblock:
            if to8bit(f[0].props.OrigDiff) > adb1 and to8bit(f[1].props.YNextDiff) > adb1d:
                return fast
            else:
                return clip
        # Applica o scarta i vari deblock in base ai valori, dal più debole al più forte
        if to8bit(f[0].props.OrigDiff) > adb1 and to8bit(f[1].props.YNextDiff) > adb1d:
            out = weakdeblock
        if to8bit(f[0].props.OrigDiff) > adb2 and to8bit(f[1].props.YNextDiff) > adb2d:
            out = mediumdeblock
        if to8bit(f[0].props.OrigDiff) > adb3 and to8bit(f[1].props.YNextDiff) > adb3d:
            out = strongdeblock
        return out

    # def fix_red(
    #     f, 
    #     unfiltered, 
    #     autodeblock
    # ):
    #     if (to8bit(f[0].props.YAverage) > 50 and to8bit(f[0].props.YAverage) < 130
    #             and to8bit(f[1].props.UAverage) > 95 and to8bit(f[1].props.UAverage) < 130
    #             and to8bit(f[2].props.VAverage) > 130 and to8bit(f[2].props.VAverage) < 155):
    #         return unfiltered
    #     return autodeblock

    # if redfix and fastdeblock:
    #     raise ValueError('AutoDeblock: You cannot set both "redfix" and "fastdeblock" to True!')

    # Sostituire con PlanesT
    if planes is None:
        planes = []
        if deblocky: planes.append(0)
        if deblockuv: planes.extend([1,2])

    # orig è una edgemask, che significa orig lo sa solo jesus
    orig = core.std.Prewitt(clip)
    # Se x è maggiore o uguale di edgevalue (def:24) allora restituisci maxvalue altrimenti x
    # È quasi un binarize ma i valori sotto edgevalue rimangono uguali
    orig = core.std.Expr(orig, f"x {edgevalue} >= {maxvalue} x ?")
    # Usa una modalità di RemoveGrain deprecata, USARE std.Median
    orig_d = orig.rgvs.RemoveGrain(4).rgvs.RemoveGrain(4)

    # Passa la clip con un po' meno grana a vsdenoise.deblock_qed
    predeblock = deblock_qed(clip.rgvs.RemoveGrain(2).rgvs.RemoveGrain(2))
    # tbsize è la Temporal Dimension, sigma 8 default
    fast = core.dfttest.DFTTest(predeblock, tbsize=1)

    # Se questi vengono svolti anche se fast = True farebbe molto ridere, sigma regola la forza, ne vengono fatti diversi
    # Analogamente al nostro approccio su Adaptive Denoise si potrebbe usare un signolo DFT e una mask per la forza
    weakdeblock = core.dfttest.DFTTest(predeblock, sigma=db1, tbsize=1, planes=planes)
    mediumdeblock = core.dfttest.DFTTest(predeblock, sigma=db2, tbsize=1, planes=planes)
    strongdeblock = core.dfttest.DFTTest(predeblock, sigma=db3, tbsize=1, planes=planes)

    # Prende le differenze di statistiche tra edgemask prima e dopo la Median
    difforig = core.std.PlaneStats(orig, orig_d, prop='Orig')
    # Prende le differenze di statistiche tra la clip e il frame successivo
    diffnext = core.std.PlaneStats(clip, clip.std.DeleteFrames([0]), prop='YNext')
    # Frame eval che prende la clip, gli effettua eval_deblock_strength passandogli le statistiche
    autodeblock = core.std.FrameEval(clip, partial(eval_deblock_strength, fastdeblock=fastdeblock,
                                     clip=clip, fast=fast, weakdeblock=weakdeblock,
                                     mediumdeblock=mediumdeblock, strongdeblock=strongdeblock),
                                     prop_clip=[difforig,diffnext])

    # if redfix:
    #     clip = core.std.PlaneStats(clip, prop='Y')
    #     clip_u = core.std.PlaneStats(clip, plane=1, prop='U')
    #     clip_v = core.std.PlaneStats(clip, plane=2, prop='V')
    #     autodeblock = core.std.FrameEval(unfiltered, partial(fix_red, unfiltered=unfiltered,
    #                                      autodeblock=autodeblock), prop_clip=[clip,clip_u,clip_v])

    return autodeblock


#TODO
#TEMPORANEA
def autodb_dpir(
    clip: vs.VideoNode,
    edgevalue: int = 24,
    strs: Sequence[float] = [10, 50, 75],
    thrs: Sequence[tuple[float, float, float]] = [(1.5, 2.0, 2.0), (3.0, 4.5, 4.5), (5.5, 7.0, 7.0)],
    matrix: Matrix | int | None = None,
    edgemasker: Callable[[vs.VideoNode], vs.VideoNode] | None = None,
    kernel: KernelT = Catrom,
    cuda: bool | Literal['trt'] | None = None,
    return_mask: bool = False,
    write_props: bool = False,
    **vsdpir_args: Any
) -> vs.VideoNode:
    """
    Rewrite of fvsfunc.AutoDeblock that uses vspdir instead of dfttest to deblock.

    This function checks for differences between a frame and an edgemask with some processing done on it,
    and for differences between the current frame and the next frame.
    For frames where both thresholds are exceeded, it will perform deblocking at a specified strength.
    This will ideally be frames that show big temporal *and* spatial inconsistencies.

    Thresholds and calculations are added to the frameprops to use as reference when setting the thresholds.

    Keep in mind that dpir is not perfect; it may cause weird, black dots to appear sometimes.
    If that happens, you can perform a denoise on the original clip (maybe even using dpir's denoising mode)
    and grab the brightest pixels from your two clips. That should return a perfectly fine clip.

    Thanks `Vardë <https://github.com/Ichunjo>`_, `louis <https://github.com/tomato39>`_,
    `Setsugen no ao <https://github.com/Setsugennoao>`_!

    Dependencies:

    * `vs-mlrt <https://github.com/AmusementClub/vs-mlrt>`_

    :param clip:            Clip to process.
    :param edgevalue:       Remove edges from the edgemask that exceed this threshold (higher means more edges removed).
    :param strs:            A list of DPIR strength values (higher means stronger deblocking).
                            You can pass any arbitrary number of values here.
                            Sane deblocking strengths lie between 1–20 for most regular deblocking.
                            Going higher than 50 is not recommended outside of very extreme cases.
                            The amount of values in strs and thrs need to be equal.
    :param thrs:            A list of thresholds, written as [(EdgeValRef, NextFrameDiff, PrevFrameDiff)].
                            You can pass any arbitrary number of values here.
                            The amount of values in strs and thrs need to be equal.
    :param matrix:          Enum for the matrix of the Clip to process.
                            See :py:attr:`lvsfunc.types.Matrix` for more info.
                            If `None`, gets matrix from the "_Matrix" prop of the clip unless it's an RGB clip,
                            in which case it stays as `None`.
    :param edgemasker:      Edgemasking function to use for calculating the edgevalues.
                            Default: Prewitt.
    :param kernel:          py:class:`vskernels.Kernel` object used for conversions between YUV <-> RGB.
                            This can also be the string name of the kernel
                            (Default: py:class:`vskernels.Bicubic(0, 0.5)`).
    :param cuda:            Used to select backend.
                            Use CUDA if True, CUDA TensorRT if 'trt', else CPU OpenVINO if False.
                            If ``None``, it will detect your system's capabilities
                            and select the fastest backend.
    :param return_mask:     Return the mask used for calculating the edgevalues.
    :param write_props:     whether to write verbose props.
    :param vsdpir_args:     Additional args to pass to :py:func:`lvsfunc.deblock.vsdpir`.

    :return:                Deblocked clip with different strengths applied based on the given parameters.

    :raises ValueError:     Unequal number of ``strength``s and ``thr``s passed.
    """

    assert check_variable(clip, "autodb_dpir")

    def _eval_db(
        n: int, f: Sequence[vs.VideoFrame],
        clip: vs.VideoNode, db_clips: Sequence[vs.VideoNode],
        nthrs: Sequence[tuple[float, float, float]]
    ) -> vs.VideoNode:
        evref_diff, y_next_diff, y_prev_diff = [
            get_prop(f[i], prop, float)
            for i, prop in zip(range(3), ['EdgeValRefDiff', 'YNextDiff', 'YPrevDiff'])
        ]

        f_type = get_prop(f[0], '_PictType', str)

        if f_type == 'I':
            y_next_diff = (y_next_diff + evref_diff) / 2

        out = clip
        nthr_used = (-1., ) * 3
        for dblk, nthr in zip(db_clips, nthrs):
            if all(p > t for p, t in zip([evref_diff, y_next_diff, y_prev_diff], nthr)):
                out = dblk
                nthr_used = nthr

        if write_props:
            for prop_name, prop_val in zip(
                ['Adb_EdgeValRefDiff', 'Adb_YNextDiff', 'Adb_YPrevDiff',
                 'Adb_EdgeValRefDiffThreshold', 'Adb_YNextDiffThreshold', 'Adb_YPrevDiffThreshold'],
                [evref_diff, y_next_diff, y_prev_diff] + list(nthr_used)
            ):
                out = out.std.SetFrameProp(prop_name, floatval=max(prop_val * 255, -1))

        return out

    if len(strs) != len(thrs):
        raise CustomValueError(
            f"You must pass an equal amount of values to strength {len(strs)} and thrs {len(thrs)}!",
            autodb_dpir, f"{len(strs)} != {len(thrs)}"
        )

    if edgemasker is None:
        edgemasker = core.std.Prewitt

    kernel = Kernel.ensure_obj(kernel)

    if vsdpir_args.get('fp16', None):
        warn("autodb_dpir: fp16 has been known to cause issues! It's highly recommended to set it to False!")

    vsdpir_final_args = dict[str, Any](cuda=cuda, fp16=vsdpir_args.pop('fp16', False))
    vsdpir_final_args |= vsdpir_args
    vsdpir_final_args.pop('strength', None)

    nthrs = [tuple(x / 255 for x in thr) for thr in thrs]

    is_rgb = clip.format.color_family is vs.RGB

    if not is_rgb:
        if matrix is None:
            matrix = get_prop(clip.get_frame(0), "_Matrix", int)

        targ_matrix = Matrix(matrix)

        rgb = kernel.resample(clip, format=vs.RGBS, matrix_in=targ_matrix)
    else:
        rgb = clip

    maxvalue = (1 << rgb.format.bits_per_sample) - 1  # type:ignore[union-attr]
    evref = edgemasker(rgb)
    evref = expr_func(evref, f"x {edgevalue} >= {maxvalue} x ?")
    evref_rm = evref.std.Median().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])

    if return_mask:
        return kernel.resample(evref_rm, format=clip.format, matrix=targ_matrix if not is_rgb else None)

    diffevref = core.std.PlaneStats(evref, evref_rm, prop='EdgeValRef')
    diffnext = core.std.PlaneStats(rgb, rgb.std.DeleteFrames([0]), prop='YNext')
    diffprev = core.std.PlaneStats(rgb, rgb[0] + rgb, prop='YPrev')

    db_clips = [
        dpir.DEBLOCK(rgb, strength=st, **vsdpir_final_args)
        .std.SetFrameProp('Adb_DeblockStrength', intval=int(st)) for st in strs
    ]

    debl = core.std.FrameEval(
        rgb, partial(_eval_db, clip=rgb, db_clips=db_clips, nthrs=nthrs),
        prop_clip=[diffevref, diffnext, diffprev]
    )

    return kernel.resample(debl, format=clip.format, matrix=targ_matrix if not is_rgb else None)