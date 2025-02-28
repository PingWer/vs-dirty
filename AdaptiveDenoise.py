import vapoursynth as vs
from vsdenoise import MVToolsPresets, Prefilter, mc_degrain, BM3DCuda, nl_means, MVTools, MotionMode, SADMode, MVTools, SADMode, MotionMode, Profile
from vstools import get_y, get_u, get_v
from vstools.enums import color
from vsmasktools import adg_mask


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
        Denoise adattivo con parametri default per film scan (16mm).

        Vengono passati 3 denoiser, mc_degrain (luma), NLMeans (chroma) e BM3DCuda (luma di nuovo).
        NLMeans prende come riferimento mc_degrain per eliminare le macchie di sporco / scanner noise dalla clip,
        mc_degrain ha quindi effetto solo sul luma, che viene successivamente passato a BM3DCuda per un secondo passaggio.
        Se precision = True, allora BM3DCuda riceve come riferimento un nuovo mc_degrain sulla base della clip già pulita.

        Le lumamask garantiscono il passaggio del denoiser solamente nelle zone del frame che sono più luminose, per non
        perdere dettaglio nelle zone scure.

        :param clip:                Clip to process.
        :param thsad:               Thsad for mc_degrain (luma denoise strength and chroma ref).
                                    Reccomended values: 300-800
        :param tr1:                 Temporal radius for the first mc_degrain and NLMeans. Reccomended values: 2-4
        :param tr2:                 Temporal radius for BM3DCuda (always) and the second mc_degrain (if precision = True).
                                    Reccomended values: 2-3
        :param sigma:               Sigma for BM3DCuda (luma denoise strength). Reccomended values: 3-10
        :param luma_mask_weaken1:   Modify how much dark spots should be denoised. Lower values means stronger denoise.
                                    Reccomended values: 0.6-0.9
        :param luma_mask_weaken2:   Only used if precision = True. Modify how much dark spots should be denoised on BM3DCuda.
                                    Lower values means stronger denoise. Reccomended values: 0.6-0.9
        :param chroma_strength:     Strength for NLMeans (chroma denoise strength). Reccomended values: 0.5-2
        :param precision:           If True a second reference and mask is made for BM3DCuda. Very slow.
        :param show_mask:           1 = Show the first lumamask, 2 = Show the second lumamask (if precision = True).

        :return:                    Denoised clip or luma_mask if show_mask is 1 or 2.
        """
    
    if precision == True and luma_mask_weaken2 == None:
        luma_mask_weaken2 = luma_mask_weaken1

    core = vs.core

    if clip.format.color_family not in {vs.YUV}:
        raise ValueError('AdaptiveDenoise: only YUV format not supported')

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
