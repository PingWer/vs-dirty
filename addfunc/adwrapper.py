import vapoursynth as vs
core = vs.core

def mini_BM3D(clip, profile, **kwargs):
    """
    BM3D mini wrapper.

    :param clip:            Clip to process.
    :param profile:         Precision. Accepted values: "LC", "HIGH".
    :return:                Denoised clip.
    """
    if profile == "LC":
        block_step=6
        bm_range=9
        ps_range=4
    elif profile == "HIGH":
        block_step=3
        bm_range=16
        ps_range=7
    
    try:
        return core.bm3dcuda_rtc.BM3Dv2(clip, **kwargs, block_step=block_step, bm_range=bm_range, ps_range=ps_range, fast=False)
    except Exception:
        return core.bm3dcpu.BM3Dv2(clip, **kwargs, block_step=block_step, bm_range=bm_range, ps_range=ps_range)