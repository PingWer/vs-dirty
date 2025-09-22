import vapoursynth as vs
core = vs.core

def mini_BM3D(
    clip: vs.VideoNode, 
    profile: str = "LC", 
    accel: str = None,
    ref_gen: bool = False,
    **kwargs):
    """
    BM3D mini wrapper.

    :param clip:            Clip to process. Must be 32 bit float format.
    :param profile:         Precision. Accepted values: "FAST", "LC", "HIGH".
    :param accel:           Choose the hardware acceleration. Accepted values: "cuda_rtc", "cuda", "hip", "cpu".
    :param ref_gen:         Generate a reference clip for block-matching, you can also pass ref with kwargs.
                            If true while you passed a ref clip it will be used for the new ref.
    :return:                Denoised clip.
    """
    if profile == "FAST" or "fast":
        block_step=[8,7,8,7]
        bm_range=[9,9,7,7]
        ps_range=[4,5]
    elif profile == "LC" or "lc":
        block_step=[6,5,6,5]
        bm_range=[9,9,9,9]
        ps_range=[4,5]
    elif profile == "HIGH" or "high":
        block_step=[3,2,3,2]
        bm_range=[16,16,16,16]
        ps_range=[7,8]
    else:
        raise ValueError("mini_BM3D: Profile not recognized.")
    
    kwargs = dict(kwargs, block_step=block_step, bm_range=bm_range, ps_range=ps_range)
    
    if ref_gen:
        if accel == "cuda_rtc" or None:
            try:
                ref = core.bm3dcuda_rtc.BM3Dv2(clip, **kwargs, fast=False)
            except Exception:
                try:
                    ref = core.bm3dhip.BM3Dv2(clip, **kwargs, fast=False)
                except Exception:
                    ref = core.bm3dcpu.BM3Dv2(clip, **kwargs)
        elif accel == "cuda":
            ref = core.bm3dcuda.BM3Dv2(clip, **kwargs, fast=False)
        elif accel == "hip":
            ref = core.bm3dhip.BM3Dv2(clip, **kwargs, fast=False)
        elif accel == "cpu":
            ref = core.bm3dcpu.BM3Dv2(clip, **kwargs)
            
        kwargs = dict(kwargs, ref=ref)


    if accel == "cuda_rtc" or None:
        try:
            return core.bm3dcuda_rtc.BM3Dv2(clip, **kwargs, fast=False)
        except Exception:
            try:
                return core.bm3dhip.BM3Dv2(clip, **kwargs, fast=False)
            except Exception:
                return core.bm3dcpu.BM3Dv2(clip, **kwargs)
    elif accel == "cuda":
        return core.bm3dcuda.BM3Dv2(clip, **kwargs, fast=False)
    elif accel == "hip":
        return core.bm3dhip.BM3Dv2(clip, **kwargs, fast=False)
    elif accel == "cpu":
        return core.bm3dcpu.BM3Dv2(clip, **kwargs)