import vapoursynth as vs
from vstools import depth, scale_value

def dirty_fix(
    clip: vs.VideoNode,
    columns: list[int] = None,
    col_val: list[int] = None,
    rows: list[int] = None,
    row_val: list[int] = None,
    prot_val: list[int] = [16, 235],
    min_val: int = 16,
    max_val: int = 235
) -> vs.VideoNode:
    """
    Fix for dirty lines by Man3500
    :param clip: Clip to process (YUV/GRAY 16bit, if not will be internally converted in 16bit with void dither).
    """

    core = vs.core

    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise ValueError('dirty_fix: only YUV and GRAY formats are supported')

    if clip.format.bits_per_sample != 16:
        clip = depth(clip, 16)