import vapoursynth as vs
from vstools import depth, scale_value, ColorRange

def dirty_fix(
    clip: vs.VideoNode,
    columns: dict[int, int] = None,
    rows: dict[int, int] = None,
    prot_val: list[int] = [16, 235],
    video_range: list[int] = [16, 235],
    min_val: int = 16,
    max_val: int = 235
) -> vs.VideoNode:
    """
    Fix for dirty lines by Man3500
    :param clip: Clip to process (YUV/GRAY 32bit, if not will be internally converted in 32bit with void dither).
    """
    core = vs.core

    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise ValueError('dirty_fix: only YUV and GRAY formats are supported')

    if clip.format.bits_per_sample != 32:
        clip = depth(clip, 32)

    min_range = 0.1
    max_range = 0.9
    range_used = max_range - min_range
    orig_transfer = clip.get_frame(0).props.get("_Transfer")

    clip = clip.resize.Point(transfer_in=orig_transfer, transfer=8)  # Original Transfer to Linear

    clip.set_output(1)

    if rows is not None:
        row_expr = ""
        for row_num, row_val in rows:
            fun_pos = f"{max_range} {row_val} {range_used} 100 / * - {min_range} -"
            # fun_pos ="0.9"
            rekt_positive = f"{fun_pos} <= 0 x {min_range} - 0.01 / x {min_range} - {fun_pos} / {range_used} * {min_range} + ?"
            # rekt_positive = "0.1"
            rowstr = f"Y {row_num} = {rekt_positive} "
            row_expr = row_expr + rowstr

        row_expr = row_expr + "x "
        for _ in range(len(rows)):
            row_expr = row_expr + "? "
    clip = clip.akarin.Expr(row_expr)

    print(row_expr)

    clip.set_output(2)

    clip = clip.resize.Point(transfer_in=8, transfer=orig_transfer) # Linear to Original Transfer

    return clip