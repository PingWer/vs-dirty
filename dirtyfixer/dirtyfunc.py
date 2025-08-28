import vapoursynth as vs
from vstools import depth, scale_value, ColorRange

def dirty_fix(
    clip: vs.VideoNode,
    rows: dict[int, int] = {},
    columns: dict[int, int] = {},
    video_range: list[int] = [16, 235],
    var: int = 1,
    verbose: bool = False
) -> vs.VideoNode:
    """
    Fix for dirty lines by Man3500, This script amplifies by a percentage the value of the column/row.

    :param clip: Clip to process (GRAY 32bit, if not will be internally converted in 32bit with void dither).
    :param rows: Receives (row_number, row_value), negative row values will darken the selected row.
    :param columns: Receives (column_number, column_value), negative column values will darken the selected column.
    :param video_range: This will clip the output values to the selected range.
    :param var: Function in beta, sane values: 0.5-10.
    :param verbose: For debugging.

    :return: Fixed clip.
    """

    if clip.format.color_family not in {vs.GRAY}:
        raise ValueError('dirty_fix: only GRAY formats are supported')

    if clip.format.bits_per_sample != 32:
        clip = depth(clip, 32)

    min_range = scale_value(video_range[0], 8, 32, ColorRange.FULL, ColorRange.FULL)
    max_range = scale_value(video_range[1], 8, 32, ColorRange.FULL, ColorRange.FULL)

    expr = ""

    if rows:
        for row_num, row_val in rows:
            row_val = row_val/100
            row_strength = f"1 {row_val} x 1 + x {var} + / * +"
            rowstr = f"Y {row_num} = x {row_strength} * {min_range} {max_range} clip "
            expr = expr + rowstr
            if verbose:
                print(f"row_number {row_num} has value {row_val}")
    
    if columns:
        for col_num, col_val in columns:
            col_val = col_val/100
            col_strength = f"1 {col_val} x 1 + x {var} + / * +"
            colstr = f"X {col_num} = x {col_strength} * {min_range} {max_range} clip "
            expr = expr + colstr
            if verbose:
                print(f"column_number {col_num} has value {col_val}")
    
    expr = expr + "x "

    for _ in range(len(rows)):
        expr = expr + "? "
    for _ in range(len(columns)):
        expr = expr + "? "

    if verbose:
        print("final expr: " + expr, end="\r\n")
    clip = clip.akarin.Expr(expr)

    return clip