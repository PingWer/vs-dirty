import vapoursynth as vs
core = vs.core
from typing import SupportsIndex
from vstools import depth, PlanesT, get_y, get_v, get_u, get_r, get_g, get_b

def plane(
    clip: vs.VideoNode,
    index: SupportsIndex
    ) -> vs.VideoNode:
    return vs.core.std.ShufflePlanes(clip, index.__index__(), vs.GRAY)

