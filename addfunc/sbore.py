import vapoursynth as vs
from typing import Optional, List
from vstools import PlanesT

core = vs.core

if not (hasattr(vs.core, 'fmtc') or hasattr(vs.core, 'bore')):
    raise ImportError("'fmtc' and 'libbore' are mandatory. Make sure the DLLs are present in the plugins folder.")


def dirtyline_mask(clip: vs.VideoNode, thickness: int = 1) -> vs.VideoNode:
    from vstools import get_y

    black = get_y(core.std.BlankClip(clip, color=[0]))

    white = get_y(core.std.BlankClip(clip, color=[65535]))

    top = white.std.Crop(bottom=clip.height - thickness)

    bottom = white.std.Crop(top=clip.height - thickness)

    mask = core.std.StackVertical([top, core.std.Crop(black, top=thickness, bottom=thickness), bottom])
    return get_y(mask)


def bore(
        clip : vs.VideoNode,
        strength: Optional[float] = 1.0,
        thickness: List[int] = [1,1,1,1],
        plane: PlanesT = 0, 
        show_mask: bool = False 
        ) -> vs.VideoNode:

    return "suca"