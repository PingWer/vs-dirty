import vapoursynth as vs
from typing import List
from vstools import PlanesT
from vstools import get_y, get_u, get_v, depth

core = vs.core

if not (hasattr(vs.core, 'fmtc') or hasattr(vs.core, 'bore')):
    raise ImportError("'fmtc' and 'libbore' are mandatory. Make sure the DLLs are present in the plugins folder.")

def bore(
        clip : vs.VideoNode,
        thickness: List[int] = [1,1,1,1],
        plane: PlanesT = 0,
        singlePlane = True
        ) -> vs.VideoNode:
    """
    Apply bore filter to clip edges to remove dirty lines.
    
    Processes selected planes at their native resolution to avoid 
    unnecessary scaling artifacts. Uses full resolution for luma,
    native chroma resolution for chroma planes.

    Only bore.SinglePlane and bore.MultiPlane are implemented, the other functions will probably never be implemented.

    :param clip:        Input clip (YUV or GRAY, RGB not supported)
    :param thickness:   List of border thicknesses to process. [top, bottom, left, right]. 0 means no processing.
    :param plane:       Plane(s) to process.
    :param singlePlane: If True uses bore.SinglePlane, otherwise bore.MultiPlane
    :return:            Processed clip with corrected borders, same format as input      
    """

    if clip.format.color_family == vs.RGB or clip.format.color_family == vs.GRAY:
        raise ValueError("easy_bore: RGB and GRAY clips are not supported yet.")
    
    if plane is None:
        raise ValueError("easy_bore: plane cannot be None.")

    if isinstance(plane, int):
        plane = [plane]

    if len(thickness) == 2:
        t, b = thickness
        l, r = 0, 0
    elif len(thickness) == 4:
        t, b, l, r = thickness
    else:
        t = thickness[0] if len(thickness) > 0 else 0
        b = thickness[1] if len(thickness) > 1 else t
        l = thickness[2] if len(thickness) > 2 else 0
        r = thickness[3] if len(thickness) > 3 else l

    fixY = get_y(clip)
    fixU = get_u(clip)
    fixV = get_v(clip)

    if 0 in plane:
        upclip = clip.resize.Bicubic(format=vs.YUV444PS)
        if singlePlane:
            fixY = get_y(depth(core.bore.SinglePlane(upclip, top=t, bottom=b, left=l, right=r, plane=0), clip.format.bits_per_sample))
        else:
            fixY = get_y(depth(core.bore.MultiPlane(upclip, top=t, bottom=b, left=l, right=r, plane=0), clip.format.bits_per_sample))
    
    if 1 in plane:
        upclip = clip.resize.Bicubic(width=get_u(clip).width, height=get_u(clip).height, format=vs.YUV444PS)
        upclip = core.std.ShufflePlanes([upclip, depth(clip,32), depth(clip,32)], planes=[0, 1, 2], colorfamily=vs.YUV)
        if singlePlane:
            fixU = get_u(depth(core.bore.SinglePlane(upclip, top=t, bottom=b, left=l, right=r, plane=1), clip.format.bits_per_sample))
        else:
            fixU = get_u(depth(core.bore.MultiPlane(upclip, top=t, bottom=b, left=l, right=r, plane=1), clip.format.bits_per_sample))
    
    if 2 in plane:
        upclip = clip.resize.Bicubic(width=get_v(clip).width, height=get_v(clip).height, format=vs.YUV444PS)
        upclip = core.std.ShufflePlanes([upclip, depth(clip,32), depth(clip,32)], planes=[0, 1, 2], colorfamily=vs.YUV)
        if singlePlane:
            fixV = get_v(depth(core.bore.SinglePlane(upclip, top=t, bottom=b, left=l, right=r, plane=2), clip.format.bits_per_sample))
        else:
            fixV = get_v(depth(core.bore.MultiPlane(upclip, top=t, bottom=b, left=l, right=r, plane=2), clip.format.bits_per_sample))

    merged = core.std.ShufflePlanes([fixY, fixU, fixV], planes=[0, 0, 0], colorfamily=vs.YUV)

    return merged