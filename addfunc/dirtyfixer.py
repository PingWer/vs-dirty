import vapoursynth as vs
from typing import List
from vstools import PlanesT
from vstools import get_y, get_u, get_v, depth

core = vs.core

if not (hasattr(vs.core, 'fmtc') or hasattr(vs.core, 'bore')):
    raise ImportError("'fmtc' and 'libbore' are mandatory. Make sure the DLLs are present in the plugins folder.")

def bore(
        clip : vs.VideoNode,
        ythickness: List[int] = None,
        uthickness: List[int] = None,
        vthickness: List[int] = None,
        planes: PlanesT = 0,
        singlePlane = True,
        **kwargs
        ) -> vs.VideoNode:
    """
    Apply bore filter to clip edges to remove dirty lines.
    
    Processes selected planes at their native resolution to avoid 
    unnecessary scaling artifacts. Uses full resolution for luma,
    native chroma resolution for chroma planes.

    Only bore.SinglePlane and bore.MultiPlane are implemented, the other functions will probably never be implemented.

    :param clip:        Input clip (YUV or GRAY, RGB not supported)
    :param ythickness:   List of luma border thicknesses to process. [top, bottom, left, right]. 0 means no processing.
    :param uthickness:   List of chroma U border thicknesses to process. [top, bottom, left, right]. 0 means no processing. if None, uses ythickness or vthickness.
    :param vthickness:   List of chroma V border thicknesses to process. [top, bottom, left, right]. 0 means no processing. if None, uses ythickness or uthickness.
    :param planes:      Plane(s) to process.
    :param singlePlane: If True uses bore.SinglePlane, otherwise bore.MultiPlane
    :param thickness:    Alternate name for ythickness.
    :return:            Processed clip with corrected borders, same format as input      
    """

    thickness = kwargs.get("thickness")
    if thickness is not None:
        ythickness = thickness

    if ythickness is None:
        ythickness = [1,1,1,1]

    if clip.format.color_family == vs.RGB:
        raise ValueError("easy_bore: RGB clips are not supported.")     
    
    if planes is None:
        raise ValueError("easy_bore: planes cannot be None.")

    if isinstance(planes, int):
        planes = [planes]

    def _get_tblr(thick):
        if len(thick) == 2:
            t, b = thick
            l, r = 0, 0
        elif len(thick) == 4:
            t, b, l, r = thick
        else:
            t = thick[0] if len(thick) > 0 else 0
            b = thick[1] if len(thick) > 1 else t
            l = thick[2] if len(thick) > 2 else 0
            r = thick[3] if len(thick) > 3 else l
        return t, b, l, r

    if uthickness is None:
        if vthickness is not None:
            uthickness = vthickness
        else:
            uthickness = ythickness
    if vthickness is None:
        if uthickness is not None:
            vthickness = uthickness
        else:
            vthickness = ythickness

    yt, yb, yl, yr = _get_tblr(ythickness)
    ut, ub, ul, ur = _get_tblr(uthickness)
    vt, vb, vl, vr = _get_tblr(vthickness)

    if clip.format.color_family == vs.GRAY:
        upclip = clip.resize.Bicubic(format=vs.YUV444PS)
        fixY = get_y(depth(core.bore.SinglePlane(upclip, top=yt, bottom=yb, left=yl, right=yr, plane=0), clip.format.bits_per_sample))
        return fixY

    fixY = get_y(clip)
    fixU = get_u(clip)
    fixV = get_v(clip)

    if 0 in planes:
        upclip = clip.resize.Bicubic(format=vs.YUV444PS)
        if singlePlane:
            fixY = get_y(depth(core.bore.SinglePlane(upclip, top=yt, bottom=yb, left=yl, right=yr, plane=0), clip.format.bits_per_sample))
        else:
            fixY = get_y(depth(core.bore.MultiPlane(upclip, top=yt, bottom=yb, left=yl, right=yr, plane=0), clip.format.bits_per_sample))
    
    if 1 in planes:
        upclip = clip.resize.Bicubic(width=get_u(clip).width, height=get_u(clip).height, format=vs.YUV444PS)
        upclip = core.std.ShufflePlanes([upclip, depth(clip,32), depth(clip,32)], planes=[0, 1, 2], colorfamily=vs.YUV)
        if singlePlane:
            fixU = get_u(depth(core.bore.SinglePlane(upclip, top=ut, bottom=ub, left=ul, right=ur, plane=1), clip.format.bits_per_sample))
        else:
            fixU = get_u(depth(core.bore.MultiPlane(upclip, top=ut, bottom=ub, left=ul, right=ur, plane=1), clip.format.bits_per_sample))
    
    if 2 in planes:
        upclip = clip.resize.Bicubic(width=get_v(clip).width, height=get_v(clip).height, format=vs.YUV444PS)
        upclip = core.std.ShufflePlanes([upclip, depth(clip,32), depth(clip,32)], planes=[0, 1, 2], colorfamily=vs.YUV)
        if singlePlane:
            fixV = get_v(depth(core.bore.SinglePlane(upclip, top=vt, bottom=vb, left=vl, right=vr, plane=2), clip.format.bits_per_sample))
        else:
            fixV = get_v(depth(core.bore.MultiPlane(upclip, top=vt, bottom=vb, left=vl, right=vr, plane=2), clip.format.bits_per_sample))

    merged = core.std.ShufflePlanes([fixY, fixU, fixV], planes=[0, 0, 0], colorfamily=vs.YUV)

    return merged