import vapoursynth as vs

from vstools import PlanesT
from vstools import depth
from .adutils import plane

core = vs.core

if not (hasattr(vs.core, "fmtc") and hasattr(vs.core, "bore")):
    raise ImportError(
        "'fmtc' and 'libbore' are mandatory. Make sure the DLLs are present in the plugins folder."
    )

type Thickness = tuple[int, int, int, int]


def bore(
    clip: vs.VideoNode,
    planes: PlanesT = 0,
    ythickness: Thickness | None = None,
    uthickness: Thickness | None = None,
    vthickness: Thickness | None = None,
    singlePlane: bool = True,
) -> vs.VideoNode:
    """
    Apply bore filter to clip's edges to remove dirty lines.

    Processes selected planes at their native resolution to avoid
    unnecessary scaling artifacts. Uses full resolution for luma,
    native chroma resolution for chroma planes.

    Only bore.SinglePlane and bore.MultiPlane are implemented, the other functions will probably never be implemented.

    :param clip:         Input clip (YUV or GRAY, RGB not supported).
    :param planes:       Which planes to process. Defaults to Y.
    :param ythickness:   Tuple of luma border thicknesses to process. (top, bottom, left, right). 0 means no processing. Default: (1, 1, 1, 1).
    :param uthickness:   Tuple of chroma U border thicknesses to process. (top, bottom, left, right). 0 means no processing. if None, uses ythickness or vthickness if is not None.
    :param vthickness:   Tuple of chroma V border thicknesses to process. (top, bottom, left, right). 0 means no processing. if None, uses ythickness or uthickness if is not None.
    :param singlePlane:  If True uses bore.SinglePlane, otherwise bore.MultiPlane. MultiPlane cannot be used with GRAY clips.

    :return:             Processed clip with corrected borders, same format as input
    """

    if clip.format.color_family == vs.RGB:
        raise ValueError("bore: RGB clips are not supported.")

    if planes is None:
        raise ValueError("bore: planes cannot be None.")

    def _parse_thickness(thick: Thickness | None) -> Thickness:
        """Parse input into Thickness tuple."""
        if thick is None:
            return (0, 0, 0, 0)

        if isinstance(thick, tuple):
            t_len = len(thick)
            if t_len == 4:
                return thick
            else:
                raise ValueError(
                    f"Thickness sequence must have exactly 4 elements (top, bottom, left, right), got {t_len}"
                )

        raise ValueError(
            f"Thickness must be (top, bottom, left, right), or None, got {type(thick)}"
        )

    p_y = _parse_thickness(ythickness) if ythickness is not None else (1, 1, 1, 1)

    if uthickness is None and vthickness is None:
        p_u = p_y
        p_v = p_y
    elif uthickness is None and vthickness is not None:
        p_v = _parse_thickness(vthickness)
        p_u = p_v
    elif uthickness is not None and vthickness is None:
        p_u = _parse_thickness(uthickness)
        p_v = p_u
    else:
        p_u = _parse_thickness(uthickness)
        p_v = _parse_thickness(vthickness)

    if isinstance(planes, int):
        planes = [planes]

    if singlePlane:
        splits = [plane(clip, i) for i in range(clip.format.num_planes)]
        processed_splits = []

        for i, p in enumerate(splits):
            if i in planes:
                if i == 0:
                    thick = p_y
                elif i == 1:
                    thick = p_u
                else:
                    thick = p_v

                p_float = depth(p, 32, dither_type="none")

                p_bored = core.bore.SinglePlane(
                    p_float,
                    top=thick[0],
                    bottom=thick[1],
                    left=thick[2],
                    right=thick[3],
                    plane=0,
                )

                p_out = depth(p_bored, clip.format.bits_per_sample, dither_type="none")
                processed_splits.append(p_out)
            else:
                processed_splits.append(p)

        return core.std.ShufflePlanes(
            processed_splits, [0] * len(processed_splits), clip.format.color_family
        )

    else:
        if clip.format.color_family == vs.GRAY:
            raise ValueError("MultiPlane cannot be used with GRAY clips")

        fixY = plane(clip, 0)
        fixU = plane(clip, 1) if clip.format.num_planes > 1 else None
        fixV = plane(clip, 2) if clip.format.num_planes > 2 else None

        if 0 in planes:
            upclip = clip.resize.Bicubic(format=vs.YUV444PS)
            fixY = plane(
                depth(
                    core.bore.MultiPlane(
                        upclip,
                        top=p_y[0],
                        bottom=p_y[1],
                        left=p_y[2],
                        right=p_y[3],
                        plane=0,
                    ),
                    clip.format.bits_per_sample,
                ),
                0,
            )

        if 1 in planes and fixU is not None:
            upclip = clip.resize.Bicubic(
                width=fixU.width, height=fixU.height, format=vs.YUV444PS
            )
            upclip = core.std.ShufflePlanes(
                [upclip, depth(clip, 32), depth(clip, 32)],
                planes=[0, 1, 2],
                colorfamily=vs.YUV,
            )
            fixU = plane(
                depth(
                    core.bore.MultiPlane(
                        upclip,
                        top=p_u[0],
                        bottom=p_u[1],
                        left=p_u[2],
                        right=p_u[3],
                        plane=1,
                    ),
                    clip.format.bits_per_sample,
                ),
                1,
            )

        if 2 in planes and fixV is not None:
            upclip = clip.resize.Bicubic(
                width=fixV.width, height=fixV.height, format=vs.YUV444PS
            )
            upclip = core.std.ShufflePlanes(
                [upclip, depth(clip, 32), depth(clip, 32)],
                planes=[0, 1, 2],
                colorfamily=vs.YUV,
            )
            fixV = plane(
                depth(
                    core.bore.MultiPlane(
                        upclip,
                        top=p_v[0],
                        bottom=p_v[1],
                        left=p_v[2],
                        right=p_v[3],
                        plane=2,
                    ),
                    clip.format.bits_per_sample,
                ),
                2,
            )

        clips_to_merge = [fixY]
        if fixU:
            clips_to_merge.append(fixU)
        if fixV:
            clips_to_merge.append(fixV)

        if clip.format.color_family == vs.GRAY:
            return fixY

        return core.std.ShufflePlanes(
            clips_to_merge,
            planes=[0] * len(clips_to_merge),
            colorfamily=clip.format.color_family,
        )
