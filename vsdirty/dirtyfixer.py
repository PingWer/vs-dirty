import vapoursynth as vs

from vstools import PlanesT
from vstools import depth
from .adutils import plane

core = vs.core

type Thickness = tuple[int, int, int, int]


def bore(
    clip: vs.VideoNode,
    planes: PlanesT | None = None,
    ythickness: Thickness | None = None,
    uthickness: Thickness | None = None,
    vthickness: Thickness | None = None,
    singlePlane: bool = True,
    **kwargs,
) -> vs.VideoNode:
    """
    Apply bore filter to clip's edges to remove dirty lines.

    Processes selected planes at their native resolution to avoid
    unnecessary scaling artifacts. Uses full resolution for luma,
    native chroma resolution for chroma planes.

    Only bore.SinglePlane and bore.MultiPlane are implemented, the other functions will probably never be implemented.

    :param clip:         Input clip (YUV or GRAY, RGB not supported).
    :param planes:       Which planes to process. Defaults to None. If None, planes to process depend strictly on passed thickness parameters (ythickness, uthickness, vthickness).
    :param ythickness:   Tuple of luma border thicknesses to process. (top, bottom, left, right). 0 means no processing. Default: (1, 1, 1, 1) if no thickness is specified.
    :param uthickness:   Tuple of chroma U border thicknesses to process. (top, bottom, left, right). 0 means no processing. If None and planes is specified, falls back to vthickness (if provided) or ythickness.
    :param vthickness:   Tuple of chroma V border thicknesses to process. (top, bottom, left, right). 0 means no processing. If None and planes is specified, falls back to uthickness (if provided) or ythickness.
    :param singlePlane:  If True uses bore.SinglePlane, otherwise bore.MultiPlane. MultiPlane cannot be used with GRAY clips.

    :return:             Processed clip with corrected borders, same format as input
    """

    if clip.format.color_family == vs.RGB:
        raise ValueError("bore: RGB clips are not supported.")

    if "plane" in kwargs:
        if planes is not None:
            raise ValueError("bore: cannot specify both 'planes' and 'plane'.")
        planes = kwargs.pop("plane")

    if kwargs:
        raise ValueError(f"bore: unexpected keyword arguments: {list(kwargs.keys())}")


    def _parse_thickness(thick: Thickness | None) -> Thickness:
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

    if planes is None:
        planes_dict: dict[int, Thickness] = {}
        if ythickness is not None:
            planes_dict[0] = _parse_thickness(ythickness)
        if uthickness is not None:
            planes_dict[1] = _parse_thickness(uthickness)
        if vthickness is not None:
            planes_dict[2] = _parse_thickness(vthickness)

        if not planes_dict:
            planes_dict[0] = (1, 1, 1, 1)
    else:
        if isinstance(planes, int):
            planes_list = [planes]
        else:
            planes_list = list(planes)

        if clip.format.color_family == vs.GRAY and any(p > 0 for p in planes_list):
            raise ValueError("bore: chroma planes cannot be specified for GRAY clips.")

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

        planes_dict = {}
        for p_idx in planes_list:
            if p_idx == 0:
                planes_dict[0] = p_y
            elif p_idx == 1:
                planes_dict[1] = p_u
            elif p_idx == 2:
                planes_dict[2] = p_v

    if singlePlane:
        splits = [plane(clip, i) for i in range(clip.format.num_planes)]
        processed_splits = []

        for i, p in enumerate(splits):
            if i in planes_dict:
                thick = planes_dict[i]
                if thick == (0, 0, 0, 0):
                    processed_splits.append(p)
                    continue

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

        if clip.format.color_family == vs.GRAY:
            return processed_splits[0]

        return core.std.ShufflePlanes(
            processed_splits, [0] * len(processed_splits), clip.format.color_family
        )

    else:
        if clip.format.color_family == vs.GRAY:
            raise ValueError("bore: MultiPlane cannot be used with GRAY clips.")

        if clip.format.subsampling_w == 0 and clip.format.subsampling_h == 0:
            work_clip = depth(clip, 32, dither_type="none")
            for p_idx in [0, 1, 2]:
                if p_idx in planes_dict and planes_dict[p_idx] != (0, 0, 0, 0):
                    thick = planes_dict[p_idx]
                    work_clip = core.bore.MultiPlane(
                        work_clip,
                        top=thick[0],
                        bottom=thick[1],
                        left=thick[2],
                        right=thick[3],
                        plane=p_idx,
                    )
            return depth(work_clip, clip.format.bits_per_sample, dither_type="none")

        if 0 in planes_dict and planes_dict[0] != (0, 0, 0, 0):
            thick = planes_dict[0]
            upclip = clip.resize.Bicubic(
                format=vs.YUV444PS,
                filter_param_a=0,
                filter_param_b=0,
            )
            bored_y = core.bore.MultiPlane(
                upclip,
                top=thick[0],
                bottom=thick[1],
                left=thick[2],
                right=thick[3],
                plane=0,
            )
            fixY = plane(
                depth(bored_y, clip.format.bits_per_sample, dither_type="none"), 0
            )
        else:
            fixY = plane(clip, 0)

        needs_u = (
            1 in planes_dict
            and planes_dict[1] != (0, 0, 0, 0)
            and clip.format.num_planes > 1
        )
        needs_v = (
            2 in planes_dict
            and planes_dict[2] != (0, 0, 0, 0)
            and clip.format.num_planes > 2
        )

        if needs_u or needs_v:
            u_orig = plane(clip, 1)
            v_orig = plane(clip, 2)
            luma_down = plane(clip, 0).resize.Bicubic(
                width=u_orig.width,
                height=u_orig.height,
                format=vs.GRAYS,
                filter_param_a=0,
                filter_param_b=0,
            )
            u_float = depth(u_orig, 32, dither_type="none")
            v_float = depth(v_orig, 32, dither_type="none")
            chroma_clip = core.std.ShufflePlanes(
                [luma_down, u_float, v_float],
                planes=[0, 0, 0],
                colorfamily=vs.YUV,
            )

            if needs_u:
                thick = planes_dict[1]
                chroma_clip = core.bore.MultiPlane(
                    chroma_clip,
                    top=thick[0],
                    bottom=thick[1],
                    left=thick[2],
                    right=thick[3],
                    plane=1,
                )

            if needs_v:
                thick = planes_dict[2]
                chroma_clip = core.bore.MultiPlane(
                    chroma_clip,
                    top=thick[0],
                    bottom=thick[1],
                    left=thick[2],
                    right=thick[3],
                    plane=2,
                )

            chroma_res = depth(
                chroma_clip, clip.format.bits_per_sample, dither_type="none"
            )
            fixU = plane(chroma_res, 1) if needs_u else u_orig
            fixV = plane(chroma_res, 2) if needs_v else v_orig
        else:
            fixU = plane(clip, 1) if clip.format.num_planes > 1 else None
            fixV = plane(clip, 2) if clip.format.num_planes > 2 else None

        # 3. Unione finale dei piani a risoluzione nativa
        clips_to_merge = [fixY]
        if fixU is not None:
            clips_to_merge.append(fixU)
        if fixV is not None:
            clips_to_merge.append(fixV)

        return core.std.ShufflePlanes(
            clips_to_merge,
            planes=[0] * len(clips_to_merge),
            colorfamily=clip.format.color_family,
        )
