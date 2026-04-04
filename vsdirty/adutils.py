import vapoursynth as vs
from typing import SupportsIndex, Optional, List, Tuple, Union
from vstools import PlanesT


def plane(clip: vs.VideoNode, index: SupportsIndex) -> vs.VideoNode:
    """
    Returns a plane from the given clip at the specified index.

    :param clip: The input video clip.
    :param index: The index of the plane to return.

    :return: A new video clip containing only the specified plane.
    """
    return vs.core.std.ShufflePlanes(clip, index.__index__(), vs.GRAY)


def scale_binary_value(
    clip: vs.VideoNode,
    value: float,
    return_int: bool = True,
    bit: Optional[int] = None,
) -> float:
    """
    Scales a value based on the bit depth of the clip.

    :param clip:         Clip to process.
    :param value:        Value to scale (0.0 - 1.0).
    :param return_int:   Whether to return an integer value. Default is True (will be ignore if the input clip is Float).
    :param bit:          Bit depth of the clip. If None, the bit depth of the clip will be used. Default is None.

    :return:             Scaled value.
    """
    if bit is None and clip is not None:
        if clip.format is None:
            raise ValueError("scale_binary_value: Clip must have a defined format.")

        if clip.format.bits_per_sample is None:
            raise ValueError("scale_binary_value: Clip must have a defined bit depth.")

        bit = clip.format.bits_per_sample

    if not (0.0 <= value <= 1.0):
        raise ValueError("scale_binary_value: Value must be between 0.0 and 1.0.")

    if clip.format.sample_type == vs.FLOAT or bit == 32:
        # For float clips, return the value as is
        return value

    max_val = (1 << bit) - 1

    if return_int:
        return int(value * max_val)
    else:
        return value * max_val


def _detect_ranges(diff: vs.VideoNode, thr: float) -> List[Tuple[int, int]]:
    core = vs.core
    stats = core.std.PlaneStats(diff)

    indices = []
    warned_missing = False
    for n in range(diff.num_frames):
        f = stats.get_frame(n)
        pmax = f.props.get("PlaneStatsMax")

        if pmax is None:
            if not warned_missing:
                print(f"Warning: PlaneStatsMax missing at frame {n}")
                warned_missing = True
            continue

        if pmax > thr:
            indices.append(n)

    detected_ranges = []
    if indices:
        start = prev = indices[0]
        for i in indices[1:]:
            if i == prev + 1:
                prev = i
            else:
                detected_ranges.append((start, prev))
                start = prev = i
        detected_ranges.append((start, prev))

    return detected_ranges


def diff_and_swap(
    clipa: vs.VideoNode,
    clipb: vs.VideoNode,
    thr: float = 35000.0,
    discard_first: int = 0,
    discard_last: int = 0,
    ranges: Optional[Union[List[int], List[Tuple[int, int]]]] = None,
    precise_swapping: bool = False,
):
    """
    Compares two clips (clipa and clipb) frame by frame based on Luma.
    Identifies ranges of frames where the maximum absolute difference exceeds a certain threshold.

    This function is useful for "patching" a source (clipb) with frames from
    another source (clipa) where differences are significant (e.g. artifacts,
    corruption, or different versions).

    :param clipa:           The "correction" clip (e.g. WEB).
    :param clipb:           The "base" clip (e.g. BD).
    :param thr:             16-bit threshold applied to PlaneStatsMax. Default 35000.
    :param discard_first:   Number of detected ranges to discard from the beginning (ignored if explicit ranges are used).
    :param discard_last:    Number of detected ranges to discard from the end (ignored if explicit ranges are used).
    :param ranges:          Manual range control.
                                  - None: Detects ranges, prints the map and uses ALL of them.
                                  - [-1]: Discards ALL ranges (returns original clipb, no calculation).
                                  - [int, ...]: Detects ranges, prints the map and uses only those at the specified indices.
                                  - [(start, end), ...]: Uses ONLY these explicit ranges, SKIPS DETECTION.
    :param precise_swapping: If True, replaces only the differing pixels (diff > threshold -> expand -> blur -> mask).
                                  If False (default), replaces the entire frame in the detected ranges.

    :return:                A tuple `(merged, selected)` where `merged` is clipb with replacements from clipa, and
                               `selected` is a clip containing only the frames from clipa used for replacement
                               (or None if no frames were swapped).
    """
    from vstools import depth, replace_ranges

    core = vs.core

    if not isinstance(clipa, vs.VideoNode) or not isinstance(clipb, vs.VideoNode):
        raise vs.Error("diff_and_swap: both inputs must be VideoNode")

    if (
        not isinstance(discard_first, int)
        or not isinstance(discard_last, int)
        or discard_first < 0
        or discard_last < 0
    ):
        raise vs.Error(
            "diff_and_swap: discard_first/discard_last must be non-negative integers"
        )

    if clipa.format.id != clipb.format.id:
        raise vs.Error("diff_and_swap: both clips must have the same format")

    if clipa.format.color_family != vs.YUV or clipb.format.color_family != vs.YUV:
        raise vs.Error("diff_and_swap: only YUV clips are supported")

    clipa = depth(clipa, 16)
    clipb = depth(clipb, 16)

    min_frames = min(clipa.num_frames, clipb.num_frames)
    clipa_t = (
        clipa.std.Trim(0, min_frames - 1) if clipa.num_frames != min_frames else clipa
    )
    clipb_t = (
        clipb.std.Trim(0, min_frames - 1) if clipb.num_frames != min_frames else clipb
    )

    final_ranges = []
    explicit_ranges = False

    if ranges is not None and len(ranges) == 1 and ranges[0] == -1:
        return clipb, None

    if ranges is not None and len(ranges) > 0 and isinstance(ranges[0], tuple):
        explicit_ranges = True
        final_ranges = ranges
        print(f"diff_and_swap: Using explicit ranges: {final_ranges}")

    if precise_swapping:
        diff_clip = core.std.Expr([clipa_t, clipb_t], "x y - abs")

        binary_thr = scale_binary_value(
            diff_clip, 4 / 255, bit=diff_clip.format.bits_per_sample
        )
        mask = diff_clip.std.Binarize(binary_thr)
        mask = mask.std.Maximum().std.BoxBlur(hradius=1, vradius=1)

        replacement = core.std.MaskedMerge(clipb_t, clipa_t, mask)
    else:
        diff_clip = core.std.Expr([plane(clipa_t, 0), plane(clipb_t, 0)], "x y - abs")
        replacement = clipa_t

    if not explicit_ranges:
        detected_ranges = _detect_ranges(diff_clip, thr)

        if detected_ranges:
            if discard_first:
                detected_ranges = detected_ranges[discard_first:]
            if discard_last:
                if discard_last >= len(detected_ranges):
                    detected_ranges = []
                else:
                    detected_ranges = detected_ranges[
                        : len(detected_ranges) - discard_last
                    ]

        if detected_ranges:
            print("diff_and_swap Detected Range Map:")
            for i, (s, e) in enumerate(detected_ranges):
                print(f"Range {i}: {s} - {e} (Frames: {e - s + 1})")
        else:
            print("diff_and_swap: No ranges detected.")

        if ranges is None:
            final_ranges = detected_ranges
        else:
            for idx in ranges:
                if not isinstance(idx, int):
                    print(f"diff_and_swap Warning: Invalid index '{idx}' ignored.")
                    continue
                if 0 <= idx < len(detected_ranges):
                    final_ranges.append(detected_ranges[idx])
                else:
                    print(
                        f"diff_and_swap Warning: Index {idx} out of bounds of detected ranges."
                    )

    if final_ranges:
        merged = replace_ranges(clipb_t, replacement, final_ranges)
        segments = [replacement.std.Trim(a, b) for a, b in final_ranges]
        selected = core.std.Splice(segments)
    else:
        merged = clipb_t
        selected = None

    return merged, selected


def smart_CAS(
    clip: vs.VideoNode,
    sharpness: float = 0.5,
    strength: float = 0.5,
    texture: bool = False,
    mask: Union[vs.VideoNode, int, None] = None,
    # character_masking: bool = False,
    planes: PlanesT = 0,
    show_mask: bool = False,
    **kwargs,
) -> vs.VideoNode:
    """
    Applies Contrast Adaptive Sharpening (CAS) with optional masking and strength control.

    :param clip:         Input video clip.
    :param sharpness:    Sharpening intensity (0.0 - 1.0). Default is 0.5.
    :param strength:     Mixing factor to dilute the sharpening effect (0.0 - 1.0).
                         0.0 means full CAS force, while 1.0 returns the original clip.
                         Default is 0.5.
    :param texture:      If True, uses hd_flatmask for masking. If False, uses advanced_edgemask.
                         Ignored if a custom mask is provided. Default is False.
    :param mask:         Custom mask for sharpening. If None, a mask is generated.
                         The sharpened clip is applied where the mask is white (light).
                         If set to 0 (int), returns the sharpened clip without any masking.
    :param planes:       List of planes to process. Default is plane 0 (Luma).
    :param show_mask:    If True, returns the mask instead of the sharpened clip.
    :param kwargs:       Additional arguments passed to the mask generation functions.

    :return:             Sharpened video clip with masking applied to detail areas (unless mask=0).
    """
    core = vs.core
    from vsrgtools import gauss_blur

    cassed = core.cas.CAS(clip, sharpness=sharpness, planes=planes, opt=0)

    if strength != 0.0:
        cassed = core.std.Merge(cassed, clip, weight=strength)

    if isinstance(mask, vs.VideoNode):
        if show_mask:
            return mask
        return core.std.MaskedMerge(clip, cassed, mask, planes=planes)
    elif mask is None:
        mask_expand = kwargs.pop("expand", -1)

        if texture:
            from .admask import hd_flatmask

            mask = hd_flatmask(clip, expand=mask_expand, **kwargs)
            mask = gauss_blur(mask, sigma=1, planes=planes)
            if show_mask:
                return mask
            return core.std.MaskedMerge(clip, cassed, mask, planes=planes)
        else:
            from .admask import advanced_edgemask

            mask = advanced_edgemask(clip, expand=mask_expand, **kwargs)
            mask = gauss_blur(mask, sigma=1, planes=planes)
            if show_mask:
                return mask
            return core.std.MaskedMerge(clip, cassed, mask, planes=planes)
    elif isinstance(mask, int) and mask == 0:
        return cassed
    else:
        raise vs.Error("smart_CAS: mask must be a VideoNode or None")
