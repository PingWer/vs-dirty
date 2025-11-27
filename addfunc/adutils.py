import vapoursynth as vs
from typing import SupportsIndex

def plane(
    clip: vs.VideoNode,
    index: SupportsIndex
    ) -> vs.VideoNode:
    return vs.core.std.ShufflePlanes(clip, index.__index__(), vs.GRAY)

def scale_binary_value(
        clip: vs.VideoNode,
        value: float,
        return_int: bool = True,
        )-> float:
    """
    Scales a value based on the bit depth of the clip.

    :param clip:         Clip to process.
    :param value:        Value to scale (0.0 - 1.0).
    :param return_int:   Whether to return an integer value. Default is True (will be ignore if the input clip is Float).
    :return:             Scaled value.
    """
    if clip.format is None:
        raise ValueError("scale_binary_value: Clip must have a defined format.")
    
    if clip.format.bits_per_sample is None:
        raise ValueError("scale_binary_value: Clip must have a defined bit depth.")
    
    if not (0.0 <= value <= 1.0):
        raise ValueError("scale_binary_value: Value must be between 0.0 and 1.0.")
    
    if clip.format.sample_type == vs.FLOAT:
        # For float clips, return the value as is
        return value
    
    max_val = (1 << clip.format.bits_per_sample) - 1

    if return_int:
        return int(value * max_val)
    else:
        return value * max_val

