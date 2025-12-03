import vapoursynth as vs
from typing import SupportsIndex, Optional

def plane(
    clip: vs.VideoNode,
    index: SupportsIndex
    ) -> vs.VideoNode:
    return vs.core.std.ShufflePlanes(clip, index.__index__(), vs.GRAY)

def scale_binary_value(
        clip: vs.VideoNode,
        value: float,
        return_int: bool = True,
        bit: Optional[int] = None,
        )-> float:
    """
    Scales a value based on the bit depth of the clip.

    :param clip:         Clip to process.
    :param value:        Value to scale (0.0 - 1.0).
    :param return_int:   Whether to return an integer value. Default is True (will be ignore if the input clip is Float).
    :param bit:          Bit depth of the clip. If None, the bit depth of the clip will be used. Default is None. c
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
    
    if clip.format.sample_type == vs.FLOAT or bit==32:
        # For float clips, return the value as is
        return value

    max_val = (1 << bit) - 1

    if return_int:
        return int(value * max_val)
    else:
        return value * max_val

