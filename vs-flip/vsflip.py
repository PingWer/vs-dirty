try:
    import vapoursynth as vs
except ImportError:
    raise ImportError('Vapoursynth R71> is required. Download it via: pip install vapoursynth')

import flip_evaluator as flip
from matplotlib.pyplot import imsave
import numpy as np
import ctypes
from vstools import frame2clip

core = vs.core

def frame_to_numpyArray(frame: vs.VideoNode) -> np.ndarray:
    """
    Convert a VapourSynth 1 frame VideoNode to a NumPy array.
    
    :param frame: The VapourSynth 1 frame VideoNode to convert.
    :return: A NumPy array representation of the frame.
    """
    frame = frame.get_frame(0)
    height, width = frame.height, frame.width

    def read_plane(plane):
        ptr = frame.get_read_ptr(plane)
        stride = frame.get_stride(plane)
        row_size = width * 4 

        c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)
        byte_ptr = ctypes.cast(ptr, c_ubyte_p)

        buf = bytearray(height * row_size)

        for y in range(height):
            src_offset = y * stride
            dst_offset = y * row_size
            ctypes.memmove(
                (ctypes.c_ubyte * row_size).from_buffer(buf, dst_offset),
                ctypes.addressof(byte_ptr.contents) + src_offset,
                row_size
            )
        return np.frombuffer(buf, dtype=np.float32).reshape((height, width))

    r = read_plane(0)
    g = read_plane(1)
    b = read_plane(2)

    return np.stack((r, g, b), axis=-1)

def numpy_to_frame(np_array: np.ndarray) -> vs.VideoFrame:
    """
    Convert a a NumPy array to VapourSynth VideoFrame.
    
    :param np_array: A 2D NumPy array representation of the frame .
    :return: The VapourSynth VideoFrame in GrayScaleS.
    """
    assert np_array.ndim == 2 and np_array.dtype == np.float32, "Input must be a 2D NumPy array of type float32."
    height, width = np_array.shape

    clip = core.std.BlankClip(format=vs.GRAYS, width=width, height=height, length=1)
    
    frame = clip.get_frame(0).copy()

    ptr = frame.get_write_ptr(0)
    stride = frame.get_stride(0)

    c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)
    dst = ctypes.cast(ptr, c_ubyte_p)

    row_bytes = width * 4
    for y in range(height):
        row = np_array[y].tobytes()
        offset = y * stride
        ctypes.memmove(ctypes.addressof(dst.contents) + offset, row, row_bytes)

    return frame


def vsflip_frame (
        ref_clip:vs.VideoNode,
        test_clip:vs.VideoNode,
        range:str="LDR",
        ref_frame: int = 0,
        test_frame: int = 0,
        parameters: dict = {"vc": [0.5, 3840, 0.6], "tonemapper": "ACES"},
        save_flip_error_mask: bool = False,
        debug: bool = False
)-> vs.VideoNode:
    """
    Compare two frames using the FLIP metric.

    :param ref_clip:                The reference VapourSynth VideoNode.
    :param test_clip:               The test VapourSynth VideoNode.   
    :param range:                   The range of the video, either "LDR" or "HDR".
    :param ref_frame:               The frame number of the reference clip to compare. Default is 0.
    :param test_frame:              The frame number of the test clip to compare. Default is 0.
    :param parameters:              A dictionary of parameters for the FLIP evaluation. Default is {"vc": [0.5, 3840, 0.6], "tonemapper": "ACES"}).
    :param save_flip_error_mask:    If True, saves the FLIP error mask as png in the script folder. Default is False.
    :param debug:                   If True, prints debug information. Default is False.
    :return:                        A VapourSynth 1 frame VideoNode containing the FLIP error map in GrayScaleS.
    """
    
    if range not in ["LDR", "HDR"]:
        raise ValueError("Range must be either 'LDR' or 'HDR'.")
    
    frame_test=test_clip[test_frame]
    frame_ref=ref_clip[ref_frame]

    if debug:
        print("Reference Frame Properties:\n")
        print(frame_ref)
        print("\nTest Frame Properties:\n")
        print(frame_test)

    if frame_ref.format.id != vs.RGBS or frame_test.format.id != vs.RGBS:
        frame_ref= core.resize.Bicubic(frame_ref, format=vs.RGBS)
        frame_test = core.resize.Bicubic(frame_test, format=vs.RGBS)

    np_ref = frame_to_numpyArray(frame_ref)
    np_test = frame_to_numpyArray(frame_test)

    flipErrorMap, meanFLIPError, parameters = flip.evaluate(np_ref, np_test, range, applyMagma=False, parameters=parameters)
    flipErrorMap = np.squeeze(flipErrorMap)

    if debug:
        print("Mean FLIP error: ", round(meanFLIPError, 6), "\n")

        print("The following parameters were used:")
        for key in parameters:
            val = parameters[key]
            if isinstance(val, float):
                val = round(val, 4)
            print("\t%s: %s" % (key, str(val)))

    if save_flip_error_mask:
        imsave(f"vsflip_error_map_{ref_frame}_{test_frame}.png", flipErrorMap, cmap="gray")

    print(flipErrorMap.max(), flipErrorMap.min())

    frame= numpy_to_frame(flipErrorMap)
    return frame2clip(frame)

def vsflipVideo(
        ref_clip:vs.VideoNode,
        test_clip:vs.VideoNode,
        range:str="LDR",
)-> vs.VideoNode:
    

    return True



# np_ref = r"vs-flip\ref2.png"
# np_test = r"vs-flip\test2.png"

# flipErrorMap, meanFLIPError, parameters = flip.evaluate(np_ref, np_test, "LDR", applyMagma=False)
# flipErrorMap = np.squeeze(flipErrorMap)

# print("Mean FLIP error: ", round(meanFLIPError, 6), "\n")

# print("The following parameters were used:")
# for key in parameters:
#     val = parameters[key]
#     if isinstance(val, float):
#         val = round(val, 4)
#     print("\t%s: %s" % (key, str(val)))

# plt.imsave("flip_error_map2.png", flipErrorMap, cmap="gray")
