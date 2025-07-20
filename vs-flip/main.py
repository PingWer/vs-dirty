try:
    import vapoursynth as vs
except ImportError:
    raise ImportError('Vapoursynth R71> is required. Download it via: pip install vapoursynth')

import flip_evaluator as flip
import matplotlib.pyplot as plt
import numpy as np
import ctypes

core = vs.core

def frame_to_numpyArray(frame: vs.VideoNode) -> np.ndarray:
    """
    Convert a VapourSynth clip to a NumPy array.
    
    :param clip: The VapourSynth video node to convert.
    :return: A NumPy array representation of the clip.
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


def vsflip (
        ref_clip:vs.VideoNode,
        test_clip:vs.VideoNode,
        range:str="LDR",
        ref_frame: int = 0,
        test_frame: int = 0
)-> vs.VideoNode:
    """
    Vuole come input due VideoNode (non ho testato se vano bene res diverse) e devi anche specificare il frame da annalizzare di entrambe, ovviamente scegli lo stesso
    """
    
    if range not in ["LDR", "HDR"]:
        raise ValueError("Range must be either 'LDR' or 'HDR'.")
    
    frame_test=test_clip[test_frame]
    frame_ref=ref_clip[ref_frame]

    frame_ref= core.resize.Bicubic(frame_ref, format=vs.RGBS)
    frame_test = core.resize.Bicubic(frame_test, format=vs.RGBS)

    np_ref = frame_to_numpyArray(frame_ref)
    np_test = frame_to_numpyArray(frame_test)

    flipErrorMap, meanFLIPError, parameters = flip.evaluate(np_ref, np_test, range, applyMagma=False)
    flipErrorMap = np.squeeze(flipErrorMap)

    print("Mean FLIP error: ", round(meanFLIPError, 6), "\n")

    print("The following parameters were used:")
    for key in parameters:
        val = parameters[key]
        if isinstance(val, float):
            val = round(val, 4)
        print("\t%s: %s" % (key, str(val)))

    plt.imsave("flip_error_map2.png", flipErrorMap, cmap="gray")

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
