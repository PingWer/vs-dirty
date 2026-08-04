import vapoursynth as vs

from typing import Optional
from vstools import PlanesT
from vsscale import Backend as BackendV2

core = vs.core

from .mini_nlm import mini_NLM
from .mini_bm3d import mini_BM3D

class adenoise:
    """
    Preset class for _adaptive_denoiser.

    Intensive Adaptive Denoise.

    Three denoisers are applied: mc_degrain (luma), NLMeans/CBM3D (chroma), and BM3D (luma).
    NLMeans/CBM3D uses mc_degrain as reference to remove dirt spots and scanner noise from the clip,
    while mc_degrain affects only the luma, which is then passed to BM3D for a second denoising pass.
    If precision = True, a series of masks are created to enhance the denoise strength on flat areas avoiding textured area.

    Luma masks ensure that denoising is applied mostly to the brighter areas of the frame, preserving details in darker regions while cleaning them as much as possible.
    Note: Luma masks are more sensitive to variations than the sigma value for the final result.

    :param clip:                Clip to process (YUV or GRAY 16bit, if not will be internally converted in 16bit).
    :param thsad:               Thsad for mc_degrain (luma denoise strength and chroma ref).
                                Recommended values: 300-800
    :param tr:                  Temporal radius for temporal consistency across al the filter involved.
                                Recommended values: 2-3.
    :param sigma:               Sigma for BM3D (luma denoise strength).
                                Recommended values: 3-6.
    :param luma_mask_weaken:    Controls how much dark spots should be denoised. Lower values mean stronger overall denoise.
                                Recommended values: 0.6-0.9
    :param luma_mask_thr:       Threshold that determines what is considered bright and what is dark in the luma mask.
                                Recommended values: 0.15-0.25
    :param chroma_denoise:      Denoiser strength and type for chroma. NLMeans/CBM3D/ArtCNN.
                                Reccomended strength values: 0.5-2. If not given, 1.0 is used (or 0.8 for ArtCNN).
                                When using ArtCNN, the strength is a Merge value between the denoised chroma and original chroma for detail retention, only values between 0 and 1 are accepted.
                                Accepted denoiser types: "nlm", "cbm3d", "artcnn". If not given, nlm is used.
    :param precision:           If True, a flat mask is created to enhance the denoise strenght on flat areas avoiding textured area (95% accuracy).
    :param fast:                If True, uses a 3x faster version of flatmask (85% accuracy). Default: True.
    :param chroma_masking:      WIP If True, enables specific chroma masking for U/V planes.
    :param luma_over_texture:   Multiplier for the luma mask in precision mode. Lower value means more importance to textured areas, higher value means more importance to luma levels.
                                Accepted values: 0.0-1.0
    :param kwargs_flatmask:     Additional arguments for flatmask creation.
                                dict values (check hd_flatmask for more info):
                                sigma1: This value should be decided based on the details level of the clip and how much grain and noise is present. Usually 1 for really textured clip, 2-3 for a normal clip, 4-5 for a clip with strong noise or grain. By default sigma1 is sigma+1.
                                texture_strength: Texture strength for mask (0-inf). Values above 1 decrese the strength of the texture in the mask, lower values increase it. The max value is theoretical infinite, but there is no gain after some point.
                                edges_strength: Edges strength for mask (0-1). Basic multiplier for edges strength.
    :param show_mask:           1 = Show the first luma mask, 2 = Show the textured luma mask, 3 = Show the complete luma mask, 4 = Show the Chroma U Plane mask (if chroma_masking = True), 5 = Show the Chroma V Plane mask (if chroma_masking = True). Any other value returns the denoised clip.

    :return:                    16bit denoised clip. If show_mask is 1, 2, 3, 4 or 5, returns a tuple (denoised_clip, mask).
    """

    @classmethod
    def _adaptive_denoiser(
        cls,
        clip: vs.VideoNode,
        thsad: int = 500,
        tr: int = 2,
        sigma: float = 6,
        luma_mask_weaken: float = 0.75,
        luma_mask_thr: float = 0.196,
        chroma_denoise: float | str | tuple[float, str] = [1.0, "nlm"],
        precision: bool = True,
        fast: bool = True,
        chroma_masking: bool = False,
        luma_over_texture: float = 0.4,
        kwargs_flatmask: Optional[dict] = {},
        show_mask: int = 0,
        backend: Optional[BackendV2] = None,
        **kwargs,
    ) -> tuple[vs.VideoNode, vs.VideoNode]:

        from vstools import depth
        from vsdenoise import (
            Prefilter,
            mc_degrain,
            nl_means,
            MVTools,
            SearchMode,
        )
        from .admask import luma_mask_ping, luma_mask_man, hd_flatmask
        from .adutils import plane

        core = vs.core

        selected_mask = None

        if clip.format.color_family not in {vs.YUV, vs.GRAY}:
            raise ValueError(
                "adaptive_denoiser: only YUV and GRAY formats are supported"
            )

        clip = depth(clip, 16, dither_type="none")

        lumamask = luma_mask_ping(clip, thr=luma_mask_thr)
        darken_luma_mask = core.akarin.Expr([lumamask], f"x {luma_mask_weaken} *")

        if show_mask == 1:
            selected_mask = darken_luma_mask

        # Degrain
        if "is_bm3d_only" not in kwargs:
            mvtools = MVTools(clip)
            vectors = mvtools.analyze(
                blksize=16,
                tr=tr,
                overlap_div=2,
                lsad=300,
                search=SearchMode.UMH,
                mvlambda=0,
                satd=True
            )
            mfilter = mini_BM3D(
                clip, sigma=sigma * 1.25, radius=tr, profile="LC", planes=0
            )
            degrain = mc_degrain(
                clip,
                prefilter=Prefilter.DFTTEST,
                blksize=8,
                mfilter=mfilter,
                thsad=thsad,
                vectors=vectors,
                tr=tr,
                limit=1,
            )
        else:
            degrain = clip

        if precision and luma_over_texture != 1:
            flatmask_defaults = {
                "sigma1": sigma + 1,
                "texture_strength": 1,
                "edges_strength": 0.05,
            }
            flatmask = hd_flatmask(
                degrain, fast=fast, **(flatmask_defaults | kwargs_flatmask)
            )
            if show_mask == 2:
                selected_mask = flatmask

            final_mask = core.std.Merge(
                flatmask, darken_luma_mask, weight=luma_over_texture
            )
        else:
            final_mask = darken_luma_mask

        if show_mask == 3:
            selected_mask = final_mask

        if "is_digital" in kwargs:
            denoised = mini_BM3D(
                plane(clip, 0),
                ref=plane(degrain, 0),
                sigma=sigma,
                radius=tr,
                profile="HIGH",
            )
        else:
            denoised = mini_BM3D(
                plane(degrain, 0), sigma=sigma, radius=tr, profile="HIGH"
            )
        y_denoised = core.std.MaskedMerge(
            denoised, plane(clip, 0), final_mask
        )  # denoise applied to darker areas

        if clip.format.color_family != vs.GRAY:
            # Chroma denoise
            if isinstance(chroma_denoise, str):
                chroma_denoise = [1.0, chroma_denoise]
            elif isinstance(chroma_denoise, float):
                chroma_denoise = [chroma_denoise, "nlm"]

            if chroma_denoise[0] <= 0:
                chroma_denoised = clip
            else:
                if chroma_denoise[1] == "nlm":
                    chroma_denoised = nl_means(
                        clip, h=chroma_denoise[0], tr=tr, ref=degrain, planes=[1, 2]
                    )
                elif chroma_denoise[1] == "cbm3d":
                    chroma_denoised = mini_BM3D(
                        clip,
                        sigma=chroma_denoise[0],
                        radius=tr,
                        ref=degrain,
                        planes=[1, 2],
                    )
                elif chroma_denoise[1] == "artcnn":
                    from vsscale import ArtCNN

                    chroma_denoised = depth(
                        ArtCNN.R8F64_JPEG420(backend=backend).scale(depth(clip, 32)),
                        clip.format.bits_per_sample,
                    )
                    chroma_denoised = chroma_denoised.resize.Bilinear(
                        format=clip.format.id
                    )
                    chroma_denoised = core.std.ShufflePlanes(
                        [clip, chroma_denoised], planes=[0, 1, 2], colorfamily=vs.YUV
                    )

                    weights = [
                        0,
                        chroma_denoise[0]
                        if isinstance(chroma_denoise[0], float)
                        else 0.8,
                    ]
                    if clip.format.num_planes > 2:
                        weights.append(
                            chroma_denoise[0]
                            if isinstance(chroma_denoise[0], float)
                            else 0.8
                        )

                    chroma_denoised = core.std.Merge(
                        clip,
                        chroma_denoised,
                        weight=weights,
                    )
                else:
                    raise ValueError(
                        f"Only 'nlm', 'cbm3d' and 'artcnn' are supported for chroma denoising, got: {chroma_denoise[1]}"
                    )

            if (
                chroma_masking and chroma_denoise[0] > 0
            ) and clip.format.color_family == vs.YUV:
                u = plane(clip, 1)
                u_mask = luma_mask_man(u, t=1.5, s=2, a=0)
                u_masked = core.std.MaskedMerge(u, plane(chroma_denoised, 1), u_mask)
                v = plane(clip, 2)
                v_mask = luma_mask_man(v, t=1.5, s=2, a=0)
                v_masked = core.std.MaskedMerge(v, plane(chroma_denoised, 2), v_mask)
                chroma_denoised = core.std.ShufflePlanes(
                    clips=[chroma_denoised, u_masked, v_masked],
                    planes=[0, 0, 0],
                    colorfamily=vs.YUV,
                )

                if show_mask == 4:
                    selected_mask = v_mask
                elif show_mask == 5:
                    selected_mask = u_mask

        if clip.format.color_family == vs.GRAY:
            final = y_denoised
        else:
            final = core.std.ShufflePlanes(
                clips=[y_denoised, chroma_denoised, chroma_denoised],
                planes=[0, 1, 2],
                colorfamily=vs.YUV,
            )
        return final, selected_mask

    # Presets
    @staticmethod
    def scan65mm(
        clip: vs.VideoNode,
        thsad: int = 200,
        tr: int = 2,
        sigma: float = 2,
        fast: bool = True,
        luma_mask_weaken: float = 0.9,
        luma_mask_thr: float = 0.196,
        chroma_denoise: float | str | tuple[float, str] = [0.5, "nlm"],
        precision: bool = True,
        chroma_masking: bool = False,
        show_mask: int = 0,
        luma_over_texture: float = 0.4,
        kwargs_flatmask: dict = {},
        backend: Optional[BackendV2] = None,
    ) -> vs.VideoNode:
        """changes: thsad=200, sigma=2, luma_mask_weaken=0.9, chroma_strength=0.5"""
        denoised = adenoise._adaptive_denoiser(
            clip,
            thsad,
            tr,
            sigma,
            luma_mask_weaken,
            luma_mask_thr,
            chroma_denoise,
            precision,
            fast,
            chroma_masking,
            luma_over_texture,
            kwargs_flatmask,
            show_mask,
            backend=backend,
        )
        if show_mask in [1, 2, 3, 4, 5]:
            return denoised
        return denoised[0]

    @staticmethod
    def scan35mm(
        clip: vs.VideoNode,
        thsad: int = 400,
        tr: int = 2,
        sigma: float = 4,
        fast: bool = True,
        luma_mask_weaken: float = 0.8,
        luma_mask_thr: float = 0.196,
        chroma_denoise: float | str | tuple[float, str] = [0.7, "nlm"],
        precision: bool = True,
        chroma_masking: bool = False,
        show_mask: int = 0,
        luma_over_texture: float = 0.4,
        kwargs_flatmask: dict = {},
        backend: Optional[BackendV2] = None,
    ) -> vs.VideoNode:
        """changes: thsad=400, sigma=4, luma_mask_weaken=0.8, chroma_strength=0.7"""
        denoised = adenoise._adaptive_denoiser(
            clip,
            thsad,
            tr,
            sigma,
            luma_mask_weaken,
            luma_mask_thr,
            chroma_denoise,
            precision,
            fast,
            chroma_masking,
            luma_over_texture,
            kwargs_flatmask,
            show_mask,
            backend=backend,
        )
        if show_mask in [1, 2, 3, 4, 5]:
            return denoised
        return denoised[0]

    @staticmethod
    def scan16mm(
        clip: vs.VideoNode,
        thsad: int = 600,
        tr: int = 2,
        sigma: float = 8,
        fast: bool = True,
        luma_mask_weaken: float = 0.75,
        luma_mask_thr: float = 0.196,
        chroma_denoise: float | str | tuple[float, str] = [1.0, "nlm"],
        precision: bool = True,
        chroma_masking: bool = False,
        show_mask: int = 0,
        luma_over_texture: float = 0.4,
        kwargs_flatmask: dict = {},
        backend: Optional[BackendV2] = None,
    ) -> vs.VideoNode:
        denoised = adenoise._adaptive_denoiser(
            clip,
            thsad,
            tr,
            sigma,
            luma_mask_weaken,
            luma_mask_thr,
            chroma_denoise,
            precision,
            fast,
            chroma_masking,
            luma_over_texture,
            kwargs_flatmask,
            show_mask,
            backend=backend,
        )
        if show_mask in [1, 2, 3, 4, 5]:
            return denoised
        return denoised[0]

    @staticmethod
    def scan8mm(
        clip: vs.VideoNode,
        thsad: int = 800,
        tr: int = 2,
        sigma: float = 12,
        fast: bool = False,
        luma_mask_weaken: float = 0.75,
        luma_mask_thr: float = 0.196,
        chroma_denoise: float | str | tuple[float, str] = [1.5, "nlm"],
        precision: bool = True,
        chroma_masking: bool = False,
        show_mask: int = 0,
        luma_over_texture: float = 0.4,
        kwargs_flatmask: dict = {},
        backend: Optional[BackendV2] = None,
    ) -> vs.VideoNode:
        """changes: thsad=800, sigma=12, luma_over_texture=0.4, fast=False"""
        denoised = adenoise._adaptive_denoiser(
            clip,
            thsad,
            tr,
            sigma,
            luma_mask_weaken,
            luma_mask_thr,
            chroma_denoise,
            precision,
            fast,
            chroma_masking,
            luma_over_texture,
            kwargs_flatmask,
            show_mask,
            backend=backend,
        )
        if show_mask in [1, 2, 3, 4, 5]:
            return denoised
        return denoised[0]

    @staticmethod
    def digital(
        clip: vs.VideoNode,
        thsad: int = 300,
        tr: int = 2,
        sigma: float = 3,
        fast: bool = True,
        luma_mask_weaken: float = 0.75,
        luma_mask_thr: float = 0.196,
        chroma_denoise: float | str | tuple[float, str] = [1.0, "nlm"],
        precision: bool = True,
        chroma_masking: bool = False,
        show_mask: int = 0,
        luma_over_texture: float = 0.2,
        kwargs_flatmask: dict = {},
        backend: Optional[BackendV2] = None,
    ) -> vs.VideoNode:
        """changes: thsad=300, sigma=3, luma_over_texture=0.2"""
        denoised = adenoise._adaptive_denoiser(
            clip,
            thsad,
            tr,
            sigma,
            luma_mask_weaken,
            luma_mask_thr,
            chroma_denoise,
            precision,
            fast,
            chroma_masking,
            luma_over_texture,
            kwargs_flatmask,
            show_mask,
            is_digital=True,
            backend=backend,
        )
        if show_mask in [1, 2, 3, 4, 5]:
            return denoised
        return denoised[0]

    @staticmethod
    def bm3d(
        clip: vs.VideoNode,
        thsad: int = 500,
        tr: int = 2,
        sigma: float = 3,
        fast: bool = True,
        luma_mask_weaken: float = 0.75,
        luma_mask_thr: float = 0.196,
        chroma_denoise: float | str | tuple[float, str] = [1.0, "nlm"],
        precision: bool = True,
        chroma_masking: bool = False,
        show_mask: int = 0,
        luma_over_texture: float = 0.2,
        kwargs_flatmask: dict = {},
        backend: Optional[BackendV2] = None,
    ) -> vs.VideoNode:
        """changes: sigma=3, luma_over_texture=0.2"""
        denoised = adenoise._adaptive_denoiser(
            clip,
            thsad,
            tr,
            sigma,
            luma_mask_weaken,
            luma_mask_thr,
            chroma_denoise,
            precision,
            fast,
            chroma_masking,
            luma_over_texture,
            kwargs_flatmask,
            show_mask,
            is_bm3d_only=True,
            backend=backend,
        )
        if show_mask in [1, 2, 3, 4, 5]:
            return denoised
        return denoised[0]

    @staticmethod
    def default(
        clip: vs.VideoNode,
        thsad: int = 500,
        tr: int = 2,
        sigma: float = 6,
        fast: bool = True,
        luma_mask_weaken: float = 0.75,
        luma_mask_thr: float = 0.196,
        chroma_denoise: float | str | tuple[float, str] = [1.0, "nlm"],
        precision: bool = True,
        chroma_masking: bool = False,
        show_mask: int = 0,
        luma_over_texture: float = 0.4,
        kwargs_flatmask: dict = {},
        backend: Optional[BackendV2] = None,
    ) -> vs.VideoNode:
        """default profile"""
        denoised = adenoise._adaptive_denoiser(
            clip,
            thsad,
            tr,
            sigma,
            luma_mask_weaken,
            luma_mask_thr,
            chroma_denoise,
            precision,
            fast,
            chroma_masking,
            luma_over_texture,
            kwargs_flatmask,
            show_mask,
            backend=backend,
        )
        if show_mask in [1, 2, 3, 4, 5]:
            return denoised
        return denoised[0]


# Ported from fvsfunc
def auto_deblock(
    clip: vs.VideoNode,
    planes: PlanesT = [0, 1, 2],
    sigma: int = 15,
    tbsize: int = 1,
    luma_mask_strength: float = 0.9,
    pre: bool = False,
    accel: Optional[str] = None,
    mask_type: int = 0,
) -> vs.VideoNode:
    """
    Deblocker 8x8 and other.

    :param clip:                Clip to process (YUV 16bit, if not will be internally converted in 16bit).
    :param planes:              Which planes to process. Defaults to all planes.
    :param sigma:               Sigma value for dfttest deblock.
    :param tbsize:              Length of the temporal dimension (i.e. number of frames).
    :param luma_mask_strength:  Mask strength multiplier. Lower values mean stronger overall deblock.
    :param pre:                 If True, applies a preliminary deblocking with vsdenoise.deblock_qed.
    :param accel:               GPU acceleration method, Accepted values are "cuda" or "opencl".
    :param mask_type:           Mask type to use.
    """

    from .admask import luma_mask_ping, luma_mask_man, luma_mask
    from vsdenoise import deblock_qed
    from vstools import depth
    from dfttest2 import DFTTest

    if clip.format.color_family not in [vs.YUV]:
        raise TypeError("AutoDeblock: clip must be YUV color family!")

    clip = depth(clip, 16, dither_type="none")

    if pre:
        clip = deblock_qed(clip, planes=planes)

    if accel is not None:
        if accel.lower() == "cuda":
            deblock = core.vszipcu.DFTTest(clip, sigma=sigma, tbsize=tbsize, planes=planes)
        elif accel.lower() == "opencl":
            deblock = core.vszipcl.DFTTest(clip, sigma=sigma, tbsize=tbsize, planes=planes)
        else:
            raise ValueError(
                f"AutoDeblock: Invalid accel value '{accel}', accepted values are 'cuda' or 'opencl'."
            )
    else:
        deblock = DFTTest(clip, sigma=sigma, tbsize=tbsize, planes=planes)

    if mask_type == 0:
        lumamask = luma_mask(clip)
    elif mask_type == 1:
        lumamask = luma_mask_man(clip)
    else:
        lumamask = luma_mask_ping(clip)
    darken_luma_mask = core.std.Expr([lumamask], f"x {luma_mask_strength} *")
    final = core.std.MaskedMerge(deblock, clip, darken_luma_mask, planes=planes)

    return final


def msaa2x(
    clip: vs.VideoNode,
    ref: Optional[vs.VideoNode] = None,
    show_mask: bool = False,
    edgemask: Optional[vs.VideoNode] = None,
    sigma: float = 3,
    thr: float = None,
    strength: float = None,
    planes: PlanesT = 0,
    backend: Optional[BackendV2] = None,
    **kwargs,
) -> vs.VideoNode | tuple[vs.VideoNode, vs.VideoNode]:
    """
    Upscales only the edges with AI (ArtCNN DN) and downscales them.

    :param clip:            Clip to process (YUV or Grayscale).
    :param planes:          Which planes to process. Defaults to Y.
    :param ref:             Reference clip used to create the edgemask (should be a denoised clip). If None, clip will be used and will be denoised with adenoise.digital to prevent edge detail loss, but remove grain and noise.
    :param show_mask:       If True, returns a tuple containing the processed clip and the mask used.
    :param edgemask:        Pre-computed edgemask. If None, it will be computed internally.
    :param sigma:           Sigma used for edge fixing during antialiasing (remove dirty spots and blocking) only if ref is None.
    :param thr:             Threshold used for Binarize the clip, only 0-1 value area allowed. If None, no Binarize will be applied.
    :param strength:        Strength of the final merge between the original clip and the upscaled clip. 0-1 values accepted.
    :param kwargs:          Accepts advanced_edgemask arguments.
    """
    from vsscale import ArtCNN
    from vstools import depth
    from .admask import advanced_edgemask
    from .adutils import scale_binary_value, plane

    if isinstance(planes, int):
        planes = [planes]
    if clip.format.color_family == vs.GRAY:
        planes = [0]

    if clip.format.color_family == vs.RGB:
        raise ValueError("msaa2x: clip must be YUV or Gray color family!")

    clip = depth(clip, 16, dither_type="none")

    if edgemask is None:
        if ref is None:
            ref = adenoise.digital(
                clip,
                sigma=sigma,
                precision=False,
                chroma_denoise=[(0 if (1 in planes or 2 in planes) else 2), "cbm3d"],
                backend=backend,
            )

        if len(planes) == 1:
            edgemask = advanced_edgemask(plane(ref, 0), **kwargs)
        else:
            masks = [
                advanced_edgemask(plane(ref, p), **kwargs)
                if p in planes
                else plane(ref, p).std.BlankClip()
                for p in range(3)
            ]
            edgemask = core.std.ShufflePlanes(
                masks, planes=[0, 0, 0], colorfamily=ref.format.color_family
            )

    if thr is not None and thr != 0:
        edgemask = edgemask.std.Binarize(
            threshold=scale_binary_value(edgemask, thr, return_int=True)
        )

    clip_f32 = depth(clip, 32)
    upscaled = depth(
        ArtCNN.C4F32_DN(backend=backend).supersample(clip_f32, 2),
        clip.format.bits_per_sample,
    )
    downscaled = core.resize.Bicubic(upscaled, clip.width, clip.height)
    aa = core.std.MaskedMerge(clip, downscaled, edgemask, planes=0)

    if 1 in planes or 2 in planes:
        lefted = aa.resize.Spline36(src_left=-0.5)
        aa = core.std.ShufflePlanes(
            [aa, lefted, lefted], planes=[0, 1, 2], colorfamily=clip.format.color_family
        )
        aa = depth(
            ArtCNN.R8F64_Chroma(backend=backend).scale(depth(aa, 32)),
            clip.format.bits_per_sample,
        )
        chroma_downscaled = core.resize.Bicubic(aa, clip.width / 2, clip.height / 2)
        u = plane(chroma_downscaled, 1)
        v = plane(chroma_downscaled, 2)
        if 0 not in planes:
            downscaled = clip
        all_downscaled = core.std.ShufflePlanes(
            [downscaled, u, v], planes=[0, 0, 0], colorfamily=clip.format.color_family
        )
        aa = core.std.MaskedMerge(clip, all_downscaled, edgemask, planes=planes)

    if strength is not None or strength != 0:
        aa = core.std.Merge(aa, clip, weight=strength)

    if show_mask:
        return aa, edgemask

    return aa