try:
    import vapoursynth as vs
except ImportError:
    raise ImportError('Vapoursynth R71> is required. Download it via: pip install vapoursynth')

try:
    from vsdenoise import Prefilter, mc_degrain, nl_means, MVTools, SearchMode, MotionMode, SADMode, MVTools, SADMode, MotionMode, deblock_qed
    from vstools import get_y, get_u, get_v, PlanesT, depth
    from vsmasktools import Morpho
    from typing import Optional
    from .admask import flat_mask, luma_mask_ping, luma_mask_man, luma_mask
    from .adwrapper import mini_BM3D
except ImportError:
    raise ImportError('vsdenoise, vstools, vsmasktools are required. Download them via: pip install vsjetpack. Other depedencies can be found here: https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack' )

core = vs.core
if not (hasattr(core, 'dfttest') or hasattr(core, 'fmtc') or hasattr(core, 'akarin')):
    raise ImportError("'dfttest', 'fmtc' and 'akarin' are mandatory. Make sure the DLLs are present in the plugins folder.")


#la desc dei preset la puoi mettere solo qui dentro il codice, 
#ma non verrà mostrata nella preview di vscode (limitazione di vscode)
#Stessa cosa vale per la desc di _adpative_denoiser, non si vedrà se non entrando dentro la funzione, 
#l'alternativa è metterla come desc della classe adenoise, scegli tu 
_PRESETS = {
    "scan65mm": dict(thsad=200, sigma=2, luma_mask_weaken=0.9, chroma_strength=0.5),
    "scan35mm": dict(thsad=400, sigma=4, luma_mask_weaken=0.8, chroma_strength=0.7),
    "scan16mm": dict(thsad=600, sigma=8),
    "scan8mm" : dict(tr2=2, chroma_strength=1.5),
    "digital" : dict(thsad=300, sigma=3, texture_penalty=1)
}

class adenoise:
    """Preset class for _adaptive_denoiser."""
    pass

#genera a runtime i metodi, non lo puoi togliere
for name, params in _PRESETS.items():
    safe_name = name if name[0].isalpha() else f"_{name}"
    def make_preset(preset_params):
        def wrapper(cls, clip: vs.VideoNode, **kwargs):
            final = {**preset_params, **kwargs}
            return _adaptive_denoiser(clip, **final)
        return classmethod(wrapper)
    setattr(adenoise, safe_name, make_preset(params))


def _adaptive_denoiser (
    clip: vs.VideoNode,
    thsad: int = 800,
    tr: int = 2,
    tr2: int = 1,
    sigma: float = 12,
    luma_mask_weaken: float = 0.75,
    luma_mask_thr: float = 50,
    chroma_strength: float = 1.0,
    precision: bool = True,
    chroma_masking: bool = False,
    show_mask: int = 0,
    flat_penalty: float = 0.5,
    texture_penalty: float = 1.1,
) -> vs.VideoNode:
    """
    Intensive Adaptive Denoise with default parameters for film scans (16mm).

    Three denoisers are applied: mc_degrain (luma), NLMeans (chroma), and BM3D (luma).
    NLMeans uses mc_degrain as reference to remove dirt spots and scanner noise from the clip,
    while mc_degrain affects only the luma, which is then passed to BM3D for a second denoising pass.
    If precision = True, BM3D receives a new mc_degrain reference based on the already cleaned clip (slower).

    Luma masks ensure that denoising is applied only to the brighter areas of the frame, preserving details in darker regions while cleaning them as much as possible.
    Note: Luma masks are more sensitive to variations than the sigma value for the final result.

    :param clip:                Clip to process (YUV 16bit, if not will be internally converted in 16bit with void dither).
    :param thsad:               Thsad for mc_degrain (luma denoise strength and chroma ref).
                                Recommended values: 300-800
    :param tr:                  Temporal radius for mvtools and nlm.
                                Recommended values: 2-3 (1 means no temporal denoise).
    :param tr2:                 Temporal radius for BM3D.
                                Recommended values: 1-2 (0 means no temporal denoise).
    :param sigma:               Sigma for BM3D (luma denoise strength).
                                Recommended values: 3-10. If precision = True, you can safely double he value used.
    :param luma_mask_weaken:    Controls how much dark spots should be denoised. Lower values mean stronger denoise.
                                Recommended values: 0.6-0.9
    :param luma_mask_thr:       Mi chiamo ping e non metto le descrizioni.
    :param chroma_strength:     Strength for NLMeans (chroma denoise strength).
                                Recommended values: 0.5-2
    :param precision:           If True, a flat mask is created to enhance the denoise strenght on flat areas avoiding textured area (90% accuracy).
    :param mask_type:           0 = Standard Luma mask, 1 = Custom Luma mask (more linear) , 2 = Custom Luma mask (less linear).
    :param show_mask:           1 = Show the first luma mask, 2 = Show the Chroma V Plane mask (if chroma_masking = True), 3 = Show the Chroma U Plane mask (if chroma_masking = True), 4 = Show the flatmask.

    :return:                    16bit denoised clip or luma_mask if show_mask is 1, 2 or 3.
    """
    

    core = vs.core

    if clip.format.color_family not in {vs.YUV}:
        raise ValueError('adaptive_denoiser: only YUV formats are supported')

    if clip.format.bits_per_sample != 16:
        clip = depth(clip, 16)

    lumamask = luma_mask_ping(clip, thr=luma_mask_thr)
    darken_luma_mask = core.std.Expr([lumamask], f"x {luma_mask_weaken} *")

    #Denoise
    mvtools = MVTools(clip)
    vectors = mvtools.analyze(blksize=16, tr=tr, overlap=8, lsad=300, search=SearchMode.UMH, truemotion=MotionMode.SAD, dct=SADMode.MIXED_SATD_DCT)
    mfilter = mini_BM3D(clip=get_y(clip), sigma=sigma*2, radius=1, profile="LC", planes=0)
    mfilter = core.std.ShufflePlanes(clips=[mfilter, get_u(clip), get_v(clip)], planes=[0,0,0], colorfamily=vs.YUV)
    ref = mc_degrain(clip, prefilter=Prefilter.DFTTEST, mfilter=mfilter, thsad=thsad, vectors=vectors, tr=tr)

    if precision:
        flatmask = flat_mask(ref, tr=tr2, sigma=sigma)
        if show_mask == 4:
            return flatmask
        darken_luma_mask = core.std.Expr(
        [darken_luma_mask, flatmask],
        f"y 65535 = x {flat_penalty} * x {texture_penalty} * ?")
        
        darken_luma_mask = Morpho.deflate(Morpho.inflate(darken_luma_mask)) # Inflate+Deflate for smoothing

    denoised = mini_BM3D(get_y(ref), sigma=sigma, radius=tr2, profile="HIGH", planes=0)
    luma = get_y(core.std.MaskedMerge(denoised, get_y(clip), darken_luma_mask, planes=0)) ##denoise applied to darker areas

    if show_mask == 1:
        return darken_luma_mask


    #Chroma NLMeans
    if chroma_strength <= 0:
        chroma_denoised = clip
    else:
        chroma_denoised = nl_means(clip, h=chroma_strength, tr=tr, ref=ref, planes=[1,2])
    
    #TODO
    #chroma mask fine tuning
    v_mask = None #Per evitare UnboundLocalError
    u_mask = None

    if chroma_masking or chroma_strength<=0:
        v=get_v(clip)
        v_mask= luma_mask_man(v, t=1.5, s=2, a=0)
        v_masked = core.std.MaskedMerge(v, get_v(chroma_denoised), v_mask)
        u=get_u(clip)
        u_mask= luma_mask_man(u, t=1.5, s=2, a=0)
        u_masked = core.std.MaskedMerge(u, get_u(chroma_denoised), u_mask)
        chroma_denoised = core.std.ShufflePlanes(clips=[chroma_denoised, u_masked, v_masked], planes=[0,0,0], colorfamily=vs.YUV)
    
    if show_mask == 2:
        return v_mask
    elif show_mask == 3:
        return u_mask
    
    final = core.std.ShufflePlanes(clips=[luma, get_u(chroma_denoised), get_v(chroma_denoised)], planes=[0,0,0], colorfamily=vs.YUV)
    return final

#TODO
#Ported from fvsfunc 
def auto_deblock(
    clip: vs.VideoNode,
    # edgevalue: int = 24,
    sigma: int = 15,
    tbsize: int = 1,
    luma_mask_strength: float = 0.9,
    mask_type: int = 0,
    planes: PlanesT = None
) -> vs.VideoNode:
    """
    Funzione che si spera funzioni, dovrebbe fare deblock MPEG2 ma essendo portata da avisynth di eoni fa dubito lo faccia bene.
    """

    core=vs.core
    try:
        from functools import partial
    except ImportError:
        raise ImportError('functools is required')

    if clip.format.color_family not in [vs.YUV]:
        raise TypeError("AutoDeblock: clip must be YUV color family!")

    if clip.format.bits_per_sample != 16:
        clip = depth(clip, 16)

    # Scale values to handle high bit depths
    # shift = clip.format.bits_per_sample - 8
    # edgevalue = edgevalue << shift
    # maxvalue = (1 << clip.format.bits_per_sample) - 1

    # orig è una edgemask, che significa orig lo sa solo jesus
    # orig = core.std.Prewitt(clip)
    # Se x è maggiore o uguale di edgevalue (def:24) allora restituisci maxvalue altrimenti x
    # È quasi un binarize ma i valori sotto edgevalue rimangono uguali
    # orig = core.std.Expr(orig, f"x {edgevalue} >= {maxvalue} x ?")
    # Doppia Median sulla edgemask
    # orig_d = orig.std.Median().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])

    # Passa la clip con un po' meno grana a vsdenoise.deblock_qed
    predeblock = deblock_qed(clip.rgvs.RemoveGrain(2).rgvs.RemoveGrain(2), planes=planes)

    # Se questi vengono svolti anche se fast = True farebbe molto ridere, sigma regola la forza, ne vengono fatti diversi
    # Analogamente al nostro approccio su Adaptive Denoise si potrebbe usare un signolo DFT e una mask per la forza
    deblock = core.dfttest.DFTTest(predeblock, sigma=sigma, tbsize=tbsize, planes=planes)

    # Prende le differenze di statistiche tra edgemask prima e dopo la Median
    # difforig = core.std.PlaneStats(orig, orig_d, prop='Orig')
    # Prende le differenze di statistiche tra la clip e il frame successivo
    # diffnext = core.std.PlaneStats(clip, clip.std.DeleteFrames([0]), prop='YNext')
    # Frame eval che prende la clip, gli effettua eval_deblock_strength passandogli le statistiche

    # Da implementare per usare le informazioni per regolare la mask
    # autodeblock = core.std.FrameEval(clip, partial(eval_deblock_strength,
    #                                  clip=clip, deblock=deblock),
    #                                  prop_src=[difforig,diffnext])
    
    if (mask_type == 0):
        lumamask = luma_mask(clip)
    elif (mask_type == 1):
        lumamask = luma_mask_man(clip)
    else:
        lumamask = luma_mask_ping(clip)
    darken_luma_mask = core.std.Expr([lumamask], f"x {luma_mask_strength} *")
    final = core.std.MaskedMerge(deblock, clip, darken_luma_mask, planes=planes)

    return final

#TODO
def increase_dynamic(
    clip: vs.VideoNode,
    t: float = 0.7,
    s: float = 50,
    a: float = 300,
)-> vs.VideoNode:
    """
    Makes dark areas more dark, and white areas more white.
    
    You can use it if you lost dynamic but it's very subjective, using this filter will raise many eyebrows.
    """
    
    core=vs.core

    lumamask = luma_mask_man(clip, t=t, s=s, a=a)
    lumamask = lumamask.std.Invert()
    u = get_u(clip)
    v = get_v(clip)
    lumamask = core.std.ShufflePlanes(clips=[lumamask, u, v], planes=[0,0,0], colorfamily=vs.YUV)
    clip = core.std.Merge(clip, lumamask, weight=0.1)
    return clip

def deblock_old(
    clip: vs.VideoNode,
    thrvalue: int = 22,
    show_mask: bool = False
)-> vs.VideoNode:
    """
    Teoricamente allo stato attuale deblock_qed fa la stessa cosa o quasi
    """
        
    core=vs.core

    maxvalue = (1 << clip.format.bits_per_sample) - 1
    thr = (thrvalue / 255) * maxvalue

    y, u, v = clip.std.SplitPlanes()
    ymask = luma_mask_man(y, t=1, a=20, s=20)
    # umask = luma_mask_man(u, t=1, a=20, s=20)
    # vmask = luma_mask_man(v, t=1, a=20, s=20)
    umask = u
    vmask = v

    diag = "x x[1,1] - 2 / abs x x[1,-1] - 2 / abs + x x[-1,1] - 2 / abs x x[-1,-1] - 2 / abs + +"
    hor = "x x[1,0] - abs x x[-1,0] - abs + x x[2,0] - 2 / abs x x[-2,0] - 2 / abs + + x x[3,0] - 4 / abs x x[-3,0] - 4 / abs + +"
    ver = "x x[0,1] - abs x x[0,-1] - abs + x x[0,2] - 2 / abs x x[0,-2] - 2 / abs + + x x[0,3] - 4 / abs x x[0,-3] - 4 / abs + +"
    ymask1 = core.akarin.Expr([ymask], f"{hor} {diag} +").std.Binarize(thr)
    ymask2 = core.akarin.Expr([ymask], f"{ver} {diag} +").std.Binarize(thr)

    umask1 = core.akarin.Expr([umask], f"{hor} {diag} +").std.Binarize(thr)
    umask2 = core.akarin.Expr([umask], f"{ver} {diag} +").std.Binarize(thr)

    vmask1 = core.akarin.Expr([vmask], f"{hor} {diag} +").std.Binarize(thr)
    vmask2 = core.akarin.Expr([vmask], f"{ver} {diag} +").std.Binarize(thr)

    ymask = core.std.Expr([ymask1, ymask2], "x y max")
    umask = core.std.Expr([umask1, umask2], "x y max")
    vmask = core.std.Expr([vmask1, vmask2], "x y max")

    y1 = deblock_qed(y, quant=(30,32))
    y1 = core.dfttest.DFTTest(y1, sigma=25, tbsize=1, sosize=8)
    y1 = core.std.MaskedMerge(y, y1, ymask)
    u1 = deblock_qed(u, quant=(30,32))
    u1 = core.dfttest.DFTTest(u1, sigma=25, tbsize=1, sosize=8)
    u1 = core.std.MaskedMerge(u, u1, umask)
    v1 = deblock_qed(v, quant=(30,32))
    v1 = core.dfttest.DFTTest(v1, sigma=25, tbsize=1, sosize=8)
    v1 = core.std.MaskedMerge(v, v1, vmask)

    if show_mask is True:
        return core.std.ShufflePlanes(clips=[ymask, umask, vmask], planes=[0,0,0], colorfamily=vs.YUV)
    return core.std.ShufflePlanes(clips=[y1, u1, v1], planes=[0,0,0], colorfamily=vs.YUV)  

def deblock(
    clip: vs.VideoNode
)-> vs.VideoNode:
    
    core = vs.core

    # stringhe di resto già pronte per 4x4 e 8x8, esempio se voglio il valore del primo pixel nel blocco 8x8 sarà "{hor8x8} 0 ="
    hor8x8 = "X 8 %"
    hor4x4 = "X 4 %"
    ver8x8 = "Y 8 %"
    ver4x4 = "Y 4 %"

    # COSTRUZIONE DELLA MASK
    # due metodi per rilevare differenze di luma ai bordi, due clip di valori, una orizzontale l'altra verticale
    sumhor8_pxdiff = "x x[0,-1] - abs x[1,0] x[1,-1] - abs + x[2,0] x[2,-1] - abs x[3,0] x[3,-1] - abs + + x[4,0] x[4,-1] - abs x[5,0] x[5,-1] - abs + x[6,0] x[6,-1] - abs x[7,0] x[7,-1] - abs + + +"
    sumhor8_pxdiff2 = "x[0,1] x[0,-2] - abs x[1,1] x[1,-2] - abs + x[2,1] x[2,-2] - abs x[3,1] x[3,-2] - abs + + x[4,1] x[4,-2] - abs x[5,1] x[5,-2] - abs + x[6,1] x[6,-2] - abs x[7,1] x[7,-2] - abs + + +"
    sumhor8_diffpx = "x x[1,0] + x[2,0] + x[3,0] + x[4,0] + x[5,0] + x[6,0] + x[7,0] + x[0,-1] x[1,-1] + x[2,-1] + x[3,-1] + x[4,-1] + x[5,-1] + x[6,-1] + x[7,-1] + - abs"
    sumhor4_diffpx = "x x[1,0] + x[2,0] + x[3,0] + x[0,-1] x[1,-1] + x[2,-1] + x[3,-1] + - abs"
    sumver8_pxdiff = "x x[-1,0] - abs x[0,1] x[-1,1] - abs + x[0,2] x[-1,2] - abs x[0,3] x[-1,3] - abs + + x[0,4] x[-1,4] - abs x[0,5] x[-1,5] - abs + x[0,6] x[-1,6] - abs x[0,7] x[-1,7] - abs + + +"
    sumver8_pxdiff2 = "x[1,0] x[-2,0] - abs x[1,1] x[-2,1] - abs + x[1,2] x[-2,2] - abs x[1,3] x[-2,3] - abs + + x[1,4] x[-2,4] - abs x[1,5] x[-2,5] - abs + x[1,6] x[-2,6] - abs x[1,7] x[-2,7] - abs + + +"
    sumver8_diffpx = "x x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5] + x[0,6] + x[0,7] + x[-1,0] x[-1,1] + x[-1,2] + x[-1,3] + x[-1,4] + x[-1,5] + x[-1,6] + x[-1,7] + - abs"
    sumver4_diffpx = "x x[0,1] + x[0,2] + x[0,3] + x[-1,0] x[-1,1] + x[-1,2] + x[-1,3] + - abs"

    horblockvalue = core.akarin.Expr([clip], f"{hor8x8} 0 = {ver8x8} 0 = {sumhor8_pxdiff} {sumhor8_pxdiff2} 0.33 * + {sumhor8_diffpx} + 2 / 0 ? 0 ?")
    horblockvalue4 = core.akarin.Expr([clip], f"{hor4x4} 0 = {ver4x4} 0 = {sumhor4_diffpx} 0 ? 0 ?")
    verblockvalue = core.akarin.Expr([clip], f"{hor8x8} 0 = {ver8x8} 0 = {sumver8_pxdiff} {sumver8_pxdiff2} 0.33 * + {sumver8_diffpx} + 2 / 0 ? 0 ?")
    verblockvalue4 = core.akarin.Expr([clip], f"{hor4x4} 0 = {ver4x4} 0 = {sumver4_diffpx} 0 ? 0 ?")

    #allungamento dei valori verticalmente e orizzontalmente
    h1 = f"{hor8x8} 1 = x[-1,0] x ?"
    h2 = f"{hor8x8} 2 = x[-2,0] {h1} ?"
    h3 = f"{hor8x8} 3 = x[-3,0] {h2} ?"
    h4 = f"{hor8x8} 4 = x[-4,0] {h3} ?"
    h5 = f"{hor8x8} 5 = x[-5,0] {h4} ?"
    h6 = f"{hor8x8} 6 = x[-6,0] {h5} ?"
    h7 = f"{hor8x8} 7 = x[-7,0] {h6} ?"
    horblockmask = core.akarin.Expr([horblockvalue], f"{h7}")
    h1 = f"{hor4x4} 1 = x[-1,0] x ?"
    h2 = f"{hor4x4} 2 = x[-2,0] {h1} ?"
    h3 = f"{hor4x4} 3 = x[-3,0] {h2} ?"
    horblockmask4 = core.akarin.Expr([horblockvalue4], f"{h3}")
    v1 = f"{ver8x8} 1 = x[0,-1] x ?"
    v2 = f"{ver8x8} 2 = x[0,-2] {v1} ?"
    v3 = f"{ver8x8} 3 = x[0,-3] {v2} ?"
    v4 = f"{ver8x8} 4 = x[0,-4] {v3} ?"
    v5 = f"{ver8x8} 5 = x[0,-5] {v4} ?"
    v6 = f"{ver8x8} 6 = x[0,-6] {v5} ?"
    v7 = f"{ver8x8} 7 = x[0,-7] {v6} ?"
    verblockmask = core.akarin.Expr([verblockvalue], f"{v7}")
    v1 = f"{ver4x4} 1 = x[0,-1] x ?"
    v2 = f"{ver4x4} 2 = x[0,-2] {v1} ?"
    v3 = f"{ver4x4} 3 = x[0,-3] {v2} ?"
    verblockmask4 = core.akarin.Expr([verblockvalue4], f"{v3}")

    hormaskshift_top1 = horblockmask.resize.Point(clip.width, clip.height, src_top=1)
    hormaskshift_top2 = horblockmask.resize.Point(clip.width, clip.height, src_top=2)
    hormaskshift_bot1 = horblockmask.resize.Point(clip.width, clip.height, src_top=-1)
    vermaskshift_left1 = verblockmask.resize.Point(clip.width, clip.height, src_left=1)
    vermaskshift_left2 = verblockmask.resize.Point(clip.width, clip.height, src_left=2)
    vermaskshift_rig1 = verblockmask.resize.Point(clip.width, clip.height, src_left=-1)

    #fusione delle mask orizzontali e verticali, poi due shift per allargare la mask 4x4 da 1px a 2px
    blockmask4 = core.akarin.Expr([horblockmask4, verblockmask4], "x y max")
    blockmaskshift1_4 = blockmask4.resize.Point(blockmask4.width, blockmask4.height, src_left=1)
    blockmaskshift2_4 = blockmask4.resize.Point(blockmask4.width, blockmask4.height, src_top=1)
    blockmaskfull8 = core.akarin.Expr([horblockmask, verblockmask, hormaskshift_top1, hormaskshift_top2, hormaskshift_bot1, vermaskshift_left1, vermaskshift_left2, vermaskshift_rig1], "x y max z max a max b max c max d max e max")
    blockmaskfull4 = core.akarin.Expr([blockmask4, blockmaskshift1_4, blockmaskshift2_4], "x y max z max")

    #potenziamento della mask, potrebbe essere utile inserire un thr variabile prima di questo potenziamento
    blockmaskfull8 = core.std.Expr([blockmaskfull8], "x 0.4 pow 700 *")
    blockmaskfull4 = core.std.Expr([blockmaskfull4], "x 0.4 pow 350 *")

    # DEBLOCK 4X4

    # dalla funzione 12: D = ( 3(s[0] – s[–1]) – ( s[1] – s[–2] ) ) / 2
    deltahor_0 = f"{hor4x4} 0 = x x[-1,0] - 3 * x[1,0] x[-2,0] - - 2 / 0 ?"
    deltahor_n1 = f"{hor4x4} 3 = x x[1,0] - 3 * x[-1,0] x[2,0] - - 2 / {deltahor_0} ?"
    deltaver_0 = f"{ver4x4} 0 = x x[0,-1] - 3 * x[0,1] x[0,-2] - - 2 / 0 ?"
    deltaver_n1 = f"{ver4x4} 3 = x x[0,1] - 3 * x[0,-1] x[0,2] - - 2 / {deltaver_0} ?"
    
    # nella funzione 13 c'è una funzione di clipping molto importante, modificata in quanto non abbiamo qp
    thr = 2000
    # funzione 14: s'[i] = s[i] – D (N – i) / (2N + 1), per un blocco 4x4 è costante: s'[i] = s[i] – D 1/3
    cost4x4 = 1/3
    f14_hor = f"x {deltahor_n1} {cost4x4} * -"
    f14_ver = f"x {deltaver_n1} {cost4x4} * -"
    deltahor = core.akarin.Expr([clip], f"{deltahor_n1} abs {thr} < {f14_hor} {deltahor_n1} 0 > x {thr} - x {thr} + ? ?")
    deltahor = core.akarin.Expr([deltahor], f"{hor4x4} 3 = x {hor4x4} 0 = x 0 ? ?")
    deltaver = core.akarin.Expr([clip], f"{deltaver_n1} abs {thr} < {f14_ver} {deltaver_n1} 0 > x {thr} - x {thr} + ? ?")
    deltaver = core.akarin.Expr([deltaver], f"{ver4x4} 3 = x {ver4x4} 0 = x 0 ? ?")
    delta = core.akarin.Expr([deltahor, deltaver], "x y max")
    deblock4x4 = core.std.MaskedMerge(clipa=clip, clipb=delta, mask=blockmaskfull4)

    # DEBLOCK 8X8

    # dalla funzione 12: D = ( 3(s[0] – s[–1]) – ( s[1] – s[–2] ) ) / 2
    deltahor_0 = f"{hor8x8} 0 = x x[-1,0] - 3 * x[1,0] x[-2,0] - - 2 / 0 ?"
    deltahor_n1 = f"{hor8x8} 7 = x x[1,0] - 3 * x[-1,0] x[2,0] - - 2 / {deltahor_0} ?"
    deltahor_1 = f"{hor8x8} 1 = x x[-1,0] - 3 * x[1,0] x[-2,0] - - 2 / 0 ?"
    deltahor_n2 = f"{hor8x8} 6 = x x[1,0] - 3 * x[-1,0] x[2,0] - - 2 / {deltahor_1} ?"
    deltaver_0 = f"{ver8x8} 0 = x x[0,-1] - 3 * x[0,1] x[0,-2] - - 2 / 0 ?"
    deltaver_n1 = f"{ver8x8} 7 = x x[0,1] - 3 * x[0,-1] x[0,2] - - 2 / {deltaver_0} ?"
    deltaver_1 = f"{ver8x8} 1 = x x[0,-1] - 3 * x[0,1] x[0,-2] - - 2 / 0 ?"
    deltaver_n2 = f"{ver8x8} 6 = x x[0,1] - 3 * x[0,-1] x[0,2] - - 2 / {deltaver_1} ?"

    # nella funzione 13 c'è una funzione di clipping molto importante, modificata in quanto non abbiamo qp
    thr = 2000
    # funzione 14: s'[i] = s[i] – D (N – i) / (2N + 1)
    f14_hor0 = f"x {deltahor_n1} 0.4 * -"
    f14_hor1 = f"x {deltahor_n2} 0.2 * -"
    f14_ver0 = f"x {deltaver_n1} 0.4 * -"
    f14_ver1 = f"x {deltaver_n2} 0.2 * -"
    deltahor = core.akarin.Expr([clip], f"{deltahor_n1} abs {thr} < {f14_hor0} {deltahor_n1} 0 > x {thr} - x {thr} + ? ?")
    deltahor = core.akarin.Expr([deltahor], f"{deltahor_n2} abs {thr} < {f14_hor1} {deltahor_n2} 0 > x {thr} - x {thr} + ? ?")
    deltahor = core.akarin.Expr([deltahor], f"{hor8x8} 7 = x {hor8x8} 6 = x {hor8x8} 1 = x {hor8x8} 0 = x 0 ? ? ? ?")
    deltaver = core.akarin.Expr([clip], f"{deltaver_n1} abs {thr} < {f14_ver0} {deltaver_n1} 0 > x {thr} - x {thr} + ? ?")
    deltaver = core.akarin.Expr([deltaver], f"{deltaver_n2} abs {thr} < {f14_ver1} {deltaver_n2} 0 > x {thr} - x {thr} + ? ?")
    deltaver = core.akarin.Expr([deltaver], f"{ver8x8} 7 = x {ver8x8} 6 = x {ver8x8} 1 = x {ver8x8} 0 = x 0 ? ? ? ?")
    delta = core.akarin.Expr([deltahor, deltaver], "x y max")
    deblock8x8 = core.std.MaskedMerge(clipa=clip, clipb=delta, mask=blockmaskfull8)

    # unione cancellando il deblock 4x4 dove c'è il blocco 8x8
    deleted4 = core.akarin.Expr([deblock4x4], f"{hor8x8} 5 = x {hor8x8} 4 = x {hor8x8} 3 = x {hor8x8} 2 = x 0 ? ? ? ?")
    deleted4 = core.akarin.Expr([deleted4], f"{ver8x8} 5 = x {ver8x8} 4 = x {ver8x8} 3 = x {ver8x8} 2 = x 0 ? ? ? ?")
    deleted8 = core.akarin.Expr([deblock8x8], f"{hor8x8} 7 = x {hor8x8} 6 = x {hor8x8} 1 = x {hor8x8} 0 = x {ver8x8} 7 = x {ver8x8} 6 = x {ver8x8} 1 = x {ver8x8} 0 = x 0 ? ? ? ? ? ? ? ?")
    union = core.akarin.Expr([deleted8, deleted4], "x y max")

    return union

#TODO
def msaa2x(
    clip: vs.VideoNode,
    thrmask: int = 4000
) -> vs.VideoNode:
    from vsscale import ArtCNN
    from addfunc import adfunc, admask #wtf

    denoise = adfunc.adenoise.scan65mm(clip)
    emask = admask.edgemask(denoise, sigma=50, blur_radius=2)
    emask = emask.std.BinarizeMask(threshold=thrmask)
    emask = Morpho.erosion(emask)
    upsc = ArtCNN().C4F32().scale(denoise, clip.width*2, clip.height*2)
    aa = core.resize.Spline16(upsc, clip.width, clip.height)
    merged = core.std.MaskedMerge(clip, aa, emask)
    return merged