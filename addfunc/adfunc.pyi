import vapoursynth as vs #non serve ma mi secca vedere robe rosse

class adenoise:
    @staticmethod
    def __init__(): ...
    
    @staticmethod
    def scan65mm(clip:vs.VideoNode, 
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
    )->vs.VideoNode: ...

    @staticmethod
    def scan35mm(clip:vs.VideoNode, 
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
    )->vs.VideoNode: ...

    @staticmethod
    def scan16mm(clip:vs.VideoNode, 
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
    )->vs.VideoNode: ...

    @staticmethod
    def scan8mm(clip:vs.VideoNode, 
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
    )->vs.VideoNode: ...

    @staticmethod
    def digital(clip:vs.VideoNode, 
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
    )->vs.VideoNode: ...