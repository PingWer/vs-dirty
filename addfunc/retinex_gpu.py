from __future__ import annotations
import vapoursynth as vs
from vstools import get_y, depth

core = vs.core

def retinex_torch(
    clip: vs.VideoNode,
    sigma: list[float] = [25, 80, 250],
    lower_thr: float = 0.001,
    upper_thr: float = 0.001,
    fast: bool = True
) -> vs.VideoNode:
    """
    Bottleneck causato dal trasferimento dei frame tra CPU e GPU, paradossalmente più lento della CPU,
    ma utile in contesti in cui la GPU è potenzialmente vuota.
    """
    try:
        import torch
        import torch.nn.functional as F
        import torchvision.transforms.functional as TF
        import numpy as np
    except ImportError:
        raise ImportError("retinex_torch requires 'torch', 'torchvision' and 'numpy'.")

    if not torch.cuda.is_available():
        raise RuntimeError("retinex_torch requires a CUDA GPU.")

    device = torch.device('cuda')

    # Ensure Luma Float
    luma = get_y(clip)
    if luma.format.sample_type != vs.FLOAT:
        luma = depth(luma, 32)
        
    sigmas = sorted(sigma)
    sigmas_run = sigmas[:-1] if fast else sigmas

    def _process_frame(n, f):
        # Copy VS Frame (CPU) -> Torch Tensor (GPU)
        frame = f[0] if isinstance(f, list) else f
        
        try:
             arr_cpu = np.array(frame.get_read_array(0), copy=False)
        except (AttributeError, Exception):
             arr_cpu = np.asarray(frame[0])
        
        t_img = torch.from_numpy(arr_cpu).to(device, non_blocking=True)
        
        t_min = t_img.min()
        t_max = t_img.max()
        t_range = t_max - t_min
        
        if t_range < 1e-6:
            return frame.copy()
            
        t_norm = (t_img - t_min) / t_range
        
        # MSR Logic
        t_input = t_norm.unsqueeze(0).unsqueeze(0)
        h, w = t_norm.shape
        
        t_acc = torch.zeros_like(t_norm)

        for s in sigmas_run:
            if fast and s > 6:
                ratio = max(1.0, s / 3.0)
                th, tw = max(1, int(h / ratio)), max(1, int(w / ratio))
                
                t_down = F.interpolate(t_input, size=(th, tw), mode='bilinear', align_corners=False)
                
                k_down = int(2 * int(3.0 * 3.0) + 1) | 1
                t_blurred_down = TF.gaussian_blur(t_down, kernel_size=[k_down, k_down], sigma=[3.0, 3.0])

                # Upscale
                t_blurred = F.interpolate(t_blurred_down, size=(h, w), mode='bilinear', align_corners=False)
                t_blur_res = t_blurred.squeeze(0).squeeze(0)
            else:
                 # Full res blur
                 k_full = min(int(2 * int(3.0 * s) + 1) | 1, 301)
                 t_blurred = TF.gaussian_blur(t_input, kernel_size=[k_full, k_full], sigma=[float(s), float(s)])
                 t_blur_res = t_blurred.squeeze(0).squeeze(0)

            t_acc += (t_norm / (t_blur_res + 1e-6)) + 1.0

            
        if fast:
            t_acc += t_norm + 1.0
            
        # Log Mean
        t_msr = torch.log(t_acc) / len(sigmas)
        
        # Final Balance (same as vszip)
        if lower_thr > 0 or upper_thr > 0:
             t_min_out = torch.quantile(t_msr, lower_thr)
             t_max_out = torch.quantile(t_msr, 1.0 - upper_thr)
        else:
             t_min_out = t_msr.min()
             t_max_out = t_msr.max()
             
        t_msr = torch.clamp(t_msr, min=t_min_out, max=t_max_out)
             
        t_range_out = t_max_out - t_min_out
        
        if t_range_out < 1e-6:
             return frame.copy()
        
        t_out = (t_msr - t_min_out) / t_range_out
             
        # 5. Download GPU -> CPU -> Frame
        arr_out = t_out.cpu().numpy()
        
        f_out = frame.copy()
        
        try:
            np.copyto(np.array(f_out.get_write_array(0), copy=False), arr_out)
        except (AttributeError, Exception):
            np.copyto(np.asarray(f_out[0]), arr_out)
            
        return f_out

    return luma.std.ModifyFrame(clips=[luma], selector=_process_frame)
