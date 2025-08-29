import shutil
import os
from typing import List, Dict, Tuple, Optional
import subprocess



def check_ffmpeg_in_path(ffmpeg_path: Optional[str]) -> str:
    if ffmpeg_path is not None:
        if not os.path.isfile(ffmpeg_path):
            raise FileNotFoundError(f"the ffmpeg's path provided is not valid: {ffmpeg_path}")
        return ffmpeg_path
    else:
        ffmpeg_in_path = shutil.which("ffmpeg")
        if ffmpeg_in_path is None:
            raise FileNotFoundError("ffmpeg not found in system PATH. Please install ffmpeg or provide a path with -f.")
        return ffmpeg_in_path
    
def get_framerate(video_path: str,) -> Tuple[float, int, int]:
    cmd = [
        "ffprobe", "-v", "0",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    rate = result.stdout.strip()
    if not rate:
        raise RuntimeError("Unable to detect framerate with ffprobe")
    nums = rate.split('/')
    if len(nums) == 2:
        return float(nums[0]) / float(nums[1]), int(nums[0]), int(nums[1])
    else:
        return float(nums[0]), 0, 0
    
def parse_fps(fps_str: str) -> float:
    fps_str = fps_str.strip()
    if "/" in fps_str:
        num, den = fps_str.split("/")
        try:
            return float(num) / float(den)
        except ValueError:
            raise ValueError(f"Invalid FPS fraction: {fps_str}")
    else:
        try:
            return float(fps_str)
        except ValueError:
            raise ValueError(f"Invalid FPS value: {fps_str}")