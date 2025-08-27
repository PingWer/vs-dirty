import re
import subprocess
import os
import json
import argparse
import sys
import random
import gzip
import shutil
import numpy as np

'''
JSON STRUCTURE
{
    "info": {
        "total": <total number of frames>,
        "valid": <number of frames with non-empty qp_grid>,
        "empty": <number of frames with empty qp_grid>
    },
    "frames": [
        {
            "frame": <frame number (starting from 0)>,
            "type": "<frame type, e.g., I/P/B/UNKNOWN>",
            "qp_grid": [
                [<qp>, <qp>, ...],   // Row of the QP grid
                ...
            ]
        },
        ...
    ]
}
'''

def check_ffmpeg_in_path(ffmpeg_path):
    if ffmpeg_path is not None:
        if not os.path.isfile(ffmpeg_path):
            print(f"Error: the ffmpeg's path provided is not valid: {ffmpeg_path}")
            sys.exit(1)
        return ffmpeg_path
    else:
        ffmpeg_in_path = shutil.which("ffmpeg")
        if ffmpeg_in_path is None:
            print("Error: ffmpeg not found in system PATH. Please install ffmpeg and make sure it is in the PATH or specify the path with -f.")
            sys.exit(1)
        return ffmpeg_in_path

def get_framerate(video_path):
    cmd = [
        "ffprobe", "-v", "0",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    rate = result.stdout.strip()
    nums = rate.split('/')
    if len(nums) == 2:
        return float(nums[0]) / float(nums[1])
    else:
        return float(nums[0])

def parse_fps(fps_str):
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

def parse_qp_log(log_path):
    print("Parsing QP log file")
    frames = []
    current_frame = None
    parsing_started = False
    skip_next = False
    frame_idx = 0

    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            if not parsing_started:
                if "New frame, type: " in line:
                    parsing_started = True
                else:
                    continue

            if "New frame, type: " in line:
                if current_frame:
                    frames.append(current_frame)
                    frame_idx += 1

                match = re.search(r"New frame, type: (\w+)", line)
                frame_type = match.group(1) if match else "UNKNOWN"
                current_frame = {
                    'frame': frame_idx,
                    'type': frame_type,
                    'qp_grid': [],
                    'status': 'VALID'
                }
                skip_next = True
                continue

            if skip_next:
                skip_next = False
                continue

            if any(s in line for s in ['nal_unit_type', 'detected', 'lavf', 'Stream', 'cur_dts']):
                continue

            parts = line.split(']', 1)
            content = parts[-1].strip() if len(parts) > 1 else line.strip()

            if re.match(r'^\s*\d+\s+\d{2,}', content):
                numbers = re.sub(r'^\s*\d+\s*', '', content)
                numbers = re.sub(r'\D', '', numbers)
                qp_values = []
                for i in range(0, len(numbers), 2):
                    qp = numbers[i:i+2]
                    if len(qp) == 2 and qp.isdigit():
                        qp_values.append(int(qp))

                if qp_values and current_frame is not None:
                    current_frame['qp_grid'].append(qp_values)
                elif current_frame is not None and current_frame['qp_grid']:
                    current_frame['status'] = 'CORRUPTED'

        if current_frame:
            frames.append(current_frame)
    for frame in frames:
        frame.pop('status', None)

    return frames

def save_qp_report(frames, output_file, compression):
    print("Saving and compressing (if not disable) JSON file")
    stats = {
        'total': len(frames),
        'valid': len([f for f in frames if f.get('qp_grid')]),
        'empty': len([f for f in frames if not f.get('qp_grid')]),
    }
    report = {
        "info": stats,
        "frames": frames
    }
    if compression:
        gz_file = output_file + ".gz" if not output_file.endswith(".gz") else output_file
        with gzip.open(gz_file, 'wt', encoding='utf-8') as f:
            json.dump(report, f, separators=(',', ':'), ensure_ascii=False)
        return report, gz_file
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, separators=(',', ':'), ensure_ascii=False)
        return report, output_file

def generateQPFromVideo(input_video, log_path="debug_frameqp.txt", ffmpeg_path="ffmpeg"):
    cmd = [
        ffmpeg_path,
        "-threads", "1",
        "-hide_banner",
        "-debug", "qp",
        "-i", input_video,
        "-f", "null",
        "-"
    ]
    print("Running ffmpeg\n The process may take minutes depending on video length.")
    with open(log_path, "w", encoding="utf-8") as logfile:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=logfile)

    if os.path.isfile(log_path) and os.path.getsize(log_path) > 0:
        return True
    else:
        print(f"Error: {log_path} was not correctly generated.")
        return False

def write_qp_to_yuv(frames, output_yuv_path):
    if not frames:
        print("No frames to write!")
        return
    # Dimensione basata sulla grandezza della grid (sarà sempre esatta)
    height = len(frames[0]['qp_grid'])
    width = len(frames[0]['qp_grid'][0])
    frame_size = width * height
    num_frames = len(frames)

    print(f"Writing {num_frames} frames to YUV400 (8bit, 6bit effective) -> {output_yuv_path}")
    with open(output_yuv_path, "wb") as f:
        f.write(b'\x00' * (frame_size * num_frames))

    with open(output_yuv_path, "r+b") as f:
        for frame in frames:
            qp_grid = np.array(frame['qp_grid'], dtype=np.uint8)
            qp_grid = np.clip(qp_grid, 0, 63)  # 6-bit effective
            f.seek(frame['frame'] * frame_size)
            f.write(qp_grid.tobytes())

def write_qp_to_mkv(frames, output_mkv_path, fps, ffmpeg_path="ffmpeg"):
    if not frames:
        raise ValueError("No frames to write!")
    
    height = len(frames[0]['qp_grid'])
    width = len(frames[0]['qp_grid'][0])
    
    print(f"Writing {len(frames)} frames to temporary raw YUV file...")
    
    # File temporaneo YUV da cui generare MKV (evito la stream diretta per questioni di buffering e possibile rottura di cazzo da parte dei newline del terminale)
    tmp_yuv = "tmp_qp.yuv"
    with open(tmp_yuv, "wb") as f:
        for frame in frames:
            qp_grid = np.array(frame['qp_grid'], dtype=np.uint8)
            qp_grid = np.clip(qp_grid, 0, 63)
            f.write(qp_grid.tobytes())
    
    print(f"Encoding raw YUV into MKV with FFV1 -> {output_mkv_path}")
    cmd = [
        ffmpeg_path,
        "-y",
        "-f", "rawvideo",
        "-pixel_format", "gray",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", tmp_yuv,
        "-c:v", "ffv1",
        output_mkv_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("FFmpeg error:\n", result.stderr)
    else:
        print(f"MKV generated successfully: {output_mkv_path}")
    
    os.remove(tmp_yuv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse QP log, generate JSON or directly YUV file.")
    parser.add_argument('-v', '--video', type=str, help='Input video file')
    parser.add_argument('-l', '--log', type=str, help='Input QP log file')
    parser.add_argument('-o', '--output', type=str, default="qp_parsed_grid.json", help='Output JSON/YUV file')
    parser.add_argument('-c', '--compress', type=bool, default=True, help='Use gzip for JSON output')
    parser.add_argument('-f', '--ffmpeg_path', type=str, default=None, help='Path to ffmpeg executable (optional)')
    parser.add_argument('-a', '--auto', type=bool, default=True, help='If True, generates only mkv file from QP grids (default True)')
    parser.add_argument('--fps', type=str, default=None, help='Frame rate to use for MKV (required if only log is provided)')
    args = parser.parse_args()

    input_video = args.video
    input_log = args.log
    output_file = args.output
    output_yuv= "output_qp"
    compression = args.compress
    ffmpeg_path = check_ffmpeg_in_path(args.ffmpeg_path)

    frames = None

    if input_log:
        if args.fps:
            fps = parse_fps(args.fps)
            print(f"Using provided FPS: {fps}")
        else:
            print("Error: FPS must be provided if only log is given.")
            sys.exit(1)
        frames = parse_qp_log(input_log)
    elif input_video:
        fps = get_framerate(args.video)
        if generateQPFromVideo(input_video, log_path="debug_frameqp.txt", ffmpeg_path=ffmpeg_path):
            input_log = "debug_frameqp.txt"
            frames = parse_qp_log(input_log)
        else:
            print(f"Failed to generate QP log from video: {input_video}")
            sys.exit(1)
    else:
        print("Provide at least a log (-l) or a video (-v)")
        sys.exit(1)

    if frames is None or not frames:
        print("No frames parsed, aborting.")
        sys.exit(1)

    if args.auto:
        mkv_path = output_yuv + ".mkv"
        write_qp_to_mkv(frames, mkv_path, fps, ffmpeg_path)
        sys.exit(0)

    report, compressed_file = save_qp_report(frames, output_file, compression)
    print(f"QP grid generated as: {compressed_file}")

    random_frame = random.choice(frames)
    print("\n--- Random Frame from the parsed data ---")
    print(f"Frame: {random_frame.get('frame')}")
    print(f"Type: {random_frame.get('type')}")
    print(f"QP Grid: {random_frame.get('qp_grid')}")
