import re
import subprocess
import os
import json
import argparse
import sys
import random
import gzip
import shutil
import sys

'''
JSON STRUCTURE
{
    "info": {
        "total": <total number of frames>,
        "valid": <number of frames with non-empty or corrupted qp_grid>,
        "empty": <number of frames with empty qp_grid>
    },
    "frames": [
        {
            "frame": <frame number (starting from 0)>,
            "type": "<frame type, e.g., I/P/B/UNKNOWN>",
            "qp_grid": [
                [<qp>, <qp>, ...],   // Row of the QP grid (array of int)
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
        frame.pop('status', None) #Non serve lo status attualmente, ma in futuro potrebbe essere utile se ci sono frame corrotti di QP

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
    """
    Runs ffmpeg to generate the QP log file from the video.
    Returns True if the file was generated successfully and is not empty.
    """
    cmd = [
        ffmpeg_path,
        "-threads", "1",
        "-hide_banner",
        "-debug", "qp",
        "-i", input_video,
        "-f", "null",
        "-"
    ]
    print("Running ffmpeg\n The proccess is very slow and could take minutes or tens of minutes depending on the video length.")
    with open(log_path, "w", encoding="utf-8") as logfile:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=logfile)

    if os.path.isfile(log_path) and os.path.getsize(log_path) > 0:
        return True
    else:
        print(f"Error: {log_path} was not correctly generated.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse QP log or generate it from video using ffmpeg.")
    parser.add_argument('-v', '--video', type=str, help='Input video file')
    parser.add_argument('-l', '--log', type=str, help='Input QP log file')
    parser.add_argument('-o', '--output', type=str, default="qp_parsed_grid.json", help='Output JSON file')
    parser.add_argument('-c', '--compress', type=bool, default=True, help='Use gzip compression for the output JSON file (default: True). Set to False to disable (not suggested).')
    parser.add_argument('-f', '--ffmpeg_path', type=str, default=None, help='Path to ffmpeg executable (optional)')
    args = parser.parse_args()

    input_video = args.video
    input_log = args.log
    output_file = args.output
    compression = args.compress
    ffmpeg_path = check_ffmpeg_in_path(args.ffmpeg_path)

    if not input_video and not input_log:
        print("Usage: python parser264_qp.py -v <input_video> -l <input_log> -o <output_file> -c <compress> -f <ffmpeg_path>\nAt least one between -v or -l must be provided.")
        sys.exit(1)
    elif not input_video and input_log:
        frames = parse_qp_log(input_log)
        report, compressed_file = save_qp_report(frames, output_file, compression)
        print(f"QP grid generated as: {compressed_file}")
    else:
        if generateQPFromVideo(input_video, log_path="debug_frameqp.txt", ffmpeg_path=ffmpeg_path):
            input_log = "debug_frameqp.txt"
            frames = parse_qp_log(input_log)
            report, compressed_file = save_qp_report(frames, output_file, compression)
            print(f"QP grid generated as: {compressed_file}")
        else:
            print(f"Failed to generate QP log from video: {input_video}\n Ffmpeg error may have occurred.")
            sys.exit(1)

    # Stampa un frame casuale dal JSON per log di sicurezza
    if compression:
        with gzip.open(compressed_file, 'rt', encoding='utf-8') as f:
            report = json.load(f)
    else:
        with open(output_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
    frames = report.get('frames', [])
    if frames:
        random_frame = random.choice(frames)
        print("\n--- Random Frame from the json to ensure correct data parsing ---")
        print(f"Frame: {random_frame.get('frame')}")
        print(f"Type: {random_frame.get('type')}")
        print(f"QP Grid: {random_frame.get('qp_grid')}")
    else:
        print("No frame found. WTF?")


