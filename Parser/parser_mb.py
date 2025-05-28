import re

def parse_mb_log(log_path):
    frames = []
    current_frame = None
    parsing_started = False
    skip_next = False

    base_mb_chars = {'I', 'P', 'B', 'S', 'i', 'd', 'D', '>', '<', 'X'}
    extra_mb_chars = {'|', '-', '+', '|'}
    valid_mb_chars = {
        'I', 'P', 'B', 'S', 'i', 'd', 'D', '>', '<', 'X',
        ">|", ">-", ">+", "X-", "X+", "X|", "<|", "<-"
    }

    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            if not parsing_started:
                if "New frame, type: " in line:
                    parsing_started = True
                else:
                    continue

            # Salva il nal_unit_type e nal_ref_idc del frame corrente
            if "nal_unit_type:" in line:
                if current_frame is not None:
                    # Rimuove il contesto AVC
                    clean_nal_info = re.sub(r'\[.*?\]', '', line).strip()
                    current_frame['nal_info'] = clean_nal_info
                continue

            if "New frame, type: " in line:
                if current_frame:
                    frames.append(current_frame)
                
                match = re.search(r"New frame, type: (\w+)", line)
                current_frame = {
                    'type': match.group(1) if match else "UNKNOWN",
                    'grid': [],
                    'status': 'VALID',
                    'nal_info': ''
                }
                skip_next = True
                continue

            if skip_next:
                skip_next = False
                continue

            # rimozione inizio riga
            parts = line.split(']', 1)
            content = parts[-1].strip() if len(parts) > 1 else line.strip()

            if re.match(r'^\s*\d+', content):
                clean_content = re.sub(r'^\s*\d+\s*', '', content)
                tokens = clean_content.split()
                mb_types = []
                i = 0
                while i < len(tokens):
                    token = tokens[i]
                    if token in base_mb_chars and i + 1 < len(tokens) and tokens[i + 1] in extra_mb_chars:
                        combined = token + tokens[i + 1]
                        if combined in valid_mb_chars:
                            mb_types.append(combined)
                            i += 2
                            continue
                    if token in valid_mb_chars:
                        mb_types.append(token)
                    i += 1

                if mb_types:
                    if current_frame is not None:
                        current_frame['grid'].append(mb_types)
                elif current_frame is not None and current_frame['grid']:
                    current_frame['status'] = 'CORRUPTED'

        if current_frame:
            frames.append(current_frame)

    return frames

def save_grid_report(frames, output_file):
    stats = {
        'total': len(frames),
        'valid': len([f for f in frames if f['status'] == 'VALID']),
        'empty': len([f for f in frames if not f['grid']]),
        'corrupted': len([f for f in frames if f.get('status') == 'CORRUPTED'])
    }
# il ";" serve per i futuri parser per fargli skippare le righe
    with open(output_file, 'w') as f:
        f.write(";Macroblock Analysis Report\n")
        f.write(";"+"="*40 + "\n")
        f.write(f";Total Frames Processed: {stats['total']}\n")
        f.write(f";Valid Frames: {stats['valid']}\n")
        f.write(f";Corrupted Frames: {stats['corrupted']}\n")
        f.write(f";Empty Frames: {stats['empty']}\n\n")
        f.write(";"+"="*80 + "\n\n")

        for idx, frame in enumerate(frames):
            status = f" [{frame['status']}]" if frame['status'] != 'VALID' else ""
            nal_info = f" {frame['nal_info']}" if frame.get('nal_info') else ""
            f.write(f"Frame {idx+1} - Type: {frame['type']}{status}{nal_info}\n")
            
            if frame['grid']:
                for y, row in enumerate(frame['grid'], 1):
                    f.write(f"{' '.join(row)}\n")
            else:
                f.write("No macroblock data\n")
            
            f.write("\n;" + "="*80 + "\n\n")

if __name__ == "__main__":
    import sys
    input_log = sys.argv[1] if len(sys.argv) > 1 else "debug_framemb.txt"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "mb_parsed_grid.txt"
    
    frames = parse_mb_log(input_log)
    save_grid_report(frames, output_file)
    print(f"Report generato: {output_file}")