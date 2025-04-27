import re

def parse_mb_log(log_path):
    frames = []
    current_frame = None
    parsing_started = False
    skip_next = False
    valid_mb_chars = {'I', 'P', 'B', 'S', 'i', 'd', 'D', '>', '<', 'X'}

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
                
                current_frame = {
                    'type': re.search(r"New frame, type: (\w+)", line).group(1),
                    'grid': [],
                    'status': 'VALID'
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
                # Pulizia avanzata
                clean_content = re.sub(r'^\s*\d+\s*', '', content)  # Rimuove numero iniziale
                clean_content = re.sub(r'[^A-Za-z> <IX]', '', clean_content)  # Rimuove caratteri non validi
                mb_types = [c for c in clean_content if c in valid_mb_chars]
                
                if mb_types:
                    current_frame['grid'].append(mb_types)
                elif current_frame['grid']:
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
        f.write(";="*40 + "\n")
        f.write(f";Total Frames Processed: {stats['total']}\n")
        f.write(f";Valid Frames: {stats['valid']}\n")
        f.write(f";Corrupted Frames: {stats['corrupted']}\n")
        f.write(f";Empty Frames: {stats['empty']}\n\n")
        f.write(";"+"="*80 + "\n\n")

        for idx, frame in enumerate(frames):
            status = f" [{frame['status']}]" if frame['status'] != 'VALID' else ""
            f.write(f"Frame {idx+1} - Type: {frame['type']}{status}\n")
            
            if frame['grid']:
                for y, row in enumerate(frame['grid'], 1):
                    f.write(f"{' '.join(row)}\n")
            else:
                f.write("No macroblock data\n")
            
            f.write("\n;" + "="*80 + "\n\n")

if __name__ == "__main__":
    import sys
    input_log = sys.argv[1] if len(sys.argv) > 1 else "debug_frame.txt"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "mb_parsed_grid.txt"
    
    frames = parse_mb_log(input_log)
    save_grid_report(frames, output_file)
    print(f"Report generato: {output_file}")