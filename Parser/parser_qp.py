import re

def parse_qp_log(log_path):
    frames = []
    current_frame = None
    parsing_started = False
    skip_next = False

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
                
                match = re.search(r"New frame, type: (\w+)", line)
                frame_type = match.group(1) if match else "UNKNOWN"
                current_frame = {
                    'type': frame_type,
                    'qp_grid': [],
                    'status': 'VALID'
                }
                skip_next = True
                continue

            if skip_next:
                skip_next = False
                continue

            # Filtraggio delle righe non rilevanti
            if any(s in line for s in ['nal_unit_type', 'detected', 'lavf', 'Stream', 'cur_dts']):
                continue

            # Estrazione dati QP
            parts = line.split(']', 1)
            content = parts[-1].strip() if len(parts) > 1 else line.strip()

            if re.match(r'^\s*\d+\s+\d{2,}', content):
                # Estrai solo la parte numerica
                numbers = re.sub(r'^\s*\d+\s*', '', content)  # Rimuove l'indice iniziale
                numbers = re.sub(r'\D', '', numbers)  # Rimuove caratteri non numerici
                
                # Dividi in valori QP a 2 cifre (assicura che qp non abbia più di 2 cifre)
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

    return frames
#conserva e stampa i risultati sul file
def save_qp_report(frames, output_file):
    stats = {
        'total': len(frames),
        'valid': len([f for f in frames if f['status'] == 'VALID']),
        'empty': len([f for f in frames if not f['qp_grid']]),
        'corrupted': len([f for f in frames if f.get('status') == 'CORRUPTED'])
    }
# il ";" serve per i futuri parser per fargli skippare le righe
    with open(output_file, 'w') as f:
        f.write(";QP Value Analysis Report\n")
        f.write(";"+"="*40 + "\n")
        f.write(f";Total Frames Processed: {stats['total']}\n")
        f.write(f";Valid Frames: {stats['valid']}\n")
        f.write(f";Corrupted Frames: {stats['corrupted']}\n")
        f.write(f";Empty Frames: {stats['empty']}\n\n")
        f.write(";"+"="*80 + "\n\n")

        for idx, frame in enumerate(frames):
            status = f" [{frame['status']}]" if frame['status'] != 'VALID' else ""
            f.write(f"Frame {idx+1} - Type: {frame['type']}{status}\n")
            
            if frame['qp_grid']:
                for y, row in enumerate(frame['qp_grid'], 1):
                    formatted_row = ' '.join(f"{qp:02}" for qp in row)
                    f.write(f"{formatted_row}\n")
            else:
                f.write("No QP data\n")
            
            f.write("\n;" + "="*80 + "\n\n")

if __name__ == "__main__":
    import sys
    input_log = sys.argv[1] if len(sys.argv) > 1 else "debug_frameqp.txt"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "qp_parsed_grid.txt"
    
    frames = parse_qp_log(input_log)
    save_qp_report(frames, output_file)
    print(f"Report generato: {output_file}")