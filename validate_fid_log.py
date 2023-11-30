import os
import re
from heapq import nlargest


def find_top_epochs_in_log(log_file_path):
    """Analisa um arquivo de log para encontrar as 5 melhores épocas com base nos menores valores de FID."""
    fid_epochs = []

    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(r'Epoch (\d+).*FID: ([\d.]+)', line)
            if match:
                epoch, fid = int(match.group(1)), match.group(2)
                if fid.lower() != 'inf':
                    fid = float(fid)
                    fid_epochs.append((fid, epoch))

    top_epochs = sorted(fid_epochs)[:5]
    return top_epochs


def search_logs_in_directory(directory):
    log_info = {}

    for root, dirs, files in os.walk(directory):
        if '_old' in dirs:
            dirs.remove('_old')

        for file in files:
            if file.endswith('.log'):
                parent_folder = os.path.basename(os.path.dirname(root))
                top_epochs = find_top_epochs_in_log(os.path.join(root, file))
                log_info[parent_folder] = top_epochs

    return log_info


def main():
    directory = 'src/data'
    log_info = search_logs_in_directory(directory)

    for folder_name, epochs in log_info.items():
        print(f"- {folder_name}")
        for fid, epoch in epochs:
            print(f"    - epoch {epoch} - FID {fid:.2f}")


if __name__ == "__main__":
    main()
