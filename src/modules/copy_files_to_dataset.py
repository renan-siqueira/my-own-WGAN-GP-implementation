import os
import shutil

def copy_files_with_interval(source_folder, target_folder, interval):
    """
    Copies one file for every 'interval' files from source_folder to target_folder.
    """
    
    files = sorted([f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))])

    selected_files = files[::interval]

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file in selected_files:
        source_path = os.path.join(source_folder, file)
        target_path = os.path.join(target_folder, file)
        shutil.copy2(source_path, target_path)
        print(f'Copied: {file}')


def main(path_original_files, path_dataset, interval):
    copy_files_with_interval(path_original_files, path_dataset, interval)
