"""extract vital presets from .zip .rar .7z .vitalbank archives"""
import os
import shutil
import tempfile
from pathlib import Path

import patoolib
import rootutils
from tqdm import tqdm

root = rootutils.find_root(search_from=os.path.dirname(os.path.abspath(__file__)), indicator=".project-root")


def handle_folder(folder_path, save_path):
    folder_path = Path(folder_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Gather all files first to show a progress bar
    files_to_process = list(folder_path.rglob('*'))
    print(f"Found {len(files_to_process)} files in {folder_path}")

    for file_path in tqdm(files_to_process, desc="Processing files"):
        if not file_path.is_file():
            continue
        process_file(file_path, save_path)


def process_file(file_path: Path, save_path: Path):
    if file_path.name.startswith("._") or file_path.name == ".DS_Store":
        return

    suffix = file_path.suffix.lower()

    # Direct .vital files
    if suffix == '.vital':
        try:
            dest = save_path / file_path.name
            if not dest.exists():
                shutil.copy2(file_path, dest)
        except Exception:
            pass

    # Archives
    elif suffix in ['.zip', '.rar', '.7z', '.tar', '.gz', '.tgz', '.vitalbank']:
        extract_archive(file_path, save_path)


def extract_archive(archive_path: Path, save_path: Path):
    """Extract archive to temp dir and recursively find .vital files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Handle .vitalbank by treating it as a zip
            target_path = archive_path
            if archive_path.suffix.lower() == '.vitalbank':
                temp_zip = Path(temp_dir) / (archive_path.stem + ".zip")
                shutil.copy2(archive_path, temp_zip)
                target_path = temp_zip

            patoolib.extract_archive(str(target_path), outdir=temp_dir, verbosity=-1)

            temp_path = Path(temp_dir)
            for extracted_file in temp_path.rglob('*'):
                if extracted_file.is_file():
                    process_file(extracted_file, save_path)

        except Exception as e:
            print(f"  Failed to extract {archive_path.name}: {e}")


if __name__ == '__main__':
    vital_sources_folder = root / 'data' / 'presets' / 'vital_raw'
    vital_out_folder = root / 'data' / 'presets' / 'vital'

    print(f"Source: {vital_sources_folder}")
    print(f"Dest: {vital_out_folder}")

    if vital_sources_folder.exists():
        handle_folder(vital_sources_folder, vital_out_folder)
    else:
        print("Source folder not found!")
