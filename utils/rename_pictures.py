import os
import argparse

def rename_pictures(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort()
    numbered_files = [f for f in files if f.split('.')[0].isdigit()]

    for i, file in enumerate(numbered_files):
        file_name = int(file.split('.')[0])
        extension = file.split('.')[1]
        new_name = f"{file_name:04d}.{extension}"
        print(f"Zmieniam nazwę pliku {file} na {new_name}")
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zmiana nazw plików w folderze")
    parser.add_argument("folder", help="Folder z plikami")
    args = parser.parse_args()

    folder = args.folder
    rename_pictures(folder)