import os
import argparse

def get_last_number(data_dir):
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    files.sort()
    numbered_files = [f for f in files if f.split('.')[0].isdigit()]
    last_number = 0
    if numbered_files:
        last_number = int(numbered_files[-1].split('.')[0])

    return last_number

def main():
    parser = argparse.ArgumentParser(description="Numerowanie plików")
    parser.add_argument("folder", help="Folder z danymi z QGIS plikami do numerowania")
    args = parser.parse_args()

    folder_path = args.folder

    data_dir = os.path.join("data", "data")

    if not os.path.isdir(folder_path):
        print(f"Podana ścieżka '{folder_path}' nie jest katalogiem.")
        return

    new_number = get_last_number(data_dir) + 1

    folders = os.listdir(folder_path)
    if not folders:
        print("Folder jest pusty")
    else:
        for folder in folders:
            if folder != "done":
                print(f"Numerowanie plików w folderze: {folder}")
                cur_folder_path = os.path.join(folder_path, folder)
                files = [f for f in os.listdir(cur_folder_path) if os.path.isfile(os.path.join(cur_folder_path, f))]
                files.sort()

                unnumbered_files = [f for f in files if not f.split('.')[0].isdigit()]

                for file_name in unnumbered_files:
                    file_extension = os.path.splitext(file_name)[1]
                    new_name = f"{new_number:03d}{file_extension}"
                    old_path = os.path.join(cur_folder_path, file_name)
                    new_path = os.path.join(data_dir, new_name)
                    new_number += 1
                    os.rename(old_path, new_path)

        print("Numerowanie zakończone!")

    # files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # files.sort()
    #
    # unnumbered_files = [f for f in files if not f.split('.')[0].isdigit()]
    #
    # start_number = get_last_number()
    #
    # for index, file_name in enumerate(unnumbered_files):
    #     file_extension = os.path.splitext(file_name)[1]
    #     new_name = f"{start_number + index:03d}{file_extension}"
    #     old_path = os.path.join(folder_path, file_name)
    #     new_path = os.path.join(folder_path, new_name)
    #     os.rename(old_path, new_path)
    #
    # print("Numerowanie zakończone!")

if __name__ == "__main__":
    main()