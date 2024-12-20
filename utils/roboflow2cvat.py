import cv2 as cv
import numpy as np
import os
import argparse

def find_highest_picture_number(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort()
    numbered_files = [f for f in files if f.split('.')[0].isdigit()]
    last_number = 0
    if numbered_files:
        last_number = int(numbered_files[-1].split('.')[0])

    return last_number

def roboflow2cvat(input_dir, output_dir):
    dirs = ['train', 'valid', 'test']
    file_number = find_highest_picture_number(os.path.join(output_dir, 'annotated data', 'all_in_one')) + 1
    for d in dirs:
        files = os.listdir(os.path.join(input_dir, d))
        for file in files:
            if file.endswith('mask.png'):
                img = cv.imread(os.path.join(input_dir, d, file), cv.IMREAD_GRAYSCALE)
                # Change so there is a new rgb image that that is fully red in places that img has value 2, fully blue in places that img has value 1
                rgb_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                rgb_img[img == 2, 2] = 255
                rgb_img[img == 1, 0] = 255

                # Get file name before _mask.png
                base_img_name = file.split('_mask.png')[0] + '.jpg'
                base_img = cv.imread(os.path.join(input_dir, d, base_img_name))
                if base_img is None:
                    print(f"Nie znaleziono pliku {base_img_name}")
                    continue

                file_name = f"{file_number:03d}.png"
                file_number += 1

                # Save the images
                base_img_output_path = os.path.join(output_dir, 'data', file_name)
                os.makedirs(os.path.dirname(base_img_output_path), exist_ok=True)
                cv.imwrite(base_img_output_path, base_img)

                output_path = os.path.join(output_dir, 'annotated data', 'all_in_one', file_name)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv.imwrite(output_path, rgb_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Konwersja danych z Roboflow do formatu CVAT")
    parser.add_argument("data", help="Folder z danymi z Roboflow")
    parser.add_argument("output", help="Folder wyjściowy")
    args = parser.parse_args()

    data = args.data
    output = args.output

    roboflow2cvat(data, output)

