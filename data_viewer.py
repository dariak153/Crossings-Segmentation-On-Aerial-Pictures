import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from PIL import Image
import glob


class SegmentationDataViewer:
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.file_pairs = self.get_file_pairs()

        if not self.file_pairs:
            print("Nie znaleziono sparowanych obrazów i masek.")
            exit()

        self.current_index = 0
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(bottom=0.25)

        self.create_buttons()
        self.create_textbox()
        self.display_current()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def get_file_pairs(self):
        images = sorted([f for f in os.listdir(self.images_dir)
                         if f.lower().endswith(('.png'))])
        masks = sorted([f for f in os.listdir(self.masks_dir)
                        if f.lower().endswith(('.png'))])

        image_set = set(images)
        mask_set = set(masks)
        common_files = image_set & mask_set

        file_pairs = sorted(list(common_files))
        print(f"Znaleziono {len(file_pairs)} sparowanych obrazów i masek.")
        return file_pairs

    def display_current(self):
        img_name = self.file_pairs[self.current_index]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('RGB')
        except Exception as e:
            print(f"Błąd podczas ładowania plików {img_name}: {e}")
            return

        self.axs[0].clear()
        self.axs[0].imshow(image)
        self.axs[0].set_title(f"Obraz: {img_name}")
        self.axs[0].axis('off')

        self.axs[1].clear()
        self.axs[1].imshow(mask)
        self.axs[1].set_title(f"Maska: {img_name}")
        self.axs[1].axis('off')

        self.fig.canvas.draw_idle()

    def create_buttons(self):
        axprev = plt.axes([0.1, 0.1, 0.1, 0.05])
        axnext = plt.axes([0.8, 0.1, 0.1, 0.05])
        axdelete = plt.axes([0.45, 0.1, 0.1, 0.05])

        self.bprev = Button(axprev, 'Poprzedni')
        self.bnext = Button(axnext, 'Następny')
        self.bdelete = Button(axdelete, 'Usuń')

        self.bprev.on_clicked(self.prev)
        self.bnext.on_clicked(self.next)
        self.bdelete.on_clicked(self.delete_current)

    def create_textbox(self):
        axbox = plt.axes([0.4, 0.175, 0.2, 0.05])
        self.text_box = TextBox(axbox, 'Przejdź do indeksu', initial="")
        self.text_box.on_submit(self.go_to_sample)

    def go_to_sample(self, text):
        try:
            index = int(text) - 1
            if 0 <= index < len(self.file_pairs):
                self.current_index = index
                self.display_current()
            else:
                print(f"Indeks poza zakresem. Wprowadź wartość między 1 a {len(self.file_pairs)}.")
        except ValueError:
            print("Wprowadź liczbę")

    def next(self, event=None):
        if self.current_index < len(self.file_pairs) - 1:
            self.current_index += 1
            self.display_current()

    def prev(self, event=None):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current()

    def delete_current(self, event=None):
        if not self.file_pairs:
            return
        img_name = self.file_pairs[self.current_index]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        try:
            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(mask_path):
                os.remove(mask_path)

            del self.file_pairs[self.current_index]

            if self.current_index >= len(self.file_pairs):
                self.current_index = len(self.file_pairs) - 1

            if self.file_pairs:
                self.display_current()
            else:
                plt.close(self.fig)
        except Exception as e:
            print(f"Błąd podczas usuwania plików {img_name}: {e}")

    def on_key_press(self, event):
        if event.key == 'd':
            self.delete_current()
        elif event.key == 'n':
            self.next()
        elif event.key == 'p':
            self.prev()
        elif event.key == 'g':
            try:
                index = int(input("Wprowadź indeks do przejścia: ")) - 1
                if 0 <= index < len(self.file_pairs):
                    self.current_index = index
                    self.display_current()
                else:
                    print(f"Indeks poza zakresem. Wprowadź wartość między 1 a {len(self.file_pairs)}.")
            except ValueError:
                print("Wprowadź liczbę.")


def main():
    base_dir = 'data'
    images_dir = os.path.join(base_dir, 'data')
    masks_dir = os.path.join(base_dir, 'annotated data', 'all_in_one')

    if not os.path.exists(images_dir):
        print(f"Folder z obrazami nie istnieje: {images_dir}")
        exit()
    if not os.path.exists(masks_dir):
        print(f"Folder z maskami nie istnieje: {masks_dir}")
        exit()

    viewer = SegmentationDataViewer(images_dir=images_dir, masks_dir=masks_dir)
    plt.show()


if __name__ == "__main__":
    main()
