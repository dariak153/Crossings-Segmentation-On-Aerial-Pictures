import os

def get_last_number():
    while True:
        try:
            last_number = int(input("Podaj ostatni numer pliku, od którego zacząć numerację: "))
            return last_number
        except ValueError:
            print("bląd liczba całkowita")


folder_path = os.path.join("data", "data")


files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
files.sort()

unnumbered_files = [f for f in files if not f.split('.')[0].isdigit()]

start_number = get_last_number()


for index, file_name in enumerate(unnumbered_files):
    file_extension = os.path.splitext(file_name)[1]
    new_name = f"{start_number + index:03d}{file_extension}"
    old_path = os.path.join(folder_path, file_name)
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)

print("Numerowanie zakończone!")
