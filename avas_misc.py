import tkinter as tk
from tkinter import filedialog
import os, sys, config
from avasplit import detect_and_extract_profiles

def select_input_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        initialdir="./input/",
        title="Select Input Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    return file_path


def main():
    if len(sys.argv) > 1:
        input_image = sys.argv[1]
    else:
        input_image = select_input_image()
    if not input_image:
        print("No image selected. Exiting.")
        return
    output_dir = os.path.join("./output", os.path.basename(input_image) + ".Z")
    os.makedirs(output_dir, exist_ok=True)

    profile_regions, total_profiles, extracted_profiles, gif_files = detect_and_extract_profiles(input_image, output_dir, config.GIF_DURATION, config.QR_CODE_URL, config.INCLUDE_QR_CODE)

    print(f"Total profiles: {total_profiles}")
#    print(f"GIF files saved: {', '.join(gif_files)}")

if __name__ == "__main__":
    main()
