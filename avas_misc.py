import tkinter as tk
from tkinter import filedialog
import os
from avasplit import detect_and_extract_profiles

def select_input_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        initialdir="./in/",
        title="Select Input Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    return file_path

def main():
    input_image = select_input_image()
    if not input_image:
        print("No image selected. Exiting.")
        return

    output_dir = os.path.join("./output", os.path.basename(input_image))
    os.makedirs(output_dir, exist_ok=True)

    url = "https://example.com"  # Replace with your desired URL
    profiles, total_contours, extracted_profiles, gif_files = detect_and_extract_profiles(input_image, output_dir, url)

    print(f"Total contours detected: {total_contours}")
    print(f"GIF files saved: {', '.join(gif_files)}")

if __name__ == "__main__":
    main()
