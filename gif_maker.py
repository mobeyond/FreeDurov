import config
from PIL import Image, ImageDraw
import cv2
import numpy as np
import os, traceback
from PIL import Image

def round_corners(image, corner_ratio=config.CORNER_ROUNDING_RATIO):
    # Calculate radius based on the smaller dimension of the image
    radius = int(min(image.size) * corner_ratio)
    
    # Create a mask for rounded corners
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), image.size], radius, fill=255)
    
    # Ensure the image has an alpha channel
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Apply the mask to the image's alpha channel
    image_rgba = image.copy()
    image_rgba.putalpha(mask)
    
    return image_rgba

def polish_image(image, corner_ratio=config.CORNER_ROUNDING_RATIO):
    # Convert PIL Image to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(cv_image, (0, 0), 3)
    
    # Sharpen the image
    sharpened = cv2.addWeighted(cv_image, 1.5, blurred, -0.5, 0)
    
    # Convert back to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGRA2RGBA))
    
    # Apply rounded corners
    return round_corners(pil_image, corner_ratio)

def create_qr_code(url, size):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)
    qr_image = qr.make_image(fill_color="black", back_color="white")
    qr_image = qr_image.resize(size)
    return qr_image

def create_gif_from_profiles(profiles, output_dir, gif_duration=config.GIF_DURATION, url=config.QR_CODE_URL, include_qr=config.INCLUDE_QR_CODE):
    print(f"Creating GIF with include_qr set to: {include_qr}")
    if not profiles:
        print("No profiles to create GIF from.")
        return []

    # Sort profiles by size (area) in descending order
    sorted_profiles = sorted(profiles, key=lambda x: x.shape[0] * x.shape[1], reverse=True)
    gif_files = []

    for i in range(0, len(sorted_profiles), config.MAX_PROFILES_PER_GIF):
        gif_profiles = sorted_profiles[i:i+config.MAX_PROFILES_PER_GIF]
        gif_count = i // config.MAX_PROFILES_PER_GIF + 1
        
        if not gif_profiles:
            print(f"No valid images for GIF {gif_count}")
            continue

        # Determine the largest size in the current group and scale it up by 1.5
        max_height = int(max(profile.shape[0] for profile in gif_profiles) * 1.5)
        max_width = int(max(profile.shape[1] for profile in gif_profiles) * 1.5)
        max_size = (max_width, max_height)

        pil_images = []
        
        # This block should only execute if include_qr is True
        if include_qr:
            qr_image = create_qr_code(url, max_size)
            qr_image = qr_image.convert("RGBA")
            qr_image = polish_image(qr_image)
            pil_images.append(qr_image)
        
        # Add profile images
        for profile in gif_profiles:
            pil_image = Image.fromarray(cv2.cvtColor(profile, cv2.COLOR_BGR2RGBA))
            pil_image = pil_image.resize(max_size, Image.LANCZOS)
            pil_image = polish_image(pil_image)
            pil_images.append(pil_image)
        
        # Add spy image as the last frame
        try:
            base_image = Image.open(config.BASE_IMAGE_PATH).convert("RGBA")
            base_image = base_image.resize(max_size, Image.LANCZOS)
            base_image = polish_image(base_image)
            pil_images.append(base_image)
        except Exception as e:
            print(f"Error processing base image: {str(e)}")
            traceback.print_exc()
        
        # Before saving, create a new list of images with a white background
        images_with_background = []
        for img in pil_images:
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.paste(img, (0, 0), img)
            images_with_background.append(bg.convert("RGB"))

        gif_path = os.path.join(output_dir, f'profiles_gif_{gif_count}.gif')
        try:
            images_with_background[0].save(
                gif_path,
                save_all=True,
                append_images=images_with_background[1:],
                duration=gif_duration,
                loop=0,  # 0 means loop indefinitely
                optimize=False,
                disposal=2  # Clear the frame before rendering the next one
            )
            print(f"Created GIF: {gif_path}")
            gif_files.append(gif_path)
        except Exception as e:
            print(f"Error creating GIF {gif_count}: {str(e)}")
            traceback.print_exc()

    return gif_files