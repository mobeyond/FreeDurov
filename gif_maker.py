import config
from PIL import Image, ImageDraw
import cv2
import numpy as np
import os, traceback
from PIL import Image
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Optional QR code support
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False
    logger.warning("qrcode module not available. QR code generation will be disabled.")

def round_corners(image, corner_ratio=config.CORNER_ROUNDING_RATIO):
    radius = int(min(image.size) * corner_ratio)
    
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), image.size], radius, fill=255)
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    image_rgba = image.copy()
    image_rgba.putalpha(mask)
    
    return image_rgba

def polish_image(image, corner_ratio=config.CORNER_ROUNDING_RATIO):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
    
    blurred = cv2.GaussianBlur(cv_image, (0, 0), 3)
    
    sharpened = cv2.addWeighted(cv_image, 1.5, blurred, -0.5, 0)
    
    pil_image = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGRA2RGBA))
    
    return round_corners(pil_image, corner_ratio)

def create_qr_code(url, size):
    if not QR_AVAILABLE:
        logger.warning("QR code generation requested but qrcode module not available")
        # Return a placeholder image
        placeholder = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(placeholder)
        draw.text((size[0]//4, size[1]//2), "QR Not Available", fill='black')
        return placeholder
    
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)
    qr_image = qr.make_image(fill_color="black", back_color="white")
    qr_image = qr_image.resize(size, Image.LANCZOS)
    return qr_image

def create_gif_from_profiles(profiles, output_dir, gif_duration=config.GIF_DURATION, url=config.QR_CODE_URL, include_qr=config.INCLUDE_QR_CODE):
    logger.info(f"Creating GIF with include_qr set to: {include_qr}")
    if not profiles:
        logger.warning("No profiles to create GIF from.")
        return []

    # Validate profile dimensions
    valid_profiles = []
    for i, profile in enumerate(profiles):
        try:
            if isinstance(profile, np.ndarray) and profile.ndim == 3:
                # Convert to uint8 if needed
                if profile.dtype != np.uint8:
                    profile = cv2.convertScaleAbs(profile)
                # Ensure RGB format (remove alpha channel if present)
                if profile.shape[2] == 4:
                    profile = cv2.cvtColor(profile, cv2.COLOR_RGBA2RGB)
                # Resize if necessary
                if max(profile.shape[0], profile.shape[1]) > 200:
                    profile = cv2.resize(profile, (200, 200))
                valid_profiles.append(profile)
            else:
                logger.warning(f"Invalid profile {i} shape: {profile.shape if isinstance(profile, np.ndarray) else type(profile)}")
        except Exception as e:
            logger.error(f"Error processing profile {i}: {str(e)}")

    if not valid_profiles:
        logger.warning("No valid profiles after validation.")
        return []

    sorted_profiles = sorted(profiles, key=lambda x: x.shape[0] * x.shape[1], reverse=True)
    gif_files = []

    for i in range(0, len(sorted_profiles), config.MAX_PROFILES_PER_GIF):
        gif_profiles = sorted_profiles[i:i+config.MAX_PROFILES_PER_GIF]
        gif_count = i // config.MAX_PROFILES_PER_GIF + 1
        
        if not gif_profiles:
            logger.warning(f"No valid images for GIF {gif_count}")
            continue

        max_height = int(max(profile.shape[0] for profile in gif_profiles) * 1.5)
        max_width = int(max(profile.shape[1] for profile in gif_profiles) * 1.5)
        max_size = (max_width, max_height)

        pil_images = []
        
        if include_qr:
            qr_image = create_qr_code(url, max_size)
            qr_image = qr_image.convert("RGBA")
            qr_image = polish_image(qr_image)
            pil_images.append(qr_image)
        
        # In the loop where images are processed
        for profile in gif_profiles:
            # Add depth conversion
            if profile.dtype != np.uint8:
                profile = cv2.convertScaleAbs(profile)
            # Fix color conversion
            pil_image = Image.fromarray(cv2.cvtColor(profile, cv2.COLOR_BGR2RGB))
            pil_image = pil_image.resize(max_size, Image.LANCZOS)
            pil_image = polish_image(pil_image)
            pil_image = polish_image(pil_image)
            pil_images.append(pil_image)
        
        try:
            base_image = Image.open(config.BASE_IMAGE_PATH).convert("RGBA")
            base_image = base_image.resize(max_size, Image.LANCZOS)
            base_image = polish_image(base_image)
            pil_images.append(base_image)
        except Exception as e:
            logger.error(f"Error processing base image: {str(e)}")
            traceback.print_exc()
        
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
                loop=3,  # 0 means loop indefinitely
                optimize=False,
                disposal=2  # Clear the frame before rendering the next one
            )
            logger.info(f"Created GIF: {gif_path}")
            gif_files.append(gif_path)
        except Exception as e:
            logger.error(f"Error creating GIF {gif_count}: {str(e)}")
            traceback.print_exc()

    return gif_files