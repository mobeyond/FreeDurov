import config
from PIL import Image
import qrcode
import os, cv2, traceback

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

        # Determine the largest size in the current group
        max_height = max(profile.shape[0] for profile in gif_profiles)
        max_width = max(profile.shape[1] for profile in gif_profiles)
        max_size = (max_width, max_height)

        pil_images = []
        
        # This block should only execute if include_qr is True
        if include_qr:
            qr_image = create_qr_code(url, max_size)
            qr_image = qr_image.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=256)
            pil_images.append(qr_image)
        
        # Add profile images
        for profile in gif_profiles:
            pil_image = Image.fromarray(cv2.cvtColor(profile, cv2.COLOR_BGR2RGB))
            pil_image = pil_image.resize(max_size, Image.LANCZOS)
            pil_image = pil_image.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=256)
            pil_images.append(pil_image)
        
        # Add spy image as the last frame
        try:
            base_image = Image.open(config.BASE_IMAGE_PATH)
            base_image = base_image.resize(max_size, Image.LANCZOS)
            base_image = base_image.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=256)
            pil_images.append(base_image)
        except Exception as e:
            print(f"Error processing spy image: {str(e)}")
            traceback.print_exc()
        
        gif_path = os.path.join(output_dir, f'profiles_gif_{gif_count}.gif')
        try:
            pil_images[0].save(
                gif_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=gif_duration,
                loop=0,  # 0 means loop indefinitely
                optimize=False
            )
            print(f"Created GIF: {gif_path}")
            gif_files.append(gif_path)
        except Exception as e:
            print(f"Error creating GIF {gif_count}: {str(e)}")
            traceback.print_exc()

    return gif_files