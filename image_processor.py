import config
from avasplit import detect_and_extract_profiles
import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def process_avasplit(image_data):
    try:
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        temp_image_path = os.path.join(config.TEMP_DIR, "input_image.png")
        
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError(config.ERROR_NO_IMAGE)
        
        # Explicitly save as PNG
        is_success, im_buf_arr = cv2.imencode(".png", img)
        if is_success:
            im_buf_arr.tofile(temp_image_path)
            logger.info(f"Saved temporary image to {temp_image_path}")
        else:
            raise ValueError("Failed to save temporary image")

        _, _, _, gif_files = detect_and_extract_profiles(temp_image_path, config.OUTPUT_DIR, config.QR_CODE_URL)
        
        return gif_files

    except Exception as e:
        logger.error(f"Error in process_avasplit: {str(e)}")
        return []