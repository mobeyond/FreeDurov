import config
from avasplit import detect_and_extract_profiles
import os
import cv2
import numpy as np
import logging
import time
logger = logging.getLogger(__name__)

def process_avasplit(image_filepath, output_path):
    try:

        _, _, _, gif_files = detect_and_extract_profiles(image_filepath, output_path, config.GIF_DURATION, config.QR_CODE_URL, config.INCLUDE_QR_CODE)

        return gif_files

    except Exception as e:
        logger.error(f"Error in process_avasplit: {str(e)}")
        return []
