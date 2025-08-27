import config
from avasplit import detect_and_extract_profiles
import os
import cv2
import numpy as np
import logging
import time
import traceback

# Set up logger
logger = logging.getLogger(__name__)

def process_avasplit(image_filepath, output_path):
    """
    Process an image file to extract avatar profiles and create GIFs.
    
    Args:
        image_filepath (str): Path to the input image file
        output_path (str): Directory path for output files
        
    Returns:
        list: List of generated GIF file paths
    """
    try:
        # Validate input parameters
        if not image_filepath or not os.path.exists(image_filepath):
            logger.error(f"Invalid or non-existent image file: {image_filepath}")
            return []
            
        if not output_path:
            logger.error("Output path not provided")
            return []
            
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Validate image file
        try:
            image = cv2.imread(image_filepath)
            if image is None:
                logger.error(f"Failed to load image from {image_filepath}")
                return []
        except Exception as e:
            logger.error(f"Error loading image {image_filepath}: {str(e)}")
            return []
        
        logger.info(f"Processing image: {image_filepath} -> {output_path}")
        start_time = time.time()
        
        # Process the image
        _, _, _, gif_files = detect_and_extract_profiles(
            image_filepath, 
            output_path, 
            config.GIF_DURATION, 
            config.QR_CODE_URL, 
            config.INCLUDE_QR_CODE
        )
        
        end_time = time.time()
        logger.info(f"Image processing completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Generated {len(gif_files)} GIF files")
        
        # Validate output files
        valid_gif_files = []
        for gif_file in gif_files:
            if os.path.exists(gif_file) and os.path.getsize(gif_file) > 0:
                valid_gif_files.append(gif_file)
                logger.debug(f"Valid GIF file: {gif_file}")
            else:
                logger.warning(f"Invalid or empty GIF file: {gif_file}")
        
        return valid_gif_files

    except Exception as e:
        logger.error(f"Error in process_avasplit: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []
