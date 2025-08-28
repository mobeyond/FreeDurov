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
    Process an image file to extract avatar profiles and save individual profile images.
    GIF creation is now deferred until after user selection.
    
    Args:
        image_filepath (str): Path to the input image file
        output_path (str): Directory path for output files
        
    Returns:
        tuple: (profile_regions, profile_filenames) - regions and saved profile image filenames
    """
    try:
        # Validate input parameters
        if not image_filepath or not os.path.exists(image_filepath):
            logger.error(f"Invalid or non-existent image file: {image_filepath}")
            return [], []
            
        if not output_path:
            logger.error("Output path not provided")
            return [], []
            
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Validate image file
        try:
            image = cv2.imread(image_filepath)
            if image is None:
                logger.error(f"Failed to load image from {image_filepath}")
                return [], []
        except Exception as e:
            logger.error(f"Error loading image {image_filepath}: {str(e)}")
            return [], []
        
        logger.info(f"Processing image: {image_filepath} -> {output_path}")
        start_time = time.time()
        
        # Process the image to extract profiles (without creating GIFs yet)
        profile_regions, profile_count, profiles, profile_filenames, _ = detect_and_extract_profiles(
            image_filepath, 
            output_path, 
            config.GIF_DURATION, 
            config.QR_CODE_URL, 
            config.INCLUDE_QR_CODE
        )
        
        end_time = time.time()
        logger.info(f"Profile extraction completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Extracted {len(profile_filenames)} profile images")
        
        # Validate profile files
        valid_profile_filenames = []
        for filename in profile_filenames:
            file_path = os.path.join(output_path, filename)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                valid_profile_filenames.append(filename)
                logger.debug(f"Valid profile file: {filename}")
            else:
                logger.warning(f"Invalid or empty profile file: {filename}")
        
        return profile_regions, valid_profile_filenames

    except Exception as e:
        logger.error(f"Error in process_avasplit: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return [], []

def create_gifs_from_selected_profiles(selected_profile_filenames, output_path):
    """
    Create GIF files from selected profile images.
    
    Args:
        selected_profile_filenames (list): List of selected profile image filenames
        output_path (str): Directory path containing profile images and for output files
        
    Returns:
        list: List of generated GIF file paths
    """
    try:
        if not selected_profile_filenames:
            logger.warning("No profile filenames provided for GIF creation.")
            return []
            
        if not output_path or not os.path.exists(output_path):
            logger.error(f"Invalid output path: {output_path}")
            return []
        
        logger.info(f"Creating GIFs from {len(selected_profile_filenames)} selected profiles")
        start_time = time.time()
        
        # Load selected profile images
        selected_profiles = []
        for filename in selected_profile_filenames:
            file_path = os.path.join(output_path, filename)
            if os.path.exists(file_path):
                profile_image = cv2.imread(file_path)
                if profile_image is not None:
                    selected_profiles.append(profile_image)
                    logger.debug(f"Loaded profile: {filename}")
                else:
                    logger.warning(f"Failed to load profile image: {filename}")
            else:
                logger.warning(f"Profile file not found: {file_path}")
        
        if not selected_profiles:
            logger.warning("No valid selected profile images found.")
            return []
        
        # Import gif_maker here to avoid circular imports
        from gif_maker import create_gif_from_profiles
        
        # Create GIFs from selected profiles
        gif_files = create_gif_from_profiles(
            selected_profiles, 
            output_path, 
            config.GIF_DURATION, 
            config.QR_CODE_URL, 
            config.INCLUDE_QR_CODE
        )
        
        end_time = time.time()
        logger.info(f"GIF creation completed in {end_time - start_time:.2f} seconds")
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
        logger.error(f"Error in create_gifs_from_selected_profiles: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []
