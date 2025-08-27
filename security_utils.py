"""
Security utilities for FreeDurov application.
Provides input validation, file sanitization, and secure file handling.
"""

import os
import re
import hashlib
import mimetypes
from pathlib import Path
from typing import Optional, Union, List
import logging

logger = logging.getLogger(__name__)

# Security constants
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.tif'}
ALLOWED_MIME_TYPES = {
    'image/png', 'image/jpeg', 'image/gif', 'image/webp',
    'image/bmp', 'image/tiff', 'image/x-ms-bmp'
}

# Dangerous file patterns
DANGEROUS_PATTERNS = [
    r'\.\./',  # Directory traversal
    r'\\\.\\',  # Windows directory traversal
    r'[<>:"|?*]',  # Invalid filename characters
    r'^\.',  # Hidden files
    r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$',  # Windows reserved names
]

def validate_filename(filename: str) -> bool:
    """
    Validate that a filename is safe and allowed.
    
    Args:
        filename: The filename to validate
        
    Returns:
        bool: True if filename is safe, False otherwise
    """
    if not filename or not isinstance(filename, str):
        return False
    
    # Check length
    if len(filename) > 255:
        logger.warning(f"Filename too long: {len(filename)} characters")
        return False
    
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            logger.warning(f"Dangerous pattern found in filename: {filename}")
            return False
    
    # Check extension
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Disallowed file extension: {ext}")
        return False
    
    return True

def validate_file_content(file_path: str) -> bool:
    """
    Validate file content by checking MIME type and basic file structure.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        bool: True if file content is valid, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            return False
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"File too large: {file_size} bytes")
            return False
        
        if file_size == 0:
            logger.warning("Empty file")
            return False
        
        # Check MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type not in ALLOWED_MIME_TYPES:
            logger.warning(f"Invalid MIME type: {mime_type}")
            return False
        
        # Additional validation: try to read as image
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                img.verify()  # Verify it's a valid image
        except Exception as e:
            logger.warning(f"Invalid image file: {e}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating file content: {e}")
        return False

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing dangerous characters and patterns.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        str: Sanitized filename
    """
    if not filename:
        return "unknown_file"
    
    # Remove dangerous characters
    sanitized = re.sub(r'[<>:"|?*\\]', '_', filename)
    
    # Remove directory traversal attempts
    sanitized = re.sub(r'\.\./?', '', sanitized)
    
    # Remove leading dots and spaces
    sanitized = sanitized.lstrip('. ')
    
    # Limit length
    if len(sanitized) > 200:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:195] + ext
    
    # Ensure we have a valid extension
    if not any(sanitized.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        sanitized += '.png'
    
    return sanitized or "unknown_file.png"

def secure_path_join(base_path: str, *paths: str) -> str:
    """
    Safely join paths ensuring the result stays within the base directory.
    
    Args:
        base_path: The base directory path
        *paths: Additional path components
        
    Returns:
        str: Safe joined path
        
    Raises:
        ValueError: If the resulting path would escape the base directory
    """
    base_path = os.path.abspath(base_path)
    joined_path = os.path.join(base_path, *paths)
    resolved_path = os.path.abspath(joined_path)
    
    # Ensure the resolved path is within the base directory
    if not resolved_path.startswith(base_path):
        raise ValueError(f"Path traversal attempt detected: {joined_path}")
    
    return resolved_path

def generate_safe_filename(original_filename: str, unique: bool = True) -> str:
    """
    Generate a safe filename, optionally making it unique.
    
    Args:
        original_filename: Original filename
        unique: Whether to make the filename unique using hash
        
    Returns:
        str: Safe filename
    """
    sanitized = sanitize_filename(original_filename)
    
    if unique:
        # Add hash to make it unique
        name, ext = os.path.splitext(sanitized)
        file_hash = hashlib.md5(original_filename.encode()).hexdigest()[:8]
        sanitized = f"{name}_{file_hash}{ext}"
    
    return sanitized

def validate_directory_path(path: str, base_dirs: List[str]) -> bool:
    """
    Validate that a directory path is within allowed base directories.
    
    Args:
        path: Directory path to validate
        base_dirs: List of allowed base directories
        
    Returns:
        bool: True if path is valid, False otherwise
    """
    try:
        abs_path = os.path.abspath(path)
        
        for base_dir in base_dirs:
            abs_base = os.path.abspath(base_dir)
            if abs_path.startswith(abs_base):
                return True
        
        logger.warning(f"Directory path outside allowed bases: {path}")
        return False
        
    except Exception as e:
        logger.error(f"Error validating directory path: {e}")
        return False

def safe_file_removal(file_path: str, base_dirs: List[str]) -> bool:
    """
    Safely remove a file with proper validation.
    
    Args:
        file_path: Path to file to remove
        base_dirs: List of allowed base directories
        
    Returns:
        bool: True if file was removed, False otherwise
    """
    try:
        if not validate_directory_path(os.path.dirname(file_path), base_dirs):
            logger.warning(f"Attempted to remove file outside allowed directories: {file_path}")
            return False
        
        if os.path.exists(file_path) and os.path.isfile(file_path):
            os.remove(file_path)
            logger.info(f"Safely removed file: {file_path}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error removing file {file_path}: {e}")
        return False

def get_file_hash(file_path: str) -> Optional[str]:
    """
    Get SHA256 hash of a file for integrity checking.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: SHA256 hash of the file, or None if error
    """
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hash: {e}")
        return None