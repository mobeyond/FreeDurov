import os
from typing import Union

# Base Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Environment variable helper
def get_env_var(key: str, default: Union[str, int, bool, float, None] = None, var_type: type = str):
    """
    Get environment variable with type conversion and default fallback.
    """
    value = os.environ.get(key)
    if value is None:
        return default
    
    if var_type == bool:
        return value.lower() in ('true', '1', 'yes', 'on')
    elif var_type == int:
        try:
            return int(value)
        except ValueError:
            return default
    elif var_type == float:
        try:
            return float(value)
        except ValueError:
            return default
    return value

# Directory Configuration
TEMP_DIR = get_env_var('FREEDUROV_TEMP_DIR', os.path.join(BASE_DIR, "static/uploads"))
OUTPUT_DIR = get_env_var('FREEDUROV_OUTPUT_DIR', os.path.join(BASE_DIR, "output"))
BASE_IMAGE_PATH = get_env_var('FREEDUROV_BASE_IMAGE', os.path.join(BASE_DIR, "base.png"))

# Application Configuration
DEBUG = get_env_var('FREEDUROV_DEBUG', False, bool)
EXPORT_INTERMEDIARY_IMAGES = get_env_var('FREEDUROV_EXPORT_INTERMEDIARY', True, bool)
LOG_LEVEL = get_env_var('FREEDUROV_LOG_LEVEL', "INFO")
LOG_FILE = get_env_var('FREEDUROV_LOG_FILE', os.path.join(BASE_DIR, "app.log"))

# Error Messages
ERROR_NO_IMAGE = "No image data received"
ERROR_NO_PROFILES = "No significant info detected in the image"

# Network Configuration
DOMAIN_NAME = get_env_var('FREEDUROV_DOMAIN', "bing.com")
HOST = get_env_var('FREEDUROV_HOST', "0.0.0.0")
PORT = get_env_var('FREEDUROV_PORT', 9111, int)

# Image Processing Parameters
MIN_CONTOUR_RATIO = get_env_var('FREEDUROV_MIN_CONTOUR_RATIO', 0.0005, float)
MIN_CONTOUR_AREA = get_env_var('FREEDUROV_MIN_CONTOUR_AREA', 225, int)
ASPECT_RATIO_RANGE = (0.65, 1.35)  # Acceptable aspect ratio range for contours
SQR_OR_CIRC = get_env_var('FREEDUROV_SQR_OR_CIRC', 0.747, float)

# Corner Processing
CORNER_RADIUS = get_env_var('FREEDUROV_CORNER_RADIUS', 15, int)
CORNER_ROUNDING_RATIO = get_env_var('FREEDUROV_CORNER_ROUNDING_RATIO', 0.15, float)

# GIF Configuration
GIF_DURATION = get_env_var('FREEDUROV_GIF_DURATION', 1000, int)  # Duration for each frame (ms)
MAX_PROFILES_PER_GIF = get_env_var('FREEDUROV_MAX_PROFILES_PER_GIF', 100, int)
INCLUDE_QR_CODE = get_env_var('FREEDUROV_INCLUDE_QR_CODE', False, bool)
INCLUDE_CANNY = get_env_var('FREEDUROV_INCLUDE_CANNY', True, bool)
QR_CODE_URL = get_env_var('FREEDUROV_QR_CODE_URL', "https://douban.com")

# HTML Configuration
HTML_TITLE = get_env_var('FREEDUROV_HTML_TITLE', "GROUP GOES GIF")
HTML_HEADER_COLOR = get_env_var('FREEDUROV_HTML_HEADER_COLOR', "#4CAF50")
HTML_BACKGROUND_COLOR = get_env_var('FREEDUROV_HTML_BACKGROUND_COLOR', "#f0f0f0")

# Server Ports
HTTP_PORT = get_env_var('FREEDUROV_HTTP_PORT', 80, int)
HTTPS_PORT = get_env_var('FREEDUROV_HTTPS_PORT', 443, int)

# SSL Configuration
SSL_CERT_FILE = get_env_var('FREEDUROV_SSL_CERT_FILE', "cert.pem")
SSL_KEY_FILE = get_env_var('FREEDUROV_SSL_KEY_FILE', "key.pem")

# Ensure required directories exist
if isinstance(TEMP_DIR, str):
    os.makedirs(TEMP_DIR, exist_ok=True)
if isinstance(OUTPUT_DIR, str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
