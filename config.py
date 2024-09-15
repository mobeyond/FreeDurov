import os

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp_output")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
BASE_IMAGE_PATH = os.path.join(BASE_DIR, "assets", "base.png")

# Logging configuration
DEBUG = False
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(BASE_DIR, "app.log")
# Error messages
ERROR_NO_IMAGE = "No image data received"
ERROR_NO_PROFILES = "No significant info detected in the image"

# Server configuration
HOST = "0.0.0.0"
PORT = 9111

# Image processing settings
MIN_CONTOUR_RATIO = 0.0005  # Minimum ratio for a contour to be considered
MIN_CONTOUR_AREA = 225  # Minimum area for a contour to be considered
ASPECT_RATIO_RANGE = (0.65, 1.35)  # Acceptable aspect ratio range for contours
SQR_OR_CIRC = 0.747

# Corner rounding settings
CORNER_RADIUS = 15  # Adjust this value to change the roundness of corners

# GIF generation settings
GIF_DURATION = 800  # Duration for each frame in the GIF (milliseconds)
MAX_PROFILES_PER_GIF = 10  # Maximum number of profiles to include in a single GIF
# QR Feature flags
INCLUDE_QR_CODE = False  # Set to True to enable QR code embedding
INCLUDE_CANNY = True
# QR code settings (used only if INCLUDE_QR_CODE is True)
QR_CODE_URL = "https://douban.com"

# HTML templates
HTML_TITLE = "GROUP GOES GIF"
HTML_HEADER_COLOR = "#4CAF50"
HTML_BACKGROUND_COLOR = "#f0f0f0"
