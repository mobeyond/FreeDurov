import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp_output")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
BASE_IMAGE_PATH = os.path.join(BASE_DIR, "base.png")

DEBUG = False
EXPORT_INTERMEDIARY_IMAGES = False
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(BASE_DIR, "app.log")
ERROR_NO_IMAGE = "No image data received"
ERROR_NO_PROFILES = "No significant info detected in the image"

DOMAIN_NAME = "eff.org"
HOST = "0.0.0.0"
PORT = 9111

MIN_CONTOUR_RATIO = 0.0005  # Minimum ratio for a contour to be considered
MIN_CONTOUR_AREA = 225  # Minimum area for a contour to be considered
ASPECT_RATIO_RANGE = (0.65, 1.35)  # Acceptable aspect ratio range for contours
SQR_OR_CIRC = 0.747

CORNER_RADIUS = 15  # Adjust this value to change the roundness of corners

CORNER_ROUNDING_RATIO = 0.15

GIF_DURATION = 1000  # Duration for each frame in the GIF (milliseconds)
MAX_PROFILES_PER_GIF = 100  # Maximum number of profiles to include in a single GIF
INCLUDE_QR_CODE = False  # Set to True to enable QR code embedding
INCLUDE_CANNY = True
QR_CODE_URL = "https://https://zh.annas-archive.org/"

HTML_TITLE = "GROUP GOES GIF"
HTML_HEADER_COLOR = "#4CAF50"
HTML_BACKGROUND_COLOR = "#f0f0f0"

HTTP_PORT = 80
HTTPS_PORT = 443  # Changed from 8443 to 1984 as per your previous setup

SSL_CERT_FILE = "cert.pem"
SSL_KEY_FILE = "key.pem"
