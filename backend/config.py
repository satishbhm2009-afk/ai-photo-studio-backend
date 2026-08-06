import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMP_DIR = os.path.join(BASE_DIR, "temp")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MAX_UPLOAD_SIZE = 500 * 1024 * 1024

DEFAULT_INTERVAL = 5

DEFAULT_TOP_FRAMES = 10

DEFAULT_PROMPT = ""

ENABLE_AI_ENHANCEMENT = True

DEVICE = "cpu"

IMAGE_FORMAT = ".jpg"

JPEG_QUALITY = 95
