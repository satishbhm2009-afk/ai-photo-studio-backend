import cv2
import numpy as np
import base64

def variance_of_laplacian(image):
    """Calculate sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def brightness_score(image):
    """Score brightness (0-100). Ideal: 80-180."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    if 80 <= mean <= 180:
        return 100.0
    elif mean < 80:
        return max(0, (mean / 80) * 100)
    else:
        return max(0, 100 - ((mean - 180) / 75) * 100)

def encode_image_to_base64(image, quality=92):
    """Encode OpenCV image to base64 JPEG string."""
    _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer).decode('utf-8')
