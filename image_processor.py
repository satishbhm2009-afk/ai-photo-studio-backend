"""
==================================================
SOFTONIX AI PHOTO STUDIO
OpenCV Image Processor
==================================================
"""

import cv2
import numpy as np

from config import (
    ENABLE_AUTO_CONTRAST,
    ENABLE_DENOISE,
    ENABLE_SHARPEN,
    ENABLE_COLOR_BOOST,
    JPEG_QUALITY,
    PNG_COMPRESSION
)


# ==================================================
# AUTO CONTRAST
# ==================================================

def auto_contrast(image):

    if not ENABLE_AUTO_CONTRAST:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))

    return cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR
    )


# ==================================================
# NOISE REDUCTION
# ==================================================

def reduce_noise(image):

    if not ENABLE_DENOISE:
        return image

    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        8,
        8,
        7,
        21
    )


# ==================================================
# SHARPEN
# ==================================================

def sharpen(image):

    if not ENABLE_SHARPEN:
        return image

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(
        image,
        -1,
        kernel
    )


# ==================================================
# COLOR BOOST
# ==================================================

def color_boost(image):

    if not ENABLE_COLOR_BOOST:
        return image

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(hsv)

    s = cv2.add(s, 15)

    hsv = cv2.merge((h, s, v))

    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


# ==================================================
# COMPLETE AI PIPELINE
# ==================================================

def process_image(input_path, output_path):

    image = cv2.imread(str(input_path))

    if image is None:
        raise ValueError("Unable to read image.")

    image = auto_contrast(image)

    image = reduce_noise(image)

    image = sharpen(image)

    image = color_boost(image)

    extension = output_path.suffix.lower()

    if extension in [".jpg", ".jpeg"]:

        cv2.imwrite(
            str(output_path),
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY
            ]
        )

    elif extension == ".png":

        cv2.imwrite(
            str(output_path),
            image,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                PNG_COMPRESSION
            ]
        )

    else:

        cv2.imwrite(
            str(output_path),
            image
        )

    return output_path
