import cv2
import numpy as np
from backend.config import settings
from backend.logger import logger

class QualityFilter:
    @staticmethod
    def is_blurry(image: np.ndarray, threshold: float = None) -> bool:
        if threshold is None:
            threshold = settings.BLUR_THRESHOLD
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold

    @staticmethod
    def brightness_score(image: np.ndarray) -> float:
        """Return brightness score (0-1) based on mean pixel intensity."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        return mean / 255.0

    @staticmethod
    def is_too_dark_or_bright(image: np.ndarray, dark_thresh=0.1, bright_thresh=0.9) -> bool:
        score = QualityFilter.brightness_score(image)
        return score < dark_thresh or score > bright_thresh