import cv2
import numpy as np
from typing import Dict, Any, List

class QualityAnalyzer:
    """Evaluates image sharpness, blur, contrast, and visual metrics."""

    @staticmethod
    def calculate_laplacian_blur(image: np.ndarray) -> float:
        """Computes variance of Laplacian operator. Higher = sharper, Lower = blurry."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def calculate_contrast_score(image: np.ndarray) -> float:
        """Calculates standard deviation of grayscale channel for contrast metric."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(gray.std())

    @staticmethod
    def calculate_brightness_balance(image: np.ndarray) -> float:
        """Evaluates brightness uniformity to reject overly exposed/dark frames."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_val = float(gray.mean())
        # Score penalty for extreme dark (<40) or extreme bright (>215)
        ideal_target = 128.0
        deviation = abs(mean_val - ideal_target)
        return float(max(0.0, 100.0 - (deviation * 0.7)))

    @classmethod
    def evaluate_frame(cls, image_path: str) -> Dict[str, Any]:
        image = cv2.imread(image_path)
        if image is None:
            return {
                "path": image_path,
                "blur_score": 0.0,
                "contrast_score": 0.0,
                "brightness_score": 0.0,
                "is_blurry": True
            }

        blur_val = cls.calculate_laplacian_blur(image)
        contrast_val = cls.calculate_contrast_score(image)
        brightness_val = cls.calculate_brightness_balance(image)

        # Thresholds
        is_blurry = blur_val < 100.0  # Threshold for blur flag

        return {
            "path": image_path,
            "blur_score": blur_val,
            "contrast_score": contrast_val,
            "brightness_score": brightness_val,
            "is_blurry": is_blurry
        }