import cv2
import numpy as np
from skimage import restoration, exposure
from backend.logger import logger

class ImageEnhancer:
    @staticmethod
    def auto_white_balance(image: np.ndarray) -> np.ndarray:
        """Simple gray-world white balance."""
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(result)
        l_avg = np.mean(l)
        l = cv2.add(l, (255 - l_avg) / 2)
        result = cv2.merge((l, a, b))
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising."""
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        """Unsharp masking."""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def upscale(image: np.ndarray, factor: float = 2.0) -> np.ndarray:
        """Resize using interpolation."""
        h, w = image.shape[:2]
        new_h, new_w = int(h * factor), int(w * factor)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def color_correction(image: np.ndarray) -> np.ndarray:
        """Simple contrast stretching."""
        return exposure.rescale_intensity(image, in_range='image', out_range=(0, 255)).astype(np.uint8)

    @staticmethod
    def enhance(image: np.ndarray) -> np.ndarray:
        """Apply all enhancements in sequence."""
        enhanced = image
        enhanced = ImageEnhancer.auto_white_balance(enhanced)
        enhanced = ImageEnhancer.denoise(enhanced)
        enhanced = ImageEnhancer.sharpen(enhanced)
        enhanced = ImageEnhancer.color_correction(enhanced)
        # Optionally upscale (commented by default to save size)
        # enhanced = ImageEnhancer.upscale(enhanced, 1.5)
        return enhanced