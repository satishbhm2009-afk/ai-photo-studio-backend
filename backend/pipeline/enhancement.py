import cv2
import numpy as np
from PIL import Image, ImageEnhance

class ImageEnhancer:
    """Applies high-grade classical computer vision enhancements & resolution upscaling."""

    @staticmethod
    def apply_clahe(image: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization in LAB space."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    @staticmethod
    def apply_bilateral_denoise(image: np.ndarray) -> np.ndarray:
        """Removes sensor noise while retaining sharp edges."""
        return cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)

    @staticmethod
    def apply_unsharp_mask(image: np.ndarray) -> np.ndarray:
        """Sharpens image features via unsharp masking technique."""
        gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
        sharpened = cv2.addWeighted(image, 1.4, gaussian, -0.4, 0)
        return sharpened

    @classmethod
    def enhance_and_upscale(cls, input_image: np.ndarray, output_path: str) -> str:
        # Step 1: Classical CV steps
        clahe_img = cls.apply_clahe(input_image)
        denoised_img = cls.apply_bilateral_denoise(clahe_img)
        sharpened_img = cls.apply_unsharp_mask(denoised_img)

        # Write intermediate image
        cv2.imwrite(output_path, sharpened_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Step 2: PIL Micro Tuning & Lanczos 2x Upscaling Hook
        pil_img = Image.open(output_path)

        # Saturation & Color Boost
        color_enhancer = ImageEnhance.Color(pil_img)
        pil_img = color_enhancer.enhance(1.12)

        # Sharpness Boost
        sharp_enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = sharp_enhancer.enhance(1.15)

        # Contrast Micro Adjustment
        contrast_enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = contrast_enhancer.enhance(1.05)

        # 2x High-Quality Upscale
        width, height = pil_img.size
        pil_img = pil_img.resize((width * 2, height * 2), Image.Resampling.LANCZOS)

        # Save Final Enhanced Image
        pil_img.save(output_path, format="JPEG", quality=98)
        return output_path