import cv2
import numpy as np
import logging

logger = logging.getLogger("image_processor")

class AIImageEnhancer:
    @staticmethod
    def enhance_image(input_path: str, output_path: str, quality: int = 95) -> bool:
        """
        Enhances image using OpenCV:
        1. CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space
        2. Fast NlMeans Denoising for high quality noise reduction
        3. Unsharp Masking for precise sharpening
        4. Subtle Color Boost in HSV space
        """
        try:
            # Read Image
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError("Could not read image or image is corrupted.")

            # 1. CLAHE (Adaptive Histogram Equalization) in LAB Space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            enhanced_lab = cv2.merge((cl, a, b))
            img_clahe = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            # 2. Fast Non-Local Means Denoising
            denoised = cv2.fastNlMeansDenoisingColored(
                img_clahe, 
                None, 
                h=3, 
                hColor=3, 
                templateWindowSize=7, 
                searchWindowSize=21
            )

            # 3. Sharpening via Unsharp Masking
            gaussian = cv2.GaussianBlur(denoised, (0, 0), 3)
            sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

            # 4. Moderate Color Boost in HSV Space
            hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = cv2.multiply(s, 1.15)  # Boost saturation by 15%
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            
            boosted_hsv = cv2.merge((h, s, v))
            final_img = cv2.cvtColor(boosted_hsv, cv2.COLOR_HSV2BGR)

            # Save Output
            cv2.imwrite(output_path, final_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            return True

        except Exception as e:
            logger.error(f"Image enhancement error: {str(e)}")
            return False
