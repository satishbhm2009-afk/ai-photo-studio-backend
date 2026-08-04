import cv2
import numpy as np

class ImageEnhancer:
    def process(self, image):
        # 1. Denoise
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # 2. Auto white balance (simple)
        result = self._auto_white_balance(denoised)

        # 3. Sharpening (unsharp mask)
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
        sharpened = cv2.filter2D(result, -1, kernel)

        # 4. Upscale 2x (LANCZOS)
        h, w = sharpened.shape[:2]
        upscaled = cv2.resize(sharpened, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)

        return upscaled

    def _auto_white_balance(self, img):
        # Simple gray-world assumption
        b, g, r = cv2.split(img)
        avg_b = np.mean(b)
        avg_g = np.mean(g)
        avg_r = np.mean(r)
        avg = (avg_b + avg_g + avg_r) / 3
        b = cv2.addWeighted(b, avg/avg_b, 0, 0, 0)
        g = cv2.addWeighted(g, avg/avg_g, 0, 0, 0)
        r = cv2.addWeighted(r, avg/avg_r, 0, 0, 0)
        return cv2.merge([b, g, r])
