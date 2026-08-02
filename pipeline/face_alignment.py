import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

class FaceAlignerDetector:
    """Detects facial landmarks and aligns face features vertically."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    def process(self, image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image at {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        face_count = len(faces)
        if face_count == 0:
            return image, {"faces_detected": 0, "face_area_ratio": 0.0, "aligned": False}

        # Take largest detected face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        img_h, img_w = image.shape[:2]
        face_area_ratio = float((w * h) / (img_w * img_h))

        # Face Eye Alignment check
        face_roi_gray = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(face_roi_gray)

        aligned_image = image.copy()
        aligned = False

        if len(eyes) >= 2:
            # Sort eyes by x coordinate
            eyes = sorted(eyes, key=lambda e: e[0])
            e1_center = (x + eyes[0][0] + eyes[0][2] // 2, y + eyes[0][1] + eyes[0][3] // 2)
            e2_center = (x + eyes[1][0] + eyes[1][2] // 2, y + eyes[1][1] + eyes[1][3] // 2)

            # Calculate rotation angle
            dy = e2_center[1] - e1_center[1]
            dx = e2_center[0] - e1_center[0]
            angle = np.degrees(np.arctan2(dy, dx))

            # Rotate image to level eyes
            if abs(angle) > 1.0 and abs(angle) < 45.0:
                center = (int((e1_center[0] + e2_center[0]) / 2), int((e1_center[1] + e2_center[1]) / 2))
                M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
                aligned_image = cv2.warpAffine(image, M, (img_w, img_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                aligned = True

        meta = {
            "faces_detected": face_count,
            "face_area_ratio": face_area_ratio,
            "aligned": aligned
        }

        return aligned_image, meta