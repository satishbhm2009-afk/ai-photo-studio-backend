import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Dict, Any, Optional

class FaceAlignerDetector:
    """Detects faces and aligns them vertically using MediaPipe."""

    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

    def process(self, image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image at {image_path}")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)

        img_h, img_w = image.shape[:2]
        meta = {"faces_detected": 0, "face_area_ratio": 0.0, "aligned": False}
        aligned_image = image.copy()
        aligned = False

        if results.detections:
            # Take the largest face (by area)
            detections = sorted(
                results.detections,
                key=lambda d: (d.location_data.relative_bounding_box.width *
                               d.location_data.relative_bounding_box.height),
                reverse=True
            )
            detection = detections[0]
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * img_w)
            y = int(bbox.ymin * img_h)
            w = int(bbox.width * img_w)
            h = int(bbox.height * img_h)

            face_area_ratio = float((w * h) / (img_w * img_h))
            meta["face_area_ratio"] = face_area_ratio
            meta["faces_detected"] = len(detections)

            # Estimate eye centers: assume eyes are in the upper half of the face
            # This is a heuristic; for better accuracy, use MediaPipe Face Mesh.
            eye_y = y + int(0.35 * h)
            eye_x_left = x + int(0.25 * w)
            eye_x_right = x + int(0.75 * w)
            e1_center = (eye_x_left, eye_y)
            e2_center = (eye_x_right, eye_y)

            # Calculate rotation angle to level the eyes
            dy = e2_center[1] - e1_center[1]
            dx = e2_center[0] - e1_center[0]
            angle = np.degrees(np.arctan2(dy, dx))

            if abs(angle) > 1.0 and abs(angle) < 45.0:
                center = (int((e1_center[0] + e2_center[0]) / 2),
                          int((e1_center[1] + e2_center[1]) / 2))
                M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
                aligned_image = cv2.warpAffine(
                    image, M, (img_w, img_h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                aligned = True

        meta["aligned"] = aligned
        return aligned_image, meta
