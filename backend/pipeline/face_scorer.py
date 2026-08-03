import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

class FaceScorer:
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

    def score_frame(self, image: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        # 1. Sharpness
        sharpness = variance_of_laplacian(image)
        sharpness_score = min(sharpness / 500.0, 1.0) * 100

        # 2. Brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if 80 <= brightness <= 180:
            brightness_score = 100
        elif brightness < 80:
            brightness_score = max(0, (brightness / 80) * 100)
        else:
            brightness_score = max(0, 100 - ((brightness - 180) / 75) * 100)

        # 3. Face detection
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.face_detection.process(rgb)

        if not detections.detections:
            return sharpness_score * 0.5, None

        detection = max(detections.detections, key=lambda d: 
                        d.location_data.relative_bounding_box.width *
                        d.location_data.relative_bounding_box.height)
        bbox = detection.location_data.relative_bounding_box
        h, w = image.shape[:2]
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        face_w = int(bbox.width * w)
        face_h = int(bbox.height * h)

        face_area_ratio = (face_w * face_h) / (w * h)
        face_size_score = min(face_area_ratio * 200, 100)

        # 4. Face Mesh
        mesh_results = self.face_mesh.process(rgb)
        if not mesh_results.multi_face_landmarks:
            composite = (0.3 * sharpness_score + 0.2 * brightness_score + 0.3 * face_size_score)
            return composite, self._align_face(image, detection)

        landmarks = mesh_results.multi_face_landmarks[0].landmark

        # 5. Eyes open
        left_ear = self._eye_aspect_ratio(landmarks, [33, 133, 160, 158, 144, 153])
        right_ear = self._eye_aspect_ratio(landmarks, [362, 263, 387, 385, 380, 373])
        avg_ear = (left_ear + right_ear) / 2.0
        eyes_open_score = min(avg_ear * 500, 100)

        # 6. Smile
        smile_ratio = self._mouth_smile_ratio(landmarks)
        smile_score = min((smile_ratio - 1.2) * 50, 100) if smile_ratio > 1.2 else 0

        # 7. Face angle
        left_eye = self._landmark_to_pt(landmarks[33])
        right_eye = self._landmark_to_pt(landmarks[362])
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        angle_score = max(0, 100 - abs(angle) * 2)

        # Weighted composite
        composite = (
            0.20 * sharpness_score +
            0.15 * brightness_score +
            0.20 * face_size_score +
            0.15 * eyes_open_score +
            0.20 * smile_score +
            0.10 * angle_score
        )

        aligned = self._align_face(image, detection, angle)
        return composite, aligned

    # ---- helpers ----
    def _eye_aspect_ratio(self, landmarks, indices):
        pts = [self._landmark_to_pt(landmarks[i]) for i in indices]
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def _mouth_smile_ratio(self, landmarks):
        left = self._landmark_to_pt(landmarks[61])
        right = self._landmark_to_pt(landmarks[291])
        top = self._landmark_to_pt(landmarks[13])
        bottom = self._landmark_to_pt(landmarks[14])
        width = np.linalg.norm(left - right)
        height = np.linalg.norm(top - bottom)
        return width / (height + 1e-6)

    def _landmark_to_pt(self, lm):
        return np.array([lm.x, lm.y])

    def _align_face(self, image, detection, angle=None):
        if angle is None:
            h, w = image.shape[:2]
            bbox = detection.location_data.relative_bounding_box
            left_eye = (bbox.xmin + 0.2 * bbox.width, bbox.ymin + 0.3 * bbox.height)
            right_eye = (bbox.xmin + 0.8 * bbox.width, bbox.ymin + 0.3 * bbox.height)
            left_pt = (int(left_eye[0] * w), int(left_eye[1] * h))
            right_pt = (int(right_eye[0] * w), int(right_eye[1] * h))
            dy = right_pt[1] - left_pt[1]
            dx = right_pt[0] - left_pt[0]
            angle = np.degrees(np.arctan2(dy, dx))
        if abs(angle) < 1.0:
            return image
        h, w = image.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return aligned

def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()
