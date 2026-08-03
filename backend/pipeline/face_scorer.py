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
        """
        Returns (composite_score, aligned_image) for this frame.
        If no face, returns (sharpness_score, None).
        """
        # 1. Sharpness (always)
        sharpness = variance_of_laplacian(image)
        sharpness_score = min(sharpness / 500.0, 1.0) * 100  # normalise

        # 2. Brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        # ideal brightness 80-180
        brightness_score = self._brightness_score(brightness)

        # 3. Face detection and scoring
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.face_detection.process(rgb)

        if not detections.detections:
            # No face: use only sharpness (fallback)
            return sharpness_score * 0.5, None

        # Take the largest face
        detection = max(detections.detections, key=lambda d: 
                        d.location_data.relative_bounding_box.width *
                        d.location_data.relative_bounding_box.height)
        bbox = detection.location_data.relative_bounding_box
        h, w = image.shape[:2]
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        face_w = int(bbox.width * w)
        face_h = int(bbox.height * h)

        # Face size score (prefer large faces)
        face_area_ratio = (face_w * face_h) / (w * h)
        face_size_score = min(face_area_ratio * 200, 100)  # caps at 100

        # 4. Face Mesh for detailed landmarks
        mesh_results = self.face_mesh.process(rgb)
        if not mesh_results.multi_face_landmarks:
            # No mesh: skip eye/smile scoring, fallback to basic
            composite = (0.3 * sharpness_score + 0.2 * brightness_score + 0.3 * face_size_score)
            return composite, self._align_face(image, detection)

        landmarks = mesh_results.multi_face_landmarks[0].landmark

        # 5. Eyes open (EAR)
        left_ear = self._eye_aspect_ratio(landmarks, [33, 133, 160, 158, 144, 153])  # left eye
        right_ear = self._eye_aspect_ratio(landmarks, [362, 263, 387, 385, 380, 373]) # right eye
        avg_ear = (left_ear + right_ear) / 2.0
        # EAR threshold for open eye ~0.2; score up to 1.0
        eyes_open_score = min(avg_ear * 500, 100)  # if avg_ear=0.2 => 100

        # 6. Smile (mouth width/height ratio)
        smile_ratio = self._mouth_smile_ratio(landmarks)
        # typical neutral ~1.5, smile >2.5; score up to 100
        smile_score = min((smile_ratio - 1.2) * 50, 100)
        if smile_score < 0:
            smile_score = 0

        # 7. Face angle (rotation from eyes)
        # Use eye landmarks to compute tilt
        left_eye = self._landmark_to_pt(landmarks[33])
        right_eye = self._landmark_to_pt(landmarks[362])
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        angle_score = max(0, 100 - abs(angle) * 2)  # 0° gets 100, 45° gets 10

        # Weighted composite (adjust weights as needed)
        composite = (
            0.20 * sharpness_score +
            0.15 * brightness_score +
            0.20 * face_size_score +
            0.15 * eyes_open_score +
            0.20 * smile_score +
            0.10 * angle_score
        )

        # Align the face using eye tilt (Phase 3)
        aligned = self._align_face(image, detection, angle)

        return composite, aligned

    # ---- Helper methods ----

    def _brightness_score(self, mean_val):
        if 80 <= mean_val <= 180:
            return 100
        elif mean_val < 80:
            return max(0, (mean_val / 80) * 100)
        else:  # >180
            return max(0, 100 - ((mean_val - 180) / 75) * 100)

    def _eye_aspect_ratio(self, landmarks, indices):
        # indices: [p1, p2, p3, p4, p5, p6] as per MediaPipe face mesh
        pts = [self._landmark_to_pt(landmarks[i]) for i in indices]
        # vertical distances
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        horizontal = np.linalg.norm(pts[0] - pts[3])
        ear = (v1 + v2) / (2.0 * horizontal + 1e-6)
        return ear

    def _mouth_smile_ratio(self, landmarks):
        # mouth corners: 61 (left), 291 (right); top: 13, bottom: 14
        left = self._landmark_to_pt(landmarks[61])
        right = self._landmark_to_pt(landmarks[291])
        top = self._landmark_to_pt(landmarks[13])
        bottom = self._landmark_to_pt(landmarks[14])
        width = np.linalg.norm(left - right)
        height = np.linalg.norm(top - bottom)
        return width / (height + 1e-6)

    def _landmark_to_pt(self, landmark, h=None, w=None):
        return np.array([landmark.x, landmark.y])  # normalized

    def _align_face(self, image, detection, angle=None):
        # Use eye tilt to rotate image
        if angle is None:
            # compute again
            h, w = image.shape[:2]
            bbox = detection.location_data.relative_bounding_box
            # approximate eye centres (relative)
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

# Reuse sharpness function
def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()
