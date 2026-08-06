import cv2
import numpy as np
from typing import Tuple, Optional

# ✅ Use the internal Python submodule path (works reliably)
from mediapipe.python.solutions import face_detection as mp_face_detection
from mediapipe.python.solutions import face_mesh as mp_face_mesh


class FaceScorer:
    def __init__(self):
        self.detection = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        self.mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        # Face Mesh landmark indices for eyes and mouth
        self.LEFT_EYE = [33, 133, 160, 158, 144, 153]
        self.RIGHT_EYE = [362, 263, 387, 385, 380, 373]
        self.MOUTH = [61, 291, 13, 14]

    def score(self, image: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """
        Returns (quality_score, aligned_image_or_None).
        Score ranges 0-100 based on face size, eyes open, smile, angle.
        """
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face
        detections = self.detection.process(rgb)
        if not detections.detections:
            return 0.0, None

        # Take largest face
        detection = max(
            detections.detections,
            key=lambda d: d.location_data.relative_bounding_box.width *
                          d.location_data.relative_bounding_box.height
        )
        bbox = detection.location_data.relative_bounding_box
        x, y = int(bbox.xmin * w), int(bbox.ymin * h)
        fw, fh = int(bbox.width * w), int(bbox.height * h)

        # Face size score
        area_ratio = (fw * fh) / (w * h)
        size_score = min(area_ratio * 200, 100)

        # Face Mesh for detailed landmarks
        mesh_results = self.mesh.process(rgb)
        if not mesh_results.multi_face_landmarks:
            return size_score * 0.5, self._align_face(image, detection)

        landmarks = mesh_results.multi_face_landmarks[0].landmark

        # Eyes open (EAR)
        left_ear = self._ear(landmarks, self.LEFT_EYE)
        right_ear = self._ear(landmarks, self.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        eyes_score = min(avg_ear * 500, 100)

        # Smile (width/height ratio)
        smile = self._smile_ratio(landmarks)
        smile_score = min((smile - 1.2) * 50, 100) if smile > 1.2 else 0

        # Angle (rotation from eyes)
        left_pt = self._lm_to_pt(landmarks[33], w, h)
        right_pt = self._lm_to_pt(landmarks[362], w, h)
        dy = right_pt[1] - left_pt[1]
        dx = right_pt[0] - left_pt[0]
        angle = np.degrees(np.arctan2(dy, dx))
        angle_score = max(0, 100 - abs(angle) * 2)

        # Composite score
        composite = (
            0.25 * size_score +
            0.25 * eyes_score +
            0.30 * smile_score +
            0.20 * angle_score
        )

        aligned = self._align_face(image, detection, angle)
        return min(composite, 100), aligned

    # ---------- Helpers ----------
    def _lm_to_pt(self, lm, w, h):
        return np.array([int(lm.x * w), int(lm.y * h)])

    def _ear(self, landmarks, indices):
        pts = [np.array([landmarks[i].x, landmarks[i].y]) for i in indices]
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h_dist = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * h_dist + 1e-6)

    def _smile_ratio(self, landmarks):
        left = np.array([landmarks[61].x, landmarks[61].y])
        right = np.array([landmarks[291].x, landmarks[291].y])
        top = np.array([landmarks[13].x, landmarks[13].y])
        bottom = np.array([landmarks[14].x, landmarks[14].y])
        return np.linalg.norm(left - right) / (np.linalg.norm(top - bottom) + 1e-6)

    def _align_face(self, image, detection, angle=None):
        h, w = image.shape[:2]
        if angle is None:
            bbox = detection.location_data.relative_bounding_box
            left = (bbox.xmin + 0.2 * bbox.width, bbox.ymin + 0.3 * bbox.height)
            right = (bbox.xmin + 0.8 * bbox.width, bbox.ymin + 0.3 * bbox.height)
            lx, ly = int(left[0] * w), int(left[1] * h)
            rx, ry = int(right[0] * w), int(right[1] * h)
            angle = np.degrees(np.arctan2(ry - ly, rx - lx))
        if abs(angle) < 1.0:
            return image
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
