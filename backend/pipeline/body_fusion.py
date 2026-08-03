import cv2
import numpy as np
import tempfile
import os
import mediapipe as mp
from typing import List, Tuple, Dict, Optional

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# ===== Landmark indices (MediaPipe) =====
# Pose landmarks
POSE_SHOULDERS = [11, 12]
POSE_HIPS = [23, 24]
POSE_ARMS = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
POSE_LEGS = [25, 26, 27, 28, 29, 30, 31, 32]

# Face Mesh indices for specific features
LEFT_EYE_INDICES = [33, 133, 160, 158, 144, 153]
RIGHT_EYE_INDICES = [362, 263, 387, 385, 380, 373]
MOUTH_INDICES = [61, 291, 13, 14]  # left, right, top, bottom
FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# ===== Region definitions =====
# Each region maps to: (pose_indices, face_indices, scoring_method)
REGIONS = {
    "face": {
        "pose": None,
        "face": FACE_OVAL_INDICES,
        "score": "face_sharpness"
    },
    "eyes": {
        "pose": None,
        "face": LEFT_EYE_INDICES + RIGHT_EYE_INDICES,
        "score": "eyes_open"
    },
    "smile": {
        "pose": None,
        "face": MOUTH_INDICES,
        "score": "smile_ratio"
    },
    "torso": {
        "pose": POSE_SHOULDERS + POSE_HIPS,
        "face": None,
        "score": "pose_stability"
    },
    "arms": {
        "pose": POSE_ARMS,
        "face": None,
        "score": "pose_stability"
    },
    "legs": {
        "pose": POSE_LEGS,
        "face": None,
        "score": "pose_stability"
    }
}


class BodyFusionEngine:
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # ===== Scoring Functions =====

    def _face_sharpness_score(self, img: np.ndarray, face_landmarks) -> float:
        """Face sharpness using Laplacian variance over the face ROI."""
        h, w = img.shape[:2]
        # Get face bounding box from landmarks
        xs = [lm.x * w for lm in face_landmarks]
        ys = [lm.y * h for lm in face_landmarks]
        x, y = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        face_roi = img[y:y2, x:x2]
        if face_roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _eyes_open_score(self, img: np.ndarray, face_landmarks) -> float:
        """Eye Aspect Ratio (EAR) – higher means more open."""
        left_ear = self._ear(face_landmarks, LEFT_EYE_INDICES)
        right_ear = self._ear(face_landmarks, RIGHT_EYE_INDICES)
        avg_ear = (left_ear + right_ear) / 2.0
        # EAR > 0.2 is open; scale to 0–100
        return min(avg_ear * 500, 100)

    def _smile_score(self, img: np.ndarray, face_landmarks) -> float:
        """Smile ratio: mouth width / height. Larger = bigger smile."""
        left = self._lm_to_pt(face_landmarks[61])
        right = self._lm_to_pt(face_landmarks[291])
        top = self._lm_to_pt(face_landmarks[13])
        bottom = self._lm_to_pt(face_landmarks[14])
        width = np.linalg.norm(left - right)
        height = np.linalg.norm(top - bottom)
        ratio = width / (height + 1e-6)
        # Neutral ~1.5, smile ~2.5–3.0; score 0–100
        return min((ratio - 1.2) * 50, 100) if ratio > 1.2 else 0.0

    def _pose_stability_score(self, img: np.ndarray, pose_landmarks) -> float:
        """Score based on landmark confidence (MediaPipe gives visibility)."""
        scores = [lm.visibility for lm in pose_landmarks]
        return np.mean(scores) * 100

    def _ear(self, landmarks, indices):
        pts = [self._lm_to_pt(landmarks[i]) for i in indices]
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def _lm_to_pt(self, lm):
        return np.array([lm.x, lm.y])

    # ===== Main Processing =====

    def process_images(self, image_paths: List[str]) -> str:
        """
        Load images, score each region, select best per region,
        align to base, and blend seamlessly.
        """
        # 1. Load all images and detect landmarks
        image_data = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue

            # Detect pose
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(rgb)
            if not pose_results.pose_landmarks:
                continue

            # Detect face mesh
            face_results = self.face_mesh.process(rgb)
            face_lms = face_results.multi_face_landmarks[0].landmark if face_results.multi_face_landmarks else None

            image_data.append({
                "img": img,
                "pose": pose_results.pose_landmarks.landmark,
                "face": face_lms,
                "path": path,
            })

        if len(image_data) == 0:
            raise ValueError("No valid images with detectable pose.")

        # 2. Score each region for every image
        region_scores = {region: [] for region in REGIONS.keys()}
        for data in image_data:
            for region, config in REGIONS.items():
                score = 0.0
                if config["score"] == "face_sharpness" and data["face"] is not None:
                    score = self._face_sharpness_score(data["img"], data["face"])
                elif config["score"] == "eyes_open" and data["face"] is not None:
                    score = self._eyes_open_score(data["img"], data["face"])
                elif config["score"] == "smile_ratio" and data["face"] is not None:
                    score = self._smile_score(data["img"], data["face"])
                elif config["score"] == "pose_stability":
                    score = self._pose_stability_score(data["img"], data["pose"])
                else:
                    score = 0.0
                region_scores[region].append(score)

        # 3. Determine base image (best overall pose)
        overall_scores = [np.mean([region_scores[r][i] for r in REGIONS.keys()]) for i in range(len(image_data))]
        base_idx = np.argmax(overall_scores)
        base_data = image_data[base_idx]
        base_img = base_data["img"].copy()
        base_pose = base_data["pose"]
        base_shape = base_img.shape[:2]

        # 4. For each region, pick the best image index
        best_region_idx = {}
        for region in REGIONS.keys():
            scores = region_scores[region]
            best_idx = np.argmax(scores)
            best_region_idx[region] = best_idx

        # 5. Composite: start with base image
        composite = base_img.copy()

        # 6. For each region, blend the best part
        for region, best_idx in best_region_idx.items():
            if best_idx == base_idx:
                continue  # already in base

            src_data = image_data[best_idx]
            src_img = src_data["img"]
            src_pose = src_data["pose"]

            # Get mask for this region on source
            mask = self._get_region_mask(src_img.shape, src_pose, src_data["face"], region)

            # Align source image + mask to base
            aligned_src, M = self._align_image_to_base(src_img, src_pose, base_pose, base_shape)
            warped_mask = self._warp_mask(mask, M, base_shape)

            # Ensure mask has enough pixels
            if np.sum(warped_mask) < 100:
                continue

            # Poisson blending (seamless)
            center = self._mask_center(warped_mask)
            mask_3ch = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)
            composite = cv2.seamlessClone(aligned_src, composite, mask_3ch, center, cv2.NORMAL_CLONE)

        # Save result
        out_fd, out_path = tempfile.mkstemp(suffix=".jpg")
        os.close(out_fd)
        cv2.imwrite(out_path, composite)
        return out_path

    # ===== Helper Methods =====

    def _get_region_mask(self, shape, pose_lms, face_lms, region: str) -> np.ndarray:
        """Create a binary mask for a given region."""
        h, w = shape[:2]
        points = []

        config = REGIONS[region]
        if config["pose"]:
            for idx in config["pose"]:
                lm = pose_lms[idx]
                points.append([int(lm.x * w), int(lm.y * h)])
        if config["face"] and face_lms:
            for idx in config["face"]:
                lm = face_lms[idx]
                points.append([int(lm.x * w), int(lm.y * h)])

        if len(points) < 3:
            # fallback: use a generous rectangular area
            return np.ones((h, w), dtype=np.uint8) * 255

        pts = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(pts)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        return mask

    def _align_image_to_base(self, img, src_pose, base_pose, base_shape):
        """Affine transform using shoulders + hips."""
        h, w = img.shape[:2]
        base_h, base_w = base_shape
        src_pts = []
        dst_pts = []

        for idx in [11, 12, 23, 24]:  # shoulders + hips
            src_pts.append([src_pose[idx].x * w, src_pose[idx].y * h])
            dst_pts.append([base_pose[idx].x * base_w, base_pose[idx].y * base_h])

        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if M is None:
            M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        aligned = cv2.warpAffine(img, M, (base_w, base_h), borderMode=cv2.BORDER_REPLICATE)
        return aligned, M

    def _warp_mask(self, mask, M, shape):
        """Warp mask using same affine transform."""
        h, w = shape
        return cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _mask_center(self, mask):
        """Find the center of mass of the mask."""
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            return (mask.shape[1] // 2, mask.shape[0] // 2)
        return (int(np.mean(x_indices)), int(np.mean(y_indices)))


# ===== Public API =====
def fuse_best_parts(image_paths: List[str]) -> str:
    engine = BodyFusionEngine()
    return engine.process_images(image_paths)
