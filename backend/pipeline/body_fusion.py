import cv2
import numpy as np
import tempfile
import os
import mediapipe as mp
from typing import List, Tuple, Dict, Optional, Any

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# ============================================================
# 1. LANDMARK INDEX DEFINITIONS
# ============================================================

# Face Mesh indices
LEFT_EYE = [33, 133, 160, 158, 144, 153]
RIGHT_EYE = [362, 263, 387, 385, 380, 373]
MOUTH = [61, 291, 13, 14]  # left, right, top, bottom
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
             361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
             176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
             162, 21, 54, 103, 67, 109]

# Pose indices (MediaPipe Pose)
NOSE = 0
LEFT_EYE_POSE = 1
RIGHT_EYE_POSE = 2
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# ============================================================
# 2. BODY PART DEFINITIONS (Convex hulls from landmarks)
# ============================================================

BODY_PARTS = {
    "hair_head": {"pose": [NOSE, LEFT_EYE_POSE, RIGHT_EYE_POSE], "face": FACE_OVAL},
    "face": {"pose": [], "face": FACE_OVAL},
    "eyes": {"pose": [], "face": LEFT_EYE + RIGHT_EYE},
    "smile": {"pose": [], "face": MOUTH},
    "neck": {"pose": [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER], "face": []},
    "left_upper_arm": {"pose": [LEFT_SHOULDER, LEFT_ELBOW, LEFT_HIP], "face": []},
    "left_forearm": {"pose": [LEFT_ELBOW, LEFT_WRIST, LEFT_INDEX], "face": []},
    "left_hand": {"pose": [LEFT_WRIST, LEFT_INDEX, LEFT_PINKY, LEFT_THUMB], "face": []},
    "right_upper_arm": {"pose": [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_HIP], "face": []},
    "right_forearm": {"pose": [RIGHT_ELBOW, RIGHT_WRIST, RIGHT_INDEX], "face": []},
    "right_hand": {"pose": [RIGHT_WRIST, RIGHT_INDEX, RIGHT_PINKY, RIGHT_THUMB], "face": []},
    "upper_torso": {"pose": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP], "face": []},
    "lower_torso": {"pose": [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE], "face": []},
    "left_thigh": {"pose": [LEFT_HIP, LEFT_KNEE, RIGHT_HIP], "face": []},
    "left_calf": {"pose": [LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL], "face": []},
    "left_foot": {"pose": [LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX], "face": []},
    "right_thigh": {"pose": [RIGHT_HIP, RIGHT_KNEE, LEFT_HIP], "face": []},
    "right_calf": {"pose": [RIGHT_KNEE, RIGHT_ANKLE, RIGHT_HEEL], "face": []},
    "right_foot": {"pose": [RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX], "face": []},
}

# Which score function to use for each part
SCORE_MAP = {
    "hair_head": "sharpness",
    "face": "sharpness",
    "eyes": "eyes_open",
    "smile": "smile_ratio",
    "neck": "sharpness",
    "left_upper_arm": "sharpness",
    "left_forearm": "sharpness",
    "left_hand": "sharpness",
    "right_upper_arm": "sharpness",
    "right_forearm": "sharpness",
    "right_hand": "sharpness",
    "upper_torso": "sharpness",
    "lower_torso": "sharpness",
    "left_thigh": "sharpness",
    "left_calf": "sharpness",
    "left_foot": "sharpness",
    "right_thigh": "sharpness",
    "right_calf": "sharpness",
    "right_foot": "sharpness",
}

# ============================================================
# 3. MAIN ENGINE
# ============================================================

class GranularFusionEngine:
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        self.segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    # ---------- Helper: landmark to pixel ----------
    def _lm_to_pt(self, lm, w, h):
        return np.array([int(lm.x * w), int(lm.y * h)])

    # ---------- Scoring Functions ----------
    def _sharpeness_score(self, img: np.ndarray, mask: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        masked = lap * (mask / 255.0)
        valid = mask > 0
        if np.sum(valid) == 0:
            return 0.0
        return float(np.var(masked[valid]))

    def _eyes_open_score(self, img: np.ndarray, face_lms, w, h) -> float:
        # if no face_lms, return 0
        if not face_lms:
            return 0.0
        pts = [self._lm_to_pt(face_lms[i], w, h) for i in LEFT_EYE + RIGHT_EYE]
        # EAR for left eye (indices 0-5), right eye (6-11)
        def ear(pts):
            v1 = np.linalg.norm(pts[1] - pts[5])
            v2 = np.linalg.norm(pts[2] - pts[4])
            h_dist = np.linalg.norm(pts[0] - pts[3])
            return (v1 + v2) / (2.0 * h_dist + 1e-6)
        left_ear = ear(pts[0:6])
        right_ear = ear(pts[6:12])
        avg_ear = (left_ear + right_ear) / 2.0
        return min(avg_ear * 500, 100)  # 0.2 -> 100

    def _smile_score(self, img: np.ndarray, face_lms, w, h) -> float:
        if not face_lms:
            return 0.0
        left = self._lm_to_pt(face_lms[61], w, h)
        right = self._lm_to_pt(face_lms[291], w, h)
        top = self._lm_to_pt(face_lms[13], w, h)
        bottom = self._lm_to_pt(face_lms[14], w, h)
        width = np.linalg.norm(left - right)
        height = np.linalg.norm(top - bottom)
        ratio = width / (height + 1e-6)
        return min((ratio - 1.2) * 50, 100) if ratio > 1.2 else 0.0

    # ---------- Mask generation for a body part ----------
    def _get_part_mask(self, img_shape, pose_lms, face_lms, part_name: str) -> np.ndarray:
        h, w = img_shape[:2]
        config = BODY_PARTS[part_name]
        points = []

        # Add pose landmarks
        for idx in config["pose"]:
            lm = pose_lms[idx]
            points.append([int(lm.x * w), int(lm.y * h)])

        # Add face mesh landmarks
        if face_lms:
            for idx in config["face"]:
                lm = face_lms[idx]
                points.append([int(lm.x * w), int(lm.y * h)])

        if len(points) < 3:
            # fallback: large region
            return np.ones((h, w), dtype=np.uint8) * 255

        pts = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(pts)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)

        # Refine with selfie segmentation (optional, but we'll skip to keep it 100% landmark-driven)
        # If you want to trim the mask to the exact body silhouette, uncomment below:
        # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # seg_result = self.segmentation.process(rgb)
        # seg_mask = (seg_result.segmentation_mask > 0.1).astype(np.uint8) * 255
        # mask = cv2.bitwise_and(mask, seg_mask)
        return mask

    # ---------- Align image to base ----------
    def _align_to_base(self, img, src_pose, base_pose, base_shape):
        h, w = img.shape[:2]
        base_h, base_w = base_shape
        src_pts = []
        dst_pts = []
        # Use shoulders + hips + nose for robust alignment
        for idx in [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]:
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
        h, w = shape
        return cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _mask_center(self, mask):
        y, x = np.where(mask > 0)
        if len(y) == 0:
            return (mask.shape[1] // 2, mask.shape[0] // 2)
        return (int(np.mean(x)), int(np.mean(y)))

    # ---------- Public API ----------
    def process_images(self, image_paths: List[str]) -> str:
        # 1. Load all images, detect pose + face
        image_data = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pose_res = self.pose.process(rgb)
            if not pose_res.pose_landmarks:
                continue
            pose_lms = pose_res.pose_landmarks.landmark

            face_res = self.face_mesh.process(rgb)
            face_lms = face_res.multi_face_landmarks[0].landmark if face_res.multi_face_landmarks else None

            image_data.append({
                "img": img,
                "pose": pose_lms,
                "face": face_lms,
            })

        if len(image_data) == 0:
            raise ValueError("No valid images with detectable pose.")

        # 2. Select base image (best overall pose confidence)
        base_idx = np.argmax([np.mean([lm.visibility for lm in d["pose"]]) for d in image_data])
        base_data = image_data[base_idx]
        base_img = base_data["img"].copy()
        base_pose = base_data["pose"]
        base_shape = base_img.shape[:2]

        # 3. For each body part, compute scores across all images
        part_scores = {part: [] for part in BODY_PARTS.keys()}
        for i, data in enumerate(image_data):
            img = data["img"]
            h, w = img.shape[:2]
            pose_lms = data["pose"]
            face_lms = data["face"]

            for part_name in BODY_PARTS.keys():
                mask = self._get_part_mask(img.shape, pose_lms, face_lms, part_name)
                score_func = SCORE_MAP[part_name]

                if score_func == "sharpness":
                    score = self._sharpeness_score(img, mask)
                elif score_func == "eyes_open":
                    score = self._eyes_open_score(img, face_lms, w, h)
                elif score_func == "smile_ratio":
                    score = self._smile_score(img, face_lms, w, h)
                else:
                    score = 0.0

                # Boost score slightly if the part is large (to avoid tiny parts winning)
                area_ratio = np.sum(mask) / (h * w)
                boost = min(area_ratio * 50, 10)  # up to 10% bonus for larger parts
                part_scores[part_name].append(score + boost)

        # 4. Determine best index for each part
        best_part_idx = {}
        for part_name in BODY_PARTS.keys():
            scores = part_scores[part_name]
            if max(scores) == 0:
                best_part_idx[part_name] = base_idx
            else:
                best_part_idx[part_name] = np.argmax(scores)

        # 5. Composite: start with base, blend best parts
        composite = base_img.copy()

        for part_name, best_idx in best_part_idx.items():
            if best_idx == base_idx:
                continue  # Already in base

            src_data = image_data[best_idx]
            src_img = src_data["img"]
            src_pose = src_data["pose"]
            src_face = src_data["face"]

            # Get mask on source
            src_mask = self._get_part_mask(src_img.shape, src_pose, src_face, part_name)
            if np.sum(src_mask) < 100:
                continue

            # Align source image and mask to base
            aligned_src, M = self._align_to_base(src_img, src_pose, base_pose, base_shape)
            warped_mask = self._warp_mask(src_mask, M, base_shape)
            if np.sum(warped_mask) < 100:
                continue

            # Poisson blending
            center = self._mask_center(warped_mask)
            mask_3ch = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)
            composite = cv2.seamlessClone(aligned_src, composite, mask_3ch, center, cv2.NORMAL_CLONE)

        # Save result
        out_fd, out_path = tempfile.mkstemp(suffix=".jpg")
        os.close(out_fd)
        cv2.imwrite(out_path, composite)
        return out_path


# ============================================================
# 4. PUBLIC WRAPPER
# ============================================================
def fuse_best_parts(image_paths: List[str]) -> str:
    engine = GranularFusionEngine()
    return engine.process_images(image_paths)
