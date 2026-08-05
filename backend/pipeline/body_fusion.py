import cv2
import numpy as np
import tempfile
import os
import logging
import math
from typing import List, Dict, Tuple, Optional, Any

# MediaPipe
import mediapipe as mp

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# -----------------------------------------------------------------------------
# MediaPipe solutions
# -----------------------------------------------------------------------------
mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# -----------------------------------------------------------------------------
# Landmark indices (MediaPipe Pose)
# -----------------------------------------------------------------------------
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# -----------------------------------------------------------------------------
# Part definitions
# -----------------------------------------------------------------------------
PART_LANDMARKS = {
    "head": [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER],
    "neck": [LEFT_SHOULDER, RIGHT_SHOULDER, NOSE],
    "torso": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP],
    "left_arm": [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST],
    "right_arm": [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST],
    "left_forearm": [LEFT_ELBOW, LEFT_WRIST],
    "right_forearm": [RIGHT_ELBOW, RIGHT_WRIST],
    "left_hand": [LEFT_WRIST],
    "right_hand": [RIGHT_WRIST],
    "hip": [LEFT_HIP, RIGHT_HIP],
    "pelvis": [LEFT_HIP, RIGHT_HIP],
    "left_thigh": [LEFT_HIP, LEFT_KNEE],
    "right_thigh": [RIGHT_HIP, RIGHT_KNEE],
    "left_leg": [LEFT_KNEE, LEFT_ANKLE],
    "right_leg": [RIGHT_KNEE, RIGHT_ANKLE],
    "feet": [LEFT_ANKLE, RIGHT_ANKLE],
}


class BodyFusionEngine:
    """
    Production-grade body fusion engine that reconstructs the best composite
    image from multiple photos of the same person.
    """

    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.segmentation = mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
        self.weights = {
            "sharpness": 0.25,
            "brightness": 0.10,
            "contrast": 0.15,
            "noise": -0.10,
            "visibility": 0.20,
            "pose_confidence": 0.10,
            "exposure": 0.05,
            "area_coverage": 0.15,
        }
        logger.info("BodyFusionEngine initialized.")

    # -------------------------------------------------------------------------
    #  Image loading & validation
    # -------------------------------------------------------------------------
    def _load_image(self, path: str) -> Optional[np.ndarray]:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            return None
        img = cv2.imread(path)
        if img is None:
            logger.error(f"Failed to load image: {path}")
            return None
        return img

    def _validate_image(self, img: np.ndarray) -> bool:
        if img is None:
            return False
        if len(img.shape) != 3 or img.shape[2] != 3:
            logger.warning("Image is not BGR (3 channels). Rejecting.")
            return False
        h, w = img.shape[:2]
        if h < 50 or w < 50:
            logger.warning(f"Image too small: {h}x{w}. Rejecting.")
            return False
        if h > 5000 or w > 5000:
            logger.warning(f"Image too large: {h}x{w}. Rejecting.")
            return False
        if np.all(img == 0) or np.all(img == 255):
            logger.warning("Image is completely black or white. Rejecting.")
            return False
        return True

    def _resize_if_needed(self, img: np.ndarray, max_dim: int = 2048) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) <= max_dim:
            return img
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # -------------------------------------------------------------------------
    #  Pose detection
    # -------------------------------------------------------------------------
    def _detect_pose(self, img: np.ndarray) -> Optional[Any]:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            logger.warning("Pose not detected.")
            return None
        return result.pose_landmarks.landmark

    def _validate_pose(self, landmarks) -> bool:
        required = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
        for idx in required:
            lm = landmarks[idx]
            if lm.visibility < 0.5:
                logger.warning(f"Landmark {idx} visibility low: {lm.visibility}")
                return False
        return True

    # -------------------------------------------------------------------------
    #  Selfie Segmentation
    # -------------------------------------------------------------------------
    def _get_segmentation_mask(self, img: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.segmentation.process(rgb)
        mask = (result.segmentation_mask > 0.1).astype(np.uint8) * 255
        return self._clean_mask(mask)

    # -------------------------------------------------------------------------
    #  Mask cleaning utilities
    # -------------------------------------------------------------------------
    def _clean_mask(self, mask: np.ndarray, min_area: int = 500) -> np.ndarray:
        if mask is None or np.sum(mask) == 0:
            return np.zeros_like(mask) if mask is not None else np.zeros((1, 1), dtype=np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = self._fill_holes(mask)
        mask = self._largest_contour_mask(mask, min_area)
        return mask

    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        mask_ext = np.zeros((h + 2, w + 2), dtype=np.uint8)
        mask_ext[1:-1, 1:-1] = mask
        _, im_floodfill, _, _ = cv2.floodFill(mask_ext, None, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        mask_filled = mask | im_floodfill_inv[1:-1, 1:-1]
        return mask_filled

    def _largest_contour_mask(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(mask)
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < min_area:
            return np.zeros_like(mask)
        out = np.zeros_like(mask)
        cv2.drawContours(out, [largest], -1, 255, -1)
        return out

    # -------------------------------------------------------------------------
    #  Helper geometry functions
    # -------------------------------------------------------------------------
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _rotation_matrix(self, angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, -s], [s, c]])

    def _bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        y, x = np.where(mask > 0)
        if len(x) == 0:
            return (0, 0, 0, 0)
        x1, x2 = int(np.min(x)), int(np.max(x))
        y1, y2 = int(np.min(y)), int(np.max(y))
        return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)

    def _safe_rect(self, x: int, y: int, w: int, h: int, max_w: int, max_h: int) -> Tuple[int, int, int, int]:
        x = max(0, min(x, max_w - 1))
        y = max(0, min(y, max_h - 1))
        w = max(1, min(w, max_w - x))
        h = max(1, min(h, max_h - y))
        return (x, y, w, h)

    def _safe_crop(self, img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        h_img, w_img = img.shape[:2]
        x, y, w, h = self._safe_rect(x, y, w, h, w_img, h_img)
        return img[y:y + h, x:x + w]

    def _safe_roi(self, img: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
        if w <= 0 or h <= 0:
            return None
        try:
            return img[y:y + h, x:x + w]
        except Exception:
            return None

    def _clamp(self, val: int, min_val: int, max_val: int) -> int:
        return max(min_val, min(val, max_val))

    def _mask_center(self, mask: np.ndarray) -> Tuple[int, int]:
        h, w = mask.shape
        y, x = np.where(mask > 0)
        if len(x) == 0:
            return (w // 2, h // 2)
        return (int(np.mean(x)), int(np.mean(y)))

    # -------------------------------------------------------------------------
    #  Generate body part masks
    # -------------------------------------------------------------------------
    def _generate_part_masks(
        self,
        img_shape: Tuple[int, int],
        landmarks,
        silhouette: np.ndarray
    ) -> Dict[str, np.ndarray]:
        h, w = img_shape[:2]
        masks = {}
        if np.sum(silhouette) == 0:
            return {part: np.zeros((h, w), dtype=np.uint8) for part in PART_LANDMARKS}

        lm_px = {
            idx: (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            for idx in range(len(landmarks))
        }

        # ---- Head ----
        head_mask = np.zeros((h, w), dtype=np.uint8)
        if all(k in lm_px for k in [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER]):
            nose = np.array(lm_px[NOSE], dtype=np.float32)
            left = np.array(lm_px[LEFT_SHOULDER], dtype=np.float32)
            right = np.array(lm_px[RIGHT_SHOULDER], dtype=np.float32)
            center = (left + right) / 2.0
            vec = center - nose
            norm = np.linalg.norm(vec)
            if norm > 1e-3:
                top = nose - (vec / norm) * (norm * 1.3)
                pts_px = [
                    tuple(left.astype(int)),
                    tuple(right.astype(int)),
                    tuple(top.astype(int)),
                    tuple(nose.astype(int)),
                ]
            else:
                pts_px = [lm_px[NOSE], lm_px[LEFT_SHOULDER], lm_px[RIGHT_SHOULDER]]

            if len(pts_px) >= 3:
                hull = cv2.convexHull(np.array(pts_px, dtype=np.int32))
                cv2.fillPoly(head_mask, [hull], 255)
                head_mask = cv2.bitwise_and(head_mask, silhouette)
                shoulder_y = int((lm_px[LEFT_SHOULDER][1] + lm_px[RIGHT_SHOULDER][1]) / 2)
                head_mask[shoulder_y:, :] = 0
                masks["head"] = self._clean_mask(head_mask, min_area=200)
            else:
                masks["head"] = np.zeros((h, w), dtype=np.uint8)
        else:
            masks["head"] = np.zeros((h, w), dtype=np.uint8)

        # ---- Neck ----
        neck_mask = np.zeros((h, w), dtype=np.uint8)
        if all(k in lm_px for k in [LEFT_SHOULDER, RIGHT_SHOULDER, NOSE]):
            shoulder_center = (
                (lm_px[LEFT_SHOULDER][0] + lm_px[RIGHT_SHOULDER][0]) // 2,
                (lm_px[LEFT_SHOULDER][1] + lm_px[RIGHT_SHOULDER][1]) // 2,
            )
            nose = lm_px[NOSE]
            neck_pts = [
                lm_px[LEFT_SHOULDER],
                lm_px[RIGHT_SHOULDER],
                (shoulder_center[0] + 20, max(0, shoulder_center[1] - 40)),
                (nose[0], nose[1]),
            ]
            if len(neck_pts) >= 3:
                hull = cv2.convexHull(np.array(neck_pts, dtype=np.int32))
                cv2.fillPoly(neck_mask, [hull], 255)
                neck_mask = cv2.bitwise_and(neck_mask, silhouette)
                masks["neck"] = self._clean_mask(neck_mask, min_area=100)
            else:
                masks["neck"] = np.zeros((h, w), dtype=np.uint8)
        else:
            masks["neck"] = np.zeros((h, w), dtype=np.uint8)

        # ---- Torso ----
        torso_mask = np.zeros((h, w), dtype=np.uint8)
        if all(k in lm_px for k in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]):
            pts = [
                lm_px[LEFT_SHOULDER],
                lm_px[RIGHT_SHOULDER],
                lm_px[RIGHT_HIP],
                lm_px[LEFT_HIP],
            ]
            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
            cv2.fillPoly(torso_mask, [hull], 255)
            torso_mask = cv2.bitwise_and(torso_mask, silhouette)
            masks["torso"] = self._clean_mask(torso_mask, min_area=500)
        else:
            masks["torso"] = np.zeros((h, w), dtype=np.uint8)

        # ---- Arms / forearms ----
        left_arm_mask = self._create_limb_mask(
            lm_px, silhouette, [LEFT_SHOULDER, LEFT_ELBOW], width_scale=0.25
        )
        masks["left_arm"] = self._clean_mask(left_arm_mask, min_area=100)

        right_arm_mask = self._create_limb_mask(
            lm_px, silhouette, [RIGHT_SHOULDER, RIGHT_ELBOW], width_scale=0.25
        )
        masks["right_arm"] = self._clean_mask(right_arm_mask, min_area=100)

        left_forearm_mask = self._create_limb_mask(
            lm_px, silhouette, [LEFT_ELBOW, LEFT_WRIST], width_scale=0.20
        )
        masks["left_forearm"] = self._clean_mask(left_forearm_mask, min_area=50)

        right_forearm_mask = self._create_limb_mask(
            lm_px, silhouette, [RIGHT_ELBOW, RIGHT_WRIST], width_scale=0.20
        )
        masks["right_forearm"] = self._clean_mask(right_forearm_mask, min_area=50)

        # ---- Hands (FIXED) ----
        left_hand_mask = self._create_hand_mask(
            lm_px, silhouette, LEFT_WRIST, radius_scale=0.15
        )
        masks["left_hand"] = self._clean_mask(left_hand_mask, min_area=30)

        right_hand_mask = self._create_hand_mask(
            lm_px, silhouette, RIGHT_WRIST, radius_scale=0.15
        )
        masks["right_hand"] = self._clean_mask(right_hand_mask, min_area=30)

        # ---- Hip / pelvis ----
        hip_mask = np.zeros((h, w), dtype=np.uint8)
        if all(k in lm_px for k in [LEFT_HIP, RIGHT_HIP]):
            width = abs(lm_px[LEFT_HIP][0] - lm_px[RIGHT_HIP][0])
            pts = [
                lm_px[LEFT_HIP],
                lm_px[RIGHT_HIP],
                (lm_px[RIGHT_HIP][0], lm_px[RIGHT_HIP][1] + max(width, 30)),
                (lm_px[LEFT_HIP][0], lm_px[LEFT_HIP][1] + max(width, 30)),
            ]
            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
            cv2.fillPoly(hip_mask, [hull], 255)
            hip_mask = cv2.bitwise_and(hip_mask, silhouette)
            masks["hip"] = self._clean_mask(hip_mask, min_area=100)
        else:
            masks["hip"] = np.zeros((h, w), dtype=np.uint8)

        masks["pelvis"] = masks["hip"].copy()

        # ---- Thighs ----
        left_thigh_mask = self._create_limb_mask(
            lm_px, silhouette, [LEFT_HIP, LEFT_KNEE], width_scale=0.30
        )
        masks["left_thigh"] = self._clean_mask(left_thigh_mask, min_area=100)

        right_thigh_mask = self._create_limb_mask(
            lm_px, silhouette, [RIGHT_HIP, RIGHT_KNEE], width_scale=0.30
        )
        masks["right_thigh"] = self._clean_mask(right_thigh_mask, min_area=100)

        # ---- Lower legs ----
        left_leg_mask = self._create_limb_mask(
            lm_px, silhouette, [LEFT_KNEE, LEFT_ANKLE], width_scale=0.20
        )
        masks["left_leg"] = self._clean_mask(left_leg_mask, min_area=80)

        right_leg_mask = self._create_limb_mask(
            lm_px, silhouette, [RIGHT_KNEE, RIGHT_ANKLE], width_scale=0.20
        )
        masks["right_leg"] = self._clean_mask(right_leg_mask, min_area=80)

        # ---- Feet (FIXED) ----
        left_foot_mask = self._create_hand_mask(
            lm_px, silhouette, LEFT_ANKLE, radius_scale=0.18
        )
        right_foot_mask = self._create_hand_mask(
            lm_px, silhouette, RIGHT_ANKLE, radius_scale=0.18
        )
        feet_mask = cv2.bitwise_or(left_foot_mask, right_foot_mask)
        masks["feet"] = self._clean_mask(feet_mask, min_area=30)

        for k in list(masks.keys()):
            if masks[k] is None:
                masks[k] = np.zeros((h, w), dtype=np.uint8)
            else:
                masks[k] = masks[k].astype(np.uint8)

        return masks

    def _create_limb_mask(
        self,
        lm_px: Dict[int, Tuple[int, int]],
        silhouette: np.ndarray,
        point_indices: List[int],
        width_scale: float = 0.2,
    ) -> np.ndarray:
        h, w = silhouette.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(point_indices) < 2:
            return mask
        pts = [lm_px[idx] for idx in point_indices if idx in lm_px]
        if len(pts) < 2:
            return mask

        p0 = np.array(pts[0], dtype=np.float32)
        p1 = np.array(pts[1], dtype=np.float32)
        length = np.linalg.norm(p1 - p0)
        if length < 5:
            return mask

        width = max(3, int(length * width_scale))
        vec = (p1 - p0) / length
        perp = np.array([-vec[1], vec[0]])

        pts_poly = np.array(
            [
                p0 + perp * width,
                p0 - perp * width,
                p1 - perp * width,
                p1 + perp * width,
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts_poly], 255)
        mask = cv2.bitwise_and(mask, silhouette)
        return mask

    def _create_hand_mask(
        self,
        lm_px: Dict[int, Tuple[int, int]],
        silhouette: np.ndarray,
        point_idx: int,
        radius_scale: float = 0.15,
    ) -> np.ndarray:
        """Circular mask around wrist or ankle. Radius scaled by shoulder width."""
        h, w = silhouette.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if point_idx not in lm_px:
            return mask

        cx, cy = lm_px[point_idx]

        if LEFT_SHOULDER in lm_px and RIGHT_SHOULDER in lm_px:
            shoulder_dist = self._distance(lm_px[LEFT_SHOULDER], lm_px[RIGHT_SHOULDER])
            radius = int(shoulder_dist * radius_scale)
        else:
            radius = int(min(h, w) * 0.035)

        radius = max(12, min(radius, 90))

        cv2.circle(mask, (int(cx), int(cy)), radius, 255, -1)
        mask = cv2.bitwise_and(mask, silhouette)
        return mask

    # -------------------------------------------------------------------------
    #  Quality analysis
    # -------------------------------------------------------------------------
    def _compute_quality_metrics(self, img: np.ndarray, landmarks) -> Dict[str, float]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(np.var(laplacian))
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        median = cv2.medianBlur(gray, 5)
        noise = float(np.mean(np.abs(gray.astype(np.float32) - median.astype(np.float32))))
        visibilities = [lm.visibility for lm in landmarks]
        avg_visibility = float(np.mean(visibilities))
        return {
            "sharpness": sharpness,
            "brightness": brightness,
            "contrast": contrast,
            "noise": noise,
            "visibility": avg_visibility,
            "pose_confidence": avg_visibility,
            "exposure": brightness / 255.0,
            "area_coverage": 0.0,
        }

    def _score_part(
        self,
        img: np.ndarray,
        landmarks,
        part_mask: np.ndarray,
        part_name: str,
        metrics: Dict[str, float],
    ) -> float:
        if part_mask is None or np.sum(part_mask) < 50:
            return -1e9

        masked = cv2.bitwise_and(img, img, mask=part_mask)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        mask_pixels = part_mask > 0
        if not np.any(mask_pixels):
            return -1e9

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        part_sharpness = float(np.var(lap[mask_pixels])) if np.any(mask_pixels) else 0.0
        part_brightness = float(np.mean(gray[mask_pixels]))
        part_contrast = float(np.std(gray[mask_pixels]))
        median = cv2.medianBlur(gray, 5)
        part_noise = float(
            np.mean(np.abs(gray.astype(np.float32) - median.astype(np.float32))[mask_pixels])
        )

        part_indices = PART_LANDMARKS.get(part_name, [])
        if part_indices:
            vis = [landmarks[i].visibility for i in part_indices if i < len(landmarks)]
            part_vis = float(np.mean(vis)) if vis else 0.0
        else:
            part_vis = 0.0

        area_frac = float(np.sum(part_mask)) / (img.shape[0] * img.shape[1])

        def normalize(val: float, max_val: float, min_val: float = 0.0) -> float:
            return max(0.0, min(1.0, (val - min_val) / (max_val - min_val + 1e-6)))

        sharpness_norm = normalize(part_sharpness, 5000.0)
        brightness_norm = normalize(part_brightness, 255.0)
        contrast_norm = normalize(part_contrast, 128.0)
        noise_norm = 1.0 - normalize(part_noise, 50.0)
        vis_norm = part_vis
        area_norm = normalize(area_frac, 0.3)

        score = (
            self.weights["sharpness"] * sharpness_norm
            + self.weights["brightness"] * brightness_norm
            + self.weights["contrast"] * contrast_norm
            + self.weights["noise"] * noise_norm
            + self.weights["visibility"] * vis_norm
            + self.weights["pose_confidence"] * vis_norm
            + self.weights["exposure"] * brightness_norm
            + self.weights["area_coverage"] * area_norm
        )
        return score

    # -------------------------------------------------------------------------
    #  Alignment
    # -------------------------------------------------------------------------
    def _align_to_base(
        self,
        src_img: np.ndarray,
        src_landmarks,
        base_landmarks,
        base_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        h_src, w_src = src_img.shape[:2]
        h_base, w_base = base_shape

        indices = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
        src_pts = []
        dst_pts = []
        for idx in indices:
            lm_src = src_landmarks[idx]
            lm_base = base_landmarks[idx]
            src_pts.append([lm_src.x * w_src, lm_src.y * h_src])
            dst_pts.append([lm_base.x * w_base, lm_base.y * h_base])

        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if M is None:
            logger.warning("Affine estimation failed, using resize fallback.")
            aligned = cv2.resize(src_img, (w_base, h_base))
            return aligned, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        scale = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
        if scale < 0.3 or scale > 3.0:
            logger.warning(f"Transform scale {scale:.2f} extreme; using resize fallback.")
            aligned = cv2.resize(src_img, (w_base, h_base))
            return aligned, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        aligned = cv2.warpAffine(
            src_img, M, (w_base, h_base), borderMode=cv2.BORDER_REPLICATE
        )
        return aligned, M

    def _warp_mask(self, mask: np.ndarray, M: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        warped = cv2.warpAffine(
            mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        warped = (warped > 127).astype(np.uint8) * 255
        return warped

    # -------------------------------------------------------------------------
    #  Blending
    # -------------------------------------------------------------------------
    def _blend_part(
        self,
        composite: np.ndarray,
        aligned_part: np.ndarray,
        warped_mask: np.ndarray,
        base_img: np.ndarray,
        part_name: str,
    ) -> np.ndarray:
        if np.sum(warped_mask) < 100:
            return composite

        center = self._mask_center(warped_mask)
        h, w = composite.shape[:2]
        if not (0 <= center[0] < w and 0 <= center[1] < h):
            logger.warning(
                f"Invalid center {center} for part {part_name}; using alpha blend fallback."
            )
            return self._alpha_blend(composite, aligned_part, warped_mask)

        mask_3ch = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)

        try:
            blended = cv2.seamlessClone(
                aligned_part, composite, mask_3ch, center, cv2.NORMAL_CLONE
            )
            blended = self._edge_cleanup(blended, warped_mask)
            return blended
        except cv2.error as e:
            logger.warning(
                f"seamlessClone failed for {part_name}: {e}. Using alpha blend fallback."
            )
            blended = self._alpha_blend(composite, aligned_part, warped_mask)
            return self._edge_cleanup(blended, warped_mask)

    def _alpha_blend(
        self, composite: np.ndarray, aligned: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        alpha = mask.astype(np.float32) / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        blended = (
            aligned.astype(np.float32) * alpha
            + composite.astype(np.float32) * (1.0 - alpha)
        )
        return blended.astype(np.uint8)

    def _edge_cleanup(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if np.sum(mask) < 100:
            return img

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
        mask_blur = cv2.GaussianBlur(mask_clean, (11, 11), 5)
        mask_blur = mask_blur.astype(np.float32) / 255.0
        mask_blur = np.expand_dims(mask_blur, axis=2)

        smoothed = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
        band = np.logical_and(mask_blur > 0.1, mask_blur < 0.9).astype(np.float32)

        result = img.copy().astype(np.float32)
        result[band > 0] = (
            smoothed[band > 0] * 0.5 + img[band > 0] * 0.5
        ).astype(np.float32)
        return result.astype(np.uint8)

    # -------------------------------------------------------------------------
    #  Post-processing
    # -------------------------------------------------------------------------
    def _post_process(self, img: np.ndarray, base_img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        base_lab = cv2.cvtColor(base_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_base, _, _ = cv2.split(base_lab)

        mean_l = np.mean(l)
        std_l = np.std(l)
        mean_base = np.mean(l_base)
        std_base = np.std(l_base)
        l = ((l - mean_l) / (std_l + 1e-6)) * std_base + mean_base
        l = np.clip(l, 0, 255).astype(np.uint8)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        kernel = np.array(
            [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32
        ) / 5.0
        img = cv2.filter2D(img, -1, kernel)
        return img

    # -------------------------------------------------------------------------
    #  Main fusion pipeline
    # -------------------------------------------------------------------------
    def fuse(self, image_paths: List[str]) -> str:
        logger.info(f"Starting fusion with {len(image_paths)} images.")

        image_data = []
        for path in image_paths:
            img = self._load_image(path)
            if img is None:
                continue
            if not self._validate_image(img):
                continue
            img = self._resize_if_needed(img)
            landmarks = self._detect_pose(img)
            if landmarks is None:
                continue
            if not self._validate_pose(landmarks):
                continue
            silhouette = self._get_segmentation_mask(img)
            if np.sum(silhouette) < 1000:
                logger.warning(f"Silhouette too small for {path}")
                continue
            quality = self._compute_quality_metrics(img, landmarks)
            part_masks = self._generate_part_masks(img.shape, landmarks, silhouette)
            image_data.append(
                {
                    "path": path,
                    "img": img,
                    "landmarks": landmarks,
                    "silhouette": silhouette,
                    "quality": quality,
                    "part_masks": part_masks,
                }
            )

        if not image_data:
            raise ValueError("No valid images with detectable pose and segmentation.")

        logger.info(f"Loaded {len(image_data)} valid images.")

        base_idx = max(
            range(len(image_data)),
            key=lambda i: (
                image_data[i]["quality"]["visibility"] * 0.6
                + image_data[i]["quality"]["sharpness"] / 10000.0
            ),
        )
        base = image_data[base_idx]
        logger.info(f"Selected base image: {base['path']}")
        base_img = base["img"].copy()
        base_shape = base_img.shape[:2]
        base_landmarks = base["landmarks"]
        base_silhouette = base["silhouette"]
        composite = base_img.copy()

        for part_name in PART_LANDMARKS.keys():
            logger.info(f"Processing part: {part_name}")
            candidates = []
            for data in image_data:
                mask = data["part_masks"].get(part_name)
                if mask is None or np.sum(mask) < 200:
                    continue
                score = self._score_part(
                    data["img"],
                    data["landmarks"],
                    mask,
                    part_name,
                    data["quality"],
                )
                candidates.append((data, mask, score))

            if not candidates:
                logger.warning(f"No valid candidate for part {part_name}, skipping.")
                continue

            best_data, best_mask, best_score = max(candidates, key=lambda x: x[2])
            if best_data is base or best_score < 0.1:
                logger.info(
                    f"Part {part_name}: using base or score too low ({best_score:.2f})"
                )
                continue

            logger.info(
                f"Part {part_name}: selected from {best_data['path']} with score {best_score:.2f}"
            )

            aligned_img, M = self._align_to_base(
                best_data["img"],
                best_data["landmarks"],
                base_landmarks,
                base_shape,
            )

            warped_mask = self._warp_mask(best_mask, M, base_shape)
            warped_mask = cv2.bitwise_and(warped_mask, base_silhouette)

            if np.sum(warped_mask) < 200:
                logger.warning(f"Warped mask too small for {part_name}, skipping.")
                continue

            composite = self._blend_part(
                composite, aligned_img, warped_mask, base_img, part_name
            )

        composite = self._post_process(composite, base_img)

        out_fd, out_path = tempfile.mkstemp(suffix=".jpg")
        os.close(out_fd)
        cv2.imwrite(out_path, composite)
        logger.info(f"Fusion complete. Output saved to {out_path}")
        return out_path


def fuse_best_parts(image_paths: List[str]) -> str:
    """Public API endpoint."""
    engine = BodyFusionEngine()
    return engine.fuse(image_paths)


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     paths = ["photo1.jpg", "photo2.jpg", "photo3.jpg"]
#     result = fuse_best_parts(paths)
#     print("Result saved to:", result)
