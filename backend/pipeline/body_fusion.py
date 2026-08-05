import cv2
import numpy as np
import tempfile
import os
import logging
import math
import gc
from typing import List, Dict, Tuple, Optional, Any

import mediapipe as mp

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("body_fusion")
logger.setLevel(logging.INFO)

mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Landmark indices
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
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.25,
            min_tracking_confidence=0.25,
        )
        self.segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

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
        logger.info("BodyFusionEngine initialized (robust + memory-friendly)")

    # -------------------------------------------------------------------------
    # Loading & validation
    # -------------------------------------------------------------------------
    def _load_image(self, path: str) -> Optional[np.ndarray]:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            return None
        img = cv2.imread(path)
        if img is None:
            logger.error(f"cv2.imread failed: {path}")
            return None
        # Early resize to save memory
        h, w = img.shape[:2]
        max_dim = 1280
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    def _validate_image(self, img: np.ndarray) -> bool:
        if img is None or len(img.shape) != 3 or img.shape[2] != 3:
            return False
        h, w = img.shape[:2]
        return h >= 40 and w >= 40

    # -------------------------------------------------------------------------
    # Pose & segmentation
    # -------------------------------------------------------------------------
    def _detect_pose(self, img: np.ndarray) -> Optional[Any]:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        del rgb
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks.landmark

    def _validate_pose(self, landmarks) -> bool:
        # Very relaxed – only require nose + shoulders with low visibility
        required = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER]
        for idx in required:
            if landmarks[idx].visibility < 0.25:
                return False
        # Hips are nice-to-have but not mandatory
        return True

    def _get_segmentation_mask(self, img: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.segmentation.process(rgb)
        del rgb
        mask = (result.segmentation_mask > 0.08).astype(np.uint8) * 255
        mask = self._clean_mask(mask, min_area=150)
        if np.sum(mask) < 150:
            # Fallback rough hull from all visible landmarks
            mask = self._create_rough_mask_from_pose(img)
        return mask

    def _create_rough_mask_from_pose(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        landmarks = self._detect_pose(img)
        if landmarks is None:
            return np.zeros((h, w), dtype=np.uint8)
        pts = []
        for lm in landmarks:
            if lm.visibility > 0.15:
                pts.append((int(lm.x * w), int(lm.y * h)))
        if len(pts) < 3:
            return np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(np.array(pts, dtype=np.int32))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        mask = cv2.dilate(mask, kernel)
        return self._clean_mask(mask, min_area=100)

    def _clean_mask(self, mask: np.ndarray, min_area: int = 150) -> np.ndarray:
        if mask is None or np.sum(mask) == 0:
            return np.zeros_like(mask) if mask is not None else np.zeros((1, 1), dtype=np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = self._fill_holes(mask)
        return self._largest_contour_mask(mask, min_area)

    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        mask_ext = np.zeros((h + 2, w + 2), dtype=np.uint8)
        mask_ext[1:-1, 1:-1] = mask
        cv2.floodFill(mask_ext, None, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(mask_ext)
        return mask | im_floodfill_inv[1:-1, 1:-1]

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
    # Part masks
    # -------------------------------------------------------------------------
    def _generate_part_masks(self, img_shape, landmarks, silhouette):
        h, w = img_shape[:2]
        masks = {p: np.zeros((h, w), dtype=np.uint8) for p in PART_LANDMARKS}
        if np.sum(silhouette) == 0:
            return masks

        lm_px = {
            i: (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for i in range(len(landmarks))
        }

        def poly_mask(indices, expand=0):
            pts = [lm_px[i] for i in indices if i in lm_px]
            if len(pts) < 3:
                return np.zeros((h, w), dtype=np.uint8)
            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [hull], 255)
            if expand > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand, expand))
                m = cv2.dilate(m, k)
            return cv2.bitwise_and(m, silhouette)

        def limb_mask(indices, width_scale=0.22):
            pts = [lm_px[i] for i in indices if i in lm_px]
            if len(pts) < 2:
                return np.zeros((h, w), dtype=np.uint8)
            p0, p1 = np.array(pts[0], dtype=np.float32), np.array(pts[1], dtype=np.float32)
            length = np.linalg.norm(p1 - p0)
            if length < 5:
                return np.zeros((h, w), dtype=np.uint8)
            width = max(6, int(length * width_scale))
            vec = (p1 - p0) / length
            perp = np.array([-vec[1], vec[0]])
            poly = np.array([
                p0 + perp * width, p0 - perp * width,
                p1 - perp * width, p1 + perp * width
            ], dtype=np.int32)
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [poly], 255)
            return cv2.bitwise_and(m, silhouette)

        def circle_mask(idx, radius_scale=0.14):
            if idx not in lm_px:
                return np.zeros((h, w), dtype=np.uint8)
            x, y = lm_px[idx]
            # Safe radius estimate
            if LEFT_SHOULDER in lm_px and RIGHT_SHOULDER in lm_px:
                shoulder = abs(lm_px[LEFT_SHOULDER][0] - lm_px[RIGHT_SHOULDER][0])
                radius = max(8, int(shoulder * radius_scale))
            else:
                radius = max(8, int(min(h, w) * 0.03))
            radius = min(radius, 70)
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(m, (x, y), radius, 255, -1)
            return cv2.bitwise_and(m, silhouette)

        masks["head"] = self._clean_mask(poly_mask([NOSE, LEFT_SHOULDER, RIGHT_SHOULDER], 8), 80)
        masks["neck"] = self._clean_mask(poly_mask([LEFT_SHOULDER, RIGHT_SHOULDER, NOSE]), 40)
        masks["torso"] = self._clean_mask(poly_mask([LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_HIP, LEFT_HIP]), 200)
        masks["left_arm"] = self._clean_mask(limb_mask([LEFT_SHOULDER, LEFT_ELBOW]), 40)
        masks["right_arm"] = self._clean_mask(limb_mask([RIGHT_SHOULDER, RIGHT_ELBOW]), 40)
        masks["left_forearm"] = self._clean_mask(limb_mask([LEFT_ELBOW, LEFT_WRIST]), 25)
        masks["right_forearm"] = self._clean_mask(limb_mask([RIGHT_ELBOW, RIGHT_WRIST]), 25)
        masks["left_hand"] = self._clean_mask(circle_mask(LEFT_WRIST, 0.13), 8)
        masks["right_hand"] = self._clean_mask(circle_mask(RIGHT_WRIST, 0.13), 8)

        # Hip / pelvis
        hip_m = np.zeros((h, w), dtype=np.uint8)
        if LEFT_HIP in lm_px and RIGHT_HIP in lm_px:
            w_hip = abs(lm_px[LEFT_HIP][0] - lm_px[RIGHT_HIP][0])
            pts = [
                lm_px[LEFT_HIP], lm_px[RIGHT_HIP],
                (lm_px[RIGHT_HIP][0], lm_px[RIGHT_HIP][1] + max(w_hip, 20)),
                (lm_px[LEFT_HIP][0], lm_px[LEFT_HIP][1] + max(w_hip, 20)),
            ]
            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
            cv2.fillPoly(hip_m, [hull], 255)
            hip_m = cv2.bitwise_and(hip_m, silhouette)
        masks["hip"] = self._clean_mask(hip_m, 40)
        masks["pelvis"] = masks["hip"].copy()

        masks["left_thigh"] = self._clean_mask(limb_mask([LEFT_HIP, LEFT_KNEE], 0.28), 40)
        masks["right_thigh"] = self._clean_mask(limb_mask([RIGHT_HIP, RIGHT_KNEE], 0.28), 40)
        masks["left_leg"] = self._clean_mask(limb_mask([LEFT_KNEE, LEFT_ANKLE], 0.20), 25)
        masks["right_leg"] = self._clean_mask(limb_mask([RIGHT_KNEE, RIGHT_ANKLE], 0.20), 25)

        foot_m = np.zeros((h, w), dtype=np.uint8)
        for idx in (LEFT_ANKLE, RIGHT_ANKLE):
            foot_m = cv2.bitwise_or(foot_m, circle_mask(idx, 0.16))
        masks["feet"] = self._clean_mask(foot_m, 8)

        return masks

    # -------------------------------------------------------------------------
    # Quality scoring
    # -------------------------------------------------------------------------
    def _compute_quality_metrics(self, img: np.ndarray, landmarks) -> Dict[str, float]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return {
            "sharpness": float(np.var(lap)),
            "brightness": float(np.mean(gray)),
            "contrast": float(np.std(gray)),
            "noise": float(np.mean(np.abs(gray.astype(np.float32) - cv2.medianBlur(gray, 5).astype(np.float32)))),
            "visibility": float(np.mean([lm.visibility for lm in landmarks])),
            "pose_confidence": float(np.mean([lm.visibility for lm in landmarks])),
            "exposure": float(np.mean(gray) / 255.0),
        }

    def _score_part(self, img, landmarks, part_mask, part_name):
        if part_mask is None or np.sum(part_mask) < 40:
            return -1e9
        gray = cv2.cvtColor(cv2.bitwise_and(img, img, mask=part_mask), cv2.COLOR_BGR2GRAY)
        mask_pixels = part_mask > 0
        if not np.any(mask_pixels):
            return -1e9

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        part_sharpness = float(np.var(lap[mask_pixels]))
        part_brightness = float(np.mean(gray[mask_pixels]))
        part_contrast = float(np.std(gray[mask_pixels]))
        part_noise = float(np.mean(np.abs(gray.astype(np.float32) - cv2.medianBlur(gray, 5).astype(np.float32))[mask_pixels]))

        part_indices = PART_LANDMARKS.get(part_name, [])
        vis = float(np.mean([landmarks[i].visibility for i in part_indices if i < len(landmarks)])) if part_indices else 0.0
        area_frac = float(np.sum(part_mask)) / (img.shape[0] * img.shape[1])

        def norm(v, mx, mn=0.0):
            return max(0.0, min(1.0, (v - mn) / (mx - mn + 1e-6)))

        score = (
            self.weights["sharpness"] * norm(part_sharpness, 5000)
            + self.weights["brightness"] * norm(part_brightness, 255)
            + self.weights["contrast"] * norm(part_contrast, 128)
            + self.weights["noise"] * (1.0 - norm(part_noise, 50))
            + self.weights["visibility"] * vis
            + self.weights["pose_confidence"] * vis
            + self.weights["exposure"] * norm(part_brightness, 255)
            + self.weights["area_coverage"] * norm(area_frac, 0.25)
        )
        return score

    # -------------------------------------------------------------------------
    # Alignment & blending
    # -------------------------------------------------------------------------
    def _align_to_base(self, src_img, src_landmarks, base_landmarks, base_shape):
        h_src, w_src = src_img.shape[:2]
        h_base, w_base = base_shape
        indices = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
        src_pts = np.array([[src_landmarks[i].x * w_src, src_landmarks[i].y * h_src] for i in indices], dtype=np.float32)
        dst_pts = np.array([[base_landmarks[i].x * w_base, base_landmarks[i].y * h_base] for i in indices], dtype=np.float32)

        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if M is None:
            return cv2.resize(src_img, (w_base, h_base)), np.eye(2, 3, dtype=np.float32)

        scale = math.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
        if scale < 0.25 or scale > 4.0:
            return cv2.resize(src_img, (w_base, h_base)), np.eye(2, 3, dtype=np.float32)

        aligned = cv2.warpAffine(src_img, M, (w_base, h_base), borderMode=cv2.BORDER_REPLICATE)
        return aligned, M

    def _warp_mask(self, mask, M, shape):
        h, w = shape
        warped = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return (warped > 127).astype(np.uint8) * 255

    def _mask_center(self, mask):
        y, x = np.where(mask > 0)
        if len(x) == 0:
            return (mask.shape[1] // 2, mask.shape[0] // 2)
        return (int(np.mean(x)), int(np.mean(y)))

    def _alpha_blend(self, composite, aligned, mask):
        alpha = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        return (aligned.astype(np.float32) * alpha + composite.astype(np.float32) * (1 - alpha)).astype(np.uint8)

    def _edge_cleanup(self, img, mask):
        if np.sum(mask) < 80:
            return img
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_c = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_c = cv2.morphologyEx(mask_c, cv2.MORPH_OPEN, kernel)
        mask_blur = cv2.GaussianBlur(mask_c, (11, 11), 5)
        mask_blur = (mask_blur.astype(np.float32) / 255.0)[:, :, np.newaxis]
        smoothed = cv2.edgePreservingFilter(img, flags=1, sigma_s=50, sigma_r=0.4)
        return (img.astype(np.float32) * (1 - mask_blur) + smoothed.astype(np.float32) * mask_blur).astype(np.uint8)

    def _blend_part(self, composite, aligned_part, warped_mask, part_name):
        if np.sum(warped_mask) < 80:
            return composite
        center = self._mask_center(warped_mask)
        h, w = composite.shape[:2]
        if not (0 <= center[0] < w and 0 <= center[1] < h):
            return self._edge_cleanup(self._alpha_blend(composite, aligned_part, warped_mask), warped_mask)

        mask_3ch = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)
        try:
            blended = cv2.seamlessClone(aligned_part, composite, mask_3ch, center, cv2.NORMAL_CLONE)
            return self._edge_cleanup(blended, warped_mask)
        except cv2.error:
            return self._edge_cleanup(self._alpha_blend(composite, aligned_part, warped_mask), warped_mask)

    def _post_process(self, img, base_img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        base_lab = cv2.cvtColor(base_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_base, _, _ = cv2.split(base_lab)
        mean_l, std_l = np.mean(l), np.std(l)
        mean_b, std_b = np.mean(l_base), np.std(l_base)
        l = ((l.astype(np.float32) - mean_l) / (std_l + 1e-6)) * std_b + mean_b
        l = np.clip(l, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32) / 5.0
        return cv2.filter2D(img, -1, kernel)

    # -------------------------------------------------------------------------
    # Main pipeline
    # -------------------------------------------------------------------------
    def fuse(self, image_paths: List[str]) -> str:
        logger.info(f"Starting fusion with {len(image_paths)} images")

        # ---------- First pass: collect usable images ----------
        candidates = []
        for path in image_paths:
            logger.info(f"Checking: {path}")
            img = self._load_image(path)
            if img is None or not self._validate_image(img):
                logger.warning("  → rejected: load / size")
                continue

            landmarks = self._detect_pose(img)
            if landmarks is None:
                logger.warning("  → rejected: pose not detected")
                del img
                continue

            if not self._validate_pose(landmarks):
                logger.warning("  → rejected: low landmark visibility")
                del img, landmarks
                continue

            silhouette = self._get_segmentation_mask(img)
            area = int(np.sum(silhouette > 0))
            if area < 150:
                logger.warning(f"  → rejected: silhouette too small ({area})")
                del img, landmarks, silhouette
                continue

            quality = self._compute_quality_metrics(img, landmarks)
            candidates.append({
                "path": path,
                "img": img,
                "landmarks": landmarks,
                "silhouette": silhouette,
                "quality": quality,
            })
            logger.info(f"  → ACCEPTED (vis={quality['visibility']:.2f}, area={area})")
            gc.collect()

        if not candidates:
            raise ValueError(
                "No valid images with detectable pose and segmentation.\n"
                "Tips:\n"
                "  • Use clear full-body or upper-body photos\n"
                "  • Avoid extreme side views / heavy occlusion\n"
                "  • Convert HEIC/WebP → JPG first\n"
                "  • Check the log lines above for the exact rejection reason"
            )

        logger.info(f"Accepted {len(candidates)} images")

        # ---------- Select base ----------
        base_idx = max(
            range(len(candidates)),
            key=lambda i: candidates[i]["quality"]["visibility"] * 0.6
            + candidates[i]["quality"]["sharpness"] / 8000.0,
        )
        base = candidates[base_idx]
        logger.info(f"Base image: {base['path']}")

        base_img = base["img"].copy()
        base_shape = base_img.shape[:2]
        base_landmarks = base["landmarks"]
        base_silhouette = base["silhouette"]
        composite = base_img.copy()

        # ---------- Score every part once ----------
        part_best = {p: {"score": -1e9, "idx": base_idx, "mask": None} for p in PART_LANDMARKS}

        for idx, data in enumerate(candidates):
            masks = self._generate_part_masks(data["img"].shape, data["landmarks"], data["silhouette"])
            for part_name, mask in masks.items():
                if mask is None or np.sum(mask) < 40:
                    continue
                score = self._score_part(data["img"], data["landmarks"], mask, part_name)
                if score > part_best[part_name]["score"]:
                    part_best[part_name] = {
                        "score": score,
                        "idx": idx,
                        "mask": mask.copy(),
                    }
            # free temporary masks
            del masks
            gc.collect()

        # ---------- Blend better parts ----------
        for part_name, info in part_best.items():
            if info["idx"] == base_idx or info["score"] < 0.05:
                continue

            src = candidates[info["idx"]]
            logger.info(f"Blending {part_name} from {src['path']} (score={info['score']:.3f})")

            aligned, M = self._align_to_base(
                src["img"], src["landmarks"], base_landmarks, base_shape
            )
            warped = self._warp_mask(info["mask"], M, base_shape)
            warped = cv2.bitwise_and(warped, base_silhouette)

            if np.sum(warped) > 80:
                composite = self._blend_part(composite, aligned, warped, part_name)

            del aligned, warped
            gc.collect()

        # ---------- Finish ----------
        composite = self._post_process(composite, base_img)

        out_fd, out_path = tempfile.mkstemp(suffix=".jpg")
        os.close(out_fd)
        cv2.imwrite(out_path, composite, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"Fusion complete → {out_path}")
        return out_path


def fuse_best_parts(image_paths: List[str]) -> str:
    engine = BodyFusionEngine()
    return engine.fuse(image_paths)
