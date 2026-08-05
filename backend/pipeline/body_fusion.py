import cv2
import numpy as np
import tempfile
import os
import mediapipe as mp
from typing import List, Dict, Tuple, Optional, Any

# MediaPipe solutions
mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Landmark indices (common names)
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

# Part definitions: list of landmark indices used to compute the part center
PART_LANDMARKS = {
    "head": [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER],
    "torso": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP],
    "left_arm": [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST],
    "right_arm": [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST],
    "left_leg": [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE],
    "right_leg": [RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE],
}


class BodyFusionEngine:
    """Production‑ready body part fusion engine."""

    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.segmentation = mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )

    # ----------------------------------------------------------------------
    #  Image loading & preprocessing
    # ----------------------------------------------------------------------
    def _load_image(self, path: str) -> Optional[np.ndarray]:
        """Load image and convert to BGR. Return None if invalid."""
        img = cv2.imread(path)
        if img is None:
            return None
        return img

    def _get_segmentation_mask(self, img: np.ndarray) -> np.ndarray:
        """Return binary silhouette mask (0/255) from selfie segmentation."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.segmentation.process(rgb)
        mask = (result.segmentation_mask > 0.1).astype(np.uint8) * 255
        return mask

    def _detect_pose(self, img: np.ndarray) -> Optional[Any]:
        """Run pose detection; return landmarks object or None."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks.landmark

    def _compute_part_centers(self, landmarks) -> Dict[str, Tuple[float, float]]:
        """Compute the mean (x, y) of the landmarks for each part."""
        centers = {}
        h, w = 1, 1  # normalized coordinates
        for part, indices in PART_LANDMARKS.items():
            xs = [landmarks[i].x for i in indices]
            ys = [landmarks[i].y for i in indices]
            centers[part] = (float(np.mean(xs)), float(np.mean(ys)))
        return centers

    def _compute_quality_scores(
        self, img: np.ndarray, landmarks
    ) -> Dict[str, float]:
        """
        Compute quality metrics for the entire image.
        Returns a dict with: sharpness, brightness, pose_confidence, visibility.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Sharpness: variance of Laplacian
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Brightness: mean pixel value
        brightness = np.mean(gray)
        # Pose confidence: average visibility of all landmarks
        visibilities = [lm.visibility for lm in landmarks]
        avg_visibility = float(np.mean(visibilities))
        # Overall confidence (we also consider detection confidence, but it's not per-landmark)
        # We'll just combine into a score.
        return {
            "sharpness": laplacian_var,
            "brightness": brightness,
            "visibility": avg_visibility,
            "pose_confidence": avg_visibility,  # proxy
        }

    # ----------------------------------------------------------------------
    #  Part mask generation (using segmentation + landmark-based division)
    # ----------------------------------------------------------------------
    def _generate_part_masks(
        self, img_shape: Tuple[int, int], landmarks, silhouette: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Create binary mask (0/255) for each body part.
        Uses silhouette as base and splits it using landmark positions.
        """
        h, w = img_shape[:2]
        masks = {part: np.zeros((h, w), dtype=np.uint8) for part in PART_LANDMARKS}

        # If silhouette is empty, return zeros
        if np.sum(silhouette) == 0:
            return masks

        # Get landmark coordinates in pixel space
        lm_coords = {}
        for idx in range(len(landmarks)):
            lm = landmarks[idx]
            lm_coords[idx] = (int(lm.x * w), int(lm.y * h))

        # Horizontal cut lines: shoulders and hips
        shoulder_y = int((lm_coords[LEFT_SHOULDER][1] + lm_coords[RIGHT_SHOULDER][1]) / 2)
        hip_y = int((lm_coords[LEFT_HIP][1] + lm_coords[RIGHT_HIP][1]) / 2)

        # Vertical middle: average x of shoulders and hips
        mid_x = int((lm_coords[LEFT_SHOULDER][0] + lm_coords[RIGHT_SHOULDER][0] +
                     lm_coords[LEFT_HIP][0] + lm_coords[RIGHT_HIP][0]) / 4)

        # 1. Head: silhouette above shoulder_y, but only in the upper region
        head_mask = silhouette.copy()
        # Remove everything below shoulder_y
        if shoulder_y > 0:
            head_mask[shoulder_y:, :] = 0
        # Also limit to a reasonable width around the center of shoulders
        head_center_x = int((lm_coords[LEFT_SHOULDER][0] + lm_coords[RIGHT_SHOULDER][0]) / 2)
        head_width = int((lm_coords[RIGHT_SHOULDER][0] - lm_coords[LEFT_SHOULDER][0]) * 1.2)
        if head_width > 0:
            left_bound = max(0, head_center_x - head_width // 2)
            right_bound = min(w, head_center_x + head_width // 2)
            head_mask[:, :left_bound] = 0
            head_mask[:, right_bound:] = 0
        masks["head"] = head_mask

        # 2. Torso: between shoulder_y and hip_y, but exclude arms (keep central part)
        torso_mask = silhouette.copy()
        torso_mask[:shoulder_y, :] = 0
        if hip_y < h:
            torso_mask[hip_y:, :] = 0
        # Narrow to central region (use vertical lines at shoulders)
        shoulder_left = lm_coords[LEFT_SHOULDER][0]
        shoulder_right = lm_coords[RIGHT_SHOULDER][0]
        hip_left = lm_coords[LEFT_HIP][0]
        hip_right = lm_coords[RIGHT_HIP][0]
        # Use a polygon to define torso boundaries
        poly_points = np.array([
            [shoulder_left, shoulder_y],
            [shoulder_right, shoulder_y],
            [hip_right, hip_y],
            [hip_left, hip_y]
        ], dtype=np.int32)
        torso_poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(torso_poly_mask, [poly_points], 255)
        torso_mask = cv2.bitwise_and(torso_mask, torso_poly_mask)
        masks["torso"] = torso_mask

        # 3. Left arm: from shoulder to wrist, on left side
        left_arm_mask = np.zeros((h, w), dtype=np.uint8)
        # Define a polygon: shoulder, elbow, wrist, and some width
        if all(k in lm_coords for k in [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]):
            shoulder = lm_coords[LEFT_SHOULDER]
            elbow = lm_coords[LEFT_ELBOW]
            wrist = lm_coords[LEFT_WRIST]
            # Add some width by offsetting perpendicular to the arm direction
            # Simplified: use a fixed width based on torso size
            arm_width = int(abs(shoulder[1] - hip_y) * 0.2)
            # Compute direction from shoulder to wrist
            vec = np.array(wrist) - np.array(shoulder)
            if np.linalg.norm(vec) > 0:
                perp = np.array([-vec[1], vec[0]]) / np.linalg.norm(vec) * arm_width
            else:
                perp = np.array([arm_width, 0])
            # Build a polygon around the arm points
            pts = [
                shoulder,
                shoulder + perp.astype(int),
                elbow + perp.astype(int),
                wrist + perp.astype(int),
                wrist - perp.astype(int),
                elbow - perp.astype(int),
                shoulder - perp.astype(int)
            ]
            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(left_arm_mask, [pts], 255)
        # Intersect with silhouette
        left_arm_mask = cv2.bitwise_and(left_arm_mask, silhouette)
        # Keep only left side of body (x < mid_x roughly) to avoid overlap
        left_arm_mask[:, mid_x:] = 0
        masks["left_arm"] = left_arm_mask

        # 4. Right arm: similar
        right_arm_mask = np.zeros((h, w), dtype=np.uint8)
        if all(k in lm_coords for k in [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]):
            shoulder = lm_coords[RIGHT_SHOULDER]
            elbow = lm_coords[RIGHT_ELBOW]
            wrist = lm_coords[RIGHT_WRIST]
            arm_width = int(abs(shoulder[1] - hip_y) * 0.2)
            vec = np.array(wrist) - np.array(shoulder)
            if np.linalg.norm(vec) > 0:
                perp = np.array([-vec[1], vec[0]]) / np.linalg.norm(vec) * arm_width
            else:
                perp = np.array([arm_width, 0])
            pts = [
                shoulder,
                shoulder + perp.astype(int),
                elbow + perp.astype(int),
                wrist + perp.astype(int),
                wrist - perp.astype(int),
                elbow - perp.astype(int),
                shoulder - perp.astype(int)
            ]
            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(right_arm_mask, [pts], 255)
        right_arm_mask = cv2.bitwise_and(right_arm_mask, silhouette)
        right_arm_mask[:, :mid_x] = 0
        masks["right_arm"] = right_arm_mask

        # 5. Left leg: from hip to ankle, left side
        left_leg_mask = np.zeros((h, w), dtype=np.uint8)
        if all(k in lm_coords for k in [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]):
            hip = lm_coords[LEFT_HIP]
            knee = lm_coords[LEFT_KNEE]
            ankle = lm_coords[LEFT_ANKLE]
            leg_width = int(abs(hip[1] - ankle[1]) * 0.15)
            vec = np.array(ankle) - np.array(hip)
            if np.linalg.norm(vec) > 0:
                perp = np.array([-vec[1], vec[0]]) / np.linalg.norm(vec) * leg_width
            else:
                perp = np.array([leg_width, 0])
            pts = [
                hip,
                hip + perp.astype(int),
                knee + perp.astype(int),
                ankle + perp.astype(int),
                ankle - perp.astype(int),
                knee - perp.astype(int),
                hip - perp.astype(int)
            ]
            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(left_leg_mask, [pts], 255)
        left_leg_mask = cv2.bitwise_and(left_leg_mask, silhouette)
        left_leg_mask[:, mid_x:] = 0
        masks["left_leg"] = left_leg_mask

        # 6. Right leg
        right_leg_mask = np.zeros((h, w), dtype=np.uint8)
        if all(k in lm_coords for k in [RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]):
            hip = lm_coords[RIGHT_HIP]
            knee = lm_coords[RIGHT_KNEE]
            ankle = lm_coords[RIGHT_ANKLE]
            leg_width = int(abs(hip[1] - ankle[1]) * 0.15)
            vec = np.array(ankle) - np.array(hip)
            if np.linalg.norm(vec) > 0:
                perp = np.array([-vec[1], vec[0]]) / np.linalg.norm(vec) * leg_width
            else:
                perp = np.array([leg_width, 0])
            pts = [
                hip,
                hip + perp.astype(int),
                knee + perp.astype(int),
                ankle + perp.astype(int),
                ankle - perp.astype(int),
                knee - perp.astype(int),
                hip - perp.astype(int)
            ]
            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(right_leg_mask, [pts], 255)
        right_leg_mask = cv2.bitwise_and(right_leg_mask, silhouette)
        right_leg_mask[:, :mid_x] = 0
        masks["right_leg"] = right_leg_mask

        # Clean up: remove small isolated regions (optional)
        for part in masks:
            masks[part] = self._clean_mask(masks[part])

        return masks

    def _clean_mask(self, mask: np.ndarray, min_area: int = 100) -> np.ndarray:
        """Remove small contours and keep largest connected component."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(mask)
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < min_area:
            return np.zeros_like(mask)
        cleaned = np.zeros_like(mask)
        cv2.drawContours(cleaned, [largest], -1, 255, -1)
        return cleaned

    # ----------------------------------------------------------------------
    #  Alignment
    # ----------------------------------------------------------------------
    def _align_to_base(
        self,
        src_img: np.ndarray,
        src_landmarks,
        base_landmarks,
        base_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align source image to base using affine transform.
        Returns (aligned_image, transform_matrix).
        If alignment fails, returns a copy of src_img resized to base_shape and identity matrix.
        """
        h_src, w_src = src_img.shape[:2]
        h_base, w_base = base_shape

        # Use a set of keypoints: nose, shoulders, hips
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

        # Estimate affine transform (scale, rotation, translation)
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if M is None:
            # Fallback: simple resize to base dimensions (no alignment)
            aligned = cv2.resize(src_img, (w_base, h_base))
            return aligned, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        # Warp source image
        aligned = cv2.warpAffine(
            src_img,
            M,
            (w_base, h_base),
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned, M

    def _warp_mask(self, mask: np.ndarray, M: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Warp a binary mask using affine transform."""
        h, w = shape
        warped = cv2.warpAffine(
            mask,
            M,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        # Threshold to ensure binary
        warped = (warped > 127).astype(np.uint8) * 255
        return warped

    # ----------------------------------------------------------------------
    #  Blending
    # ----------------------------------------------------------------------
    def _blend_part(
        self,
        composite: np.ndarray,
        aligned_part: np.ndarray,
        warped_mask: np.ndarray,
        base_img: np.ndarray,
        part_name: str,
    ) -> np.ndarray:
        """
        Blend a single part into the composite.
        Uses seamlessClone if possible, else alpha blending.
        """
        if np.sum(warped_mask) < 500:
            # Mask too small, skip
            return composite

        # Ensure mask is 3-channel for seamlessClone (they expect 3-channel for cloning)
        mask_3ch = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)

        # Compute center of mask for seamlessClone
        y, x = np.where(warped_mask > 0)
        if len(x) == 0 or len(y) == 0:
            return composite
        center = (int(np.mean(x)), int(np.mean(y)))

        # Validate center is within image bounds
        h, w = composite.shape[:2]
        if not (0 <= center[0] < w and 0 <= center[1] < h):
            # Fallback to alpha blending
            return self._alpha_blend(composite, aligned_part, warped_mask)

        # Try seamlessClone
        try:
            result = cv2.seamlessClone(
                aligned_part,
                composite,
                mask_3ch,
                center,
                cv2.NORMAL_CLONE,
            )
            return result
        except cv2.error as e:
            # If seamlessClone fails, fallback to alpha blending
            print(f"seamlessClone failed for {part_name}: {e}. Falling back to alpha blend.")
            return self._alpha_blend(composite, aligned_part, warped_mask)

    def _alpha_blend(
        self,
        composite: np.ndarray,
        aligned_part: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Simple alpha blending using mask as alpha channel."""
        alpha = mask.astype(np.float32) / 255.0
        alpha = np.expand_dims(alpha, axis=2)  # shape (h,w,1)
        blended = (aligned_part.astype(np.float32) * alpha +
                   composite.astype(np.float32) * (1 - alpha))
        return blended.astype(np.uint8)

    # ----------------------------------------------------------------------
    #  Score computation for part selection
    # ----------------------------------------------------------------------
    def _score_part_candidate(
        self,
        img: np.ndarray,
        landmarks,
        part_mask: np.ndarray,
        part_name: str,
    ) -> float:
        """
        Compute a quality score for a specific part in a specific image.
        Combines sharpness, brightness, visibility, and mask area.
        """
        # Extract region of interest (ROI) from the image using the mask
        # We'll compute metrics only on the masked region
        masked_region = cv2.bitwise_and(img, img, mask=part_mask)
        gray = cv2.cvtColor(masked_region, cv2.COLOR_BGR2GRAY)
        # Only consider pixels where mask > 0
        mask_pixels = part_mask > 0
        if np.sum(mask_pixels) == 0:
            return -1e9

        # Sharpness (Laplacian variance) on masked region
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian[mask_pixels]) if np.any(mask_pixels) else 0

        # Brightness: mean gray value on mask
        brightness = np.mean(gray[mask_pixels]) if np.any(mask_pixels) else 0

        # Visibility of landmarks for this part (average visibility of the landmarks used)
        part_indices = PART_LANDMARKS[part_name]
        vis = [landmarks[i].visibility for i in part_indices]
        avg_vis = np.mean(vis)

        # Area of mask (normalized)
        area = np.sum(part_mask) / (img.shape[0] * img.shape[1])

        # Combine into a score (weights are tunable)
        score = (0.4 * sharpness / 1000.0 +   # scale sharpness
                 0.2 * brightness / 255.0 +
                 0.3 * avg_vis +
                 0.1 * area)
        return score

    # ----------------------------------------------------------------------
    #  Main fusion pipeline
    # ----------------------------------------------------------------------
    def fuse(self, image_paths: List[str]) -> str:
        """
        Main entry point. Returns path to the fused image.
        """
        # 1. Load all images and detect pose / segmentation
        image_data = []
        for path in image_paths:
            img = self._load_image(path)
            if img is None:
                continue
            landmarks = self._detect_pose(img)
            if landmarks is None:
                continue
            silhouette = self._get_segmentation_mask(img)
            if np.sum(silhouette) < 1000:  # too small person
                continue

            quality = self._compute_quality_scores(img, landmarks)
            image_data.append({
                "path": path,
                "img": img,
                "landmarks": landmarks,
                "silhouette": silhouette,
                "quality": quality,
                "part_masks": self._generate_part_masks(img.shape, landmarks, silhouette),
            })

        if not image_data:
            raise ValueError("No valid images with detectable pose and segmentation.")

        # 2. Select base image (best overall quality)
        base_idx = max(
            range(len(image_data)),
            key=lambda i: image_data[i]["quality"]["visibility"] * 0.5 +
                          image_data[i]["quality"]["sharpness"] / 10000.0,
        )
        base = image_data[base_idx]
        base_img = base["img"].copy()
        base_shape = base_img.shape[:2]
        base_landmarks = base["landmarks"]
        base_silhouette = base["silhouette"]
        composite = base_img.copy()

        # 3. For each part, select the best source and blend
        parts = list(PART_LANDMARKS.keys())

        for part in parts:
            # Gather candidates: for each image, compute the part mask and score
            candidates = []
            for data in image_data:
                mask = data["part_masks"].get(part)
                if mask is None or np.sum(mask) < 500:
                    continue
                score = self._score_part_candidate(
                    data["img"],
                    data["landmarks"],
                    mask,
                    part,
                )
                candidates.append((data, mask, score))

            if not candidates:
                # No valid candidate for this part; use base part (already in composite)
                continue

            # Select best candidate by score
            best_data, best_mask, best_score = max(candidates, key=lambda x: x[2])

            # Skip if best is the base image (already there) or score too low
            if best_data is base:
                continue
            if best_score < 0.1:  # threshold
                continue

            # Align the best source image to base
            aligned_img, M = self._align_to_base(
                best_data["img"],
                best_data["landmarks"],
                base_landmarks,
                base_shape,
            )

            # Warp the part mask accordingly
            warped_mask = self._warp_mask(best_mask, M, base_shape)

            # Optionally intersect with base silhouette to avoid bleeding outside person
            warped_mask = cv2.bitwise_and(warped_mask, base_silhouette)

            # Blend the part into composite
            composite = self._blend_part(
                composite,
                aligned_img,
                warped_mask,
                base_img,
                part,
            )

        # 4. Fill any remaining holes with base image (just to be safe)
        #   Actually composite already has base content, so not needed.

        # 5. Save result
        out_fd, out_path = tempfile.mkstemp(suffix=".jpg")
        os.close(out_fd)
        cv2.imwrite(out_path, composite)
        return out_path


def fuse_best_parts(image_paths: List[str]) -> str:
    """API‑compatible entry point."""
    engine = BodyFusionEngine()
    return engine.fuse(image_paths)
