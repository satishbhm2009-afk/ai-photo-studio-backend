import cv2
import numpy as np
import tempfile
import os
import mediapipe as mp
from typing import List, Tuple, Dict

mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Landmarks
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

class BodyFusionEngine:
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    def _get_silhouette(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.segmentation.process(rgb)
        return (results.segmentation_mask > 0.1).astype(np.uint8) * 255

    def _get_region_mask(self, shape, landmarks, part_name):
        h, w = shape[:2]
        # Define region points (simplified: head, torso, arms, legs)
        # This is a simplified version – for production, expand to 15+ regions
        pts = []
        if part_name == "head":
            indices = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER]
        elif part_name == "torso":
            indices = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
        elif part_name == "left_arm":
            indices = [LEFT_SHOULDER, 13, 15]  # shoulder, elbow, wrist
        elif part_name == "right_arm":
            indices = [RIGHT_SHOULDER, 14, 16]
        elif part_name == "left_leg":
            indices = [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]
        elif part_name == "right_leg":
            indices = [RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]
        else:
            return np.ones((h, w), dtype=np.uint8) * 255

        for idx in indices:
            lm = landmarks[idx]
            pts.append([int(lm.x * w), int(lm.y * h)])

        if len(pts) < 3:
            return np.ones((h, w), dtype=np.uint8) * 255

        hull = cv2.convexHull(np.array(pts, dtype=np.int32))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        return mask

    def _align_to_base(self, img, src_pose, base_pose, base_shape):
        h, w = img.shape[:2]
        base_h, base_w = base_shape
        src_pts = []
        dst_pts = []
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

    def fuse(self, image_paths: List[str]) -> str:
        # Load images and detect poses
        image_data = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pose_res = self.pose.process(rgb)
            if not pose_res.pose_landmarks:
                continue
            image_data.append({
                "img": img,
                "pose": pose_res.pose_landmarks.landmark,
                "silhouette": self._get_silhouette(img)
            })

        if not image_data:
            raise ValueError("No valid images with detectable pose.")

        # Base = best pose confidence
        base_idx = np.argmax([np.mean([lm.visibility for lm in d["pose"]]) for d in image_data])
        base = image_data[base_idx]
        base_img = base["img"].copy()
        base_pose = base["pose"]
        base_shape = base_img.shape[:2]
        base_silhouette = base["silhouette"]

        parts = ["head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"]
        composite = base_img.copy()

        for part in parts:
            best_score = -1
            best_idx = base_idx
            for i, data in enumerate(image_data):
                mask = self._get_region_mask(data["img"].shape, data["pose"], part)
                score = np.sum(mask)
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx == base_idx:
                continue

            src = image_data[best_idx]
            src_mask = self._get_region_mask(src["img"].shape, src["pose"], part)
            if np.sum(src_mask) < 500:
                continue

            aligned_src, M = self._align_to_base(src["img"], src["pose"], base_pose, base_shape)
            warped_mask = self._warp_mask(src_mask, M, base_shape)
            warped_mask = cv2.bitwise_and(warped_mask, base_silhouette)

            if np.sum(warped_mask) < 500:
                continue

            center = self._mask_center(warped_mask)
            mask_3ch = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)
            composite = cv2.seamlessClone(aligned_src, composite, mask_3ch, center, cv2.NORMAL_CLONE)

        # Fill gaps with base
        gray = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
        empty = gray == 0
        composite[empty] = base_img[empty]

        out_fd, out_path = tempfile.mkstemp(suffix=".jpg")
        os.close(out_fd)
        cv2.imwrite(out_path, composite)
        return out_path


def fuse_best_parts(image_paths: List[str]) -> str:
    engine = BodyFusionEngine()
    return engine.fuse(image_paths)
