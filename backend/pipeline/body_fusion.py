import cv2
import numpy as np
import tempfile
import os
import mediapipe as mp
from typing import List, Tuple, Dict, Optional

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Landmarks
NOSE = 0
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

LEFT_EYE = [33, 133, 160, 158, 144, 153]
RIGHT_EYE = [362, 263, 387, 385, 380, 373]
MOUTH = [61, 291, 13, 14]

class BodyRegionParser:
    @staticmethod
    def split_mask(silhouette: np.ndarray, landmarks, img_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        h, w = img_shape[:2]
        mask = silhouette.copy()
        regions = {}

        lms = {idx: np.array([landmarks[idx].x * w, landmarks[idx].y * h]) for idx in range(33)}

        head_y = min(lms[LEFT_SHOULDER][1], lms[RIGHT_SHOULDER][1]) - 0.1 * h
        shoulder_y = (lms[LEFT_SHOULDER][1] + lms[RIGHT_SHOULDER][1]) / 2
        hip_y = (lms[LEFT_HIP][1] + lms[RIGHT_HIP][1]) / 2
        knee_y = (lms[LEFT_KNEE][1] + lms[RIGHT_KNEE][1]) / 2
        ankle_y = (lms[LEFT_ANKLE][1] + lms[RIGHT_ANKLE][1]) / 2
        torso_center_x = (lms[LEFT_SHOULDER][0] + lms[RIGHT_SHOULDER][0]) / 2

        def slice_mask(y_top, y_bottom, x_left=0, x_right=w):
            y_top = max(0, int(y_top))
            y_bottom = min(h, int(y_bottom))
            x_left = max(0, int(x_left))
            x_right = min(w, int(x_right))
            if y_top >= y_bottom or x_left >= x_right:
                return np.zeros((h, w), dtype=np.uint8)
            region = np.zeros((h, w), dtype=np.uint8)
            region[y_top:y_bottom, x_left:x_right] = mask[y_top:y_bottom, x_left:x_right]
            return region

        mid_torso = (shoulder_y + hip_y) / 2
        regions["head"] = slice_mask(0, shoulder_y)
        regions["chest"] = slice_mask(shoulder_y, mid_torso)
        regions["waist"] = slice_mask(mid_torso, hip_y)

        left_arm_x = (lms[LEFT_SHOULDER][0] + lms[LEFT_ELBOW][0]) / 2
        regions["left_upper_arm"] = slice_mask(shoulder_y, (shoulder_y + lms[LEFT_ELBOW][1])/2, 0, left_arm_x)
        regions["left_forearm"] = slice_mask((shoulder_y + lms[LEFT_ELBOW][1])/2, lms[LEFT_WRIST][1], 0, left_arm_x)
        regions["left_hand"] = slice_mask(lms[LEFT_WRIST][1], lms[LEFT_WRIST][1] + 0.15*h, 0, left_arm_x)

        right_arm_x = (lms[RIGHT_SHOULDER][0] + lms[RIGHT_ELBOW][0]) / 2
        regions["right_upper_arm"] = slice_mask(shoulder_y, (shoulder_y + lms[RIGHT_ELBOW][1])/2, right_arm_x, w)
        regions["right_forearm"] = slice_mask((shoulder_y + lms[RIGHT_ELBOW][1])/2, lms[RIGHT_WRIST][1], right_arm_x, w)
        regions["right_hand"] = slice_mask(lms[RIGHT_WRIST][1], lms[RIGHT_WRIST][1] + 0.15*h, right_arm_x, w)

        regions["left_thigh"] = slice_mask(hip_y, (hip_y + knee_y)/2, 0, torso_center_x)
        regions["left_calf"] = slice_mask((hip_y + knee_y)/2, ankle_y, 0, torso_center_x)
        regions["left_foot"] = slice_mask(ankle_y, h, 0, torso_center_x)

        regions["right_thigh"] = slice_mask(hip_y, (hip_y + knee_y)/2, torso_center_x, w)
        regions["right_calf"] = slice_mask((hip_y + knee_y)/2, ankle_y, torso_center_x, w)
        regions["right_foot"] = slice_mask(ankle_y, h, torso_center_x, w)

        # Overlap dilation
        kernel = np.ones((10, 10), np.uint8)
        for name in regions:
            regions[name] = cv2.dilate(regions[name], kernel, iterations=1)

        return regions


class RegionScorer:
    @staticmethod
    def sharpness(img: np.ndarray, mask: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        masked = lap * (mask / 255.0)
        valid = mask > 0
        if np.sum(valid) < 100:
            return 0.0
        return float(np.var(masked[valid]))

    @staticmethod
    def eyes_open(img: np.ndarray, face_lms, w: int, h: int) -> float:
        if not face_lms:
            return 0.0
        def ear(pts):
            v1 = np.linalg.norm(pts[1] - pts[5])
            v2 = np.linalg.norm(pts[2] - pts[4])
            h_dist = np.linalg.norm(pts[0] - pts[3])
            return (v1 + v2) / (2.0 * h_dist + 1e-6)
        left_pts = [np.array([face_lms[i].x * w, face_lms[i].y * h]) for i in LEFT_EYE]
        right_pts = [np.array([face_lms[i].x * w, face_lms[i].y * h]) for i in RIGHT_EYE]
        left_ear = ear(left_pts)
        right_ear = ear(right_pts)
        return min(((left_ear + right_ear) / 2.0) * 500, 100)

    @staticmethod
    def smile_ratio(img: np.ndarray, face_lms, w: int, h: int) -> float:
        if not face_lms:
            return 0.0
        left = np.array([face_lms[61].x * w, face_lms[61].y * h])
        right = np.array([face_lms[291].x * w, face_lms[291].y * h])
        top = np.array([face_lms[13].x * w, face_lms[13].y * h])
        bottom = np.array([face_lms[14].x * w, face_lms[14].y * h])
        width = np.linalg.norm(left - right)
        height = np.linalg.norm(top - bottom)
        ratio = width / (height + 1e-6)
        return min((ratio - 1.2) * 50, 100) if ratio > 1.2 else 0.0


class NoGapFusionEngine:
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        self.segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    def _get_silhouette(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.segmentation.process(rgb)
        mask = (results.segmentation_mask > 0.1).astype(np.uint8) * 255
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

    def process_images(self, image_paths: List[str]) -> str:
        image_data = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pose_res = self.pose.process(rgb)
            if not pose_res.pose_landmarks:
                continue
            face_res = self.face_mesh.process(rgb)
            face_lms = face_res.multi_face_landmarks[0].landmark if face_res.multi_face_landmarks else None
            silhouette = self._get_silhouette(img)
            image_data.append({
                "img": img,
                "pose": pose_res.pose_landmarks.landmark,
                "face": face_lms,
                "silhouette": silhouette,
            })

        if len(image_data) == 0:
            raise ValueError("No valid images with detectable pose.")

        base_idx = np.argmax([np.mean([lm.visibility for lm in d["pose"]]) for d in image_data])
        base_data = image_data[base_idx]
        base_img = base_data["img"].copy()
        base_pose = base_data["pose"]
        base_shape = base_img.shape[:2]
        base_silhouette = base_data["silhouette"]

        region_names = ["head", "chest", "waist", 
                        "left_upper_arm", "left_forearm", "left_hand",
                        "right_upper_arm", "right_forearm", "right_hand",
                        "left_thigh", "left_calf", "left_foot",
                        "right_thigh", "right_calf", "right_foot"]

        all_regions = []
        for data in image_data:
            regions = BodyRegionParser.split_mask(data["silhouette"], data["pose"], data["img"].shape)
            all_regions.append(regions)

        scorer = RegionScorer()
        region_scores = {name: [] for name in region_names}
        for i, data in enumerate(image_data):
            img = data["img"]
            h, w = img.shape[:2]
            face_lms = data["face"]
            for name in region_names:
                mask = all_regions[i][name]
                if np.sum(mask) < 200:
                    score = 0.0
                else:
                    if name == "head":
                        s = scorer.sharpness(img, mask)
                        eye = scorer.eyes_open(img, face_lms, w, h) if face_lms else 0
                        smile = scorer.smile_ratio(img, face_lms, w, h) if face_lms else 0
                        score = 0.5*s + 0.25*eye + 0.25*smile
                    else:
                        score = scorer.sharpness(img, mask)
                area_bonus = min((np.sum(mask) / (h * w)) * 20, 10)
                region_scores[name].append(score + area_bonus)

        best_idx = {}
        for name in region_names:
            scores = region_scores[name]
            if max(scores) == 0:
                best_idx[name] = base_idx
            else:
                best_idx[name] = np.argmax(scores)

        composite = base_img.copy()

        for name, idx in best_idx.items():
            if idx == base_idx:
                continue
            src_data = image_data[idx]
            src_img = src_data["img"]
            src_pose = src_data["pose"]
            src_mask = all_regions[idx][name]
            if np.sum(src_mask) < 200:
                continue
            aligned_src, M = self._align_to_base(src_img, src_pose, base_pose, base_shape)
            warped_mask = self._warp_mask(src_mask, M, base_shape)
            warped_mask = cv2.bitwise_and(warped_mask, base_silhouette)
            if np.sum(warped_mask) < 200:
                continue
            center = self._mask_center(warped_mask)
            mask_3ch = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)
            composite = cv2.seamlessClone(aligned_src, composite, mask_3ch, center, cv2.NORMAL_CLONE)

        gray_comp = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
        empty = (gray_comp == 0)
        if np.any(empty):
            composite[empty] = base_img[empty]

        out_fd, out_path = tempfile.mkstemp(suffix=".jpg")
        os.close(out_fd)
        cv2.imwrite(out_path, composite)
        return out_path

def fuse_best_parts(image_paths: List[str]) -> str:
    engine = NoGapFusionEngine()
    return engine.process_images(image_paths)
