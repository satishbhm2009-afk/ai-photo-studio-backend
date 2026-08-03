import cv2
import numpy as np
import tempfile
import os
import mediapipe as mp
from typing import List, Tuple, Dict

mp_pose = mp.solutions.pose

# Landmark indices for body regions (MediaPipe Pose)
REGIONS = {
    "face": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "neck": [11, 12, 23, 24],
    "chest": [11, 12, 13, 14, 23, 24],
    "stomach": [23, 24, 25, 26],
    "hips": [23, 24, 25, 26, 27, 28],
    "legs": [25, 26, 27, 28, 29, 30, 31, 32],
    "arms": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
}

def get_landmarks(image: np.ndarray):
    """Detect pose landmarks. Returns list of landmarks or None."""
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if not results.pose_landmarks:
            return None
        return results.pose_landmarks.landmark

def get_region_mask(image_shape, landmarks, region_indices):
    """Create a binary mask for a region defined by convex hull of keypoints."""
    h, w = image_shape[:2]
    points = []
    for idx in region_indices:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        points.append([x, y])
    points = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(points)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    return mask

def compute_region_sharpness(image: np.ndarray, mask: np.ndarray) -> float:
    """Laplacian variance only inside the mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    masked = laplacian * (mask / 255.0)
    valid = mask > 0
    if np.sum(valid) == 0:
        return 0.0
    return float(np.var(masked[valid]))

def align_image_to_base(img, src_landmarks, dst_landmarks, dst_shape):
    """
    Align `img` to the base image coordinate system using affine transform
    based on shoulders (11,12) and hips (23,24).
    Returns (aligned_img, affine_matrix).
    """
    h, w = img.shape[:2]
    dst_h, dst_w = dst_shape
    src_pts = []
    dst_pts = []
    for idx in [11, 12, 23, 24]:
        src_lm = src_landmarks[idx]
        dst_lm = dst_landmarks[idx]
        src_pts.append([src_lm.x * w, src_lm.y * h])
        dst_pts.append([dst_lm.x * dst_w, dst_lm.y * dst_h])
    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None:
        # fallback to identity
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    aligned = cv2.warpAffine(img, M, (dst_w, dst_h), borderMode=cv2.BORDER_REPLICATE)
    return aligned, M

def warp_mask(mask: np.ndarray, M: np.ndarray, dst_shape: Tuple[int, int]) -> np.ndarray:
    """Warp a mask using the same affine transform."""
    dst_h, dst_w = dst_shape
    warped = cv2.warpAffine(mask, M, (dst_w, dst_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped

def fuse_best_parts(image_paths: List[str]) -> str:
    """
    Load images, detect pose, score each region by sharpness,
    align the best region to the base image, and blend using Poisson blending.
    """
    # Load all images and their landmarks
    images = []
    landmarks_list = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        lm = get_landmarks(img)
        if lm is None:
            continue
        images.append(img)
        landmarks_list.append(lm)

    if len(images) == 0:
        raise ValueError("No valid images with detectable pose.")

    # Choose base image: the one with highest overall sharpness
    overall_scores = [cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                      for img in images]
    base_idx = np.argmax(overall_scores)
    base_img = images[base_idx].copy()
    base_landmarks = landmarks_list[base_idx]
    base_shape = base_img.shape[:2]

    # For each region, find the best image (max sharpness) and its mask
    best_region_scores = {region: -1 for region in REGIONS}
    best_region_img_idx = {region: None for region in REGIONS}
    best_region_masks = {region: None for region in REGIONS}
    best_region_landmarks = {region: None for region in REGIONS}

    # Pre-compute masks for all images and regions
    masks_all = []
    for lm in landmarks_list:
        region_masks = {}
        for region, idxs in REGIONS.items():
            region_masks[region] = get_region_mask(base_shape, lm, idxs)
        masks_all.append(region_masks)

    for region in REGIONS:
        for i, img in enumerate(images):
            mask = masks_all[i][region]
            score = compute_region_sharpness(img, mask)
            if score > best_region_scores[region]:
                best_region_scores[region] = score
                best_region_img_idx[region] = i
                best_region_masks[region] = mask
                best_region_landmarks[region] = landmarks_list[i]

    # Start with a copy of the base image
    composite = base_img.copy()

    # For each region, if the best is not the base, align and blend
    for region, best_idx in best_region_img_idx.items():
        if best_idx is None or best_idx == base_idx:
            continue

        src_img = images[best_idx]
        src_lm = best_region_landmarks[region]
        src_mask = best_region_masks[region]  # in base coordinates (we pre-computed using base shape, but that's an approximation)

        # More accurate: align the source image to base, then warp the mask (computed on source) accordingly.
        # We'll compute the mask on source image coordinates first.
        src_mask_original = get_region_mask(src_img.shape, src_lm, REGIONS[region])
        aligned_src, M = align_image_to_base(src_img, src_lm, base_landmarks, base_shape)
        # Warp the mask using the same transform
        aligned_mask = warp_mask(src_mask_original, M, base_shape)

        # Use Poisson blending to insert aligned_src into composite using aligned_mask
        # cv2.seamlessClone requires the mask to be a binary mask, and we need the center of the mask
        # Find the bounding box of the mask to get center
        y_indices, x_indices = np.where(aligned_mask > 0)
        if len(y_indices) == 0:
            continue
        center = (int(np.mean(x_indices)), int(np.mean(y_indices)))
        # Ensure mask is 3-channel for seamlessClone (it expects 3-channel mask)
        mask_3ch = cv2.cvtColor(aligned_mask, cv2.COLOR_GRAY2BGR)
        # Poisson blending
        composite = cv2.seamlessClone(aligned_src, composite, mask_3ch, center, cv2.NORMAL_CLONE)

    # Save result
    out_fd, out_path = tempfile.mkstemp(suffix=".jpg")
    os.close(out_fd)
    cv2.imwrite(out_path, composite)
    return out_path
