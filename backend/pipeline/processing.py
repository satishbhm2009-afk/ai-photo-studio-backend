import cv2
import numpy as np
import tempfile
import os
from typing import List, Tuple, Optional
fromimport cv2
import tempfile
import os
from typing import List, Tuple, Optional

from backend.pipeline.frame_sampler import sample_frames
from backend.pipeline.quality_filter import QualityFilter
from backend.pipeline.duplicate_filter import DuplicateFilter
from backend.pipeline.face_scorer import FaceScorer
from backend.pipeline.semantic_ranker import SemanticRanker
from backend.pipeline.enhancer import ImageEnhancer
from backend.pipeline.utils import variance_of_laplacian, brightness_score

def process_video_pipeline(
    video_path: str,
    num_frames: int = 10,
    interval: int = 5,
    prompt: str = "",
    enhance: bool = False
) -> List[Tuple[float, str, Optional[str]]]:
    """
    Full pipeline:
    1. Sample frames
    2. Quality filter (blur + brightness)
    3. Duplicate removal
    4. Face scoring
    5. Semantic ranking (if prompt provided)
    6. Enhancement (if requested)

    Returns: List of (final_score, original_path, enhanced_path_or_None)
    """
    # 1. Sample frames
    raw_frames = []
    for _, frame in sample_frames(video_path, interval):
        raw_frames.append(frame)

    if not raw_frames:
        raise ValueError("No frames extracted from video.")

    # 2. Quality filter (remove blurry/dark frames)
    qf = QualityFilter(min_sharpness=50.0)
    filtered_frames = [f for f in raw_frames if qf.is_acceptable(f)]

    if not filtered_frames:
        # Lower threshold and try again
        qf = QualityFilter(min_sharpness=25.0, min_brightness=10.0)
        filtered_frames = [f for f in raw_frames if qf.is_acceptable(f)]
        if not filtered_frames:
            # Last resort: take the sharpest 20% of frames
            sorted_frames = sorted(raw_frames, key=lambda f: variance_of_laplacian(f), reverse=True)
            count = max(1, int(len(sorted_frames) * 0.2))
            filtered_frames = sorted_frames[:count]

    # 3. Remove duplicates
    df = DuplicateFilter(threshold=0.92)
    unique_frames = df.filter(filtered_frames)

    if len(unique_frames) > num_frames * 2:
        # If still too many, take the sharpest ones
        unique_frames = sorted(unique_frames, key=lambda f: variance_of_laplacian(f), reverse=True)[:num_frames * 2]

    # 4. Face scoring
    scorer = FaceScorer()
    quality_scores = []
    aligned_frames = []
    for frame in unique_frames:
        score, aligned = scorer.score(frame)
        quality_scores.append(score)
        aligned_frames.append(aligned if aligned is not None else frame.copy())

    # 5. Semantic ranking (if prompt provided)
    ranker = SemanticRanker()
    ranked = ranker.rank(aligned_frames, quality_scores, prompt)

    # 6. Take top N
    top_n = ranked[:num_frames]

    # 7. Save frames and optionally enhance
    result = []
    enhancer = ImageEnhancer() if enhance else None

    for final_score, idx in top_n:
        frame = aligned_frames[idx]

        # Save original
        fd1, orig_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd1)
        cv2.imwrite(orig_path, frame)

        enh_path = None
        if enhance:
            enhanced = enhancer.process(frame)
            fd2, enh_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd2)
            cv2.imwrite(enh_path, enhanced)

        result.append((final_score, orig_path, enh_path))

    return result backend.pipeline.face_scorer import FaceScorer
from backend.pipeline.clip_scorer import CLIPScorer
from backend.pipeline.enhancer import ImageEnhancer
import warnings
warnings.filterwarnings("ignore")

def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_duplicate(frame1, frame2, threshold=0.95):
    """Check if two frames are near‑duplicates using histogram correlation."""
    hist1 = cv2.calcHist([frame1], [0], None, [64], [0,256])
    hist2 = cv2.calcHist([frame2], [0], None, [64], [0,256])
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return correlation > threshold

def extract_best_frames_with_scores(
    video_path: str,
    num_frames: int = 10,
    interval: int = 5,
    prompt: str = "",
    enhance: bool = False
) -> List[Tuple[float, str, Optional[str]]]:
    """
    Extract frames, score them with FaceScorer, optionally add CLIP similarity,
    remove duplicates, apply enhancement, and return top N.
    Returns list of (final_score, original_path, enhanced_path_or_None).
    """
    cap = cv2.VideoCapture(video_path)
    scorer = FaceScorer()
    clip = CLIPScorer() if prompt else None
    enhancer = ImageEnhancer() if enhance else None

    raw_frames = []  # list of (frame, original_score, sharpness)
    frame_count = 0
    MIN_SHARPNESS = 50.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            sharpness = variance_of_laplacian(frame)
            if sharpness < MIN_SHARPNESS:
                frame_count += 1
                continue

            # Get quality score from FaceScorer
            quality_score, aligned = scorer.score_frame(frame)
            if aligned is not None:
                frame = aligned  # use aligned version

            raw_frames.append((frame, quality_score, sharpness))
        frame_count += 1

    cap.release()

    if not raw_frames:
        raise ValueError("No valid frames extracted.")

    # Remove near-duplicates (keep highest quality)
    unique_frames = []
    for i, (frame, score, sharp) in enumerate(raw_frames):
        is_dup = False
        for j, (f2, _, _) in enumerate(unique_frames):
            if is_duplicate(frame, f2):
                is_dup = True
                break
        if not is_dup:
            unique_frames.append((frame, score, sharp))

    # If we have a prompt, compute CLIP similarity
    if clip:
        clip_scores = []
        for frame, _, _ in unique_frames:
            sim = clip.score(frame, prompt)
            clip_scores.append(sim)
        # Combine quality (0‑100) and CLIP (0‑1) into final score (weighted)
        final_scores = []
        for i, (_, qual, _) in enumerate(unique_frames):
            # Normalize quality to 0‑1, clip to [0,1]
            norm_qual = min(qual / 100.0, 1.0)
            # Final score: 0.5 * norm_qual + 0.5 * clip_scores[i]
            combined = 0.5 * norm_qual + 0.5 * clip_scores[i]
            final_scores.append(combined * 100)  # back to 0‑100 scale
    else:
        final_scores = [qual for _, qual, _ in unique_frames]

    # Sort by final score descending
    scored = list(zip(final_scores, unique_frames))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top N
    top = scored[:num_frames]

    # Save frames and apply enhancement if requested
    paths = []
    for final_score, (frame, _, _) in top:
        out_fd, out_path = tempfile.mkstemp(suffix=".jpg")
        os.close(out_fd)
        cv2.imwrite(out_path, frame)
        enh_path = None
        if enhance:
            enhanced = enhancer.process(frame)
            enh_fd, enh_path = tempfile.mkstemp(suffix=".jpg")
            os.close(enh_fd)
            cv2.imwrite(enh_path, enhanced)
        paths.append((final_score, out_path, enh_path))

    return paths
