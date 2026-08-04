import cv2
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

    return result
