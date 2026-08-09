import time
import os
import numpy as np
from backend.pipeline.frame_sampler import FrameSampler
from backend.pipeline.face_scorer import FaceScorer
from backend.pipeline.quality_filter import QualityFilter
from backend.pipeline.duplicate_filter import DuplicateFilter
from backend.pipeline.semantic_ranker import SemanticRanker
from backend.pipeline.enhancer import ImageEnhancer
from backend.utils import image_to_base64, cleanup_temp_files
from backend.config import settings
from backend.logger import logger
from typing import List, Dict, Any

def process_video(video_path: str, prompt: str, num_frames: int = None) -> Dict[str, Any]:
    """
    Full pipeline: returns dict with selected frames and metadata.
    """
    start_time = time.time()
    if num_frames is None:
        num_frames = settings.DEFAULT_NUM_FRAMES

    # 1. Frame sampling
    sampler = FrameSampler(interval_sec=settings.FRAME_SAMPLE_INTERVAL)
    raw_frames = sampler.extract_frames(video_path)
    if not raw_frames:
        return {"success": False, "error": "No frames extracted"}

    total_frames = len(raw_frames)

    # 2. Face detection & quality scoring
    face_scorer = FaceScorer()
    quality_filter = QualityFilter()
    duplicate_filter = DuplicateFilter()

    scored_frames = []  # (index, timestamp, image, scores_dict)
    for idx, ts, img in raw_frames:
        # Check blur
        if quality_filter.is_blurry(img):
            continue
        # Check brightness
        if quality_filter.is_too_dark_or_bright(img):
            continue
        # Duplicate
        if duplicate_filter.is_duplicate(img):
            continue

        # Face detection
        has_face, face_data = face_scorer.detect_faces(img)
        if not has_face:
            continue

        # Face quality metrics
        quality_metrics = face_scorer.compute_face_quality(img)
        if not quality_metrics.get("has_face", False):
            continue

        # Combine quality score
        face_score = quality_metrics.get("quality", 0.0)
        # Also sharpness and brightness from face
        sharpness = quality_metrics.get("sharpness", 0.0)
        brightness = quality_metrics.get("brightness", 0.0)
        # Overall quality = weighted combination
        quality = 0.6 * face_score + 0.2 * sharpness + 0.2 * brightness
        quality = min(max(quality, 0), 1)

        scores = {
            "face_score": face_score,
            "sharpness": sharpness,
            "brightness": brightness,
            "quality": quality,
            "raw": quality_metrics
        }
        scored_frames.append((idx, ts, img, scores))

    if not scored_frames:
        return {"success": False, "error": "No valid faces found"}

    # 3. Semantic ranking
    ranker = SemanticRanker()
    ranked = ranker.rank_frames(scored_frames, prompt)

    # 4. Select top N
    top = ranked[:num_frames]

    # 5. Enhance selected frames
    enhanced_frames = []
    for idx, ts, img, scores, semantic_score in top:
        enhanced = ImageEnhancer.enhance(img)
        # Compute final overall score: weighted combination of quality and semantic
        overall = 0.5 * scores["quality"] + 0.5 * semantic_score
        # Store
        b64 = image_to_base64(enhanced)
        enhanced_frames.append({
            "index": idx,
            "timestamp": ts,
            "score": overall,
            "quality": scores["quality"],
            "semantic": semantic_score,
            "image_base64": b64
        })

    processing_time = time.time() - start_time

    # Cleanup temp files (video_path will be removed by caller)
    # Do not delete video_path here, caller will clean up.

    return {
        "success": True,
        "total_frames": total_frames,
        "selected_frames": enhanced_frames,
        "processing_time": processing_time,
        "video_info": {
            "duration": 0,  # could extract with OpenCV
            "resolution": "unknown"
        }
    }