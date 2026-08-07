import numpy as np
from backend.pipeline.clip_scorer import ClipScorer
from backend.logger import logger

class SemanticRanker:
    def __init__(self):
        self.clip = ClipScorer()

    def rank_frames(self, frames: list, prompt: str) -> list:
        """Frames is list of (index, timestamp, image, scores_dict).
        Returns sorted list with added 'semantic' score."""
        if not frames:
            return []
        logger.info(f"Ranking {len(frames)} frames with prompt: {prompt}")
        scored = []
        for idx, ts, img, scores in frames:
            sim = self.clip.compute_similarity(img, prompt)
            scored.append((idx, ts, img, scores, sim))
        # Sort by sim descending
        scored.sort(key=lambda x: x[4], reverse=True)
        return scored