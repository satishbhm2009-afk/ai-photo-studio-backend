import numpy as np
from typing import List, Tuple
from backend.pipeline.clip_scorer import CLIPScorer

class SemanticRanker:
    def __init__(self, quality_weight: float = 0.6, semantic_weight: float = 0.4):
        self.quality_weight = quality_weight
        self.semantic_weight = semantic_weight
        self.clip = CLIPScorer()

    def rank(self, frames: List[np.ndarray], quality_scores: List[float], prompt: str) -> List[Tuple[float, int]]:
        """
        Returns list of (final_score, index) sorted descending.
        """
        if not prompt:
            # No semantic prompt → use quality scores only
            sorted_idx = sorted(range(len(quality_scores)), key=lambda i: quality_scores[i], reverse=True)
            return [(quality_scores[i], i) for i in sorted_idx]

        # Compute CLIP similarity for each frame
        semantic_scores = [self.clip.score(frame, prompt) * 100 for frame in frames]  # scale to 0-100

        final_scores = []
        for i in range(len(frames)):
            norm_quality = min(quality_scores[i] / 100.0, 1.0)
            norm_semantic = min(semantic_scores[i] / 100.0, 1.0)
            combined = (self.quality_weight * norm_quality) + (self.semantic_weight * norm_semantic)
            final_scores.append(combined * 100)

        sorted_idx = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
        return [(final_scores[i], i) for i in sorted_idx]
