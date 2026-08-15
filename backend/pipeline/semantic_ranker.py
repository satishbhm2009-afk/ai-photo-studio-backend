from backend.pipeline.clip_scorer import ClipScorer
from backend.logger import logger


class SemanticRanker:
    """
    Ranks candidate frames using CLIP.

    Input:
        [
            (
                frame_index,
                timestamp,
                image,
                scores_dict
            ),
            ...
        ]

    Output:
        [
            (
                frame_index,
                timestamp,
                image,
                scores_dict,
                semantic_score
            ),
            ...
        ]
    """

    def __init__(
        self,
        batch_size: int = 8
    ):

        self.clip = ClipScorer()

        self.batch_size = (
            max(
                1,
                int(batch_size)
            )
        )

    # =========================================================
    # RANK
    # =========================================================

    def rank_frames(
        self,
        frames: list,
        prompt: str
    ) -> list:

        if not frames:
            return []

        logger.info(
            f"Ranking {len(frames)} "
            f"candidate frames with CLIP"
        )

        images = [
            item[2]
            for item in frames
        ]

        semantic_scores = (
            self.clip.compute_batch_similarity(
                images,
                prompt,
                batch_size=self.batch_size
            )
        )

        scored = []

        for item, semantic_score in zip(
            frames,
            semantic_scores
        ):

            idx, ts, img, scores = item

            scored.append(
                (
                    idx,
                    ts,
                    img,
                    scores,
                    float(semantic_score)
                )
            )

        # Highest CLIP score first
        scored.sort(
            key=lambda item: item[4],
            reverse=True
        )

        return scored
