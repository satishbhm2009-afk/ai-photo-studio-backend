import time

from typing import (
    List,
    Dict,
    Any
)

from backend.pipeline.frame_sampler import (
    FrameSampler
)

from backend.pipeline.face_scorer import (
    FaceScorer
)

from backend.pipeline.quality_filter import (
    QualityFilter
)

from backend.pipeline.duplicate_filter import (
    DuplicateFilter
)

from backend.pipeline.semantic_ranker import (
    SemanticRanker
)

from backend.pipeline.enhancer import (
    ImageEnhancer
)

from backend.utils import (
    image_to_base64
)

from backend.config import (
    settings
)

from backend.logger import (
    logger
)


def process_video(
    video_path: str,
    prompt: str,
    num_frames: int = None
) -> Dict[str, Any]:

    """
    AI video frame extraction pipeline.

    Current stage:

        Video
          ↓
        Frame sampling
          ↓
        Quality filtering
          ↓
        Duplicate filtering
          ↓
        Face quality scoring
          ↓
        CLIP batch ranking
          ↓
        Enhancement
          ↓
        Top results

    Body/Face Fusion is intentionally added
    in the next stage.
    """

    start_time = time.time()

    # =========================================================
    # SETTINGS
    # =========================================================

    if num_frames is None:
        num_frames = (
            settings.DEFAULT_NUM_FRAMES
        )

    num_frames = max(
        1,
        min(
            int(num_frames),
            50
        )
    )

    if not prompt or not prompt.strip():

        prompt = (
            "a clear, sharp, high quality "
            "realistic photograph of a person "
            "with a clear face, open eyes, "
            "natural expression and good lighting"
        )

    logger.info(
        "========================================"
    )

    logger.info(
        "Starting AI video processing"
    )

    logger.info(
        f"Video: {video_path}"
    )

    logger.info(
        f"Prompt: {prompt}"
    )

    logger.info(
        f"Requested output frames: {num_frames}"
    )

    # =========================================================
    # 1. FRAME SAMPLING
    # =========================================================

    sampler = FrameSampler(
        interval_sec=(
            settings.FRAME_SAMPLE_INTERVAL
        )
    )

    raw_frames = (
        sampler.extract_frames(
            video_path
        )
    )

    if not raw_frames:

        return {
            "success": False,
            "error": (
                "No frames could be "
                "extracted from video"
            )
        }

    total_frames = len(
        raw_frames
    )

    logger.info(
        f"Sampled {total_frames} frames"
    )

    # =========================================================
    # 2. INITIAL FILTERING
    # =========================================================

    face_scorer = FaceScorer()

    quality_filter = (
        QualityFilter()
    )

    duplicate_filter = (
        DuplicateFilter()
    )

    scored_frames = []

    rejected_blur = 0
    rejected_exposure = 0
    rejected_duplicate = 0
    rejected_face = 0

    # ---------------------------------------------------------
    # Process every sampled frame
    # ---------------------------------------------------------

    for (
        idx,
        timestamp,
        image
    ) in raw_frames:

        try:

            # ---------------------------------------------
            # Blur
            # ---------------------------------------------

            if quality_filter.is_blurry(
                image
            ):

                rejected_blur += 1
                continue

            # ---------------------------------------------
            # Exposure
            # ---------------------------------------------

            if quality_filter.is_too_dark_or_bright(
                image
            ):

                rejected_exposure += 1
                continue

            # ---------------------------------------------
            # Duplicate
            # ---------------------------------------------

            if duplicate_filter.is_duplicate(
                image
            ):

                rejected_duplicate += 1
                continue

            # ---------------------------------------------
            # Face scoring
            # ---------------------------------------------

            face_score, aligned = (
                face_scorer.score(
                    image
                )
            )

            if face_score <= 0:

                rejected_face += 1
                continue

            # -------------------------------------------------
            # Quality values
            # -------------------------------------------------

            brightness = (
                quality_filter.brightness_score(
                    image
                )
            )

            # Convert brightness 0-1 → 0-100
            brightness_100 = (
                brightness * 100.0
            )

            # Face score is already 0-100
            face_quality = (
                max(
                    0.0,
                    min(
                        100.0,
                        float(face_score)
                    )
                )
            )

            # -------------------------------------------------
            # Sharpness
            # -------------------------------------------------

            # We already passed the blur filter.
            # Calculate an additional normalized
            # sharpness score for ranking.

            import cv2

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            sharpness_value = float(
                cv2.Laplacian(
                    gray,
                    cv2.CV_64F
                ).var()
            )

            # Normalize sharpness.
            #
            # 100 variance = approximately good
            # 500+ = very sharp
            sharpness_score = min(
                100.0,
                (
                    sharpness_value /
                    500.0
                ) * 100.0
            )

            # -------------------------------------------------
            # Combined visual quality
            # -------------------------------------------------

            quality = (

                face_quality * 0.50 +

                sharpness_score * 0.30 +

                brightness_100 * 0.20

            )

            quality = max(
                0.0,
                min(
                    100.0,
                    quality
                )
            )

            scores = {

                "face_score": (
                    face_quality / 100.0
                ),

                "face_score_100": (
                    face_quality
                ),

                "sharpness": (
                    sharpness_score / 100.0
                ),

                "sharpness_100": (
                    sharpness_score
                ),

                "brightness": (
                    brightness
                ),

                "brightness_100": (
                    brightness_100
                ),

                "quality": (
                    quality / 100.0
                ),

                "quality_100": (
                    quality
                ),

                "raw_face_score": (
                    face_score
                )
            }

            scored_frames.append(
                (
                    idx,
                    timestamp,
                    image,
                    scores
                )
            )

        except Exception as exc:

            logger.warning(
                f"Frame {idx} failed: "
                f"{exc}"
            )

            continue

    # =========================================================
    # RELEASE MEDIAPIPE
    # =========================================================

    try:
        face_scorer.detection.close()
    except Exception:
        pass

    try:
        face_scorer.mesh.close()
    except Exception:
        pass

    # =========================================================
    # NO VALID FRAMES
    # =========================================================

    if not scored_frames:

        return {
            "success": False,
            "error": (
                "No valid face frames found"
            ),

            "total_frames": (
                total_frames
            ),

            "processing_time": (
                time.time() -
                start_time
            ),

            "filter_stats": {

                "blur": rejected_blur,

                "exposure": (
                    rejected_exposure
                ),

                "duplicate": (
                    rejected_duplicate
                ),

                "face": (
                    rejected_face
                )
            }
        }

    logger.info(
        f"Valid candidate frames: "
        f"{len(scored_frames)}"
    )

    logger.info(
        f"Rejected blur: "
        f"{rejected_blur}"
    )

    logger.info(
        f"Rejected exposure: "
        f"{rejected_exposure}"
    )

    logger.info(
        f"Rejected duplicates: "
        f"{rejected_duplicate}"
    )

    logger.info(
        f"Rejected no-face: "
        f"{rejected_face}"
    )

    # =========================================================
    # 3. LIMIT CLIP CANDIDATES
    # =========================================================

    # Do not send hundreds of frames to CLIP.
    #
    # First use local visual quality to reduce candidates.

    scored_frames.sort(
        key=lambda item:
            item[3]["quality_100"],
        reverse=True
    )

    max_clip_candidates = min(
        len(scored_frames),
        max(
            20,
            num_frames * 5
        )
    )

    clip_candidates = (
        scored_frames[
            :max_clip_candidates
        ]
    )

    logger.info(
        f"Sending {len(clip_candidates)} "
        f"frames to CLIP"
    )

    # =========================================================
    # 4. CLIP SEMANTIC RANKING
    # =========================================================

    ranker = SemanticRanker(
        batch_size=8
    )

    ranked = ranker.rank_frames(
        clip_candidates,
        prompt
    )

    if not ranked:

        return {
            "success": False,
            "error": (
                "CLIP ranking returned "
                "no candidates"
            )
        }

    # =========================================================
    # 5. FINAL COMBINED RANKING
    # =========================================================

    final_ranked = []

    for (
        idx,
        timestamp,
        image,
        scores,
        semantic_score
    ) in ranked:

        quality = (
            scores["quality"]
        )

        # CLIP score is 0-1
        semantic = max(
            0.0,
            min(
                1.0,
                float(
                    semantic_score
                )
            )
        )

        # Quality + CLIP
        final_score = (

            quality * 0.55 +

            semantic * 0.45

        )

        final_ranked.append(
            (
                idx,
                timestamp,
                image,
                scores,
                semantic,
                final_score
            )
        )

    final_ranked.sort(
        key=lambda item:
            item[5],
        reverse=True
    )

    # =========================================================
    # 6. TOP RESULTS
    # =========================================================

    top = final_ranked[
        :num_frames
    ]

    enhanced_frames = []

    for (
        idx,
        timestamp,
        image,
        scores,
        semantic_score,
        final_score
    ) in top:

        try:

            enhanced = (
                ImageEnhancer.enhance(
                    image
                )
            )

            # ---------------------------------------------
            # Encode
            # ---------------------------------------------

            b64 = image_to_base64(
                enhanced
            )

            enhanced_frames.append({

                "index": int(idx),

                "timestamp": float(
                    timestamp
                ),

                "score": float(
                    final_score
                ),

                "quality": float(
                    scores["quality"]
                ),

                "semantic": float(
                    semantic_score
                ),

                "image_base64": b64

            })

        except Exception as exc:

            logger.warning(
                f"Enhancement failed "
                f"for frame {idx}: "
                f"{exc}"
            )

            continue

    # =========================================================
    # CLEANUP
    # =========================================================

    processing_time = (
        time.time() -
        start_time
    )

    logger.info(
        f"AI processing completed "
        f"in {processing_time:.2f}s"
    )

    logger.info(
        "========================================"
    )

    # =========================================================
    # RESULT
    # =========================================================

    return {

        "success": True,

        "total_frames": (
            total_frames
        ),

        "selected_frames": (
            enhanced_frames
        ),

        "processing_time": (
            processing_time
        ),

        "video_info": {

            "duration": 0,

            "resolution": (
                "unknown"
            )
        }
    }
