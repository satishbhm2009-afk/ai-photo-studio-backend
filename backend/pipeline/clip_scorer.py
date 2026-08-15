import cv2
import numpy as np
import torch

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from backend.config import settings
from backend.logger import logger


class ClipScorer:
    """
    CLIP based image/text similarity scorer.

    Important:
    - Model is loaded only once.
    - Supports batch scoring.
    - CPU/GPU is selected from settings.
    """

    def __init__(self):
        self.device = torch.device(
            settings.CLIP_DEVICE
        )

        logger.info(
            f"Loading CLIP model: "
            f"{settings.CLIP_MODEL_NAME}"
        )

        self.processor = (
            CLIPProcessor.from_pretrained(
                settings.CLIP_MODEL_NAME
            )
        )

        self.model = (
            CLIPModel.from_pretrained(
                settings.CLIP_MODEL_NAME
            )
            .to(self.device)
        )

        self.model.eval()

        logger.info(
            f"CLIP model loaded on "
            f"{self.device}"
        )

    # =========================================================
    # IMAGE CONVERSION
    # =========================================================

    @staticmethod
    def _to_pil(image: np.ndarray) -> Image.Image:

        if image is None:
            raise ValueError(
                "Image is None"
            )

        if image.size == 0:
            raise ValueError(
                "Image is empty"
            )

        if len(image.shape) == 2:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )

        elif image.shape[2] == 4:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2RGB
            )

        else:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        return Image.fromarray(
            image
        ).convert("RGB")

    # =========================================================
    # SINGLE IMAGE
    # =========================================================

    def compute_similarity(
        self,
        image: np.ndarray,
        prompt: str
    ) -> float:

        results = self.compute_batch_similarity(
            [image],
            prompt
        )

        if not results:
            return 0.0

        return results[0]

    # =========================================================
    # BATCH IMAGE SCORING
    # =========================================================

    def compute_batch_similarity(
        self,
        images: list,
        prompt: str,
        batch_size: int = 8
    ) -> list:

        if not images:
            return []

        if not prompt:
            prompt = (
                "a clear, sharp, high quality "
                "realistic photograph of a person"
            )

        all_scores = []

        total = len(images)

        logger.info(
            f"CLIP ranking {total} images "
            f"with batch size {batch_size}"
        )

        for start in range(
            0,
            total,
            batch_size
        ):

            batch = images[
                start:start + batch_size
            ]

            pil_images = []

            for image in batch:

                try:
                    pil_images.append(
                        self._to_pil(image)
                    )

                except Exception as exc:

                    logger.warning(
                        f"CLIP image conversion "
                        f"failed: {exc}"
                    )

                    # Keep list alignment
                    pil_images.append(
                        Image.new(
                            "RGB",
                            (224, 224),
                            "black"
                        )
                    )

            try:

                inputs = self.processor(
                    text=[prompt],
                    images=pil_images,
                    return_tensors="pt",
                    padding=True
                )

                inputs = {
                    key: value.to(
                        self.device
                    )
                    for key, value
                    in inputs.items()
                }

                with torch.no_grad():

                    outputs = self.model(
                        **inputs
                    )

                    logits = (
                        outputs
                        .logits_per_image
                    )

                    # One text prompt + N images.
                    # Shape is normally [N, 1].
                    if logits.ndim == 2:
                        logits = logits[:, 0]

                    # Convert logits to a bounded
                    # confidence-like value.
                    scores = torch.sigmoid(
                        logits
                    )

                    scores = (
                        scores
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(
                            np.float32
                        )
                    )

                all_scores.extend(
                    [
                        float(
                            np.clip(
                                score,
                                0.0,
                                1.0
                            )
                        )
                        for score in scores
                    ]
                )

            except Exception as exc:

                logger.exception(
                    "CLIP batch scoring failed"
                )

                # Do not crash entire video pipeline.
                all_scores.extend(
                    [0.0] * len(batch)
                )

        return all_scores

    # =========================================================
    # RANK IMAGES
    # =========================================================

    def rank_images(
        self,
        images: list,
        prompt: str,
        batch_size: int = 8
    ) -> list:

        scores = (
            self.compute_batch_similarity(
                images,
                prompt,
                batch_size
            )
        )

        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True
        )

        return ranked
