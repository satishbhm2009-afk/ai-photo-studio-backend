import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import cv2
from backend.config import settings
from backend.logger import logger

class ClipScorer:
    def __init__(self):
        self.device = torch.device(settings.CLIP_DEVICE)
        self.model = CLIPModel.from_pretrained(settings.CLIP_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_NAME)
        self.model.eval()
        logger.info(f"CLIP model loaded on {self.device}")

    def compute_similarity(self, image: np.ndarray, prompt: str) -> float:
        """Return cosine similarity between image and prompt (0-1)."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        inputs = self.processor(text=[prompt], images=pil_img, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # image-text similarity
            similarity = torch.sigmoid(logits_per_image).item()  # normalize to 0-1
        return similarity