import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPScorer:
    def __init__(self):
        self.model = None
        self.processor = None

    def _load(self):
        if self.model is None:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def score(self, image, text_prompt):
        """
        Returns similarity score between image and text (0‑1).
        """
        if not text_prompt or not text_prompt.strip():
            return 0.0

        self._load()  # load model only if needed

        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=[text_prompt], images=pil_img, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            sim = torch.sigmoid(logits_per_image).item()

        return sim
