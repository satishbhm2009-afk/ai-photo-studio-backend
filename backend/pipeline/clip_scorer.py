import torch
import cv2
import numpy as np
from PIL import Image

class CLIPScorer:
    def __init__(self):
        self.model = None
        self.processor = None

    def _load(self):
        if self.model is None:
            from transformers import CLIPProcessor, CLIPModel
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def score(self, image: np.ndarray, text_prompt: str) -> float:
        if not text_prompt or not text_prompt.strip():
            return 0.0

        self._load()
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=[text_prompt], images=pil_img, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            sim = torch.sigmoid(outputs.logits_per_image).item()

        return sim
