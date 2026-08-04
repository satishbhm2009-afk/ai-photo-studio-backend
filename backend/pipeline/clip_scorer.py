import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image

class CLIPScorer:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def score(self, image, text_prompt):
        """
        Returns similarity score between image and text (0‑1).
        """
        if not text_prompt.strip():
            return 0.0
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=[text_prompt], images=pil_img, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # image-text similarity
            sim = torch.sigmoid(logits_per_image).item()
        return sim
