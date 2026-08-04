import numpy as np
from backend.pipeline.utils import variance_of_laplacian, brightness_score

class QualityFilter:
    def __init__(self, min_sharpness: float = 50.0, min_brightness: float = 20.0):
        self.min_sharpness = min_sharpness
        self.min_brightness = min_brightness

    def is_acceptable(self, frame: np.ndarray) -> bool:
        sharpness = variance_of_laplacian(frame)
        if sharpness < self.min_sharpness:
            return False

        bright = brightness_score(frame)
        if bright < self.min_brightness:
            return False

        return True
