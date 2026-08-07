import cv2
import numpy as np
from backend.config import settings
from backend.logger import logger

class SceneDetector:
    def __init__(self, threshold: float = None):
        self.threshold = threshold or settings.SCENE_CHANGE_THRESHOLD
        self.prev_hist = None

    def is_scene_change(self, frame: np.ndarray) -> bool:
        """Return True if frame is a scene change from previous."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if self.prev_hist is None:
            self.prev_hist = hist
            return False

        # Correlation or chi-square
        correlation = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL)
        self.prev_hist = hist
        # If correlation low, scene change
        return correlation < (1 - self.threshold)

    def reset(self):
        self.prev_hist = None