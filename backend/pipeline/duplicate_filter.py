import cv2
import numpy as np
from typing import List, Tuple

class DuplicateFilter:
    def __init__(self, threshold: float = 0.92):
        self.threshold = threshold

    def is_duplicate(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Histogram correlation based duplicate check."""
        hist1 = cv2.calcHist([frame1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([frame2], [0], None, [64], [0, 256])
        corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return corr > self.threshold

    def filter(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        unique = []
        for frame in frames:
            is_dup = False
            for existing in unique:
                if self.is_duplicate(frame, existing):
                    is_dup = True
                    break
            if not is_dup:
                unique.append(frame)
        return unique
