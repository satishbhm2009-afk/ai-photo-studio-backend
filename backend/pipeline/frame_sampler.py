import cv2
import numpy as np
from typing import Generator, Tuple, Optional

def sample_frames(video_path: str, interval: int = 5) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Generator that yields (frame_number, frame) from a video.
    Skips frames by `interval` to reduce processing time.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            yield frame_count, frame
        frame_count += 1
    cap.release()
