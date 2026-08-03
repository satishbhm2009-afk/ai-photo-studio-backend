import cv2
import numpy as np
import tempfile
import os

def variance_of_laplacian(image):
    """Compute the Laplacian variance (sharpness measure)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def extract_best_frame(video_path: str, interval: int = 30) -> str:
    """
    Read video, sample frames every `interval` frames,
    pick the one with highest Laplacian variance.
    Returns path to the saved best frame.
    """
    cap = cv2.VideoCapture(video_path)
    best_score = -1
    best_frame = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            score = variance_of_laplacian(frame)
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
        frame_count += 1

    cap.release()

    if best_frame is None:
        raise ValueError("No frames extracted from video.")

    # Save best frame as JPEG
    out_fd, out_path = tempfile.mkstemp(suffix=".jpg")
    os.close(out_fd)
    cv2.imwrite(out_path, best_frame)
    return out_path
