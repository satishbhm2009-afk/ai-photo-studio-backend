import cv2
import numpy as np
import tempfile
import os
from backend.pipeline.face_scorer import FaceScorer

def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def extract_best_frame(video_path: str, interval: int = 15) -> str:
    """
    Extract frames, score each using FaceScorer, pick the highest-scoring frame.
    Returns path to the best frame (aligned if face is detected).
    """
    cap = cv2.VideoCapture(video_path)
    scorer = FaceScorer()
    best_score = -1
    best_frame = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            score, aligned_frame = scorer.score_frame(frame)
            if score > best_score:
                best_score = score
                best_frame = aligned_frame if aligned_frame is not None else frame.copy()
        frame_count += 1

    cap.release()

    if best_frame is None:
        raise ValueError("No frames extracted from video.")

    # Save best (aligned) frame
    out_fd, out_path = tempfile.mkstemp(suffix=".jpg")
    os.close(out_fd)
    cv2.imwrite(out_path, best_frame)
    return out_path
