import cv2
import numpy as np
import tempfile
import os
from typing import List, Tuple
from backend.pipeline.face_scorer import FaceScorer

def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def extract_best_frames_with_scores(video_path: str, num_frames: int = 10, interval: int = 15) -> List[Tuple[float, str]]:
    cap = cv2.VideoCapture(video_path)
    scorer = FaceScorer()
    scored_frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            score, aligned_frame = scorer.score_frame(frame)
            frame_to_save = aligned_frame if aligned_frame is not None else frame.copy()
            scored_frames.append((score, frame_to_save))
        frame_count += 1

    cap.release()

    if not scored_frames:
        raise ValueError("No frames extracted.")

    scored_frames.sort(key=lambda x: x[0], reverse=True)
    top = scored_frames[:num_frames]

    paths = []
    for score, frame in top:
        out_fd, out_path = tempfile.mkstemp(suffix=".jpg")
        os.close(out_fd)
        cv2.imwrite(out_path, frame)
        paths.append((score, out_path))

    return paths
