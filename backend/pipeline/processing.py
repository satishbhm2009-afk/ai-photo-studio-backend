import cv2
import numpy as np
import tempfile
import os
from typing import List, Tuple
from backend.pipeline.face_scorer import FaceScorer

def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def extract_best_frames_with_scores(video_path: str, num_frames: int = 10, interval: int = 5) -> List[Tuple[float, str]]:
    """
    Extract frames every 'interval' frames, score each using FaceScorer,
    and return the top 'num_frames' frames that are sharp enough.
    Frames with a sharpness score below 50 are discarded.
    """
    cap = cv2.VideoCapture(video_path)
    scorer = FaceScorer()
    scored_frames = []
    frame_count = 0

    # Minimum sharpness threshold (adjust if needed)
    MIN_SHARPNESS = 50.0  # Laplacian variance value

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            # Get composite score and aligned frame
            score, aligned_frame = scorer.score_frame(frame)
            
            # Extract sharpness component from score (we stored it inside scorer.score_frame,
            # but we can recompute it quickly)
            sharpness = variance_of_laplacian(frame)
            
            # Only keep frames that are sharp enough
            if sharpness >= MIN_SHARPNESS:
                frame_to_save = aligned_frame if aligned_frame is not None else frame.copy()
                scored_frames.append((score, frame_to_save))
        frame_count += 1

    cap.release()

    if not scored_frames:
        raise ValueError("No frames extracted – the video may be too short or too blurry.")

    # Sort by score descending (highest quality first)
    scored_frames.sort(key=lambda x: x[0], reverse=True)

    # Take the top N (or fewer if less available)
    top = scored_frames[:num_frames]
    if len(top) < num_frames:
        print(f"⚠️ Only {len(top)} frames met the sharpness threshold. Returning those.")

    paths = []
    for score, frame in top:
        out_fd, out_path = tempfile.mkstemp(suffix=".jpg")
        os.close(out_fd)
        cv2.imwrite(out_path, frame)
        paths.append((score, out_path))

    return paths
