import cv2
import numpy as np
import tempfile
import os
from typing import List, Tuple
from backend.pipeline.face_scorer import FaceScorer  # We built this in Phase 2

def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def extract_best_frames(video_path: str, num_frames: int = 10, interval: int = 15) -> List[str]:
    """
    Extract frames, score each using FaceScorer, return the Top N frames.
    Returns a list of file paths to the saved JPEGs.
    """
    cap = cv2.VideoCapture(video_path)
    scorer = FaceScorer()
    scored_frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            # Get composite score and aligned frame
            score, aligned_frame = scorer.score_frame(frame)
            frame_to_save = aligned_frame if aligned_frame is not None else frame.copy()
            scored_frames.append((score, frame_to_save))
        frame_count += 1

    cap.release()

    if not scored_frames:
        raise ValueError("No frames extracted from video.")

    # Sort by score descending (highest quality first)
    scored_frames.sort(key=lambda x: x[0], reverse=True)

    # Take only the top N (or fewer if video is short)
    top_frames = scored_frames[:num_frames]
    if len(top_frames) < num_frames:
        print(f"⚠️ Video only had {len(top_frames)} valid frames. Returning {len(top_frames)}.")

    # Save frames to temporary files
    paths = []
    for i, (score, frame) in enumerate(top_frames):
        out_fd, out_path = tempfile.mkstemp(suffix=f"_frame_{i+1}.jpg")
        os.close(out_fd)
        cv2.imwrite(out_path, frame)
        paths.append(out_path)

    return paths
