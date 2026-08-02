import cv2
import os
import uuid
from typing import List, Tuple
from backend.config import settings

class FrameExtractor:
    """Extracts frames from video files at dynamic calculated intervals."""

    @staticmethod
    def extract_frames(video_path: str, session_temp_dir: str, target_count: int = 15) -> List[str]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Unable to open video stream or corrupt file.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        if total_frames <= 0:
            total_frames = 100

        # Calculate interval step
        step = max(1, total_frames // target_count)
        extracted_paths: List[str] = []
        frame_index = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or saved_count >= target_count:
                break

            if frame_index % step == 0:
                frame_filename = f"frame_{saved_count:02d}_{uuid.uuid4().hex[:6]}.jpg"
                save_path = os.path.join(session_temp_dir, frame_filename)
                
                # Write frame with high JPG quality
                cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                extracted_paths.append(save_path)
                saved_count += 1

            frame_index += 1

        cap.release()

        if not extracted_paths:
            raise RuntimeError("Frame extraction returned 0 valid frames.")

        return extracted_paths