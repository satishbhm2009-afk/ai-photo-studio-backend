import cv2
import os
import numpy as np
import uuid

class VideoBestFrameExtractor:
    @staticmethod
    def get_blur_score(image: np.ndarray) -> float:
        """
        Calculates image focus/sharpness using Variance of Laplacian.
        Higher value = Clear/Sharp Frame.
        Lower value = Blurry Frame.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @classmethod
    def extract_best_frames(cls, video_path: str, output_dir: str, top_n: int = 4, frame_skip: int = 4) -> list:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Video file ko read nahi kiya ja saka.")

        scored_frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every Nth frame to keep speed fast
            if frame_count % frame_skip == 0:
                score = cls.get_blur_score(frame)
                scored_frames.append((score, frame.copy()))

            frame_count += 1

        cap.release()

        if not scored_frames:
            return []

        # Sort frames by sharpness score (highest score first)
        scored_frames.sort(key=lambda x: x[0], reverse=True)

        saved_files = []

        # Save top_n sharpest frames
        for idx, (score, frame) in enumerate(scored_frames[:top_n]):
            file_name = f"clear_frame_{uuid.uuid4().hex[:8]}.jpg"
            save_path = os.path.join(output_dir, file_name)
            
            # Save frame in High Quality JPEG
            cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            saved_files.append({
                "filename": file_name,
                "score": round(score, 2)
            })

        return saved_files
