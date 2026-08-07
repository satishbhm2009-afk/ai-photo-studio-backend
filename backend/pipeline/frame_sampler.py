import cv2
from backend.pipeline.scene_detector import SceneDetector
from backend.logger import logger

class FrameSampler:
    def __init__(self, interval_sec: float = 0.5, max_frames: int = 500):
        self.interval_sec = interval_sec
        self.max_frames = max_frames
        self.scene_detector = SceneDetector()

    def extract_frames(self, video_path: str) -> list:
        """Extract frames at interval and scene changes.
        Returns list of (frame_index, timestamp_sec, image)."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 30.0
        interval_frames = int(fps * self.interval_sec)

        frames = []
        frame_idx = 0
        self.scene_detector.reset()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample every interval_frames or if scene change
            if frame_idx % interval_frames == 0 or self.scene_detector.is_scene_change(frame):
                timestamp = frame_idx / fps
                frames.append((frame_idx, timestamp, frame))
                if len(frames) >= self.max_frames:
                    break

            frame_idx += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames