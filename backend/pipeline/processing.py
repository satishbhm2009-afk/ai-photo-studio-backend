import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

from backend.pipeline.frame_extractor import FrameExtractor
from backend.pipeline.quality import QualityAnalyzer
from backend.pipeline.face_alignment import FaceAlignerDetector
from backend.pipeline.enhancement import ImageEnhancer
from backend.pipeline.ai_providers import get_ai_provider

class MediaProcessingPipeline:
    """Master Orchestrator Pipeline coordinating extraction, selection, and restoration."""

    def __init__(self):
        self.face_aligner = FaceAlignerDetector()

    def process_session(self, input_paths: List[str], is_video: bool, session_temp_dir: str, session_results_dir: str, session_id: str) -> Dict[str, Any]:
        # Step 1: Frame Extraction / Input collection
        candidate_frame_paths: List[str] = []
        if is_video:
            video_path = input_paths[0]
            candidate_frame_paths = FrameExtractor.extract_frames(video_path, session_temp_dir, target_count=15)
        else:
            candidate_frame_paths = input_paths

        if not candidate_frame_paths:
            raise ValueError("No valid image frames available for processing.")

        # Step 2: Quality Analysis & Face Scoring
        evaluated_frames: List[Dict[str, Any]] = []

        for frame_path in candidate_frame_paths:
            # Quality metrics
            q_metrics = QualityAnalyzer.evaluate_frame(frame_path)
            
            # Face metrics
            _, face_meta = self.face_aligner.process(frame_path)

            # Calculate unified score
            # Formula: (Blur * 0.35) + (Contrast * 0.15) + (FacesCount * 120) + (FaceAreaRatio * 400)
            composite_score = (
                (q_metrics["blur_score"] * 0.35) +
                (q_metrics["contrast_score"] * 0.15) +
                (face_meta["faces_detected"] * 120.0) +
                (face_meta["face_area_ratio"] * 400.0)
            )

            evaluated_frames.append({
                "path": frame_path,
                "score": composite_score,
                "blur_score": q_metrics["blur_score"],
                "faces_detected": face_meta["faces_detected"],
                "is_blurry": q_metrics["is_blurry"]
            })

        # Step 3: Best Frame Selection
        evaluated_frames.sort(key=lambda x: x["score"], reverse=True)
        best_frame_info = evaluated_frames[0]
        best_frame_path = best_frame_info["path"]

        # Step 4: Face Alignment on Best Frame
        aligned_img, align_meta = self.face_aligner.process(best_frame_path)

        # Step 5: Save Original Frame for Result Comparison
        orig_filename = f"original_{session_id}.jpg"
        orig_result_path = os.path.join(session_results_dir, orig_filename)
        cv2.imwrite(orig_result_path, aligned_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Step 6: Classical Enhancement & Upscaling
        interim_enhanced_path = os.path.join(session_temp_dir, f"interim_{session_id}.jpg")
        ImageEnhancer.enhance_and_upscale(aligned_img, interim_enhanced_path)

        # Step 7: AI Provider Enhancement
        final_filename = f"enhanced_{session_id}.jpg"
        final_result_path = os.path.join(session_results_dir, final_filename)
        
        provider = get_ai_provider()
        try:
            provider.enhance(interim_enhanced_path, final_result_path)
        except Exception as err:
            # Fallback to classical result if external API fails
            import shutil
            shutil.copyfile(interim_enhanced_path, final_result_path)

        return {
            "session_id": session_id,
            "original_filename": orig_filename,
            "enhanced_filename": final_filename,
            "total_frames_analyzed": len(candidate_frame_paths),
            "best_frame_score": round(best_frame_info["score"], 2),
            "faces_detected": best_frame_info["faces_detected"],
            "blur_score": round(best_frame_info["blur_score"], 2)
        }