import os
import shutil
import tempfile
import zipfile
import io
import base64
import cv2
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import Response, JSONResponse
from backend.pipeline.processing import extract_best_frames_with_scores, enhance_frames
from backend.pipeline.body_fusion import fuse_best_parts

router = APIRouter()

@router.post("/extract-top-frames")
async def extract_top_frames(
    video: UploadFile = File(...),
    num_frames: int = Query(10, ge=1, le=50),
    prompt: str = Form(""),  # optional semantic prompt
    enhance: bool = Form(False)
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(video.file, tmp)
            tmp_path = tmp.name

        # Extract and score frames
        frame_results = extract_best_frames_with_scores(tmp_path, num_frames=num_frames, prompt=prompt, enhance=enhance)

        # Each result: (score, image_path, enhanced_path if enhance else None)
        frames_data = []
        for i, (score, path, enh_path) in enumerate(frame_results):
            img = cv2.imread(enh_path if enh_path else path)
            if img is None:
                continue
            _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            b64 = base64.b64encode(buffer).decode('utf-8')
            frames_data.append({
                "index": i + 1,
                "score": round(score, 2),
                "data": f"data:image/jpeg;base64,{b64}"
            })
            os.unlink(path)
            if enh_path:
                os.unlink(enh_path)

        os.unlink(tmp_path)
        return JSONResponse(content={"frames": frames_data})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reconstruct-from-parts")
async def reconstruct_from_parts(images: list[UploadFile] = File(...)):
    # ... (unchanged from previous version)
