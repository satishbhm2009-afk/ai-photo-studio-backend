import os
import shutil
import tempfile
import base64
import cv2
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import Response, JSONResponse
from backend.pipeline.processing import extract_best_frames_with_scores
from backend.pipeline.body_fusion import fuse_best_parts

router = APIRouter()

@router.post("/extract-top-frames")
async def extract_top_frames(
    video: UploadFile = File(...),
    num_frames: int = Query(10, ge=1, le=50),
    prompt: str = Form(""),
    enhance: bool = Form(False)
):
    """
    Upload a video, extract the top N best frames.
    Optionally use a semantic prompt (e.g., "smiling") and/or enable AI enhancement.
    """
    try:
        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(video.file, tmp)
            tmp_path = tmp.name

        # Process video – returns list of (score, original_path, enhanced_path_or_None)
        frame_results = extract_best_frames_with_scores(
            tmp_path,
            num_frames=num_frames,
            prompt=prompt,
            enhance=enhance
        )

        # Build response with base64‑encoded JPEGs
        frames_data = []
        for i, (score, path, enh_path) in enumerate(frame_results):
            # Use enhanced version if available, otherwise original
            img_path = enh_path if enh_path else path
            img = cv2.imread(img_path)
            if img is None:
                continue

            _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            b64 = base64.b64encode(buffer).decode('utf-8')
            frames_data.append({
                "index": i + 1,
                "score": round(score, 2),
                "data": f"data:image/jpeg;base64,{b64}"
            })

            # Clean up temporary image files
            os.unlink(path)
            if enh_path:
                os.unlink(enh_path)

        # Remove the temporary video file
        os.unlink(tmp_path)

        return JSONResponse(content={"frames": frames_data})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reconstruct-from-parts")
async def reconstruct_from_parts(images: list[UploadFile] = File(...)):
    """
    Upload multiple partial images (different poses, body parts).
    AI will select the best parts from each and fuse them into a single seamless composite.
    """
    try:
        temp_files = []
        for img in images:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                shutil.copyfileobj(img.file, tmp)
                temp_files.append(tmp.name)

        # Run the fusion pipeline
        output_path = fuse_best_parts(temp_files)

        # Clean up uploaded temp files
        for f in temp_files:
            os.unlink(f)

        # Return the final composite image
        return Response(
            content=open(output_path, "rb").read(),
            media_type="image/jpeg"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
