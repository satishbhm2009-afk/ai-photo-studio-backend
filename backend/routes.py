import os
import shutil
import tempfile
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from backend.pipeline.processing import extract_best_frame
from backend.pipeline.body_fusion import fuse_best_parts

router = APIRouter()

@router.post("/extract-best-frame")
async def extract_best_frame_endpoint(video: UploadFile = File(...)):
    """Upload a video, returns the sharpest frame as JPEG."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(video.file, tmp)
            tmp_path = tmp.name

        best_frame_path = extract_best_frame(tmp_path)
        os.unlink(tmp_path)  # clean up video

        return FileResponse(best_frame_path, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reconstruct-from-parts")
async def reconstruct_from_parts(images: list[UploadFile] = File(...)):
    """
    Accept multiple images (e.g., face, torso, legs, etc.) or different poses.
    Returns a composite image with the best parts stitched together.
    """
    try:
        # Save all uploaded images to temp files
        temp_files = []
        for img in images:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                shutil.copyfileobj(img.file, tmp)
                temp_files.append(tmp.name)

        output_path = fuse_best_parts(temp_files)

        # Clean up temp files
        for f in temp_files:
            os.unlink(f)

        return FileResponse(output_path, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
