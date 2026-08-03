import os
import uuid
import shutil
import tempfile

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from backend.pipeline.processing import extract_best_frame
from backend.pipeline.body_fusion import fuse_best_parts

router = APIRouter()

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@router.post("/extract-best-frame")
async def extract_best_frame_endpoint(video: UploadFile = File(...)):
    """
    Upload a video and return the best frame.
    """

    try:

        # Save uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(video.file, tmp)
            video_path = tmp.name

        # AI Processing
        best_frame = extract_best_frame(video_path)

        # Delete uploaded temp video
        os.unlink(video_path)

        # Save final image
        filename = f"{uuid.uuid4()}.jpg"
        final_path = os.path.join(OUTPUT_DIR, filename)

        shutil.copy(best_frame, final_path)

        return JSONResponse({

            "success": True,

            "original_url": None,

            "enhanced_url": f"/outputs/{filename}",

            "total_frames": 0,

            "faces_detected": 0,

            "processing_time": "Completed"

        })

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/reconstruct-from-parts")
async def reconstruct_from_parts(images: list[UploadFile] = File(...)):
    """
    Merge the best body parts from multiple images.
    """

    try:

        temp_files = []

        for img in images:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                shutil.copyfileobj(img.file, tmp)
                temp_files.append(tmp.name)

        output = fuse_best_parts(temp_files)

        filename = f"{uuid.uuid4()}.jpg"

        final_path = os.path.join(OUTPUT_DIR, filename)

        shutil.copy(output, final_path)

        for f in temp_files:
            os.unlink(f)

        return JSONResponse({

            "success": True,

            "enhanced_url": f"/outputs/{filename}"

        })

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
