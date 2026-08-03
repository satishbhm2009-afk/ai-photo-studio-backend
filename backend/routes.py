import os
import shutil
import tempfile
import zipfile
import io
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import Response
from backend.pipeline.processing import extract_best_frames

router = APIRouter()

# Keep your old endpoint if you want, or replace it. I'll add a new one.
@router.post("/extract-top-frames")
async def extract_top_frames(
    video: UploadFile = File(...),
    num_frames: int = Query(10, ge=1, le=50)  # User can request 1 to 50 frames
):
    """
    Upload a video, returns a ZIP file containing the Top N best frames.
    """
    try:
        # 1. Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(video.file, tmp)
            tmp_path = tmp.name

        # 2. Extract Top N frames
        frame_paths = extract_best_frames(tmp_path, num_frames=num_frames)

        # 3. Create a ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, path in enumerate(frame_paths):
                # Add file to zip with a clean name
                arcname = f"best_frame_{i+1}.jpg"
                zf.write(path, arcname)
                os.unlink(path)  # Clean up temp frame file

        os.unlink(tmp_path)  # Clean up temp video file

        # 4. Return ZIP as a downloadable response
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=best_frames_{num_frames}.zip"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
