import os
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse

from backend.config import settings
from backend.utils import generate_unique_id, is_allowed_file, is_video_file, save_upload_file, cleanup_directory
from backend.pipeline.processing import MediaProcessingPipeline

router = APIRouter()
pipeline = MediaProcessingPipeline()

@router.get("/health")
def health_check():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "ai_provider": settings.AI_PROVIDER
    }

@router.post("/upload")
async def handle_media_upload(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="No media files uploaded.")

    session_id = generate_unique_id()
    session_upload_dir = os.path.join(settings.TEMP_DIR, session_id)
    os.makedirs(session_upload_dir, exist_ok=True)

    saved_file_paths: List[str] = []
    has_video = False

    for file in files:
        if not is_allowed_file(file.filename):
            continue

        if is_video_file(file.filename):
            has_video = True

        file_path = os.path.join(session_upload_dir, file.filename)
        await save_upload_file(file, file_path)
        saved_file_paths.append(file_path)

    if not saved_file_paths:
        cleanup_directory(session_upload_dir)
        raise HTTPException(
            status_code=400, 
            detail="No valid files saved. Ensure uploads are supported image/video formats."
        )

    if has_video and len(saved_file_paths) > 1:
        cleanup_directory(session_upload_dir)
        raise HTTPException(
            status_code=400,
            detail="Please upload either 1 Video clip OR multiple Photos (5-20), not both."
        )

    try:
        result_meta = pipeline.process_session(
            input_paths=saved_file_paths,
            is_video=has_video,
            session_temp_dir=session_upload_dir,
            session_results_dir=settings.RESULTS_DIR,
            session_id=session_id
        )

        # Schedule temp files cleanup in background
        background_tasks.add_task(cleanup_directory, session_upload_dir)

        return JSONResponse(status_code=200, content={
            "status": "success",
            "session_id": session_id,
            "original_url": f"/results/{result_meta['original_filename']}",
            "enhanced_url": f"/results/{result_meta['enhanced_filename']}",
            "total_frames": result_meta["total_frames_analyzed"],
            "score": result_meta["best_frame_score"],
            "faces_detected": result_meta["faces_detected"]
        })

    except Exception as e:
        cleanup_directory(session_upload_dir)
        raise HTTPException(status_code=500, detail=f"Processing Pipeline Error: {str(e)}")

@router.get("/results/{filename}")
def download_result_file(filename: str):
    file_path = os.path.join(settings.RESULTS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found or expired.")
    return FileResponse(file_path)