from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
import os
import tempfile
import shutil
import logging
from typing import List

from .pipeline.processing import extract_best_frame  # assume this exists
from .downloader.social_downloader import SocialMediaDownloader

router = APIRouter()
logger = logging.getLogger(__name__)

# Existing upload endpoint – keep as is
@router.post("/upload-video")
async def upload_video(video: UploadFile = File(...)):
    try:
        # Save uploaded video
        suffix = os.path.splitext(video.filename)[1] or ".mp4"
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        # Process with existing pipeline
        result_path = extract_best_frame(temp_path)
        # Cleanup temp video
        os.unlink(temp_path)
        return JSONResponse({"image_path": result_path})
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, str(e))

# NEW endpoint for social media download
@router.post("/download-video")
async def download_video_from_url(request: Request):
    try:
        data = await request.json()
        url = data.get("url")
        if not url:
            raise HTTPException(400, "Missing 'url' field")

        # Download using yt-dlp
        downloader = SocialMediaDownloader()
        video_path, metadata = downloader.download_video(url)

        # Pass to existing frame extraction function
        image_path = extract_best_frame(video_path)

        # Cleanup downloaded video
        downloader.cleanup(video_path)

        # Return image path and metadata
        return JSONResponse({
            "image_path": image_path,
            "video_info": metadata
        })

    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Download-video error: {e}")
        raise HTTPException(500, f"Internal error: {str(e)}")
