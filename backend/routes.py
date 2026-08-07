"""
API routes for AI Photo Studio Pro.

- POST /upload-video          – classic file upload
- POST /download-video        – download from URL then run the same pipeline
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.downloader.social_downloader import SocialMediaDownloader
from backend.pipeline.processing import extract_best_frames

router = APIRouter()
logger = logging.getLogger(__name__)


class DownloadRequest(BaseModel):
    url: str = Field(..., description="Public video URL (YouTube, Instagram, X, TikTok, …)")
    num_frames: int = Field(8, ge=1, le=30)
    interval: int = Field(5, ge=1, le=60)
    prompt: str = Field("")
    enhance: bool = Field(False)


def _safe_cleanup(*paths: Optional[str]) -> None:
    for p in paths:
        if p and os.path.isfile(p):
            try:
                os.unlink(p)
            except OSError:
                pass


@router.post("/upload-video")
async def upload_video(
    video: UploadFile = File(...),
    num_frames: int = Form(8),
    interval: int = Form(5),
    prompt: str = Form(""),
    enhance: bool = Form(False),
):
    temp_path: Optional[str] = None
    try:
        suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="upload_")
        os.close(fd)

        with open(temp_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        result = extract_best_frames(
            video_path=temp_path,
            num_frames=num_frames,
            interval=interval,
            prompt=prompt,
            enhance=enhance,
            return_base64=True,
        )
        return JSONResponse(result)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected upload error")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")
    finally:
        _safe_cleanup(temp_path)


@router.post("/download-video")
async def download_video_from_url(body: DownloadRequest):
    downloader = SocialMediaDownloader()
    video_path: Optional[str] = None

    try:
        logger.info("Starting download for URL: %s", body.url)
        video_path, metadata = downloader.download_video(body.url)

        result = extract_best_frames(
            video_path=video_path,
            num_frames=body.num_frames,
            interval=body.interval,
            prompt=body.prompt,
            enhance=body.enhance,
            return_base64=True,
        )
        result["video_info"] = metadata
        return JSONResponse(result)

    except ValueError as exp:
        raise HTTPException(status_code=400, detail=str(exp))
    except Exception as exp:
        logger.exception("Download-video unexpected error")
        raise HTTPException(status_code=500, detail=f"Internal error: {exp}")
    finally:
        if video_path:
            downloader.cleanup(video_path)


@router.post("/download-video-raw")
async def download_video_raw(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    url = data.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url' field")

    body = DownloadRequest(
        url=url,
        num_frames=int(data.get("num_frames", 8)),
        interval=int(data.get("interval", 5)),
        prompt=str(data.get("prompt", "")),
        enhance=bool(data.get("enhance", False)),
    )
    return await download_video_from_url(body)
