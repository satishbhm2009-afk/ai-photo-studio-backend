import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from config import settings
from video_processor import VideoBestFrameExtractor

app = FastAPI(title="AI Photo Studio - Clear Frame Extractor")

# Allow requests from Hostinger Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve result images statically
app.mount("/results", StaticFiles(directory=settings.RESULT_DIR), name="results")

@app.get("/")
def health_check():
    return {"status": "AI Video Processing Engine Active"}

@app.post("/api/extract-video-frames")
async def process_video_frames(request: Request, video: UploadFile = File(...)):
    ext = os.path.splitext(video.filename)[1].lower()
    if ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Kewal MP4, MOV, AVI ya MKV video upload karein.")

    temp_video_filename = f"temp_{video.filename}"
    temp_video_path = os.path.join(settings.UPLOAD_DIR, temp_video_filename)

    try:
        # Save video temporarily
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Extract top 4 clear frames
        results = VideoBestFrameExtractor.extract_best_frames(
            video_path=temp_video_path,
            output_dir=settings.RESULT_DIR,
            top_n=4,
            frame_skip=3
        )

        base_url = str(request.base_url).rstrip("/")
        output_data = []

        for item in results:
            output_data.append({
                "filename": item["filename"],
                "url": f"{base_url}/results/{item['filename']}",
                "sharpness": item["score"]
            })

        return {
            "success": True,
            "message": "Video analyzed successfully. Clear frames extracted.",
            "frames": output_data
        }

    except Exception as e:
        return {"success": False, "message": str(e)}

    finally:
        # Clean up temporary video file to save disk space
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
