import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse
from config import settings
from utils import validate_image_file, generate_unique_filename, cleanup_file
from image_processor import AIImageEnhancer

router = APIRouter()

@router.post("/upload")
async def upload_and_process_image(request: Request, file: UploadFile = File(...)):
    """Handles image upload, runs OpenCV processing, and returns result URL."""
    try:
        ext = validate_image_file(file)
        filename = generate_unique_filename(ext)
        
        upload_path = os.path.join(settings.UPLOAD_DIR, filename)
        result_filename = f"enhanced_{filename}"
        result_path = os.path.join(settings.RESULT_DIR, result_filename)

        # Save uploaded file
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process Image using AI Enhancer
        success = AIImageEnhancer.enhance_image(upload_path, result_path, settings.JPEG_QUALITY)

        # Cleanup original upload to save disk space
        cleanup_file(upload_path)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to process image.")

        # Build absolute URL for client download/viewing
        base_url = str(request.base_url).rstrip("/")
        result_url = f"{base_url}/results/{result_filename}"

        return {
            "success": True,
            "filename": result_filename,
            "result_url": result_url,
            "message": "Image processed successfully"
        }

    except HTTPException as http_ex:
        return {"success": False, "message": http_ex.detail}
    except Exception as e:
        return {"success": False, "message": f"An unexpected error occurred: {str(e)}"}


@router.get("/results/{filename}")
async def get_result_image(filename: str):
    """Serves the enhanced image file."""
    file_path = os.path.join(settings.RESULT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Processed image not found.")
    return FileResponse(file_path)
