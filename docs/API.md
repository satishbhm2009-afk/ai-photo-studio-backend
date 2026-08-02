# Softonix API Documentation

## `POST /api/v1/upload`
Uploads image or video assets for processing.

### Request
- **Content-Type:** `multipart/form-data`
- **Body:** `files` (Array of files)

### Response
```json
{
  "status": "success",
  "session_id": "uuid-string",
  "original_url": "/results/original_uuid.jpg",
  "enhanced_url": "/results/enhanced_uuid.jpg",
  "total_frames": 15,
  "score": 450.2,
  "faces_detected": 1
}