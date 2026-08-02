from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from config import settings
from routes import router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

# CORS configuration for Hostinger Frontend Access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production में अपनी Hostinger Domain URL यहाँ डालें
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve results statically if required
app.mount("/results", StaticFiles(directory=settings.RESULT_DIR), name="results")

# Register API Routes
app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "AI Photo Studio Backend Online"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
