from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes import router

app = FastAPI(title="AI Photo Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve processed images
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

app.include_router(router)

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "AI Photo Studio Backend Online"
    }
