# Intentionally minimal – heavy modules load only when needed
__all__ = ["process_video_pipeline", "extract_best_frames"]

def __getattr__(name):
    if name in ("process_video_pipeline", "extract_best_frames"):
        from .processing import process_video_pipeline, extract_best_frames
        return process_video_pipeline if name == "process_video_pipeline" else extract_best_frames
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
