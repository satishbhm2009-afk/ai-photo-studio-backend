from typing import Dict, Any

def extract_metadata(info: Dict) -> Dict:
    """Extract relevant metadata from yt-dlp info dict."""
    return {
        "title": info.get("title"),
        "duration": info.get("duration"),
        "resolution": f"{info.get('width')}x{info.get('height')}" if info.get('width') else None,
        "format": info.get("ext"),
        "size": info.get("filesize") or info.get("filesize_approx"),
    }