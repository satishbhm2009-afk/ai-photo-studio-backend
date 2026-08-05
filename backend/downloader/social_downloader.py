import os
import tempfile
import logging
import re
from typing import Optional, Tuple
import yt_dlp

logger = logging.getLogger(__name__)

class SocialMediaDownloader:
    """Download video from social media URLs using yt-dlp."""

    # Supported domains (for quick validation)
    SUPPORTED_DOMAINS = [
        "youtube.com", "youtu.be",
        "instagram.com",
        "facebook.com", "fb.watch",
        "twitter.com", "x.com",
        "tiktok.com"
    ]

    def __init__(self, max_file_size_mb: int = 500, timeout_seconds: int = 300):
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.timeout = timeout_seconds

    def _validate_url(self, url: str) -> bool:
        """Basic URL validation and domain check."""
        if not url or not url.startswith(("http://", "https://")):
            return False
        # Check if domain is supported
        pattern = r"https?://(?:www\.)?([^/]+)"
        match = re.search(pattern, url)
        if not match:
            return False
        domain = match.group(1)
        return any(support in domain for support in self.SUPPORTED_DOMAINS)

    def download_video(self, url: str) -> Tuple[str, dict]:
        """
        Download video to a temporary file.
        Returns (file_path, video_metadata).
        Raises ValueError on failure.
        """
        if not self._validate_url(url):
            raise ValueError(f"Unsupported or invalid URL: {url}")

        # Create a temporary file (yt-dlp will append extension)
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        os.unlink(temp_path)  # remove empty file; yt-dlp will create the real one

        ydl_opts = {
            "format": "best[ext=mp4]/best",  # prefer mp4 container
            "outtmpl": temp_path,            # yt-dlp will add extension, but we handle
            "quiet": True,
            "no_warnings": True,
            "no_playlist": True,
            "socket_timeout": self.timeout,
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            "merge_output_format": "mp4",     # ensures merging if formats split
            "max_filesize": self.max_file_size,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first to get metadata and final filename
                info = ydl.extract_info(url, download=False)
                if not info:
                    raise ValueError("Could not extract video info")

                # Fix output template: yt-dlp uses the template, but if we set outtmpl without extension,
                # it will add the extension automatically. However, to get the final path, we use ydl.prepare_filename(info)
                # But it's easier to let ydl download and then get the file.
                # We'll set outtmpl to a specific path with extension removed, then get the actual file.
                # Simpler: use a temporary directory and let ydl create files there.
                with tempfile.TemporaryDirectory() as tmpdir:
                    ydl_opts["outtmpl"] = os.path.join(tmpdir, "%(title)s.%(ext)s")
                    ydl = yt_dlp.YoutubeDL(ydl_opts)
                    # Download
                    result = ydl.download([url])
                    if result != 0:
                        raise ValueError("Download failed with error code")
                    # Find the downloaded file
                    files = os.listdir(tmpdir)
                    if not files:
                        raise ValueError("No file downloaded")
                    downloaded_path = os.path.join(tmpdir, files[0])
                    # Move to a persistent temp file (outside tmpdir)
                    final_fd, final_path = tempfile.mkstemp(suffix=os.path.splitext(files[0])[1])
                    os.close(final_fd)
                    os.rename(downloaded_path, final_path)

                    # Clean up tmpdir (will be deleted when context exits)
                    return final_path, {
                        "title": info.get("title", "Untitled"),
                        "duration": info.get("duration", 0),
                        "uploader": info.get("uploader", ""),
                        "thumbnail": info.get("thumbnail", ""),
                    }

        except yt_dlp.utils.DownloadError as e:
            raise ValueError(f"yt-dlp download error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Download failed: {str(e)}")

    def cleanup(self, path: str) -> None:
        """Delete temporary video file."""
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception as e:
            logger.warning(f"Cleanup error for {path}: {e}")
