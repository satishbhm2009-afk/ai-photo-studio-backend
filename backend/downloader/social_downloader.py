import subprocess
import os
import yt_dlp
from backend.config import settings
from backend.logger import logger
from backend.utils import get_temp_file, cleanup_temp_files
from backend.downloader.cookie_manager import CookieManager
from backend.downloader.metadata import extract_metadata
from typing import Optional, Dict

class SocialDownloader:
    @staticmethod
    def download(url: str, cookies: Optional[Dict[str, str]] = None) -> Dict:
        """
        Download video from URL, return dict with video_path, metadata, or error.
        """
        temp_file = get_temp_file(suffix=".mp4")
        cookie_file = None
        if cookies:
            cookie_file = CookieManager.load_cookies(cookies)

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'merge_output_format': 'mp4',
            'outtmpl': temp_file,
            'quiet': True,
            'no_warnings': True,
            'retries': settings.MAX_RETRIES,
            'timeout': settings.DOWNLOAD_TIMEOUT,
            'noplaylist': True,
            'ignoreerrors': False,
            'no_check_certificates': True,
            'prefer_insecure': True,
            'cookiefile': cookie_file if cookie_file else None,
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = temp_file if os.path.exists(temp_file) else None
                if not video_path:
                    raise Exception("Download failed: output file not created")
                metadata = extract_metadata(info)
                return {
                    "success": True,
                    "video_path": video_path,
                    "metadata": metadata
                }
        except Exception as e:
            logger.error(f"Download error: {e}")
            cleanup_temp_files([temp_file])
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            if cookie_file and os.path.exists(cookie_file):
                os.unlink(cookie_file)
