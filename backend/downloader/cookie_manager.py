import os
import json
from typing import Dict, Optional
from backend.logger import logger

class CookieManager:
    @staticmethod
    def load_cookies(cookie_data: Dict[str, str]) -> Optional[str]:
        """Convert cookie dict to Netscape format temp file for yt-dlp."""
        if not cookie_data:
            return None
        # Write as JSON or Netscape? yt-dlp accepts --cookies-from-browser or --cookies FILE.
        # For simplicity, we'll use --cookies-from-browser if possible, else we need to format.
        # We'll create a temp file with cookies in netscape format.
        # For this demo, we'll assume cookie_data is already in Netscape format string.
        # Actually we'll just pass as --cookies <file> where we write a simple format.
        # yt-dlp supports --cookies DATA (but we'll use file).
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".txt", text=True)
        with os.fdopen(fd, 'w') as f:
            f.write("# Netscape HTTP Cookie File\n")
            for key, value in cookie_data.items():
                # domain, flag, path, secure, expiration, name, value
                # We'll write a dummy line: domain\tTRUE\t/path\tFALSE\t0\tname\tvalue
                f.write(f".youtube.com\tTRUE\t/\tFALSE\t0\t{key}\t{value}\n")
        return path