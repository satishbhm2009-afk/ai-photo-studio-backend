import re

def validate_url(url: str) -> bool:
    """Check if URL is a supported social platform or direct video."""
    patterns = [
        r'(youtube\.com|youtu\.be)',
        r'(instagram\.com)',
        r'(facebook\.com)',
        r'(twitter\.com|x\.com)',
        r'(tiktok\.com)',
        r'(vimeo\.com)',
        r'(dailymotion\.com)',
        r'(reddit\.com)',
        r'\.mp4$',
    ]
    for pat in patterns:
        if re.search(pat, url, re.IGNORECASE):
            return True
    return False
