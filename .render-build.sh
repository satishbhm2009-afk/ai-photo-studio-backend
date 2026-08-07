#!/usr/bin/env bash
set -e
apt-get update && apt-get install -y --no-install-recommends ffmpeg
pip install -r requirements.txt
# Force headless OpenCV (MediaPipe may have pulled the full contrib package)
pip uninstall -y opencv-contrib-python opencv-python 2>/dev/null || true
pip install --no-cache-dir "opencv-python-headless>=4.10,<4.13"
