RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-contrib-python opencv-python 2>/dev/null || true \
    && pip install --no-cache-dir "opencv-python-headless>=4.10,<4.13"
