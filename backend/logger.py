```python
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


# ---------------------------------------------------------
# LOG DIRECTORY
# ---------------------------------------------------------
# Render and other cloud platforms usually do not allow
# applications to write inside /var/log.
#
# Therefore, always use a directory inside the application
# working directory.
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

LOG_DIR = BASE_DIR / "logs"

# Create logs directory safely.
# parents=True allows creation if parent directories
# don't already exist.
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# LOGGER SETUP
# ---------------------------------------------------------

def setup_logger(name: str = "ai_photo_studio") -> logging.Logger:

    logger = logging.getLogger(name)

    # Prevent duplicate handlers when modules are imported
    # multiple times.
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # -----------------------------------------------------
    # CONSOLE HANDLER
    # -----------------------------------------------------

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(console)

    # -----------------------------------------------------
    # FILE HANDLER
    # -----------------------------------------------------

    log_file = LOG_DIR / "app.log"

    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10_000_000,
            backupCount=5,
            encoding="utf-8"
        )

        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    except (PermissionError, OSError) as exc:
        # Never allow logging failure to crash the API.
        logger.warning(
            "File logging disabled: %s",
            exc
        )

    return logger


# ---------------------------------------------------------
# GLOBAL LOGGER
# ---------------------------------------------------------

logger = setup_logger()
```
