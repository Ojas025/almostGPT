import logging.config as config
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_FILE = LOG_DIR / "train.log"

def configure_logging(debug_level="INFO"):

    if not LOG_DIR.exists():
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                "datefmt": "%d-%m-%Y %H:%M:%S"
            },
            "json": {
                "()": "app.logging.formatters.JsonFormatter"
            }
        },
        "handlers": {
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 5,
                "level": "DEBUG",
                "formatter": "default",
                "filename": LOG_FILE,
                "mode": "a",
                "encoding": "utf-8"
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": debug_level
            }
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["file", "console"]
        }
    }

    config.dictConfig(LOGGING_CONFIG)