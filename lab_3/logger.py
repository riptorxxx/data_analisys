import logging
from logging.config import dictConfig
from config import LOG_FILE, LOG_LEVEL


# Настройка логгера с использованием конфига
dictConfig({
    "version": 1,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": str(LOG_FILE),                  # Используем путь из config.py
            "mode": "a"
        }
    },
    "root": {
        "level": LOG_LEVEL,                             # Используем уровень из config.py
        "handlers": ["console", "file"]
    }
})

logger = logging.getLogger(__name__)
