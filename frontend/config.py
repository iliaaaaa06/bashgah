"""Frontend settings, read from the project's root .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("FRONTEND_REQUEST_TIMEOUT", "300"))
DEFAULT_TEMPERATURE = float(os.getenv("FRONTEND_DEFAULT_TEMPERATURE", "0.2"))
MAX_TEMPERATURE = float(os.getenv("FRONTEND_MAX_TEMPERATURE", "1.5"))
HEALTH_CACHE_SECONDS = int(os.getenv("FRONTEND_HEALTH_CACHE_SECONDS", "15"))
