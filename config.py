import os
import secrets
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.environ.get("IDS_SECRET_KEY") or secrets.token_hex(32)
    DATABASE = os.environ.get("IDS_DATABASE", str(BASE_DIR / "database" / "ids.sqlite3"))
    MODEL_DIR = Path(os.environ.get("IDS_MODEL_DIR", str(BASE_DIR / "saved_models")))
    UPLOAD_DIR = Path(os.environ.get("IDS_UPLOAD_DIR", str(BASE_DIR / "uploads")))
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    TESTING = False
