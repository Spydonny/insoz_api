from pydantic_settings import BaseSettings
from os import getenv
from pathlib import Path

class Settings(BaseSettings):
    MONGO_URL: str = getenv("MONGO_URL", "mongodb+srv://test:Test123@insoz.elylyhc.mongodb.net/")
    DB_NAME: str = getenv("DB_NAME", "insozlabdev_db")
    SECRET_KEY: str = getenv("SECRET_KEY", "your_super_secret_key_here")
    ALGORITHM: str = getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "3000"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

    # RAG configuration
    RAG_PDF_DIR: str = getenv("RAG_PDF_DIR", str(Path("speech_therapy_pdfs")))
    RAG_FAISS_DIR: str = getenv("RAG_FAISS_DIR", str(Path("app") / "data" / "faiss_speech_therapy"))

settings = Settings()
