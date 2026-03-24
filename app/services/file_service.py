import uuid
from fastapi import UploadFile, HTTPException
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from bson import ObjectId
from app.dependencies import db
import os


UPLOAD_DIR = "uploads"

from google.cloud import storage

fs = AsyncIOMotorGridFSBucket(db)
BUCKET_NAME = "insoz"  
storage_client = storage.Client.from_service_account_json("app/insoz.json")
bucket = storage_client.bucket(BUCKET_NAME)

# Получение файла
async def get_file_from_gridfs(picture_id: str) -> tuple[bytes, str] | None:
    try:
        file_obj = await fs.open_download_stream(ObjectId(picture_id))
        file_data = await file_obj.read()
        content_type = file_obj.metadata.get("content_type", "application/octet-stream")
        await file_obj.close()
        return file_data, content_type
    except Exception as e:
        print(f"Ошибка при чтении из GridFS: {e}")
        return None

async def upload_and_convert_to_wav(file: UploadFile) -> str:
    """
    Загружает файл в Google Cloud Storage и возвращает его GCS путь.
    """
    filename = file.filename.lower()
    if "." not in filename:
        raise HTTPException(status_code=400, detail="Invalid file format")

    unique_id = uuid.uuid4()
    blob_name = f"uploads/{unique_id}_{filename}"  # путь внутри bucket

    try:
        # Считываем данные файла
        file_data = await file.read()

        # Загружаем в GCS
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(file_data, content_type=file.content_type)

        # Возвращаем путь к файлу в bucket
        return f"https://storage.googleapis.com/{BUCKET_NAME}/{blob_name}"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GCS upload error: {e}")
    
