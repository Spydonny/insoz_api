import uuid
from fastapi import UploadFile, HTTPException
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from bson import ObjectId
from app.dependencies import db
import os


UPLOAD_DIR = "uploads"


fs = AsyncIOMotorGridFSBucket(db)

async def save_file_to_gridfs(picture: UploadFile) -> str | None:
    if not picture:
        return None

    file_data = await picture.read()
    filename = f"{uuid.uuid4()}_{picture.filename}"
    picture_id = await fs.upload_from_stream(
        filename,
        file_data,
        metadata={"content_type": picture.content_type},
    )
    return str(picture_id)

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
    # Проверяем, есть ли расширение
    filename = file.filename.lower()
    if "." not in filename:
        raise HTTPException(status_code=400, detail="Invalid file format")

    # Уникальное имя
    unique_id = uuid.uuid4()
    file_path = os.path.join(UPLOAD_DIR, f"{unique_id}_{filename}")

    try:
        # Сохраняем оригинальный файл
        file_data = await file.read()
        with open(file_path, "wb") as f:
            f.write(file_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload error: {e}")

    return file_path