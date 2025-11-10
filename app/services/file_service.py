import uuid
from fastapi import UploadFile
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from bson import ObjectId
from app.dependencies import db

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
