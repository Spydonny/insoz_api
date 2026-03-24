from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from app.core.config import settings
import certifi

client = AsyncIOMotorClient(
    settings.MONGO_URL,
    tls=True,
    tlsCAFile=certifi.where()
)

db = client[settings.DB_NAME]
fs = AsyncIOMotorGridFSBucket(db, bucket_name="pictures")