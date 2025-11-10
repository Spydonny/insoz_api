from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGO_URL: str = "mongodb+srv://user_1:NUsqvYgm8sR7Srk@cluster0.h9bjsv6.mongodb.net/myDatabase?retryWrites=true&w=majority&appName=Cluster0"
    DB_NAME: str = "insozlabdev_db"
    SECRET_KEY: str = "your_super_secret_key_here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 3000
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

settings = Settings()
