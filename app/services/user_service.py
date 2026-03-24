from datetime import timedelta
from fastapi import HTTPException, status
from uuid import uuid4

from app.core.security import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token
)
from app.core.config import settings
from app.schemas.user import UserCreate, UserLogin, UserOut, Token, TokenRefreshRequest
from app.dependencies import db

users_collection = db["users"]
tokens_collection = db["refresh_tokens"]

async def register_user(user: UserCreate) -> dict:
    existing_user = await users_collection.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already exists"
        )

    hashed_pwd = hash_password(user.password)
    new_user = {
        "uuid": str(uuid4()),
        "username": user.username,
        "full_name": user.full_name,
        "hashed_password": hashed_pwd,
    }

    await users_collection.insert_one(new_user)
    return {"message": "User registered successfully"}

async def authenticate_user(user_data: UserLogin) -> Token:
    user = await users_collection.find_one({"username": user_data.username})
    if not user or not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["username"]}, expires_delta=access_token_expires)
    refresh_token = create_refresh_token(data={"sub": user["username"]})

    # Сохраняем refresh токен в БД
    await tokens_collection.insert_one({"username": user["username"], "refresh_token": refresh_token})

    return Token(access_token=access_token, refresh_token=refresh_token, token_type="bearer")

async def refresh_access_token(data: TokenRefreshRequest) -> Token:
    payload = decode_token(data.refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=400, detail="Invalid token type")

    username = payload.get("sub")

    token_in_db = await tokens_collection.find_one({"username": username, "refresh_token": data.refresh_token})
    if not token_in_db:
        raise HTTPException(status_code=401, detail="Invalid or revoked refresh token")

    access_token = create_access_token(data={"sub": username})
    new_refresh_token = create_refresh_token(data={"sub": username})

    # Обновляем refresh токен в БД
    await tokens_collection.update_one(
        {"_id": token_in_db["_id"]},
        {"$set": {"refresh_token": new_refresh_token}}
    )

    return Token(access_token=access_token, refresh_token=new_refresh_token, token_type="bearer")

async def logout_user(refresh_token: str):
    """Удаляет refresh токен из базы (logout)."""
    result = await tokens_collection.delete_one({"refresh_token": refresh_token})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Token not found")
    return {"message": "Logged out successfully"}

async def get_user_by_username(username: str) -> UserOut:
    user = await users_collection.find_one({"username": username})
    return UserOut(**user)
