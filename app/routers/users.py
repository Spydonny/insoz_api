from fastapi import APIRouter, Depends, HTTPException, status
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from app.core.security import decode_token
from app.core.config import settings
from app.schemas.user import UserCreate, UserLogin, UserOut, Token, TokenRefreshRequest
from app.services.user_service import (
    register_user, authenticate_user, get_user_by_username,
    refresh_access_token, logout_user
)

router = APIRouter(prefix="/users", tags=["users"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")

async def get_current_user_id(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_token(token)
    username = payload.get("sub")
    user = await get_user_by_username(username)
    if user is None:
        raise credentials_exception

    return str(user.uuid)


@router.post("/register")
async def register(user: UserCreate):
    return await register_user(user)

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user_data = UserLogin(username=form_data.username, password=form_data.password)
    return await authenticate_user(user_data)

@router.post("/refresh", response_model=Token)
async def refresh_token(request: TokenRefreshRequest):
    return await refresh_access_token(request)

@router.post("/logout")
async def logout(request: TokenRefreshRequest):
    return await logout_user(request.refresh_token)

@router.get("/me", response_model = UserOut)
async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_token(token)
    username = payload.get("sub")
    user = await get_user_by_username(username)
    if not user:
        return {"error": "User not found"}

    return user
