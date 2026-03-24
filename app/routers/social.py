from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import OAuth2PasswordBearer
from typing import List

from app.core.security import decode_token
from app.services.user_service import get_user_by_username
from app.schemas.social import (
    PostCreate, PostOut,
    CommentCreate, CommentOut,
    UserProfileOut,
)
from app.services.social_service import (
    create_post, get_posts, get_post_by_id, delete_post,
    toggle_like, get_user_posts,
    add_comment, get_comments, delete_comment,
    get_user_profile,
)

router = APIRouter(prefix="/social", tags=["social"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")


# ─── Auth dependency ────────────────────────────────────────────────────────

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


# ─── Posts ──────────────────────────────────────────────────────────────────

@router.post("/posts", response_model=PostOut, status_code=201)
async def create_new_post(
    data: PostCreate,
    user_id: str = Depends(get_current_user_id),
):
    return await create_post(user_id, data)


@router.get("/posts", response_model=List[PostOut])
async def list_posts(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    _: str = Depends(get_current_user_id),  # require auth
):
    return await get_posts(skip=skip, limit=limit)


@router.get("/posts/{post_id}", response_model=PostOut)
async def get_single_post(
    post_id: str,
    _: str = Depends(get_current_user_id),
):
    post = await get_post_by_id(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Пост не найден")
    return post


@router.delete("/posts/{post_id}", status_code=204)
async def remove_post(
    post_id: str,
    user_id: str = Depends(get_current_user_id),
):
    deleted = await delete_post(post_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Пост не найден или нет доступа")


@router.post("/posts/{post_id}/like", response_model=PostOut)
async def like_post(
    post_id: str,
    user_id: str = Depends(get_current_user_id),
):
    try:
        return await toggle_like(post_id, user_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Пост не найден")


# ─── Comments ───────────────────────────────────────────────────────────────

@router.post("/posts/{post_id}/comments", response_model=CommentOut, status_code=201)
async def add_new_comment(
    post_id: str,
    data: CommentCreate,
    user_id: str = Depends(get_current_user_id),
):
    try:
        return await add_comment(post_id, user_id, data)
    except ValueError:
        raise HTTPException(status_code=404, detail="Пост не найден")


@router.get("/posts/{post_id}/comments", response_model=List[CommentOut])
async def list_comments(
    post_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    _: str = Depends(get_current_user_id),
):
    return await get_comments(post_id, skip=skip, limit=limit)


@router.delete("/posts/{post_id}/comments/{comment_id}", status_code=204)
async def remove_comment(
    post_id: str,
    comment_id: str,
    user_id: str = Depends(get_current_user_id),
):
    deleted = await delete_comment(comment_id, post_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Комментарий не найден или нет доступа")


# ─── Profile ────────────────────────────────────────────────────────────────

@router.get("/profile/{user_id}", response_model=UserProfileOut)
async def get_profile(
    user_id: str,
    _: str = Depends(get_current_user_id),
):
    profile = await get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    return profile


@router.get("/profile/{user_id}/posts", response_model=List[PostOut])
async def get_profile_posts(
    user_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    _: str = Depends(get_current_user_id),
):
    return await get_user_posts(user_id, skip=skip, limit=limit)
