from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ─── POST ───────────────────────────────────────────────────────────────────

class PostCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)
    image_url: Optional[str] = None


class PostOut(BaseModel):
    id: str
    author_id: str
    author_name: str
    content: str
    image_url: Optional[str] = None
    likes: List[str] = []
    likes_count: int = 0
    comments_count: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


# ─── COMMENT ────────────────────────────────────────────────────────────────

class CommentCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=1000)


class CommentOut(BaseModel):
    id: str
    post_id: str
    author_id: str
    author_name: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


# ─── PROFILE ────────────────────────────────────────────────────────────────

class UserProfileOut(BaseModel):
    user_id: str
    full_name: str
    username: str
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    posts_count: int = 0
    joined_at: Optional[datetime] = None
