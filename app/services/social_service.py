from datetime import datetime, timezone
from typing import List, Optional
from bson import ObjectId

from app.dependencies import db
from app.schemas.social import PostCreate, PostOut, CommentCreate, CommentOut, UserProfileOut


# ─── Helpers ────────────────────────────────────────────────────────────────

def _oid(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except Exception:
        raise ValueError(f"Invalid ObjectId: {value}")


async def _get_user_name(user_id: str) -> str:
    """Get full_name of a user by uuid or _id."""
    user = await db["users"].find_one({"uuid": user_id})
    if user:
        return user.get("full_name") or user.get("username", "Unknown")
    return "Unknown"


def _post_doc_to_out(doc: dict) -> PostOut:
    likes = doc.get("likes", [])
    return PostOut(
        id=str(doc["_id"]),
        author_id=doc["author_id"],
        author_name=doc.get("author_name", ""),
        content=doc["content"],
        image_url=doc.get("image_url"),
        likes=likes,
        likes_count=len(likes),
        comments_count=doc.get("comments_count", 0),
        created_at=doc["created_at"],
    )


def _comment_doc_to_out(doc: dict) -> CommentOut:
    return CommentOut(
        id=str(doc["_id"]),
        post_id=doc["post_id"],
        author_id=doc["author_id"],
        author_name=doc.get("author_name", ""),
        content=doc["content"],
        created_at=doc["created_at"],
    )


# ─── Posts ──────────────────────────────────────────────────────────────────

async def create_post(user_id: str, data: PostCreate) -> PostOut:
    author_name = await _get_user_name(user_id)
    doc = {
        "author_id": user_id,
        "author_name": author_name,
        "content": data.content,
        "image_url": data.image_url,
        "likes": [],
        "comments_count": 0,
        "created_at": datetime.now(timezone.utc),
    }
    result = await db["social_posts"].insert_one(doc)
    doc["_id"] = result.inserted_id
    return _post_doc_to_out(doc)


async def get_posts(skip: int = 0, limit: int = 20) -> List[PostOut]:
    cursor = db["social_posts"].find().sort("created_at", -1).skip(skip).limit(limit)
    posts = []
    async for doc in cursor:
        posts.append(_post_doc_to_out(doc))
    return posts


async def get_post_by_id(post_id: str) -> Optional[PostOut]:
    doc = await db["social_posts"].find_one({"_id": _oid(post_id)})
    if not doc:
        return None
    return _post_doc_to_out(doc)


async def delete_post(post_id: str, user_id: str) -> bool:
    result = await db["social_posts"].delete_one(
        {"_id": _oid(post_id), "author_id": user_id}
    )
    return result.deleted_count > 0


async def toggle_like(post_id: str, user_id: str) -> PostOut:
    doc = await db["social_posts"].find_one({"_id": _oid(post_id)})
    if not doc:
        raise ValueError("Post not found")

    likes = doc.get("likes", [])
    if user_id in likes:
        likes.remove(user_id)
    else:
        likes.append(user_id)

    await db["social_posts"].update_one(
        {"_id": _oid(post_id)},
        {"$set": {"likes": likes}}
    )
    doc["likes"] = likes
    return _post_doc_to_out(doc)


async def get_user_posts(user_id: str, skip: int = 0, limit: int = 20) -> List[PostOut]:
    cursor = (
        db["social_posts"]
        .find({"author_id": user_id})
        .sort("created_at", -1)
        .skip(skip)
        .limit(limit)
    )
    posts = []
    async for doc in cursor:
        posts.append(_post_doc_to_out(doc))
    return posts


# ─── Comments ───────────────────────────────────────────────────────────────

async def add_comment(post_id: str, user_id: str, data: CommentCreate) -> CommentOut:
    # Ensure post exists
    post = await db["social_posts"].find_one({"_id": _oid(post_id)})
    if not post:
        raise ValueError("Post not found")

    author_name = await _get_user_name(user_id)
    doc = {
        "post_id": post_id,
        "author_id": user_id,
        "author_name": author_name,
        "content": data.content,
        "created_at": datetime.now(timezone.utc),
    }
    result = await db["social_comments"].insert_one(doc)
    doc["_id"] = result.inserted_id

    # Increment comments_count on post
    await db["social_posts"].update_one(
        {"_id": _oid(post_id)},
        {"$inc": {"comments_count": 1}}
    )

    return _comment_doc_to_out(doc)


async def get_comments(post_id: str, skip: int = 0, limit: int = 50) -> List[CommentOut]:
    cursor = (
        db["social_comments"]
        .find({"post_id": post_id})
        .sort("created_at", 1)
        .skip(skip)
        .limit(limit)
    )
    comments = []
    async for doc in cursor:
        comments.append(_comment_doc_to_out(doc))
    return comments


async def delete_comment(comment_id: str, post_id: str, user_id: str) -> bool:
    result = await db["social_comments"].delete_one(
        {"_id": _oid(comment_id), "author_id": user_id}
    )
    if result.deleted_count > 0:
        await db["social_posts"].update_one(
            {"_id": _oid(post_id)},
            {"$inc": {"comments_count": -1}}
        )
        return True
    return False


# ─── Profile ────────────────────────────────────────────────────────────────

async def get_user_profile(user_id: str) -> Optional[UserProfileOut]:
    user = await db["users"].find_one({"uuid": user_id})
    if not user:
        return None

    posts_count = await db["social_posts"].count_documents({"author_id": user_id})

    return UserProfileOut(
        user_id=user_id,
        full_name=user.get("full_name", ""),
        username=user.get("username", ""),
        bio=user.get("bio"),
        avatar_url=user.get("avatar_url"),
        posts_count=posts_count,
        joined_at=user.get("created_at"),
    )
