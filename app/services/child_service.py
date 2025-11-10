from app.dependencies import db
from app.schemas.children import ChildCreate, ChildDB
from bson import ObjectId
from typing import Optional
from uuid import uuid4

children_collection = db["children"]

async def create_child(child: ChildCreate, doctor_id: str) -> dict:

    new_child = child.dict()
    new_child["uuid"] = str(uuid4())
    new_child["doctor_id"] = doctor_id

    result = await children_collection.insert_one(new_child)
    return {
        "message": "Child created successfully",
        "child_id": str(result.inserted_id),
        "uuid": new_child["uuid"],
        "doctor_id": doctor_id,
        "image_id": new_child.get("picture_id")
    }

async def get_all_children() -> list[ChildDB]:

    children_cursor = children_collection.find()
    children = []
    async for child in children_cursor:
        child["_id"] = str(child["_id"])
        children.append(ChildDB(**child))
    return children

async def get_child_by_id(child_id: str) -> Optional[ChildDB]:

    child = await children_collection.find_one({"_id": ObjectId(child_id)})
    if not child:
        return None

    child["_id"] = str(child["_id"])
    return ChildDB(**child)

async def get_child_by_uuid(uuid: str) -> Optional[ChildDB]:
    child = await children_collection.find_one({"uuid": uuid})
    if not child:
        return None

    child["_id"] = str(child["_id"])
    return ChildDB(**child)

async def get_child_by_doctor_id(doctor_id: str) -> list[ChildDB]:
    children_cursor = children_collection.find({"doctor_id": doctor_id})
    children = []
    async for child in children_cursor:
        child["_id"] = str(child["_id"])
        children.append(ChildDB(**child))
    return children
