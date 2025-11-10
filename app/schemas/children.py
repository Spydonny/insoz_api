from pydantic import BaseModel
from typing import List
from uuid import UUID

class ChildBase(BaseModel):
    name: str
    age: int
    diagnosis: List[str]
    picture_id: str | None = None


class ChildCreate(ChildBase):
    pass

class ChildDB(ChildBase):
    uuid: UUID
    doctor_id: UUID

    class Config:
        orm_mode = True
