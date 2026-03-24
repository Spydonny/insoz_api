from pydantic import BaseModel, EmailStr, constr
from uuid import UUID

class UserBase(BaseModel):
    username: EmailStr
    full_name: str | None = None

class UserCreate(UserBase):
    password: constr(min_length=8, max_length=72)

class UserLogin(BaseModel):
    username: EmailStr
    password: str

class UserDB(UserBase):
    uuid: UUID
    hashed_password: str

class UserOut(UserBase):
    uuid: UUID

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class TokenRefreshRequest(BaseModel):
    refresh_token: str
