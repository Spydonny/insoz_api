from pydantic import BaseModel
from typing import List
from uuid import UUID
from datetime import datetime
from pydantic import Field

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

class ChildRecord(BaseModel):
    id: UUID
    child_uuid: UUID
    file_path: str
    uploaded_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    diagnosis_probabilities: 'DiagnosisProbability' 

class DiagnosisProbability(BaseModel):
    record_id: UUID
    rhotacism: float = Field(..., ge=0, le=1, description="Probability of rhotacism (картавость)")
    lisp: float = Field(..., ge=0, le=1, description="Probability of lisp (шипилявость)")
    general_speech_disorder: float = Field(..., ge=0, le=1, description="Probability of general speech underdevelopment (ОНР)")
    phonetic_phonemic_disorder: float = Field(..., ge=0, le=1, description="Probability of phonetic and phonemic underdevelopment (ФНР)")
    stuttering: float = Field(..., ge=0, le=1, description="Probability of stuttering (заикание)")
    aphasia: float = Field(..., ge=0, le=1, description="Probability of aphasia (афазия)")
    dysarthria: float = Field(..., ge=0, le=1, description="Probability of dysarthria (дизартрия)")
    normal: float = Field(..., ge=0, le=1, description="Probability of normal speech development")