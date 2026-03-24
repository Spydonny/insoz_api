from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class DiagnosisProbability(BaseModel):
    record_id: UUID
    rhotacism: float = Field(..., ge=0, le=1, description="Probability of rhotacism (картавость)")
    lisp: float = Field(..., ge=0, le=1, description="Probability of lisp (шепелявость)")
    general_speech_disorder: float = Field(..., ge=0, le=1, description="Probability of general speech underdevelopment (ОНР)")
    phonetic_phonemic_disorder: float = Field(..., ge=0, le=1, description="Probability of phonetic and phonemic underdevelopment (ФФНР)")
    stuttering: float = Field(..., ge=0, le=1, description="Probability of stuttering (заикание)")
    aphasia: float = Field(..., ge=0, le=1, description="Probability of aphasia (афазия)")
    dysarthria: float = Field(..., ge=0, le=1, description="Probability of dysarthria (дизартрия)")
    normal: float = Field(..., ge=0, le=1, description="Probability of normal speech development")


class PhonemeScoreItem(BaseModel):
    phoneme: str
    letter: str
    score: int = Field(..., ge=0)
    best_similarity: Optional[float] = Field(default=None, ge=0, le=1)


class PhonemeAnalysisEntry(BaseModel):
    id: str
    child_uuid: str
    analysis_type: Literal["manual", "ai"]
    language: str
    phonemes: List[str] = Field(default_factory=list)
    scores: Dict[str, int] = Field(default_factory=dict)
    results: List[PhonemeScoreItem] = Field(default_factory=list)
    max_score: int = Field(..., ge=1)
    average_score: float = Field(default=0, ge=0)
    summary: Optional[str] = None
    comment: Optional[str] = None
    audio_file_path: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ChildRecord(BaseModel):
    id: UUID
    child_uuid: UUID
    file_path: str
    uploaded_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    diagnosis_probabilities: Optional[DiagnosisProbability] = None


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

    model_config = {
        "from_attributes": True
    }


class ChildDetailResponse(ChildBase):
    uuid: UUID
    doctor_id: UUID
    records: List[ChildRecord] = Field(default_factory=list)
    phoneme_analyses: List[PhonemeAnalysisEntry] = Field(default_factory=list)

    model_config = {
        "from_attributes": True
    }


class AnalyzePhonemesRequest(BaseModel):
    language: str
    phonemes: List[str]
    max_score: int = Field(..., ge=1)


class AnalyzePhonemesResponse(BaseModel):
    analysis: PhonemeAnalysisEntry


class ManualPhonemeScoreRequest(BaseModel):
    language: str
    phonemes: List[str]
    scores: Dict[str, int]
    max_score: int = Field(..., ge=1)
    comment: Optional[str] = None


class ManualPhonemeScoreResponse(BaseModel):
    analysis: PhonemeAnalysisEntry


class ChildPhonemeAnalysesResponse(BaseModel):
    analyses: List[PhonemeAnalysisEntry] = Field(default_factory=list)
