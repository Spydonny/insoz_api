from pydantic import BaseModel, Field


class RagTherapyAnswerRequest(BaseModel):
    child_uuid: str = Field(..., description="Child UUID to personalize retrieval")
    question: str = Field(..., min_length=1, description="User question for the therapist assistant")
    k_total: int = Field(2, ge=1, le=30, description="Total retrieval budget across disorders")
    include_context: bool = Field(False, description="If true, return retrieved context blocks")


class RagTherapyAnswerResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list, description="Source PDF filenames used in context")
    context: str | None = Field(default=None, description="Returned only when include_context=true")

