from fastapi import APIRouter, Depends, HTTPException, status

from app.routers.users import get_current_user_id
from app.schemas.rag import RagTherapyAnswerRequest, RagTherapyAnswerResponse
from app.services.child_service import get_child_by_uuid
from app.services.rag_service import (
    answer_with_gemini,
    get_vector_db,
    patient_profile_from_child,
    retrieve_weighted_context,
)


router = APIRouter(prefix="/rag", tags=["rag"], dependencies=[Depends(get_current_user_id)])


@router.post("/therapy-answer", response_model=RagTherapyAnswerResponse)
async def rag_therapy_answer(
    payload: RagTherapyAnswerRequest,
    doctor_id: str = Depends(get_current_user_id),
):
    child = await get_child_by_uuid(payload.child_uuid)
    if not child:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Child not found")

    # `get_child_by_uuid` returns a Pydantic model; access attrs via dict conversion.
    child_dict = child.model_dump() if hasattr(child, "model_dump") else dict(child)

    if str(child_dict.get("doctor_id")) != doctor_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized access")

    try:
        db = get_vector_db()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )

    patient_profile = patient_profile_from_child(child_dict)
    context, sources = retrieve_weighted_context(patient_profile, db, k_total=payload.k_total)

    if not context.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No diagnosis probabilities found for this child. Add a record first to generate diagnosis weights.",
        )

    answer = answer_with_gemini(payload.question, patient_profile, context)
    return RagTherapyAnswerResponse(
        answer=answer,
        sources=sources,
        context=context if payload.include_context else None,
    )

