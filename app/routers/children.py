import json
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.routers.users import get_current_user_id
from app.schemas.children import (
    AnalyzePhonemesResponse,
    ChildCreate,
    ChildDB,
    ChildDetailResponse,
    ChildPhonemeAnalysesResponse,
    ChildRecord,
    ManualPhonemeScoreRequest,
    ManualPhonemeScoreResponse,
)
from app.services.child_service import (
    add_record_to_child_in_db,
    create_child,
    get_child_by_doctor_id,
    get_child_by_uuid,
    get_child_detail_by_uuid,
    get_child_phoneme_analyses_by_uuid,
    get_child_records_by_uuid,
    save_ai_phoneme_analysis,
    save_manual_phoneme_analysis,
)
from app.services.file_service import upload_and_convert_to_wav

router = APIRouter(prefix="/children", tags=["children"], dependencies=[Depends(get_current_user_id)])


def _parse_phonemes_form_value(raw_value: str) -> List[str]:
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list):
            phonemes = [str(item).strip() for item in parsed if str(item).strip()]
            if phonemes:
                return phonemes
    except json.JSONDecodeError:
        pass

    phonemes = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not phonemes:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Phonemes are required")
    return phonemes


async def _get_authorized_child(uuid: str, doctor_id: str):
    child = await get_child_by_uuid(uuid)
    if not child:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Child not found",
        )

    if str(child.doctor_id) != doctor_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized access",
        )

    return child


@router.post("/", response_model=dict, status_code=201)
async def create_new_child(
    name: str = Form(...),
    age: int = Form(...),
    diagnosis: List[str] = Form(...),
    picture: UploadFile = File(None),
    doctor_id: str = Depends(get_current_user_id),
):
    picture_id = None
    try:
        picture_id = await upload_and_convert_to_wav(picture) if picture else None
    except Exception as e:
        return {"error": f"Ошибка при сохранении файла: {e}"}

    child_data = ChildCreate(
        name=name,
        age=age,
        diagnosis=diagnosis,
        picture_id=picture_id,
    )

    return await create_child(child_data, doctor_id)


@router.get("/", response_model=list[ChildDB])
async def read_children(
    doctor_id: str = Depends(get_current_user_id),
):
    return await get_child_by_doctor_id(doctor_id=doctor_id)


@router.get("/id/{uuid}", response_model=ChildDetailResponse | dict)
async def read_child_by_uuid(
    uuid: str,
    doctor_id: str = Depends(get_current_user_id),
):
    await _get_authorized_child(uuid, doctor_id)
    child = await get_child_detail_by_uuid(uuid)
    if not child:
        return {"error": "Child not found"}
    return child


@router.post("/record/{uuid}", response_model=dict)
async def add_record_to_child(
    uuid: str,
    record: UploadFile = File(...),
    doctor_id: str = Depends(get_current_user_id),
):
    await _get_authorized_child(uuid, doctor_id)

    try:
        record_path = await upload_and_convert_to_wav(record)
        if not record_path:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save record",
            )

        return await add_record_to_child_in_db(uuid, record_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при добавлении записи: {e}",
        )


@router.get("/record/{uuid}", response_model=list[ChildRecord])
async def get_child_records(
    uuid: str,
    doctor_id: str = Depends(get_current_user_id),
):
    await _get_authorized_child(uuid, doctor_id)
    records = await get_child_records_by_uuid(uuid)
    if records is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No records found for this child",
        )
    return records


@router.post("/phonemes/analyze/{uuid}", response_model=AnalyzePhonemesResponse)
async def analyze_child_phonemes(
    uuid: str,
    language: str = Form(...),
    phonemes: str = Form(...),
    max_score: int = Form(...),
    record: UploadFile = File(...),
    doctor_id: str = Depends(get_current_user_id),
):
    await _get_authorized_child(uuid, doctor_id)

    if max_score < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="max_score must be greater than 0",
        )

    audio_bytes = await record.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio record is required for AI analysis",
        )

    audio_file_path = None
    try:
        await record.seek(0)
        audio_file_path = await upload_and_convert_to_wav(record)
    except Exception:
        audio_file_path = None

    try:
        analysis = await save_ai_phoneme_analysis(
            child_uuid=uuid,
            language=language,
            phonemes=_parse_phonemes_form_value(phonemes),
            max_score=max_score,
            audio_bytes=audio_bytes,
            audio_file_path=audio_file_path,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e

    return AnalyzePhonemesResponse(analysis=analysis)


@router.post("/phonemes/manual-score/{uuid}", response_model=ManualPhonemeScoreResponse)
async def submit_manual_phoneme_score(
    uuid: str,
    payload: ManualPhonemeScoreRequest,
    doctor_id: str = Depends(get_current_user_id),
):
    await _get_authorized_child(uuid, doctor_id)
    analysis = await save_manual_phoneme_analysis(uuid, payload)
    return ManualPhonemeScoreResponse(analysis=analysis)


@router.get("/phonemes/{uuid}", response_model=ChildPhonemeAnalysesResponse)
async def get_child_phoneme_analyses(
    uuid: str,
    doctor_id: str = Depends(get_current_user_id),
):
    await _get_authorized_child(uuid, doctor_id)
    analyses = await get_child_phoneme_analyses_by_uuid(uuid)
    return ChildPhonemeAnalysesResponse(analyses=analyses)
