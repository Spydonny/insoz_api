from fastapi import APIRouter, Depends, Form, File, UploadFile, HTTPException, status
from app.routers.users import get_current_user_id
from app.services.child_service import *
from app.services.file_service import save_file_to_gridfs, upload_and_convert_to_wav
from app.schemas.children import ChildDB, ChildCreate
from typing import List

router = APIRouter(prefix="/children", tags=["children"], dependencies=[Depends(get_current_user_id)])


@router.post("/", response_model=dict, status_code=201)
async def create_new_child(
    name: str = Form(...),
    age: int = Form(...),
    diagnosis: List[str] = Form(...),
    picture: UploadFile = File(None),
    doctor_id: str = Depends(get_current_user_id)
):  
    picture_id = None
    try:
        picture_id = await save_file_to_gridfs(picture) if picture else None
    except Exception as e:
        return {"error": f"Ошибка при сохранении файла: {e}"}
    
    child_data = ChildCreate(
        name=name,
        age=age,
        diagnosis=diagnosis,
        picture_id=picture_id
    )

    result = await create_child(child_data, doctor_id)
    return result


@router.get("/", response_model=list[ChildDB])
async def read_children(
    doctor_id: str = Depends(get_current_user_id)
):
    
    return await get_child_by_doctor_id(doctor_id=doctor_id)

@router.get("/id/{uuid}", response_model=ChildDB | dict)
async def read_child_by_uuid(uuid: str):
    child = await get_child_by_uuid(uuid)
    if not child:
        return {"error": "Child not found"}
    return child

@router.post("/record/{uuid}", response_model=dict)
async def add_record_to_child(
    uuid: str,
    record: UploadFile = File(...),
    doctor_id: str = Depends(get_current_user_id),
):
    child = await get_child_by_uuid(uuid)
    if not child:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Child not found"
        )

    if str(child.doctor_id) != doctor_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized access"
        )

    try:
        record_path = await upload_and_convert_to_wav(record)
        if not record_path:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save record"
            )
        
        updated_child = await add_record_to_child_in_db(uuid, record_path)
        return updated_child

    except HTTPException:
        raise  # пробрасываем уже созданное исключение дальше
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при добавлении записи: {e}"
        )

@router.get("/record/{uuid}", response_model=list[ChildRecord])
async def get_child_records(
    uuid: str,
    doctor_id: str = Depends(get_current_user_id),
):
    child = await get_child_by_uuid(uuid)
    if not child:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Child not found"
        )

    if str(child.doctor_id) != doctor_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized access"
        )

    records = await get_child_records_by_uuid(uuid)
    if records is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No records found for this child"
        )

    return records

