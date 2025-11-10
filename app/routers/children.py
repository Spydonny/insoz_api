from fastapi import APIRouter, Depends, Form, File, UploadFile
from app.routers.users import get_current_user_id
from app.services.child_service import *
from app.services.file_service import save_file_to_gridfs
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

