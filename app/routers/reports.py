from base64 import b64encode

from fastapi import APIRouter, Depends, HTTPException, status

from app.routers.users import get_current_user_id
from app.schemas.children import ChildDB, ChildRecord
from app.schemas.reports import SpeechTherapyReport, TherapyReportRequest, TherapyReportResponse
from app.services.child_service import get_child_by_uuid, get_child_records_by_uuid
from app.services.medgemma_service import MedGemmaService


router = APIRouter(tags=["reports"])

_service = MedGemmaService()


@router.post(
    "/generate-therapy-report/",
    response_model=TherapyReportResponse,
    status_code=status.HTTP_200_OK,
)
async def generate_therapy_report(
    payload: TherapyReportRequest,
    doctor_id: str = Depends(get_current_user_id),
) -> TherapyReportResponse:
    """
    Генерация AI‑отчёта по ребёнку на основе его записей и вероятностей нарушений.

    - Берём ребёнка и его записи из MongoDB;
    - Передаём в MedGemmaService;
    - Конвертируем HTML → PDF;
    - Отдаём JSON‑отчёт + PDF (base64).
    """

    child: ChildDB | None = await get_child_by_uuid(payload.child_uuid)
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    if str(child.doctor_id) != doctor_id:
        raise HTTPException(status_code=403, detail="Unauthorized access")

    records: list[ChildRecord] = await get_child_records_by_uuid(payload.child_uuid)
    if not records:
        raise HTTPException(status_code=400, detail="No records for this child")

    report: SpeechTherapyReport = await _service.generate_report(
        child=child,
        records=records,
        doctor_notes=payload.doctor_notes,
    )

    html = _service.render_html_report(child=child, report=report)

    try:
        pdf_bytes = _service.html_to_pdf(html)
    except Exception as e:  # pragma: no cover - зависит от системного wkhtmltopdf
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate PDF: {e}",
        )

    pdf_b64 = b64encode(pdf_bytes).decode("utf-8")

    return TherapyReportResponse(report=report, pdf_base64=pdf_b64)

