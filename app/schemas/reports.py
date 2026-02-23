from pydantic import BaseModel


class SpeechTherapyReport(BaseModel):
    """
    Структурированный логопедический отчёт,
    который возвращает MedGemma.
    """

    assessment: str
    diagnosis: str
    recommendations: str
    correction_plan: str
    severity: str


class TherapyReportRequest(BaseModel):
    """
    Запрос на генерацию AI‑отчёта.

    Мы передаём только UUID ребёнка и (опционально) заметки врача,
    а сами подтягиваем записи и вероятности из базы.
    """

    child_uuid: str
    doctor_notes: str | None = None


class TherapyReportResponse(BaseModel):
    """
    Ответ API: структурированный отчёт + PDF в base64.
    """

    report: SpeechTherapyReport
    pdf_base64: str

