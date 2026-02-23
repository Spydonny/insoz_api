import asyncio
import json
import re
from textwrap import dedent
from typing import List

from ollama import chat as ollama_chat

import pdfkit

from app.core.config import settings
from app.schemas.children import ChildDB, ChildRecord
from app.schemas.reports import SpeechTherapyReport


class MedGemmaService:
    """
    Сервис-обёртка над локальной моделью MedGemma в Ollama.

    Отвечает за:
    - сбор данных о ребёнке и записях;
    - формирование промпта;
    - вызов модели через ollama;
    - парсинг JSON-ответа в Pydantic-модель;
    - генерацию HTML и PDF-версии отчёта.
    """

    def __init__(self, model_name: str | None = None) -> None:
        # Если имя модели не передали явно — берём из настроек.
        # По умолчанию: alibayram/medgemma (можно поменять через MEDGEMMA_MODEL_NAME).
        self.model_name = model_name or settings.MEDGEMMA_MODEL_NAME

    async def generate_report(
        self,
        child: ChildDB,
        records: List[ChildRecord],
        doctor_notes: str | None = None,
    ) -> SpeechTherapyReport:
        prompt = self._build_prompt(child=child, records=records, doctor_notes=doctor_notes)
        raw_json = await self._call_model(prompt)

        # Приводим ключи к ожидаемой Pydantic-схеме
        # (если модель вернула чуть другие имена — аккуратно мапим).
        mapped = self._normalize_keys(raw_json)

        return SpeechTherapyReport(**mapped)

    def render_html_report(self, child: ChildDB, report: SpeechTherapyReport) -> str:
        """
        Простая HTML-верстка отчёта.
        В реальном проекте это можно перенести в шаблон Jinja2.
        """

        diagnoses_str = ", ".join(child.diagnosis) if child.diagnosis else "нет данных"

        html = f"""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8" />
            <title>Логопедический отчёт — {child.name}</title>
            <style>
                body {{
                    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                    color: #1f2933;
                    background: #fdfdfb;
                    padding: 32px;
                    line-height: 1.6;
                }}
                h1 {{
                    font-size: 28px;
                    margin-bottom: 8px;
                    color: #854d0e;
                }}
                h2 {{
                    font-size: 20px;
                    margin-top: 24px;
                    margin-bottom: 8px;
                    color: #b45309;
                }}
                .card {{
                    background: #ffffff;
                    border-radius: 12px;
                    padding: 20px;
                    margin-bottom: 16px;
                    border: 1px solid #facc15;
                }}
                .meta {{
                    font-size: 14px;
                    color: #4b5563;
                }}
                .section-text {{
                    white-space: pre-wrap;
                }}
            </style>
        </head>
        <body>
            <h1>Логопедический отчёт</h1>
            <div class="card meta">
                <div><strong>Имя:</strong> {child.name}</div>
                <div><strong>Возраст:</strong> {child.age}</div>
                <div><strong>Диагнозы (клиника):</strong> {diagnoses_str}</div>
            </div>

            <div class="card">
                <h2>Оценка состояния</h2>
                <div class="section-text">{report.assessment}</div>
            </div>

            <div class="card">
                <h2>Диагноз</h2>
                <div class="section-text">{report.diagnosis}</div>
            </div>

            <div class="card">
                <h2>Рекомендации</h2>
                <div class="section-text">{report.recommendations}</div>
            </div>

            <div class="card">
                <h2>План коррекции</h2>
                <div class="section-text">{report.correction_plan}</div>
            </div>

            <div class="card">
                <h2>Серьёзность нарушений</h2>
                <div class="section-text">{report.severity}</div>
            </div>
        </body>
        </html>
        """
        return dedent(html)

    def html_to_pdf(self, html: str) -> bytes:
        """
        Конвертация HTML → PDF.

        Требует установленного wkhtmltopdf в системе.
        """

        # pdfkit.from_string возвращает bytes, если путь = False.
        return pdfkit.from_string(html, False)

    def _build_prompt(
        self,
        child: ChildDB,
        records: List[ChildRecord],
        doctor_notes: str | None = None,
    ) -> str:
        """
        Собираем детальные данные о ребёнке и всех его записях
        (включая вероятности нарушений) и просим MedGemma вернуть
        строго валидный JSON.
        """

        records_block: list[str] = []
        for idx, rec in enumerate(records, start=1):
            probs = rec.diagnosis_probabilities
            if not probs:
                continue

            probs_lines = [
                f"  - Картавость (rhotacism): {probs.rhotacism * 100:.1f}%",
                f"  - Шепелявость (lisp): {probs.lisp * 100:.1f}%",
                f"  - Общее недоразвитие речи (general_speech_disorder): {probs.general_speech_disorder * 100:.1f}%",
                f"  - Фонетико‑фонематическое НР (phonetic_phonemic_disorder): {probs.phonetic_phonemic_disorder * 100:.1f}%",
                f"  - Заикание (stuttering): {probs.stuttering * 100:.1f}%",
                f"  - Афазия (aphasia): {probs.aphasia * 100:.1f}%",
                f"  - Дизартрия (dysarthria): {probs.dysarthria * 100:.1f}%",
                f"  - Норма (normal): {probs.normal * 100:.1f}%",
            ]

            records_block.append(
                "\n".join(
                    [
                        f"Запись #{idx} от {rec.uploaded_at}:",
                        *probs_lines,
                    ]
                )
            )

        records_text = "\n\n".join(records_block) if records_block else "Нет рассчитанных вероятностей нарушений."

        doctor_notes_text = doctor_notes.strip() if doctor_notes else "Нет дополнительных заметок врача."

        prompt = f"""
        Ты — опытный детский логопед.

        У тебя есть данные о ребёнке и результаты автоматического анализа речи
        (вероятности различных нарушений по нескольким записям).

        Задача: составить краткий, но профессиональный логопедический отчёт,
        понятный врачу-логопеду. Стиль — клинический, без лишней воды.

        1) Информация о ребёнке:
           - Имя: {child.name}
           - Возраст: {child.age}
           - Клинические диагнозы: {", ".join(child.diagnosis) if child.diagnosis else "нет данных"}

        2) Вероятности нарушений по записям (чем выше %, тем выше риск):

        {records_text}

        3) Заметки лечащего врача:
        {doctor_notes_text}

        Сформируй СТРОГО валидный JSON без комментариев и дополнительного текста.
        ТОЛЬКО JSON, без Markdown, без пояснений.

        Структура JSON (ключи на английском, текст на русском):
        {{
          "assessment": "Краткая клиническая оценка текущего состояния речи ребёнка",
          "diagnosis": "Формулировка логопедического диагноза (с опорой на вероятности и клинические данные)",
          "recommendations": "Практические рекомендации по работе с ребёнком, взаимодействию с родителями и смежными специалистами",
          "correction_plan": "Структурированный план логопедической коррекции (этапы, фокус, частота занятий)",
          "severity": "Оценка степени выраженности нарушений (легкая / умеренная / тяжелая) с кратким обоснованием"
        }}

        Убедись, что JSON можно распарсить стандартной библиотекой Python json.loads без ошибок.
        """
        return dedent(prompt)

    async def _call_model(self, prompt: str) -> dict:
        """
        Асинхронный вызов локальной модели через Ollama.
        """

        def _sync_call() -> str:
            resp = ollama_chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты — профессиональный детский логопед. "
                            "Отвечай только в виде валидного JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            # В актуальной версии клиента есть и индексный доступ, и атрибут.
            content = getattr(resp, "message", None)
            if content is not None and hasattr(content, "content"):
                return content.content

            return resp["message"]["content"]

        content = await asyncio.to_thread(_sync_call)
        text = content.strip()

        # На всякий случай вырезаем JSON из возможного обрамляющего текста.
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("MedGemma did not return JSON")

        json_str = match.group(0)
        return json.loads(json_str)

    def _normalize_keys(self, data: dict) -> dict:
        """
        Приводим разные варианты названий полей к единой схеме.
        """

        key_map = {
            "assessment": "assessment",
            "оценка": "assessment",
            "diagnosis": "diagnosis",
            "диагноз": "diagnosis",
            "recommendations": "recommendations",
            "рекомендации": "recommendations",
            "correction_plan": "correction_plan",
            "plan": "correction_plan",
            "план_коррекции": "correction_plan",
            "severity": "severity",
            "серьезность": "severity",
            "степень_выраженности": "severity",
        }

        normalized: dict[str, str] = {}
        for raw_key, value in data.items():
            key = raw_key.strip()
            target = key_map.get(key, key)
            normalized[target] = value

        # Гарантируем наличие всех ключей, даже если модель что-то пропустила.
        for required_key in ["assessment", "diagnosis", "recommendations", "correction_plan", "severity"]:
            normalized.setdefault(required_key, "")

        return normalized

