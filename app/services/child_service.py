from app.audio_model import get_audio_model
from app.dependencies import db
from app.schemas.children import (
    ChildCreate,
    ChildDB,
    ChildDetailResponse,
    ChildRecord,
    DiagnosisProbability,
    ManualPhonemeScoreRequest,
    PhonemeAnalysisEntry,
)
from app.services.phoneme_service import analyze_phonemes, get_phoneme_letter
from bson import ObjectId
from typing import List, Optional
from uuid import uuid4
from datetime import datetime
from fastapi import HTTPException
import google.generativeai as genai
from faster_whisper import WhisperModel
import json
import re
import random
import os
from dotenv import load_dotenv

load_dotenv()

children_collection = db["children"]
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

_model = None


def get_model():
    global _model
    if _model is None:
        print("Loading Whisper model...")
        _model = WhisperModel("base", device="cpu")
    return _model

def add_noise(value: float, noise_level: float = 0.05) -> float:
    """
    Adds small Gaussian-like noise and clamps result to [0, 1]
    """
    noise = random.uniform(-noise_level, noise_level)
    return max(0.0, min(1.0, value + noise))

async def create_child(child: ChildCreate, doctor_id: str) -> dict:
    new_child = child.dict()
    new_child["uuid"] = str(uuid4())
    new_child["doctor_id"] = doctor_id

    result = await children_collection.insert_one(new_child)
    return {
        "message": "Child created successfully",
        "child_id": str(result.inserted_id),
        "uuid": new_child["uuid"],
        "doctor_id": doctor_id,
        "image_id": new_child.get("picture_id"),
    }


async def get_all_children() -> list[ChildDB]:
    children_cursor = children_collection.find()
    children = []
    async for child in children_cursor:
        child["_id"] = str(child["_id"])
        children.append(ChildDB(**child))
    return children


async def get_child_by_id(child_id: str) -> Optional[ChildDB]:
    child = await children_collection.find_one({"_id": ObjectId(child_id)})
    if not child:
        return None

    child["_id"] = str(child["_id"])
    return ChildDB(**child)


async def get_child_by_uuid(uuid: str) -> ChildDB | None:
    child = await children_collection.find_one({"uuid": uuid})
    if not child:
        return None

    child["_id"] = str(child["_id"])
    return ChildDB(**child)


async def get_child_by_doctor_id(doctor_id: str) -> list[ChildDB]:
    children_cursor = children_collection.find({"doctor_id": doctor_id})
    children = []
    async for child in children_cursor:
        child["_id"] = str(child["_id"])
        children.append(ChildDB(**child))
    return children


def _build_pronunciation_summary(average_score: float, max_score: int) -> str:
    if max_score <= 0:
        return "No pronunciation score available."

    ratio = average_score / max_score
    if ratio >= 0.75:
        return "Excellent pronunciation."
    if ratio >= 0.4:
        return "Average pronunciation, some phonemes need improvement."
    return "Pronunciation needs improvement."


def _clamp_score(value: int, max_score: int) -> int:
    return max(0, min(max_score, int(value)))


def _build_phoneme_results(
    phonemes: List[str],
    scores: dict[str, int],
    similarities: dict[str, float] | None = None,
) -> list[dict]:
    results = []
    for phoneme in phonemes:
        result = {
            "phoneme": phoneme,
            "letter": get_phoneme_letter(phoneme),
            "score": scores.get(phoneme, 0),
        }
        if similarities and phoneme in similarities:
            result["best_similarity"] = round(float(similarities[phoneme]), 4)
        results.append(result)
    return results


def _build_phoneme_analysis_payload(
    child_uuid: str,
    analysis_type: str,
    language: str,
    phonemes: List[str],
    scores: dict[str, int],
    max_score: int,
    comment: str | None = None,
    audio_file_path: str | None = None,
    similarities: dict[str, float] | None = None,
) -> dict:
    average_score = round(sum(scores.values()) / len(scores), 2) if scores else 0.0
    return {
        "id": str(uuid4()),
        "child_uuid": child_uuid,
        "analysis_type": analysis_type,
        "language": language,
        "phonemes": phonemes,
        "scores": scores,
        "results": _build_phoneme_results(phonemes, scores, similarities),
        "max_score": max_score,
        "average_score": average_score,
        "summary": _build_pronunciation_summary(average_score, max_score),
        "comment": comment,
        "audio_file_path": audio_file_path,
        "created_at": datetime.utcnow().isoformat(),
    }


async def _save_phoneme_analysis(child_uuid: str, analysis_payload: dict) -> PhonemeAnalysisEntry:
    result = await children_collection.update_one(
        {"uuid": child_uuid},
        {"$push": {"phoneme_analyses": analysis_payload}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Child not found")
    return PhonemeAnalysisEntry(**analysis_payload)


def _build_child_record(record: dict) -> ChildRecord:
    diagnosis = None
    if record.get("diagnosis_probabilities"):
        diagnosis = DiagnosisProbability(**record["diagnosis_probabilities"])

    return ChildRecord(
        id=record["id"],
        child_uuid=record["child_uuid"],
        file_path=record["file_path"],
        uploaded_at=record["uploaded_at"],
        diagnosis_probabilities=diagnosis,
    )


def _build_child_detail_response(child: dict) -> ChildDetailResponse:
    records = [_build_child_record(record) for record in child.get("records", [])]
    phoneme_analyses = [
        PhonemeAnalysisEntry(**analysis)
        for analysis in child.get("phoneme_analyses", [])
    ]

    return ChildDetailResponse(
        name=child["name"],
        age=child["age"],
        diagnosis=child["diagnosis"],
        picture_id=child.get("picture_id"),
        uuid=child["uuid"],
        doctor_id=child["doctor_id"],
        records=records,
        phoneme_analyses=phoneme_analyses,
    )


async def get_child_detail_by_uuid(uuid: str) -> ChildDetailResponse | None:
    child = await children_collection.find_one({"uuid": uuid})
    if not child:
        return None
    return _build_child_detail_response(child)


async def get_child_phoneme_analyses_by_uuid(child_uuid: str) -> List[PhonemeAnalysisEntry]:
    child = await children_collection.find_one({"uuid": child_uuid})
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")
    return [PhonemeAnalysisEntry(**analysis) for analysis in child.get("phoneme_analyses", [])]


async def save_manual_phoneme_analysis(
    child_uuid: str,
    payload: ManualPhonemeScoreRequest,
) -> PhonemeAnalysisEntry:
    normalized_scores = {
        phoneme: _clamp_score(payload.scores.get(phoneme, 0), payload.max_score)
        for phoneme in payload.phonemes
    }
    analysis_payload = _build_phoneme_analysis_payload(
        child_uuid=child_uuid,
        analysis_type="manual",
        language=payload.language,
        phonemes=payload.phonemes,
        scores=normalized_scores,
        max_score=payload.max_score,
        comment=payload.comment,
    )
    return await _save_phoneme_analysis(child_uuid, analysis_payload)


async def save_ai_phoneme_analysis(
    child_uuid: str,
    language: str,
    phonemes: List[str],
    max_score: int,
    audio_bytes: bytes,
    audio_file_path: str | None = None,
) -> PhonemeAnalysisEntry:
    ai_result = analyze_phonemes(
        audio_source=audio_bytes,
        phonemes=phonemes,
        max_score=max_score,
        augment=False,
    )
    similarities = {
        result["phoneme"]: result["best_similarity"]
        for result in ai_result["results"]
    }
    analysis_payload = _build_phoneme_analysis_payload(
        child_uuid=child_uuid,
        analysis_type="ai",
        language=language,
        phonemes=phonemes,
        scores=ai_result["scores"],
        max_score=max_score,
        audio_file_path=audio_file_path,
        similarities=similarities,
    )
    return await _save_phoneme_analysis(child_uuid, analysis_payload)

def build_prompt(transcription: str) -> str:
    prompt = f"""
        Ты — эксперт по анализу речевых транскрипций (speech-language pathology + computational linguistics).

        Твоя задача — оценить вероятность наличия типов речевых нарушений по текстовой транскрипции.

        ⚠️ ВАЖНО:
        - Ты НЕ ставишь диагноз.
        - Ты не используешь медицинскую категоричность.
        - Ты выполняешь многоклассовую вероятностную оценку по наблюдаемым речевым признакам.

        ---

        # 1. Классы:

        - rhotacism (устойчивое нарушение /r/)
        - lisp (нарушение свистящих и шипящих)
        - general_speech_disorder (общее недоразвитие речи, бедная и упрощённая речь)
        - phonetic_phonemic_disorder (системные фонологические/фонетические замены)
        - stuttering (нарушение беглости речи: повторы, блоки, пролонгации)
        - aphasia (нарушение языковой структуры: аграмматизмы, бессвязность, трудности построения фраз)
        - dysarthria (моторное нарушение речи: смазанность, низкая артикуляционная чёткость)
        - normal (отсутствие устойчивых нарушений)

        ---

        # 2. Извлечение признаков:

        Определи наличие:
        - фонетических замен (речь → искажённые фонемы)
        - артикуляционных искажений
        - пропусков звуков/слогов
        - повторов (звуки, слоги, слова)
        - пролонгаций звуков
        - речевых блоков/паузации
        - грамматических ошибок
        - сниженной информативности речи
        - нарушения связности текста

        ---

        # 3. Интерпретация признаков:

        - rhotacism → изолированная проблема только /r/ 
        - lisp → системные ошибки свистящих/шипящих
        - phonetic_phonemic_disorder → множественные стабильные фонологические замены
        - stuttering → дисфлюентность при сохранной языковой структуре
        - dysarthria → сниженная артикуляционная чёткость + моторная нестабильность
        - aphasia → нарушение грамматической и смысловой организации речи
        - general_speech_disorder → бедная, упрощённая, но структурно сохранная речь
        - normal → нет устойчивых патологических паттернов

        ---

        # 4. Ограничения модели:

        - Вероятности должны быть в диапазоне [0, 1]
        - Вернуть ВСЕ классы
        - 3 знака после запятой
        - Значения НЕ обязаны суммироваться в 1 (multi-label scoring)
        - Но:
        - если один класс высок (>0.6), конкурирующие типы должны быть низкими
        - aphasia и normal не должны одновременно быть высокими
        - dysarthria и normal не должны быть одновременно высокими
        - stuttering и aphasia — слабо совместимы

        ---

        # 5. Формат ответа (СТРОГО JSON):

        {{
            "rhotacism": 0.000,
            "lisp": 0.000,
            "general_speech_disorder": 0.000,
            "phonetic_phonemic_disorder": 0.000,
            "stuttering": 0.000,
            "aphasia": 0.000,
            "dysarthria": 0.000,
            "normal": 0.000
        }}

        ---

        # 6. Вход:

        Транскрипция:
        ---
        {transcription}
        ---
        """
    return prompt

async def add_record_to_child_in_db(child_uuid: str, file_path: str) -> dict:
    child = await children_collection.find_one({"uuid": child_uuid})
    if not child:
        raise ValueError("Child not found")

    record_id = str(uuid4())
    uploaded_at = datetime.utcnow().isoformat()

    transcription_text = ""
    try:
        model = get_model()
        segments, info = model.transcribe(file_path, beam_size=5)
        transcription_text = " ".join(seg.text.strip() for seg in segments)
    except Exception as e:
        print("Transcription error:", e)

    diagnosis_probabilities = {
        "record_id": record_id,
        "rhotacism": 0.0,
        "lisp": 0.0,
        "general_speech_disorder": 0.0,
        "phonetic_phonemic_disorder": 0.0,
        "stuttering": 0.0,
        "aphasia": 0.0,
        "dysarthria": 0.0,
        "normal": 0.0,
    }

    if transcription_text:
        try:
            model_gemini = genai.GenerativeModel("models/gemini-2.5-flash")

            prompt = build_prompt(transcription_text)  # вынеси промпт отдельно

            resp = model_gemini.generate_content(prompt)
            text = resp.text

            json_match = re.search(r"\{[\s\S]*\}", text)
            if not json_match:
                raise ValueError("Invalid JSON from model")

            raw = json.loads(json_match.group(0))

            # 1. базовая загрузка без посторонней логики
            for k in diagnosis_probabilities.keys():
                if k in raw and isinstance(raw[k], (int, float)):
                    diagnosis_probabilities[k] = float(raw[k])

        except Exception as e:
            print("Gemini error:", e)


    # -----------------------------
    # POST-PROCESSING (INDEPENDENT + NOISE)
    # -----------------------------

    disease_keys = [
        "rhotacism",
        "lisp",
        "general_speech_disorder",
        "phonetic_phonemic_disorder",
        "stuttering",
        "aphasia",
        "dysarthria",
    ]

    # применяем шум независимо к каждому классу
    for k in disease_keys:
        diagnosis_probabilities[k] = add_noise(
            diagnosis_probabilities[k],
            noise_level=0.04  # можно 0.02–0.06 по вкусу
        )

    # normal тоже НЕ зависит от суммы
    diagnosis_probabilities["normal"] = add_noise(
        diagnosis_probabilities["normal"],
        noise_level=0.15
    )

    # финальное ограничение диапазона
    for k in diagnosis_probabilities:
        if isinstance(diagnosis_probabilities[k], float):
            diagnosis_probabilities[k] = round(
                max(0.0, min(1.0, diagnosis_probabilities[k])),
                3
        )

    new_record = {
        "id": record_id,
        "child_uuid": child_uuid,
        "file_path": file_path,
        "uploaded_at": uploaded_at,
        "transcription": transcription_text,
        "diagnosis_probabilities": diagnosis_probabilities,
    }

    await children_collection.update_one(
        {"uuid": child_uuid},
        {"$push": {"records": new_record}},
    )

    updated_child = await children_collection.find_one({"uuid": child_uuid})
    updated_child["_id"] = str(updated_child["_id"])

    return {
        "message": "Record added successfully",
        "child": updated_child,
        "new_record": new_record,
    }


async def get_child_records_by_uuid(child_uuid: str) -> List[ChildRecord]:
    child = await children_collection.find_one({"uuid": child_uuid})
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    records = child.get("records", [])
    return [_build_child_record(record) for record in records]
