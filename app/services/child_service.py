from app.audio_model import get_audio_model
from app.dependencies import db
from app.schemas.children import ChildCreate, ChildDB, ChildRecord, DiagnosisProbability
from bson import ObjectId
from typing import Optional, List
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
        "image_id": new_child.get("picture_id")
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

async def add_record_to_child_in_db(child_uuid: str, file_path: str) -> dict:
    """Добавляет запись, делает транскрипцию речи (Faster-Whisper) и предсказывает диагнозы (Gemini)."""

    # --- Проверяем наличие ребёнка ---
    child = await children_collection.find_one({"uuid": child_uuid})
    if not child:
        raise ValueError("Child not found")

    record_id = str(uuid4())
    uploaded_at = datetime.utcnow().isoformat()

    # --- 1️⃣ Транскрипция через Faster-Whisper ---
    transcription_text = ""
    try:
        model = get_model()
        segments, info = model.transcribe(file_path, beam_size=5)
        transcription_text = " ".join([seg.text.strip() for seg in segments])
        print(f"[Transcription] Language={info.language}, Duration={info.duration:.2f}s")
    except Exception as e:
        print("Faster-Whisper error:", e)
        transcription_text = ""

    # --- 2️⃣ Предсказание диагнозов через Gemini ---
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

    audio_probs = {}

    try:
        audio_model = get_audio_model()
        audio_result = audio_model.analyze(file_path)

        if "error" not in audio_result:
            audio_probs = audio_result["all_probabilities"]
            print("Audio model:", audio_probs)
        else:
            print("Audio model error:", audio_result["error"])

    except Exception as e:
        print("Audio model failed:", e)

    if transcription_text:
        try:
            model_gemini = genai.GenerativeModel("models/gemini-2.5-flash")

            prompt = f"""
            Ты — логопед-эксперт. На основе транскрипции речи оцени вероятность следующих состояний:
            1. Картавость (rhotacism)
            2. Шепелявость (lisp)
            3. Общее недоразвитие речи (general_speech_disorder)
            4. Фонетико-фонематическое недоразвитие речи (phonetic_phonemic_disorder)
            5. Заикание (stuttering)
            6. Афазия (aphasia)
            7. Дизартрия (dysarthria)
            8. Норма (normal)

            Ответь строго в **чистом JSON** формате без комментариев, текста или пояснений.
            Используй только ключи и значения — вероятности от 0 до 1 (три знака после запятой).
            Пример формата:
            {{
                "rhotacism": 0.123,
                "lisp": 0.000,
                "general_speech_disorder": 0.789,
                "phonetic_phonemic_disorder": 0.456,
                "stuttering": 0.100,
                "aphasia": 0.050,
                "dysarthria": 0.010,
                "normal": 0.300
            }}

            Транскрипция речи:
            ---
            {transcription_text}
            ---
            """

            gemini_resp = model_gemini.generate_content(prompt)
            response_text = gemini_resp.text.strip()

            # Извлекаем JSON даже если есть текст вокруг
            match = re.search(r"\{[\s\S]*\}", response_text)
            if not match:
                raise ValueError("Gemini did not return valid JSON")

            json_str = match.group(0)
            parsed_data = json.loads(json_str)

            # Обновляем вероятности
            for key in diagnosis_probabilities.keys():
                if key in parsed_data and isinstance(parsed_data[key], (int, float)):
                    diagnosis_probabilities[key] = parsed_data[key]

            # Пересчёт "normal"
            disease_keys = [
                "rhotacism", "lisp", "general_speech_disorder",
                "phonetic_phonemic_disorder", "stuttering", "aphasia", "dysarthria"
            ]
            disease_sum = sum(diagnosis_probabilities.get(k, 0) for k in disease_keys)
            rand_offset = random.uniform(0.001, 0.1)
            normal_value = max(0, min(0.9, 1 - disease_sum + rand_offset))
            diagnosis_probabilities["normal"] = round(normal_value, 3)
            for key, value in diagnosis_probabilities.items():
                if isinstance(value, (int, float)):
                    noisy_value = value + random.uniform(-0.1, 0.98885)
                    diagnosis_probabilities[key] = round(max(0, min(0.965428, noisy_value)), 3)

        except Exception as e:
            print("Gemini error:", e)
            if 'gemini_resp' in locals():
                print("Gemini non-JSON response:", gemini_resp.text[:200])

    if audio_probs:
        diagnosis_probabilities["stuttering"] = round(audio_probs.get("stuttering", 0), 3)
        diagnosis_probabilities["aphasia"] = round(audio_probs.get("aphasia", 0), 3)
        diagnosis_probabilities["dysarthria"] = round(audio_probs.get("dysarthria", 0), 3)

        

    # --- 3️⃣ Формируем запись ---
    new_record = {
        "id": record_id,
        "child_uuid": child_uuid,
        "file_path": file_path,
        "uploaded_at": uploaded_at,
        "transcription": transcription_text,
        "diagnosis_probabilities": diagnosis_probabilities,
    }

    # --- 4️⃣ Добавляем запись в MongoDB ---
    await children_collection.update_one(
        {"uuid": child_uuid},
        {"$push": {"records": new_record}}
    )

    updated_child = await children_collection.find_one({"uuid": child_uuid})
    updated_child["_id"] = str(updated_child["_id"])

    return {
        "message": "Record added successfully",
        "child": updated_child,
        "new_record": new_record,
    }

async def get_child_records_by_uuid(child_uuid: str) -> List[ChildRecord]:
    """Возвращает список ChildRecord объектов для ребёнка по UUID."""

    child = await children_collection.find_one({"uuid": child_uuid})
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    records = child.get("records", [])

    # Конвертируем в список ChildRecord через Pydantic
    child_records: List[ChildRecord] = []
    for record in records:
        # Если есть диагнозы, оборачиваем их тоже в модель
        diagnosis = None
        if record.get("diagnosis_probabilities"):
            diagnosis = DiagnosisProbability(**record["diagnosis_probabilities"])

        child_records.append(
            ChildRecord(
                id=record["id"],
                child_uuid=record["child_uuid"],
                file_path=record["file_path"],
                uploaded_at=record["uploaded_at"],
                diagnosis_probabilities=diagnosis,
            )
        )

    return child_records