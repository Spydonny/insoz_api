from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import google.generativeai as genai

from app.core.config import settings


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


@dataclass(frozen=True)
class RagPaths:
    pdf_dir: Path
    faiss_dir: Path


_VECTOR_DB: FAISS | None = None
_EMBEDDINGS: HuggingFaceEmbeddings | None = None


def get_rag_paths() -> RagPaths:
    base = Path.cwd()
    pdf_dir = Path(getattr(settings, "RAG_PDF_DIR", str(base / "speech_therapy_pdfs")))
    faiss_dir = Path(getattr(settings, "RAG_FAISS_DIR", str(base / "app" / "data" / "faiss_speech_therapy")))
    return RagPaths(pdf_dir=pdf_dir, faiss_dir=faiss_dir)


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
    return _EMBEDDINGS


def get_vector_db(faiss_dir: str | Path | None = None) -> FAISS:
    """
    Loads FAISS index once per process.
    Build it via `scripts/build_faiss_index.py` first.
    """
    global _VECTOR_DB
    if _VECTOR_DB is not None:
        return _VECTOR_DB

    paths = get_rag_paths()
    faiss_path = Path(faiss_dir) if faiss_dir is not None else paths.faiss_dir

    if not faiss_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{faiss_path}'. Build it first (see scripts/build_faiss_index.py)."
        )

    embeddings = _get_embeddings()
    _VECTOR_DB = FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)
    return _VECTOR_DB


def load_and_chunk_pdfs(
    directory: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Any]:
    all_chunks: list[Any] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )

    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"PDF folder does not exist: {directory}")

    for file_path in sorted(directory.glob("*.pdf")):
        try:
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()

            for doc in docs:
                doc.metadata["source_file"] = file_path.name

            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
        except Exception:
            # Skip bad PDFs rather than failing the whole build.
            continue

    return all_chunks


def build_and_save_faiss_index(
    pdf_dir: str | Path,
    faiss_dir: str | Path,
) -> dict[str, Any]:
    pdf_dir = Path(pdf_dir)
    faiss_dir = Path(faiss_dir)
    faiss_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_and_chunk_pdfs(pdf_dir)
    embeddings = _get_embeddings()
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(str(faiss_dir))

    sources = sorted({c.metadata.get("source_file") for c in chunks if getattr(c, "metadata", None)})  # type: ignore[attr-defined]
    return {"chunks": len(chunks), "sources": [s for s in sources if s]}


def _normalize_probability_to_percent(x: Any) -> int:
    try:
        v = float(x)
    except Exception:
        return 0
    if v <= 0:
        return 0
    if v > 1.0:
        # already percent-ish
        return int(max(0, min(100, v)))
    return int(max(0, min(100, round(v * 100))))


def patient_profile_from_child(child: dict) -> dict[str, Any]:
    """
    Build patient profile for RAG:
    - base `child["diagnosis"]` list is treated as high-confidence (100%)
    - augmented with probabilities from latest record.diagnosis_probabilities
    """
    diagnosis: dict[str, int] = {}

    # 1) Base diagnoses from child card (list[str]) – always included with high weight.
    for d in child.get("diagnosis") or []:
        if not d:
            continue
        key = str(d).strip()
        if not key:
            continue
        diagnosis[key] = max(diagnosis.get(key, 0), 100)

    # 2) Probabilities from the latest audio record, if present.
    records = child.get("records") or []
    latest = records[-1] if isinstance(records, list) and records else {}
    probs = (latest.get("diagnosis_probabilities") or {}) if isinstance(latest, dict) else {}

    for k, v in probs.items():
        if k == "record_id":
            continue
        percent = _normalize_probability_to_percent(v)
        if percent <= 0:
            continue
        diagnosis[k] = max(diagnosis.get(k, 0), percent)

    return {
        "age": child.get("age"),
        "diagnosis": diagnosis,
    }


def retrieve_weighted_context(patient_profile: dict[str, Any], db: FAISS, k_total: int = 10) -> tuple[str, list[str]]:
    context_blocks: list[str] = []
    sources: list[str] = []

    diagnosis = patient_profile.get("diagnosis") or {}
    if not isinstance(diagnosis, dict) or not diagnosis:
        return "", []

    for disorder, severity in diagnosis.items():
        try:
            sev = int(severity)
        except Exception:
            sev = 0
        if sev <= 0:
            continue

        k_for_disorder = max(1, int((sev / 100) * k_total))
        query = f"Therapy methods and exercises for {disorder} for age {patient_profile.get('age')}"
        results = db.similarity_search(query, k=k_for_disorder)

        for res in results:
            src = res.metadata.get("source_file")
            if src:
                sources.append(str(src))
            context_blocks.append(
                f"--- CONTEXT FOR {str(disorder).upper()} (Weight: {sev}%) ---\n"
                f"{res.page_content}\n"
                f"Source: {src}"
            )

    # de-dupe sources (preserve order)
    seen = set()
    unique_sources: list[str] = []
    for s in sources:
        if s not in seen:
            unique_sources.append(s)
            seen.add(s)

    return "\n\n".join(context_blocks), unique_sources


def answer_with_gemini(question: str, patient_profile: dict[str, Any], context: str) -> str:
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    prompt = f"""
You are a speech-language therapist assistant.
Use ONLY the provided context. If the context is insufficient, say what is missing and give safe, general guidance.

Patient:
- Age: {patient_profile.get("age")}
- Diagnosis weights (0-100): {patient_profile.get("diagnosis")}

Context:
{context}

Question:
{question}

Answer requirements:
- Provide a short, actionable plan (bulleted).
- Provide 3-7 concrete exercises/activities.
- Give frequency/duration suggestions.
- When you mention a recommendation, cite the Source filename if applicable.
""".strip()

    resp = model.generate_content(prompt)
    return (resp.text or "").strip()

