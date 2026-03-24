from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable
import librosa
import soundfile as sf

import numpy as np
import pandas as pd
import requests
import torch
import torchaudio
from scipy.spatial.distance import cosine
from transformers import HubertModel, Wav2Vec2FeatureExtractor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000
WINDOW_SEC = 1.0
MODEL_NAME = "facebook/hubert-base-ls960"
PARQUET_ROOT = Path(__file__).resolve().parents[2] / "hubert_parquets"

PHONEME_TO_LETTER = {
    "b": "б",
    "d": "д",
    "e": "е",
    "h": "һ",
    "i": "і",
    "j": "й",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "қ",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "w": "у",
    "z": "з",
    "æ": "ә",
    "ŋ": "ң",
    "ɑ": "а",
    "ɕː": "щ",
    "ə": "ә",
    "ɡ": "г",
    "ɪ": "і",
    "ʁ": "ғ",
    "ʃ": "ш",
    "ʊ": "ұ",
    "ʏ": "ү",
    "ʒ": "ж",
    "χ": "х",
}

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = HubertModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
if DEVICE == "cuda":
    model = model.half()
else:
    model = model.float()


def load_audio_from_url(url: str) -> torch.Tensor:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return load_audio_from_bytes(response.content)



import subprocess
import imageio_ffmpeg as ffmpeg
import numpy as np
import torch

def load_audio_from_bytes(audio_bytes: bytes) -> torch.Tensor:
    ffmpeg_path = ffmpeg.get_ffmpeg_exe()

    process = subprocess.Popen(
        [
            ffmpeg_path,
            "-i", "pipe:0",
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ar", str(TARGET_SR),
            "-ac", "1",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    out, _ = process.communicate(audio_bytes)

    wav = np.frombuffer(out, np.float32)

    return torch.tensor(wav, dtype=torch.float32)

    
def load_audio(audio_source: str | bytes | bytearray) -> torch.Tensor:
    if isinstance(audio_source, str):
        if audio_source.startswith(("http://", "https://")):
            return load_audio_from_url(audio_source)

        data, sr = librosa.load(audio_source, sr=TARGET_SR, mono=True)
        return torch.tensor(data, dtype=torch.float32)

    return load_audio_from_bytes(bytes(audio_source))


def augment_audio(wav: torch.Tensor) -> torch.Tensor:
    wav = wav * np.random.uniform(0.8, 1.2)
    noise = np.random.randn(len(wav)) * np.random.uniform(0.001, 0.01)
    wav = wav + torch.tensor(noise, dtype=torch.float32)
    if np.random.rand() < 0.3 and len(wav) > 2048:
        rate = np.random.uniform(0.9, 1.1)
        wav = torchaudio.functional.resample(
            wav.unsqueeze(0),
            TARGET_SR,
            int(TARGET_SR * rate),
        ).squeeze(0)
    wav = torch.clamp(wav, -1.0, 1.0)
    return wav


def get_hubert_embedding(wav: torch.Tensor) -> np.ndarray:
    inputs = feature_extractor(
        wav.numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(DEVICE)
    if DEVICE == "cuda":
        input_values = input_values.half()
    with torch.no_grad():
        hidden = model(input_values).last_hidden_state
        embedding = hidden.max(dim=1).values.squeeze(0).cpu().numpy()
    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray | torch.Tensor) -> float:
    target = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else b
    similarity = 1 - cosine(a, target)
    if not np.isfinite(similarity):
        return 0.0
    return float(similarity)


def get_target_phoneme_embedding(parquet_path: str | Path) -> np.ndarray:
    df = pd.read_parquet(parquet_path)
    embeddings = np.stack(df["embedding"].values)
    return embeddings.mean(axis=0)


def load_target_embeddings(parquet_path: str | Path, k: int = 15) -> list[torch.Tensor]:
    df = pd.read_parquet(parquet_path, columns=["embedding"])
    df = df.head(k)
    return [torch.tensor(x) for x in df["embedding"].tolist()]


def get_phoneme_letter(phoneme: str) -> str:
    return PHONEME_TO_LETTER.get(phoneme, phoneme)


def get_target_parquet_path(phoneme: str) -> Path:
    parquet_path = PARQUET_ROOT / f"{phoneme}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"No parquet found for phoneme '{phoneme}'")
    return parquet_path


def normalize_similarity(best_similarity: float | None) -> float:
    if best_similarity is None or not np.isfinite(best_similarity):
        return 0.0
    return max(0.0, min(1.0, float(best_similarity)))


def similarity_to_score(best_similarity: float | None, max_score: int) -> int:
    normalized = normalize_similarity(best_similarity)
    return int(round(normalized * max_score))


def extract_window_embeddings(
    audio_source: str | bytes | bytearray,
    augment: bool = False,
    window_sec: float = WINDOW_SEC,
) -> list[dict]:
    wav = load_audio(audio_source)

    if augment:
        wav = augment_audio(wav)

    window_len = int(window_sec * TARGET_SR)
    num_windows = max(1, len(wav) // window_len)
    window_embeddings = []

    for i in range(num_windows):
        start = i * window_len
        end = start + window_len

        segment = wav[start:end]
        if len(segment) < window_len:
            segment = torch.cat([segment, torch.zeros(window_len - len(segment))])

        window_embeddings.append(
            {
                "start_sample": start,
                "embedding": get_hubert_embedding(segment),
            }
        )

    return window_embeddings


def analyze_phoneme_from_windows(
    window_embeddings: Iterable[dict],
    target_parquet: str | Path,
    top_k: int = 15,
) -> dict:
    target_embeddings = load_target_embeddings(target_parquet, k=top_k)

    best_sim = -1.0
    best_emb = None
    best_start = 0
    all_embeddings = []

    for window_data in window_embeddings:
        emb = window_data["embedding"]
        all_embeddings.append(emb)
        sims = [cosine_similarity(emb, target_embedding) for target_embedding in target_embeddings]
        sim = max(sims) if sims else 0.0

        if sim > best_sim:
            best_sim = sim
            best_emb = emb
            best_start = window_data["start_sample"]

    return {
        "best_similarity": best_sim,
        "best_embedding": best_emb,
        "best_start_sample": best_start,
        "all_embeddings": all_embeddings,
    }


def analyze_phoneme(
    audio_source: str | bytes | bytearray,
    target_parquet: str | Path,
    augment: bool = True,
    window_sec: float = WINDOW_SEC,
    top_k: int = 15,
) -> dict:
    window_embeddings = extract_window_embeddings(
        audio_source=audio_source,
        augment=augment,
        window_sec=window_sec,
    )
    return analyze_phoneme_from_windows(
        window_embeddings=window_embeddings,
        target_parquet=target_parquet,
        top_k=top_k,
    )


def analyze_phonemes(
    audio_source: str | bytes | bytearray,
    phonemes: list[str],
    max_score: int,
    augment: bool = False,
    window_sec: float = WINDOW_SEC,
    top_k: int = 15,
) -> dict:

    window_embeddings = extract_window_embeddings(
        audio_source=audio_source,
        augment=augment,
        window_sec=window_sec,
    )

    results = []
    scores = {}

    for phoneme in phonemes:
        analysis = analyze_phoneme_from_windows(
            window_embeddings=window_embeddings,
            target_parquet=get_target_parquet_path(phoneme),
            top_k=top_k,
        )
        best_similarity = normalize_similarity(analysis["best_similarity"])
        score = similarity_to_score(best_similarity, max_score)
        scores[phoneme] = score
        results.append(
            {
                "phoneme": phoneme,
                "letter": get_phoneme_letter(phoneme),
                "score": score,
                "best_similarity": best_similarity,
            }
        )

    average_score = sum(scores.values()) / len(scores) if scores else 0.0
    return {
        "scores": scores,
        "results": results,
        "average_score": average_score,
    }
