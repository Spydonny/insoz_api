import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch import nn
import warnings
from typing import Dict, Any, Optional
import math

warnings.filterwarnings("ignore", category=UserWarning)


# ------------------------------
# КОНФИГ
# ------------------------------

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAMPLE_RATE = 16000
    MAX_SEC = 10
    MAX_LEN = SAMPLE_RATE * MAX_SEC
    CHUNK_STEP_SEC = 5
    CHUNK_STEP = SAMPLE_RATE * CHUNK_STEP_SEC
    LOAD_CHUNK_SEC = 15
    LOAD_CHUNK_SAMPLES = SAMPLE_RATE * LOAD_CHUNK_SEC


# ------------------------------
# АРХИТЕКТУРА МОДЕЛИ
# ------------------------------

class Wav2VecBiLSTM(nn.Module):
    def __init__(self, n_classes, freeze_w2v=False):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        if freeze_w2v:
            for p in self.wav2vec.parameters():
                p.requires_grad = False
        self.lstm = nn.LSTM(
            input_size=self.wav2vec.config.hidden_size, hidden_size=256,
            num_layers=2, bidirectional=True, batch_first=True
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256*2, n_classes)

    def forward(self, input_values, attention_mask):
        out = self.wav2vec(input_values, attention_mask=attention_mask)
        hidden_states = out.last_hidden_state
        input_lengths = attention_mask.sum(dim=-1)
        compressed_lengths = self.wav2vec._get_feat_extract_output_lengths(input_lengths)
        lstm_out, _ = self.lstm(hidden_states)
        lstm_out = self.dropout(lstm_out)
        summed = lstm_out.sum(dim=1)
        mean_output = summed / torch.clamp(
            compressed_lengths.to(summed.device).unsqueeze(1), min=1.0
        )
        logits = self.fc(mean_output)
        return logits


# ------------------------------
# VAD Очистка
# ------------------------------

def apply_vad_and_clean(waveform, vad_model, vad_utils):
    if vad_model is None or vad_utils is None:
        return waveform
    (get_speech_timestamps, _, _, _, collect_chunks) = vad_utils
    try:
        speech_timestamps = get_speech_timestamps(
            waveform.cpu(), vad_model, sampling_rate=Config.SAMPLE_RATE,
            min_speech_duration_ms=250, min_silence_duration_ms=100
        )
        if not speech_timestamps:
            return torch.tensor([])
        cleaned = collect_chunks(speech_timestamps, waveform.cpu())
        return cleaned
    except Exception:
        return waveform


# ------------------------------
# Утилиты инференса
# ------------------------------

def _get_logits_for_waveform(waveform, model, processor):
    if waveform.shape[0] < Config.SAMPLE_RATE * 0.5:
        return None

    # короткие куски
    if waveform.shape[0] <= Config.MAX_LEN:
        inputs = processor(
            [waveform],
            sampling_rate=Config.SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=Config.MAX_LEN,
            return_attention_mask=True
        )
        input_values = inputs.input_values.to(Config.DEVICE)
        attention_mask = inputs.attention_mask.to(Config.DEVICE)
        with torch.no_grad():
            logits = model(input_values, attention_mask)
        return logits

    # длинные куски
    chunks = []
    for start in range(0, waveform.shape[0] - Config.MAX_LEN + 1, Config.CHUNK_STEP):
        chunks.append(waveform[start:start + Config.MAX_LEN])
    if not chunks:
        chunks.append(waveform[-Config.MAX_LEN:])

    inputs = processor(
        [c.numpy() for c in chunks],
        sampling_rate=Config.SAMPLE_RATE,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=Config.MAX_LEN,
        return_attention_mask=True
    )

    input_values = inputs.input_values.to(Config.DEVICE)
    attention_mask = inputs.attention_mask.to(Config.DEVICE)
    with torch.no_grad():
        logits = model(input_values, attention_mask)
    return logits


# ------------------------------
# ГЛАВНАЯ УТИЛИТА
# ------------------------------

class AudioInferencePipeline:
    def __init__(self, model_path: str, class_names: list[str]):
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.device = Config.DEVICE

        # модель и процессор
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2VecBiLSTM(self.n_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # VAD
        try:
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True
            )
            self.vad_model.to(self.device)
        except Exception:
            self.vad_model, self.vad_utils = None, None
            warnings.warn("⚠️ VAD не загружен — будет пропущен.")

    def analyze(self, file_path: str) -> Dict[str, Any]:
        try:
            info = torchaudio.info(file_path)
            total_frames = info.num_frames
            file_sr = info.sample_rate
        except Exception as e:
            return {"error": f"Cannot read file: {e}"}

        resampler = None
        if file_sr != Config.SAMPLE_RATE:
            resampler = T.Resample(orig_freq=file_sr, new_freq=Config.SAMPLE_RATE)

        all_logits_list = []
        total_chunks = math.ceil(total_frames / (Config.LOAD_CHUNK_SEC * file_sr))

        for offset in range(0, total_frames, Config.LOAD_CHUNK_SEC * file_sr):
            num_frames = min(Config.LOAD_CHUNK_SEC * file_sr, total_frames - offset)
            try:
                chunk, sr = torchaudio.load(file_path, frame_offset=offset, num_frames=num_frames)
            except Exception:
                continue

            if resampler:
                chunk = resampler(chunk)
            if chunk.shape[0] > 1:
                chunk = chunk.mean(dim=0)

            waveform = chunk.squeeze(0)
            cleaned = apply_vad_and_clean(waveform, self.vad_model, self.vad_utils)
            if cleaned.shape[0] == 0:
                continue

            logits = _get_logits_for_waveform(cleaned, self.model, self.processor)
            if logits is not None:
                all_logits_list.append(logits.cpu())

        if not all_logits_list:
            return {"error": "No speech detected"}

        all_logits = torch.cat(all_logits_list, dim=0)
        final_logits = torch.mean(all_logits.float(), dim=0)

        probs = F.softmax(final_logits, dim=0).cpu()
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

        return {
            "predicted_label": self.class_names[pred_idx],
            "confidence": confidence,
            "all_probabilities": {
                self.class_names[i]: probs[i].item() for i in range(len(probs))
            },
        }
        