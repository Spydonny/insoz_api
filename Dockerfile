FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/uploads
RUN mkdir -p /uploads

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --retries 10 \
  --timeout 60 \
  -r requirements.txt

RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base')"

RUN python -c "from transformers import HubertModel, Wav2Vec2FeatureExtractor; \
HubertModel.from_pretrained('facebook/hubert-base-ls960'); \
Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960')"

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
