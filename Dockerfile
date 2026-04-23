FROM python:3.11-slim

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
    patchelf \
    pax-utils \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /uploads

ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface

RUN mkdir -p /cache/huggingface

COPY requirements.txt .

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --retries 10 \
  --timeout 60 \
  -r requirements.txt

# Patch execstack flag directly via Python struct manipulation
RUN python3 - <<'EOF'
import os, glob, struct

def clear_execstack(path):
    with open(path, 'r+b') as f:
        data = bytearray(f.read())
    # Check ELF magic
    if data[:4] != b'\x7fELF':
        return
    bits = data[4]  # 1=32bit, 2=64bit
    endian = '<' if data[5] == 1 else '>'
    if bits == 2:
        e_phoff = struct.unpack_from(endian + 'Q', data, 32)[0]
        e_phentsize = struct.unpack_from(endian + 'H', data, 54)[0]
        e_phnum = struct.unpack_from(endian + 'H', data, 56)[0]
        for i in range(e_phnum):
            off = e_phoff + i * e_phentsize
            p_type = struct.unpack_from(endian + 'I', data, off)[0]
            if p_type == 0x6474e551:  # PT_GNU_STACK
                p_flags = struct.unpack_from(endian + 'I', data, off + 4)[0]
                if p_flags & 0x1:  # has execute bit
                    p_flags &= ~0x1
                    struct.pack_into(endian + 'I', data, off + 4, p_flags)
                    with open(path, 'wb') as f:
                        f.write(data)
                    print(f"Patched: {path}")

for so in glob.glob('/usr/local/lib/python3.11/site-packages/ctranslate2/**/*.so*', recursive=True):
    if os.path.isfile(so) and not os.path.islink(so):
        clear_execstack(so)
EOF

RUN find /usr/local/lib/python3.11/site-packages/ctranslate2 -name "*.so*" | \
    xargs -I{} patchelf --clear-execstack {} 2>/dev/null || true

RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base')"

RUN python -c "from transformers import HubertModel, Wav2Vec2FeatureExtractor; \
HubertModel.from_pretrained('facebook/hubert-base-ls960'); \
Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960')"

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]