from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8")

path = './uploads/db18ee5f-3f3c-4dc3-b167-7c2221e346f1_whatsapp ptt 2025-11-10 at 16.19.16.ogg'

segments, info = model.transcribe(path, language="ru")

for segment in segments:
    print(segment.text)
