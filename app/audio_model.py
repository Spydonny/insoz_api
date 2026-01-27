from pathlib import Path
from app.services.ai_service import AudioInferencePipeline

AUDIO_MODEL = None

def get_audio_model():
    global AUDIO_MODEL
    if AUDIO_MODEL is None:
        AUDIO_MODEL = AudioInferencePipeline(
            model_path="models/best_model.pth",
            class_names=["stuttering", "aphasia", "dysarthria"]
        )
    return AUDIO_MODEL
