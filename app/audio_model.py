from pathlib import Path

AUDIO_MODEL = None


def get_audio_model():
    """
    Локальная аудио-модель (torch/torchaudio/transformers) — опциональна.

    Мы импортируем тяжёлые зависимости лениво, чтобы backend мог стартовать
    даже если эти библиотеки не установлены (например, чтобы пользоваться
    только MedGemma-отчётом).
    """

    global AUDIO_MODEL
    if AUDIO_MODEL is None:
        try:
            from app.services.ai_service import AudioInferencePipeline
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "AudioInferencePipeline unavailable. Install torch/torchaudio/transformers "
                "and related dependencies to enable audio analysis."
            ) from e

        AUDIO_MODEL = AudioInferencePipeline(
            model_path="models/best_model.pth",
            class_names=["stuttering", "aphasia", "dysarthria"],
        )
    return AUDIO_MODEL
