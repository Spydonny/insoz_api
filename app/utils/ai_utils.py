import torch
import torchaudio
from transformers import Wav2Vec2Processor
from torch import nn
from transformers import Wav2Vec2Model
import os
from typing import Optional


class Wav2VecBiLSTM(nn.Module):
    def __init__(self, n_classes, freeze_w2v=False):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        if freeze_w2v:
            for param in self.wav2vec.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=self.wav2vec.config.hidden_size,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256 * 2, n_classes)

    def forward(self, input_values, attention_mask):
        output = self.wav2vec(input_values, attention_mask=attention_mask)
        hidden_states = output.last_hidden_state
        
        input_lengths = attention_mask.sum(dim=-1)
        compressed_lengths = self.wav2vec._get_feat_extract_output_lengths(input_lengths)

        lstm_output, _ = self.lstm(hidden_states)
        lstm_output = self.dropout(lstm_output)

        summed_output = lstm_output.sum(dim=1)
        clamped_lengths = torch.clamp(
            compressed_lengths.to(summed_output.device).unsqueeze(1),
            min=1.0
        )
        mean_output = summed_output / clamped_lengths
        logits = self.fc(mean_output)
        return logits

    # 👇 Добавляем удобный метод инференса
    def predict_from_path(self, audio_path, processor, device="cpu"):
        """
        Возвращает вероятности по всем классам (softmax) для данного аудио.
        """
        # 1. Загружаем аудио
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)  # моно
        
        # 2. Преобразуем для модели
        inputs = processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        # 3. Прогоняем через модель
        self.eval()
        with torch.no_grad():
            logits = self(input_values, attention_mask)
            probs = torch.softmax(logits, dim=-1).cpu().squeeze(0)
        
        return probs


def load_weights(model: nn.Module, weights_path: str = "../models/best_model.pth", map_location: Optional[torch.device] = None, strict: bool = True) -> nn.Module:
    """
    Load weights from a checkpoint into `model`.

    Behavior / contract:
    - Accepts a path to a PyTorch checkpoint saved with `torch.save(state_dict)` or
      a dict containing keys like 'state_dict' or 'model_state_dict'.
    - Automatically maps to CUDA if available unless `map_location` is provided.
    - Strips a leading 'module.' prefix from keys if the checkpoint was saved
      from a DataParallel model.

    Args:
        model: the nn.Module instance to load the weights into.
        weights_path: path to the .pth checkpoint file (default ../models/best_model.pth).
        map_location: optional torch.device or string to map tensors to. If None,
            will use 'cuda' when available otherwise 'cpu'.
        strict: passed to `load_state_dict` to control strict key matching.

    Returns:
        The model with loaded weights (also moved to the chosen device).

    Raises:
        FileNotFoundError: if the weights file does not exist.
        RuntimeError: if loading fails for any other reason.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    if map_location is None:
        map_location = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    try:
        checkpoint = torch.load(weights_path, map_location=map_location)

        # Unwrap common checkpoint wrappers
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # assume the dict is the state_dict itself
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Remove DataParallel 'module.' prefix if present
        new_state = {}
        model_keys = list(model.state_dict().keys())
        for k, v in state_dict.items():
            new_k = k
            if k.startswith('module.') and not any(key.startswith('module.') for key in model_keys):
                new_k = k.replace('module.', '', 1)
            new_state[new_k] = v

        model.load_state_dict(new_state, strict=strict)
        model.to(map_location)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load weights from {weights_path}: {e}")


