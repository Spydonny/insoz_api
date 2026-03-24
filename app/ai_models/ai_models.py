import torch
import torchaudio
from transformers import Wav2Vec2Processor
from torch import nn
from transformers import Wav2Vec2Model


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
