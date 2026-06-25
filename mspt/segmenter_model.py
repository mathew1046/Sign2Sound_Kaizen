"""Frame-level sign vs idle classifier for live boundary detection."""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_INPUT_DIM = 294  # hand(84) + body(66) + face(144)


class SignFrameClassifier(nn.Module):
    """BiLSTM binary classifier: P(sign) per pose frame."""

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Return logits ``(B, T)``."""
        out, _ = self.lstm(x)
        logits = self.head(out).squeeze(-1)
        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            logits = logits.masked_fill(~mask, 0.0)
        return logits

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """``(1, T, D)`` -> ``(T,)`` sign probabilities."""
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits[0])
