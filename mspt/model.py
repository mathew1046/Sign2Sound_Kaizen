"""Multi-Stream Pose Transformer (MSPT) — memory-aware forward pass."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class StreamEncoder(nn.Module):
    """Linear projection + CLS + sinusoidal PE + transformer encoder blocks."""

    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 256,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        self.input_proj = nn.Linear(in_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len + 1, dropout=dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            for _ in range(num_layers)
        ])

    def _run_layers(self, h: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                h = checkpoint(
                    lambda inp, lyr=layer, p=pad: lyr(inp, src_key_padding_mask=p),
                    h,
                    use_reentrant=False,
                )
            else:
                h = layer(h, src_key_padding_mask=pad)
        return h

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        h = self.input_proj(x)
        cls = self.cls_token.expand(b, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.pos_enc(h)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=mask.device), mask == 0], dim=1)
        h = self._run_layers(h, pad)
        return h[:, 0]


class MCAFusion(nn.Module):
    """Hand query attends to body+face; gated concat of three CLS tokens."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 3, d_model * 3), nn.Sigmoid())

    def forward(
        self,
        cls_hand: torch.Tensor,
        cls_body: torch.Tensor,
        cls_face: torch.Tensor,
    ) -> torch.Tensor:
        ctx = torch.stack([cls_body, cls_face], dim=1)
        q = cls_hand.unsqueeze(1)
        attn_out, _ = self.attn(q, ctx, ctx)
        hand_refined = self.norm(cls_hand + attn_out.squeeze(1))
        fused = torch.cat([hand_refined, cls_body, cls_face], dim=-1)
        return fused * self.gate(fused)


class MSPT(nn.Module):
    def __init__(
        self,
        hand_dim: int = 84,
        body_dim: int = 66,
        face_dim: int = 144,
        d_model: int = 128,
        num_classes: int = 50,
        max_len: int = 128,
        joint_dropout: float = 0.3,
        use_checkpoint: bool = True,
        sequential_streams: bool = True,
    ):
        super().__init__()
        self.sequential_streams = sequential_streams
        self.use_checkpoint = use_checkpoint
        self.hand_enc = StreamEncoder(
            hand_dim, d_model=d_model, max_len=max_len, use_checkpoint=use_checkpoint
        )
        self.body_enc = StreamEncoder(
            body_dim, d_model=d_model, max_len=max_len, use_checkpoint=use_checkpoint
        )
        self.face_enc = StreamEncoder(
            face_dim, d_model=d_model, max_len=max_len, use_checkpoint=use_checkpoint
        )
        self.fusion = MCAFusion(d_model)
        self.joint_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model * 3,
                nhead=8,
                dim_feedforward=d_model * 6,
                dropout=joint_dropout,
                batch_first=True,
                activation="gelu",
            )
            for _ in range(2)
        ])
        self.joint_cls = nn.Parameter(torch.zeros(1, 1, d_model * 3))
        nn.init.trunc_normal_(self.joint_cls, std=0.02)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, num_classes),
        )

    def _encode_stream(
        self,
        enc: StreamEncoder,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return enc(x, mask)

    def _run_joint(self, seq: torch.Tensor) -> torch.Tensor:
        for layer in self.joint_layers:
            if self.use_checkpoint and self.training:
                seq = checkpoint(layer, seq, use_reentrant=False)
            else:
                seq = layer(seq)
        return seq[:, 0]

    def forward(self, hand: torch.Tensor, body: torch.Tensor, face: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Sequential stream encoding lowers peak activation memory on small GPUs.
        if self.sequential_streams:
            cls_h = self._encode_stream(self.hand_enc, hand, mask)
            cls_b = self._encode_stream(self.body_enc, body, mask)
            cls_f = self._encode_stream(self.face_enc, face, mask)
        else:
            cls_h, cls_b, cls_f = (
                self.hand_enc(hand, mask),
                self.body_enc(body, mask),
                self.face_enc(face, mask),
            )
        fused = self.fusion(cls_h, cls_b, cls_f)
        b = fused.size(0)
        seq = torch.cat([self.joint_cls.expand(b, -1, -1), fused.unsqueeze(1)], dim=1)
        return self.classifier(self._run_joint(seq))
