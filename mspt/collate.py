"""Variable-length batch collation — pad only to batch max T, not global max."""

from __future__ import annotations

import torch


def collate_mspt_batch(batch: list[dict]) -> dict:
    """Stack samples padded to the longest sequence in this batch."""
    max_t = max(int(s["length"]) for s in batch)
    bsz = len(batch)
    hand_dim = batch[0]["hand"].shape[-1]
    body_dim = batch[0]["body"].shape[-1]
    face_dim = batch[0]["face"].shape[-1]

    hand = torch.zeros(bsz, max_t, hand_dim, dtype=torch.float32)
    body = torch.zeros(bsz, max_t, body_dim, dtype=torch.float32)
    face = torch.zeros(bsz, max_t, face_dim, dtype=torch.float32)
    mask = torch.zeros(bsz, max_t, dtype=torch.float32)
    labels = torch.zeros(bsz, dtype=torch.long)
    lengths = torch.zeros(bsz, dtype=torch.long)

    for i, s in enumerate(batch):
        t = int(s["length"])
        hand[i, :t] = s["hand"][:t]
        body[i, :t] = s["body"][:t]
        face[i, :t] = s["face"][:t]
        mask[i, :t] = 1.0
        labels[i] = s["label"]
        lengths[i] = t

    return {
        "hand": hand,
        "body": body,
        "face": face,
        "mask": mask,
        "label": labels,
        "length": lengths,
    }
