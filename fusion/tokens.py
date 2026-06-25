"""Sign tokens emitted by MSPT, alphabet, and glove workers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Source = Literal["mspt", "alphabet", "glove"]


@dataclass
class SignToken:
    gloss: str
    source: Source
    confidence: float
    timestamp: float
    meta: dict[str, Any] = field(default_factory=dict)
