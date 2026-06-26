"""Pydantic models for orientation coach API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class OrientationError(BaseModel):
    feature: str
    deviation_deg: float
    direction: str
    severity: Literal["low", "medium", "high"]


class ComparisonResult(BaseModel):
    sign_id: str
    overall_result: Literal["pass", "needs_correction", "unusable"]
    errors: list[OrientationError] = Field(default_factory=list)
    message: str | None = None
    usable_frame_ratio: float = 1.0


class OrientationAttempt(BaseModel):
    timestamp: str
    overall_result: str
    error_count: int
    feedback_text: str


class OrientationProgress(BaseModel):
    gloss: str
    attempts: list[OrientationAttempt] = Field(default_factory=list)
    mastered: bool = False
    attempt_count: int = 0


class OrientationReferenceMeta(BaseModel):
    sign_id: str
    display_name: str
    display_name_ml: str | None = None
    sign_type: Literal["static", "dynamic"]
    active_hand: Literal["left", "right"]
    critical_features: list[str]
    tolerance: dict[str, float]
    num_reference_frames: int


class AnalyzeResponse(BaseModel):
    comparison: ComparisonResult
    feedback_text: str
    progress: OrientationProgress
    display_name: str
    display_name_ml: str | None = None


class FeatureVectorDict(BaseModel):
    """Serializable per-frame feature vector."""

    palm_normal: list[float]
    finger_curls: dict[str, list[float]]
    wrist_flexion_deg: float
    confidence: float
    timestamp_ms: int = 0
    active_hand: str = "right"
