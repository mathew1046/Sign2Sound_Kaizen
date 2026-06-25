"""Compare user feature sequences against reference orientation data."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from dashboard.server.orientation.features import palm_facing_direction, palm_normal_angle_deg
from dashboard.server.orientation.schemas import ComparisonResult, OrientationError

MIN_USABLE_FRAME_RATIO = 0.6
STATIC_STABILITY_MS = 500
STATIC_MOTION_THRESH = 0.015


def dtw_align(user: list[dict[str, Any]], ref: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Return list of (user_idx, ref_idx) alignment pairs."""
    na, nb = len(user), len(ref)
    if na == 0 or nb == 0:
        return []
    dp = np.full((na + 1, nb + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    back = np.zeros((na + 1, nb + 1, 2), dtype=np.int32)

    def _frame_cost(i: int, j: int) -> float:
        return _feature_distance(user[i], ref[j])

    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = _frame_cost(i - 1, j - 1)
            candidates = [
                (dp[i - 1, j] + cost, i - 1, j),
                (dp[i, j - 1] + cost, i, j - 1),
                (dp[i - 1, j - 1] + cost, i - 1, j - 1),
            ]
            best = min(candidates, key=lambda x: x[0])
            dp[i, j] = best[0]
            back[i, j] = (best[1], best[2])

    pairs: list[tuple[int, int]] = []
    i, j = na, nb
    while i > 0 and j > 0:
        pairs.append((i - 1, j - 1))
        pi, pj = int(back[i, j, 0]), int(back[i, j, 1])
        if pi == i and pj == j:
            break
        i, j = pi, pj
    pairs.reverse()
    return pairs


def _feature_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    palm = palm_normal_angle_deg(a.get("palm_normal", [0, 0, 0]), b.get("palm_normal", [0, 0, 0]))
    flex = abs(float(a.get("wrist_flexion_deg", 0)) - float(b.get("wrist_flexion_deg", 0)))
    curl = _mean_curl_distance(a.get("finger_curls", {}), b.get("finger_curls", {}))
    return palm + 0.5 * flex + 0.3 * curl


def _mean_curl_distance(ac: dict, bc: dict) -> float:
    dists = []
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        fa = ac.get(finger, [])
        fb = bc.get(finger, [])
        for i in range(min(len(fa), len(fb))):
            dists.append(abs(float(fa[i]) - float(fb[i])))
    return float(np.mean(dists)) if dists else 0.0


def _usable_frames(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [f for f in features if float(f.get("confidence", 0)) >= 0.5]


def _stability_window(
    features: list[dict[str, Any]],
    fps: float,
    window_ms: int = STATIC_STABILITY_MS,
) -> list[dict[str, Any]] | None:
    if not features:
        return None
    window_frames = max(2, int(window_ms * fps / 1000.0))
    if len(features) < window_frames:
        return features

    best_start = 0
    best_motion = float("inf")
    for start in range(0, len(features) - window_frames + 1):
        window = features[start : start + window_frames]
        motion = 0.0
        for i in range(1, len(window)):
            p0 = window[i - 1].get("palm_normal", [0, 0, 0])
            p1 = window[i].get("palm_normal", [0, 0, 0])
            motion += palm_normal_angle_deg(p0, p1)
        motion /= max(len(window) - 1, 1)
        if motion < best_motion:
            best_motion = motion
            best_start = start

    if best_motion > STATIC_MOTION_THRESH * 180:
        return features
    return features[best_start : best_start + window_frames]


def _average_features(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if not frames:
        return {}
    palm = np.mean([f.get("palm_normal", [0, 0, 0]) for f in frames], axis=0)
    norm = float(np.linalg.norm(palm))
    if norm > 1e-8:
        palm = (palm / norm).tolist()
    else:
        palm = [0.0, 0.0, 0.0]

    curls: dict[str, list[float]] = {}
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        arrays = [f.get("finger_curls", {}).get(finger, []) for f in frames]
        max_len = max((len(a) for a in arrays), default=0)
        vals = []
        for j in range(max_len):
            col = [float(a[j]) for a in arrays if j < len(a)]
            vals.append(float(np.mean(col)) if col else 0.0)
        curls[finger] = vals

    flex = float(np.mean([float(f.get("wrist_flexion_deg", 0)) for f in frames]))
    return {
        "palm_normal": palm,
        "finger_curls": curls,
        "wrist_flexion_deg": flex,
        "confidence": float(np.mean([float(f.get("confidence", 0)) for f in frames])),
    }


def _severity(deviation: float, tolerance: float) -> Literal["low", "medium", "high"]:
    ratio = deviation / max(tolerance, 1e-6)
    if ratio >= 2.0:
        return "high"
    if ratio >= 1.2:
        return "medium"
    return "low"


def _compare_averaged(
    user_avg: dict[str, Any],
    ref_avg: dict[str, Any],
    tolerance: dict[str, float],
    critical_features: list[str],
) -> list[OrientationError]:
    errors: list[OrientationError] = []

    if "palm_normal" in critical_features:
        palm_tol = float(tolerance.get("palm_normal_deg", 15))
        dev = palm_normal_angle_deg(user_avg.get("palm_normal", [0, 0, 0]), ref_avg.get("palm_normal", [0, 0, 0]))
        if dev > palm_tol:
            ref_z = float(ref_avg.get("palm_normal", [0, 0, 0])[2])
            user_z = float(user_avg.get("palm_normal", [0, 0, 0])[2])
            errors.append(
                OrientationError(
                    feature="palm_normal",
                    deviation_deg=round(dev, 1),
                    direction=palm_facing_direction(ref_z, user_z),
                    severity=_severity(dev, palm_tol),
                )
            )

    if "wrist_flexion_deg" in critical_features:
        flex_tol = float(tolerance.get("wrist_flexion_deg", 10))
        dev = abs(float(user_avg.get("wrist_flexion_deg", 0)) - float(ref_avg.get("wrist_flexion_deg", 0)))
        if dev > flex_tol:
            direction = "wrist too extended" if user_avg.get("wrist_flexion_deg", 0) > ref_avg.get("wrist_flexion_deg", 0) else "wrist too flexed"
            errors.append(
                OrientationError(
                    feature="wrist_flexion_deg",
                    deviation_deg=round(dev, 1),
                    direction=direction,
                    severity=_severity(dev, flex_tol),
                )
            )

    curl_tol = float(tolerance.get("finger_curl_deg", 12))
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        feat_name = f"{finger}_finger_curl"
        if feat_name not in critical_features and "finger_curls" not in critical_features:
            continue
        uc = user_avg.get("finger_curls", {}).get(finger, [])
        rc = ref_avg.get("finger_curls", {}).get(finger, [])
        for i, (u, r) in enumerate(zip(uc, rc)):
            dev = abs(float(u) - float(r))
            if dev > curl_tol:
                direction = "not curled enough" if float(u) > float(r) else "curled too much"
                errors.append(
                    OrientationError(
                        feature=f"{finger}_finger_curl_{i}",
                        deviation_deg=round(dev, 1),
                        direction=direction,
                        severity=_severity(dev, curl_tol),
                    )
                )

    return errors


def _aggregate_dynamic_errors(
    user: list[dict[str, Any]],
    ref: list[dict[str, Any]],
    pairs: list[tuple[int, int]],
    tolerance: dict[str, float],
    critical_features: list[str],
) -> list[OrientationError]:
    """Aggregate per-aligned-pair deviations."""
    accum: dict[str, list[float]] = {}
    directions: dict[str, list[str]] = {}

    for ui, ri in pairs:
        pair_errors = _compare_averaged(user[ui], ref[ri], tolerance, critical_features)
        for err in pair_errors:
            accum.setdefault(err.feature, []).append(err.deviation_deg)
            directions.setdefault(err.feature, []).append(err.direction)

    errors: list[OrientationError] = []
    threshold_ratio = 0.35
    for feat, devs in accum.items():
        if len(devs) < max(1, int(len(pairs) * threshold_ratio)):
            continue
        mean_dev = float(np.mean(devs))
        tol_key = "palm_normal_deg" if feat == "palm_normal" else (
            "wrist_flexion_deg" if feat == "wrist_flexion_deg" else "finger_curl_deg"
        )
        tol = float(tolerance.get(tol_key, 12))
        if mean_dev <= tol:
            continue
        direction = max(set(directions.get(feat, ["off"])), key=directions.get(feat, ["off"]).count)
        errors.append(
            OrientationError(
                feature=feat,
                deviation_deg=round(mean_dev, 1),
                direction=direction,
                severity=_severity(mean_dev, tol),
            )
        )
    return errors


def compare_sequences(
    sign_id: str,
    user_features: list[dict[str, Any]],
    reference: dict[str, Any],
    *,
    fps: float = 25.0,
) -> ComparisonResult:
    """Compare user features against a reference document."""
    ref_seq = reference.get("reference_sequence", [])
    tolerance = reference.get("tolerance", {})
    critical = reference.get("critical_features", ["palm_normal", "wrist_flexion_deg", "finger_curls"])
    sign_type = reference.get("sign_type", "dynamic")

    usable = _usable_frames(user_features)
    ratio = len(usable) / max(len(user_features), 1)
    if ratio < MIN_USABLE_FRAME_RATIO:
        return ComparisonResult(
            sign_id=sign_id,
            overall_result="unusable",
            errors=[],
            message="Hand or arm not visible enough. Reposition so your signing hand and upper body stay in frame.",
            usable_frame_ratio=round(ratio, 3),
        )

    if sign_type == "static":
        window = _stability_window(usable, fps) or usable
        user_avg = _average_features(window)
        ref_avg = _average_features(ref_seq) if ref_seq else {}
        errors = _compare_averaged(user_avg, ref_avg, tolerance, critical)
    else:
        pairs = dtw_align(usable, ref_seq)
        errors = _aggregate_dynamic_errors(usable, ref_seq, pairs, tolerance, critical)

    overall = "pass" if not errors else "needs_correction"
    return ComparisonResult(
        sign_id=sign_id,
        overall_result=overall,
        errors=errors,
        usable_frame_ratio=round(ratio, 3),
    )
