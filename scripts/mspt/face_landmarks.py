"""Expression-relevant MediaPipe face mesh indices (subset of 468)."""

from __future__ import annotations

# Lips, eyes, eyebrows — common non-manual marker regions for sign language.
FACE_IDXS: tuple[int, ...] = tuple(
    sorted(
        {
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
            70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
            300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
        }
    )
)

NUM_FACE = len(FACE_IDXS)

# Chain edges along each contour (for optional GCN; MLP path ignores these).
FACE_EDGES: list[tuple[int, int]] = []
for ring in (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
    (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35),
    (36, 37, 38, 39, 40, 41, 42, 43, 44, 45),
    (46, 47, 48, 49, 50, 51, 52, 53, 54, 55),
    (56, 57, 58, 59, 60, 61, 62, 63, 64, 65),
):
    for a, b in zip(ring[:-1], ring[1:]):
        if a < NUM_FACE and b < NUM_FACE:
            FACE_EDGES.append((a, b))
