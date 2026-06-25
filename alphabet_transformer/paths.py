from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "sign_transformer_alphabet.pth"
HAND_LANDMARKER = ROOT / "hand_landmarker.task"
