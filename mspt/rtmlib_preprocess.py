"""rtmlib wholebody preprocessing — same settings as Modal include50_rtmlib_1080."""

from __future__ import annotations

import ctypes.util
from pathlib import Path

import numpy as np

from mspt.onnx_gpu_path import configure_onnx_gpu_libs

configure_onnx_gpu_libs()

NUM_WHOLEBODY = 133
BODY_SLICE = slice(0, 17)
FOOT_SLICE = slice(17, 23)
FACE_SLICE = slice(23, 91)
LEFT_HAND_SLICE = slice(91, 112)
RIGHT_HAND_SLICE = slice(112, 133)

DET_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "yolox_x_8xb8-300e_humanart-a39d44ed.zip"
)
POSE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
    "rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip"
)
MODEL_NAME = "yolox-x + rtmw-x_384x288"
DET_INPUT_SIZE = (640, 640)
POSE_INPUT_SIZE = (288, 384)
CONF_THRESH = 0.05


def pick_best_instance(keypoints, scores) -> tuple[np.ndarray, np.ndarray]:
    kp = np.asarray(keypoints, dtype=np.float32)
    sc = np.asarray(scores, dtype=np.float32)
    if kp.size == 0:
        return (
            np.zeros((NUM_WHOLEBODY, 2), dtype=np.float32),
            np.zeros((NUM_WHOLEBODY,), dtype=np.float32),
        )

    if kp.ndim == 2 and kp.shape[0] % NUM_WHOLEBODY == 0:
        n_inst = kp.shape[0] // NUM_WHOLEBODY
        kp = kp.reshape(n_inst, NUM_WHOLEBODY, 2)
        sc = sc.reshape(n_inst, NUM_WHOLEBODY)
    elif kp.ndim == 2:
        kp = kp.reshape(1, -1, 2)
        sc = sc.reshape(1, -1)
    idx = int(np.argmax(sc.mean(axis=1)))
    return kp[idx], sc[idx]


def frame_to_wholebody_array(
    keypoints,
    scores,
    width: int,
    height: int,
) -> np.ndarray:
    """One frame -> ``(133, 4)`` with x_norm, y_norm, score, valid."""
    out = np.zeros((NUM_WHOLEBODY, 4), dtype=np.float32)
    kp, sc = pick_best_instance(keypoints, scores)
    n = min(NUM_WHOLEBODY, kp.shape[0], sc.shape[0])
    for i in range(n):
        x, y = float(kp[i, 0]), float(kp[i, 1])
        conf = float(sc[i])
        valid = float(conf > CONF_THRESH and 0 <= x <= width and 0 <= y <= height)
        out[i] = (x / max(width, 1), y / max(height, 1), conf, valid)
    return out


def split_wholebody(wholebody: np.ndarray) -> dict[str, np.ndarray]:
    """``(T, 133, 4)`` -> body/foot/face/left_hand/right_hand arrays."""
    return {
        "wholebody": wholebody,
        "body": wholebody[:, BODY_SLICE],
        "foot": wholebody[:, FOOT_SLICE],
        "face": wholebody[:, FACE_SLICE],
        "left_hand": wholebody[:, LEFT_HAND_SLICE],
        "right_hand": wholebody[:, RIGHT_HAND_SLICE],
    }


def _cudnn_available() -> bool:
    if ctypes.util.find_library("cudnn") or ctypes.util.find_library("cudnn.so.9"):
        return True
    import site
    import sys

    for base in site.getsitepackages() + [str(Path(sys.executable).parent / "lib")]:
        p = Path(base) / "nvidia" / "cudnn" / "lib"
        if any(p.glob("libcudnn.so*")):
            return True
    for lib_dir in (
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/local/cuda/lib64"),
        Path.home() / "anaconda3" / "lib",
    ):
        if any(lib_dir.glob("libcudnn.so*")):
            return True
    return False


def _probe_cuda_provider() -> bool:
    """Return True only if ONNX Runtime can actually create a CUDA session."""
    if not _cudnn_available():
        return False
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False
        # Probe with a real cached ONNX model (synthetic graphs may use unsupported IR).
        det = Path.home() / ".cache/rtmlib/hub/checkpoints/yolox_x_8xb8-300e_humanart-a39d44ed.onnx"
        if det.is_file():
            sess = ort.InferenceSession(
                str(det),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            return "CUDAExecutionProvider" in sess.get_providers()
        return bool(configure_onnx_gpu_libs())
    except Exception:
        return False


def resolve_device(prefer: str = "cuda") -> str:
    if prefer == "cpu":
        return "cpu"
    if _probe_cuda_provider():
        return "cuda"
    return "cpu"


class RtmlibWholebodyExtractor:
    """YOLOX-x + RTMW-x wholebody extractor (training-time settings)."""

    def __init__(self, device: str | None = None):
        from rtmlib import Wholebody

        requested = device or "cuda"
        self.device = resolve_device(requested)
        if self.device == "cpu" and requested != "cpu":
            print(
                "[rtmlib] WARN: onnxruntime GPU unavailable (cuDNN/CUDA libs missing) — "
                "using CPU for pose (MSPT torch inference can still use GPU)"
            )
        self._model = Wholebody(
            det=DET_URL,
            det_input_size=DET_INPUT_SIZE,
            pose=POSE_URL,
            pose_input_size=POSE_INPUT_SIZE,
            backend="onnxruntime",
            device=self.device,
            to_openpose=False,
        )

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        keypoints, scores = self._model(frame_bgr)
        return frame_to_wholebody_array(keypoints, scores, width, height)

    def close(self) -> None:
        del self._model
