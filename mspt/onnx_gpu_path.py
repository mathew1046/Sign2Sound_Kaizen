"""Put NVIDIA pip wheels (cuDNN, CUDA runtime) on the loader path for onnxruntime-gpu."""

from __future__ import annotations

import ctypes
import os
import site
import sys
from pathlib import Path

_CONFIGURED = False


def _nvidia_lib_dirs() -> list[Path]:
    roots: list[Path] = []
    candidates = list(site.getsitepackages())
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass
    candidates.append(str(Path(sys.executable).resolve().parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"))

    for base in candidates:
        nvidia = Path(base) / "nvidia"
        if nvidia.is_dir():
            roots.append(nvidia)

    lib_dirs: list[Path] = []
    for root in roots:
        for child in root.iterdir():
            lib = child / "lib"
            if lib.is_dir() and any(lib.glob("*.so*")):
                lib_dirs.append(lib.resolve())
    # Prefer cuDNN / CUDA runtime first.
    order = {"cudnn": 0, "cuda_runtime": 1, "cublas": 2, "cufft": 3, "cusparse": 4, "cusolver": 5, "nvjitlink": 6}
    lib_dirs.sort(key=lambda p: order.get(p.parent.name, 99))
    return lib_dirs


def configure_onnx_gpu_libs() -> list[str]:
    """Extend LD_LIBRARY_PATH and preload cuDNN so onnxruntime CUDA EP can load."""
    global _CONFIGURED
    if _CONFIGURED:
        return os.environ.get("ORT_NVIDIA_LIB_DIRS", "").split(":")

    added: list[str] = []
    for lib_dir in _nvidia_lib_dirs():
        s = str(lib_dir)
        if s not in added:
            added.append(s)

    if added:
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        merged = added + ([p for p in prev.split(":") if p] if prev else [])
        os.environ["LD_LIBRARY_PATH"] = ":".join(merged)
        os.environ["ORT_NVIDIA_LIB_DIRS"] = ":".join(added)
        for d in added:
            lib_path = Path(d)
            # Preload cuDNN split libs so onnxruntime_providers_cuda can resolve them.
            for so in sorted(lib_path.glob("libcudnn*.so*")):
                try:
                    ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass

    _CONFIGURED = True
    return added


def ort_session_providers(device: str) -> list:
    """ONNX Runtime providers with CUDA tuning when available."""
    import onnxruntime as ort

    if device == "cpu" or "CUDAExecutionProvider" not in ort.get_available_providers():
        return ["CPUExecutionProvider"]
    return [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider",
    ]


def recreate_ort_session(onnx_path: str, device: str):
    """Recreate an ONNX session with graph optimizations enabled."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(
        onnx_path,
        sess_options=opts,
        providers=ort_session_providers(device),
    )


def tune_rtmlib_onnx_sessions(model, device: str) -> None:
    """Replace rtmlib YOLOX/RTMPose sessions with optimized ORT sessions."""
    for attr in ("det_model", "pose_model"):
        tool = getattr(model, attr, None)
        if tool is None or getattr(tool, "backend", None) != "onnxruntime":
            continue
        onnx_path = getattr(tool, "onnx_model", None)
        if not onnx_path:
            continue
        try:
            tool.session = recreate_ort_session(onnx_path, device)
        except Exception:
            pass
