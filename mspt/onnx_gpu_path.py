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
