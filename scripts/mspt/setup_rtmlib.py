#!/usr/bin/env python3
"""Install-check rtmlib + ONNX models used for INCLUDE-50 preprocessing."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parent
CACHE = Path.home() / ".cache" / "rtmlib" / "hub" / "checkpoints"
sys.path.insert(0, str(NOTEBOOKS))

REQUIRED_PACKAGES = (
    ("rtmlib", "rtmlib"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("onnxruntime", "onnxruntime-gpu"),
)

# Matches notebooks/mspt/rtmlib_preprocess.py
EXPECTED_ONNX = (
    "yolox_x_8xb8-300e_humanart-a39d44ed.onnx",
    "rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.onnx",
)


def _check_packages() -> list[str]:
    errors: list[str] = []
    for module, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module)
        except ImportError as exc:
            errors.append(f"missing {pip_name}: {exc}")
    return errors


def _check_onnxruntime() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    info: list[str] = []
    try:
        import onnxruntime as ort
    except ImportError as exc:
        return [f"onnxruntime import failed: {exc}"], info

    providers = ort.get_available_providers()
    info.append(f"onnxruntime {ort.__version__} providers={providers}")
    from mspt.rtmlib_preprocess import _cudnn_available, _probe_cuda_provider

    if _probe_cuda_provider():
        info.append("onnxruntime CUDA: usable")
    elif "CUDAExecutionProvider" in providers:
        info.append("WARN: CUDAExecutionProvider listed but not usable (install cuDNN 9 for GPU pose)")
    else:
        info.append("onnxruntime CUDA: not available — rtmlib will use CPU")
    return errors, info


def _check_cached_models() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    info: list[str] = []
    if not CACHE.is_dir():
        errors.append(f"model cache missing: {CACHE}")
        return errors, info

    for stale in CACHE.glob("tmp*"):
        stale.unlink(missing_ok=True)
        info.append(f"removed stale partial download: {stale.name}")

    present = {p.name for p in CACHE.glob("*.onnx")}
    for name in EXPECTED_ONNX:
        if name in present:
            mb = CACHE.joinpath(name).stat().st_size / 1e6
            info.append(f"cached {name} ({mb:.0f} MB)")
        else:
            info.append(f"will download on first use: {name}")
    return errors, info


def _smoke_test() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    info: list[str] = []
    sys.path.insert(0, str(NOTEBOOKS))
    import numpy as np
    from mspt.rtmlib_preprocess import RtmlibWholebodyExtractor

    ext = RtmlibWholebodyExtractor()
    info.append(f"rtmlib device: {ext.device}")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    wb = ext.process_frame(frame)
    ext.close()
    if wb.shape != (133, 4):
        errors.append(f"unexpected output shape {wb.shape}")
    else:
        info.append(f"smoke test ok: output (133, 4) float32")
    return errors, info


def main() -> int:
    import os

    py = sys.executable
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    print("=== rtmlib setup check ===")
    print(f"python: {py}")
    if conda_env:
        print(f"conda env: {conda_env}")
    if conda_env and conda_env != "base":
        print("NOTE: using conda env {!r} — for this project use: conda activate base".format(conda_env))
    print()

    all_errors: list[str] = []
    for label, fn in (
        ("packages", _check_packages),
        ("onnxruntime", lambda: _check_onnxruntime()[0]),
        ("cache", lambda: _check_cached_models()[0]),
    ):
        errs = fn()
        all_errors.extend(errs)
        status = "OK" if not errs else "FAIL"
        print(f"[{status}] {label}")
        for e in errs:
            print(f"  - {e}")

    for label, fn in (
        ("onnxruntime", _check_onnxruntime),
        ("cache", _check_cached_models),
    ):
        _, info = fn()
        for line in info:
            print(f"  {line}")

    print("\n[run] smoke test (downloads pose model if needed)...")
    try:
        smoke_errs, smoke_info = _smoke_test()
        all_errors.extend(smoke_errs)
        for line in smoke_info:
            print(f"  {line}")
        print("  OK" if not smoke_errs else "  FAIL")
        for e in smoke_errs:
            print(f"  - {e}")
    except Exception as exc:
        all_errors.append(f"smoke test exception: {exc}")
        print(f"  FAIL: {exc}")

    if all_errors:
        print(f"\nFAILED ({len(all_errors)} issues)")
        print("Fix: pip install -r notebooks/requirements-rtmlib.txt")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
