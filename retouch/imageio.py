"""Введення/виведення. Все всередині — float32 BGR у [0..1].

Розрядність вхідного файлу запам'ятовується і повертається на запис,
щоб 16-бітний TIFF з Camera Raw не деградував до 8 біт.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_MAXV = {np.dtype("uint8"): 255.0, np.dtype("uint16"): 65535.0}


def read(path: str | Path) -> tuple[np.ndarray, np.dtype]:
    """Повертає (float32 BGR [0..1], оригінальний dtype)."""
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"не читається: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    src_dtype = img.dtype
    if src_dtype == np.float32 or src_dtype == np.float64:
        return np.clip(img.astype(np.float32), 0, 1), np.dtype("uint16")
    scale = _MAXV.get(np.dtype(src_dtype))
    if scale is None:
        raise ValueError(f"непідтримувана розрядність: {src_dtype}")
    return img.astype(np.float32) / scale, np.dtype(src_dtype)


def write(path: str | Path, img: np.ndarray, dtype: np.dtype | None = None,
          alpha: np.ndarray | None = None) -> None:
    """Пише BGR (або BGRA, якщо задано alpha) у вказаній розрядності."""
    dtype = np.dtype(dtype or "uint16")
    scale = _MAXV[dtype]
    out = np.clip(img, 0, 1)
    if alpha is not None:
        out = np.dstack([out, np.clip(alpha, 0, 1)])
    out = (out * scale + 0.5).astype(dtype)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), out):
        raise IOError(f"не записалося: {path}")


def read_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Читає маску (біла = обробляти) і підганяє під розмір кадру."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"маска не читається: {path}")
    if m.shape[:2] != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)
