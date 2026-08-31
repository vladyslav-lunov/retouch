"""Синтетична "шкіра" з відомими дефектами — для тестів без реальних фото.

Дає градієнт тону (низька частота), зернистість пор (висока частота)
і N темних плям із заданими координатами. Це дозволяє перевіряти
детектор об'єктивно: ми точно знаємо, де дефекти.
"""

from __future__ import annotations

import cv2
import numpy as np


def make_skin(h: int = 512, w: int = 512, n_spots: int = 12,
              seed: int = 7, spot_strength: float = 0.06
              ) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Повертає (float32 BGR [0..1], список (x, y, r) дефектів)."""
    rng = np.random.default_rng(seed)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    tone = 0.62 + 0.10 * (xx / w) + 0.06 * np.sin(yy / h * 3.0)
    base = np.dstack([tone * 0.78, tone * 0.88, tone])  # BGR, тепліший в R

    pores = rng.normal(0, 1, (h, w, 3)).astype(np.float32)
    pores = cv2.GaussianBlur(pores, (0, 0), 0.9)
    pores /= (pores.std() + 1e-8)
    img = np.clip(base + pores * 0.012, 0, 1)

    spots: list[tuple[int, int, int]] = []
    m = 40
    for _ in range(n_spots):
        x = int(rng.integers(m, w - m))
        y = int(rng.integers(m, h - m))
        r = int(rng.integers(3, 9))
        spots.append((x, y, r))
        blob = np.zeros((h, w), np.float32)
        cv2.circle(blob, (x, y), r, 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), r * 0.45)
        img -= blob[..., None] * np.array([0.4, 0.7, 1.0], np.float32) * spot_strength
    return np.clip(img, 0, 1).astype(np.float32), spots
