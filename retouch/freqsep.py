"""Частотне розкладання.

Працюємо в тому ж кодуванні, що й вхід (display-referred, з гамою).
Це навмисно: у лінійному просторі частотка дає інший вигляд, ніж
звична в Photoshop, і високочастотний шар перестає бути "текстурою".
"""

from __future__ import annotations

import cv2
import numpy as np


def freq_split(img: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """img: float32 BGR у [0..1]. Повертає (low, high).

    high — знакова різниця (0 = немає деталі), не зсунута на 0.5.
    radius ≈ розмір найбільшої деталі, яку ще вважаємо текстурою, у px.
    Для портрета 24 Мп це приблизно 5-8 px, для 50 Мп — 8-14 px.
    """
    low = cv2.GaussianBlur(img, (0, 0), radius, borderType=cv2.BORDER_REPLICATE)
    return low, img - low


def freq_merge(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(low + high, 0.0, 1.0)


def luma(img: np.ndarray) -> np.ndarray:
    """Яскравість (працює і зі знаковим HF-шаром). BGR -> 1 канал."""
    return 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]


def radius_for(shape: tuple[int, ...], skin_mask: np.ndarray | None = None,
               base_face: float = 1200.0, base_radius: float = 6.0,
               lo: float = 2.0, hi: float = 32.0) -> float:
    """Радіус частотки з РОЗМІРУ ОБЛИЧЧЯ, а не з мегапікселів кадру.

    Це принципово. Прив'язка до роздільності файлу ламається на кропах:
    поясний портрет і кроп голови з того самого файлу мають однакові
    мегапікселі, але зовсім різний масштаб пор. Прив'язка до ширини
    маски шкіри лишається стабільною і на кропі, і на іншій камері.

    Калібрування: обличчя шириною 1200 px -> радіус 6 px.
    """
    if skin_mask is not None and skin_mask.any():
        cols = np.nonzero(skin_mask.any(axis=0))[0]
        face_w = float(cols.max() - cols.min() + 1)
    else:
        face_w = 0.55 * min(shape[0], shape[1])  # груба здогадка без маски
    return float(np.clip(base_radius * face_w / base_face, lo, hi))
