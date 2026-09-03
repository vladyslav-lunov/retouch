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


def face_width(shape: tuple[int, ...], skin_mask: np.ndarray | None = None,
               face_w: float | None = None) -> tuple[float, str]:
    """Ширина обличчя в пікселях і те, ЗВІДКИ вона взята.

    Три джерела, у порядку спадання довіри, і різниця між ними не
    академічна — заміряно на 44 реальних кадрах (spec.md §6.3):

    **рамка детектора** — те, що треба. YuNet дає саме обличчя.

    **габарит маски шкіри** — те, що тут було, і воно бреше в обидва
    боки. На груповому кадрі маска накриває кілька людей, і габарит
    міряє відстань МІЖ обличчями: на IMG_0796 вийшло 1564 px замість
    642, тобто радіус 7.82 замість 3.21. На одиночному портреті —
    навпаки: маска вирізає очі, брови, рот і волосся, тож вужча за
    голову (388 замість 515), і радіус упирався в підлогу.

    **частка кадру** — здогадка, коли немає нічого. Лишається тільки
    щоб конвеєр не падав без маски й без детектора.
    """
    if face_w and face_w > 0:
        return float(face_w), "detector"
    if skin_mask is not None and skin_mask.any():
        cols = np.nonzero(skin_mask.any(axis=0))[0]
        return float(cols.max() - cols.min() + 1), "mask-bbox"
    return 0.55 * min(shape[0], shape[1]), "guess"


def radius_for(shape: tuple[int, ...], skin_mask: np.ndarray | None = None,
               base_face: float = 1200.0, base_radius: float = 6.0,
               lo: float = 2.0, hi: float = 32.0,
               face_w: float | None = None) -> float:
    """Радіус частотки з РОЗМІРУ ОБЛИЧЧЯ, а не з мегапікселів кадру.

    Це принципово. Прив'язка до роздільності файлу ламається на кропах:
    поясний портрет і кроп голови з того самого файлу мають однакові
    мегапікселі, але зовсім різний масштаб пор.

    `face_w` — ширина з детектора; без неї береться габарит маски, який
    на реальних кадрах помиляється до 2.4 раза (див. face_width).

    Калібрування: обличчя шириною 1200 px -> радіус 6 px.
    """
    fw, _src = face_width(shape, skin_mask, face_w)
    return float(np.clip(base_radius * fw / base_face, lo, hi))
