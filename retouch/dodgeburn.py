"""Dodge & Burn: вирівнювання низької частоти.

Лікування (blemish.py) працює з ВИСОКОЮ частотою — підміняє текстуру.
D&B працює з НИЗЬКОЮ — вирівнює тон і об'єм. Це дві половини однієї
роботи, і саме тому вони не конфліктують: одна не чіпає того, з чим
працює друга (spec.md §1).

Ідея класична, без нейромереж: беремо яскравість низької частоти,
згладжуємо її зі збереженням країв, і різницю між згладженим і
фактичним віддаємо як корекцію. Де шкіра темніша, ніж «мала б бути» за
загальною формою, — освітлюємо; де світліша — затемнюємо. Плями від
нерівного тону зникають, світлотінь обличчя лишається.

**Носій — шар 50% сірого в режимі Soft Light.** Це не примха, а точна
властивість: soft_light(base, 0.5) == base, тобто рівно сірий шар
нічого не робить. Отже шар завжди можна послабити, вимкнути або
домалювати по ньому маскою, і поза корекцією кадр лишається дослівно
тим самим. Для геометрії такого не буває (§14), а тут буває.

Згладжування — guided filter, а не гаусів. Гаусів розмиває через край
обличчя, і біля контуру D&B починає освітлювати фон. Guided filter
тримає край, бо веде згладжування за самим зображенням. У headless-збірці
opencv його немає (він у contrib), тож реалізований тут через boxFilter —
це п'ятнадцять рядків і рівно та сама формула.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .freqsep import luma


@dataclass
class DodgeBurnParams:
    radius: int = 96
    """Радіус згладжування в px. Задає МАСШТАБ нерівностей, які
    вирівнюються: менший чіпає дрібні плями тону, більший — загальну
    нерівність щоки. Не плутати з радіусом частотки (§6.3)."""

    eps: float = 0.01
    """Поріг «краю» для guided filter. Менший — сильніше тримає межі й
    менше згладжує; більший поводиться як звичайне розмиття."""

    strength: float = 0.5
    """Сила 0..1. Множник різниці, тобто прямий аналог непрозорості
    шару. 1.0 вирівнює тон повністю і виглядає пластиково."""

    limit: float = 0.12
    """Стеля корекції в одиницях яскравості [0..1]. Захист від того, щоб
    D&B не намагався витягнути тінь від носа чи волосся, прийнявши їх за
    нерівність тону."""

    feather: float = 24.0
    """Розмиття краю маски шкіри в px. Різкий край дав би видиму межу
    там, де корекція обривається."""


def guided_filter(guide: np.ndarray, src: np.ndarray,
                  radius: int, eps: float) -> np.ndarray:
    """Guided filter (He et al.), самонавідний варіант тут найчастіший.

    Реалізовано через boxFilter, бо cv2.ximgproc у headless-збірці немає.
    Формула стандартна; вся суть у тому, що коефіцієнт `a` малий там, де
    дисперсія мала (рівна ділянка — сильно згладжуємо) і близький до
    одиниці на краю (край лишаємо).
    """
    r = (max(1, int(radius)) | 1, max(1, int(radius)) | 1)
    box = lambda x: cv2.boxFilter(x, -1, r, normalize=True,   # noqa: E731
                                  borderType=cv2.BORDER_REPLICATE)
    mean_i = box(guide)
    mean_p = box(src)
    var_i = box(guide * guide) - mean_i * mean_i
    cov_ip = box(guide * src) - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def soft_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Soft Light за формулою Photoshop.

    Ключова властивість: blend == 0.5 дає рівно base. На ній тримається
    весь сенс шару 50% сірого, і вона перевіряється тестом.
    """
    b = np.clip(base, 0.0, 1.0)
    s = np.clip(blend, 0.0, 1.0)
    dark = 2.0 * b * s + b * b * (1.0 - 2.0 * s)
    light = 2.0 * b * (1.0 - s) + np.sqrt(b) * (2.0 * s - 1.0)
    return np.where(s <= 0.5, dark, light).astype(np.float32)


def gray_map(img: np.ndarray, skin_mask: np.ndarray | None = None,
             p: DodgeBurnParams | None = None) -> np.ndarray:
    """Карта 50% сірого з корекцією. 0.5 = не чіпати.

    Працює по ЯСКРАВОСТІ низької частоти: колір і текстура не наші.
    """
    p = p or DodgeBurnParams()
    lo = cv2.GaussianBlur(img, (0, 0), max(2.0, p.radius / 8.0),
                          borderType=cv2.BORDER_REPLICATE)
    y = luma(lo)
    smooth = guided_filter(y, y, p.radius, p.eps)
    delta = np.clip(smooth - y, -p.limit, p.limit) * float(p.strength)

    if skin_mask is not None:
        m = (skin_mask > 0).astype(np.uint8)
        if p.feather > 0:
            # Спершу стискаємо, потім розмиваємо. Просте розмиття
            # розповзається В ОБИДВА боки, і корекція виходить за маску:
            # заміряно 112 тисяч пікселів на фоні. З попередньою ерозією
            # розмиття дотягується щонайбільше назад до початкового краю.
            # Ерозія на feather, розмиття з сигмою feather/3: гаус
            # дотягується приблизно на 3 сигми, тобто рівно назад до
            # початкового краю. Множення на вихідну маску наприкінці —
            # гарантія, а не оздоба: перша спроба (ерозія на feather,
            # сигма feather) лишала 55 тисяч пікселів на фоні.
            k = max(1, int(round(p.feather)))
            inner = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=k)
            mf = cv2.GaussianBlur(inner.astype(np.float32), (0, 0),
                                  max(1.0, p.feather / 3.0),
                                  borderType=cv2.BORDER_REPLICATE)
            mf *= m.astype(np.float32)
        else:
            mf = m.astype(np.float32)
        delta = delta * mf

    # У Soft Light відхилення на d від 0.5 змінює яскравість приблизно
    # на d (для середніх тонів). Точна інверсія тут не потрібна: сила
    # все одно регулюється, а надточність дала б хибне відчуття контролю.
    return np.clip(0.5 + delta, 0.0, 1.0).astype(np.float32)


def apply(img: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Накласти карту як Soft Light. Три канали з однієї сірої карти."""
    g = gray if gray.ndim == 3 else np.dstack([gray] * 3)
    return soft_light(img, g)


def coverage(gray: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Де корекція справді є — щоб зібрати шар і показати покриття."""
    g = gray if gray.ndim == 2 else luma(gray)
    return (np.abs(g - 0.5) > eps).astype(np.float32)
