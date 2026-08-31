"""Проявлення: те, що йде ДО ретуші.

Це етапи 1-15 конвеєра з spec.md §16, і майже все з них уже вміє libraw.
Нашого тут рівно три речі, яких у нього немає: кроп, тон-крива і
контраст. Решта — акуратно прокинуті параметри rawpy.

Свого тон-мапу не пишемо. Це не скромність, а §4: власний демозаїк і
власна колірна наука — місяці роботи заради гіршого результату. Тон-крива
тут не «як в ACR», а рівно те, що задано точками: передбачувано і без
претензій на схожість.

**Два набори параметрів, і плутати їх не можна.** Одні застосовуються
під час декодування RAW і для TIFF не мають сенсу взагалі (експозиція до
демозаїка, баланс білого, відновлення світлів). Другі працюють з будь-яким
входом (кроп, крива, контраст). Якщо подати TIFF і задати `exposure`,
конвеєр скаже, що проігнорував його, а не зробить вигляд.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DevelopParams:
    # --- лише для RAW: застосовується під час декодування ----------------
    exposure: float = 0.0
    """Зсув експозиції в стопах. Для RAW іде в libraw ДО демозаїка, де
    світла ще не зрізані. Для TIFF ігнорується — там вони вже зрізані,
    і множення нічого не поверне."""

    white_balance: str = "camera"
    """camera | auto | custom. camera — як поставив фотограф; custom бере
    wb_multipliers. Лише для RAW."""

    wb_multipliers: tuple[float, ...] = ()
    """Множники R,G,B,G2 при white_balance=custom. Порожньо = не чіпати."""

    highlight_mode: int = 0
    """Відновлення пересвітів у libraw: 0 обрізати, 1 не чіпати,
    2 змішати, 3+ реконструкція. Лише для RAW."""

    demosaic: str = ""
    """Назва алгоритму libraw (AHD, DCB, AMAZE...). Порожньо = типовий.
    Лише для RAW."""

    noise_thr: float = 0.0
    """Поріг шумозаглушення wavelet у libraw. 0 = вимкнено. Лише для RAW."""

    saturation: float = 1.0
    """Насиченість при декодуванні. 1.0 = не чіпати. Лише для RAW."""

    # --- будь-який вхід ---------------------------------------------------
    crop: tuple[float, ...] = ()
    """Кроп у ЧАСТКАХ кадру (x0, y0, x1, y1), 0..1. У частках, а не в
    пікселях, щоб пресет переносився між кадрами різного розміру."""

    rotate: float = 0.0
    """Поворот у градусах. Кадр не розширюємо: кути обрізаються."""

    contrast: float = 0.0
    """Контраст -1..+1 як S-подібна крива навколо середини. Зручність:
    те саме можна задати точками в curve, але одним числом коротше."""

    curve: tuple = ()
    """Тон-крива точками ((x, y), ...) у [0..1], монотонна. Порожньо =
    не чіпати. Застосовується ПІСЛЯ contrast."""

    def raw_only(self) -> dict:
        """Що з цього має сенс лише для RAW — щоб було про що звітувати."""
        d = {}
        if self.exposure:
            d["exposure"] = self.exposure
        if self.white_balance != "camera":
            d["white_balance"] = self.white_balance
        if self.highlight_mode:
            d["highlight_mode"] = self.highlight_mode
        if self.demosaic:
            d["demosaic"] = self.demosaic
        if self.noise_thr:
            d["noise_thr"] = self.noise_thr
        if self.saturation != 1.0:
            d["saturation"] = self.saturation
        return d

    def touches_pixels(self) -> bool:
        return bool(self.crop or self.rotate or self.contrast or self.curve)


# ---------------------------------------------------------------------------
# RAW: у параметри libraw
# ---------------------------------------------------------------------------

def rawpy_kwargs(p: DevelopParams) -> dict:
    """DevelopParams -> аргументи rawpy.postprocess().

    Дефолти тут ті самі, що в rawread._read_rawpy: явні й відтворювані,
    бо від них залежить висока частота, а від неї — поріг детекції (§4).
    """
    import rawpy

    kw = {
        "output_bps": 16,
        "no_auto_bright": True,
        "gamma": (2.222, 4.5),
        "output_color": rawpy.ColorSpace.sRGB,
        "use_camera_wb": p.white_balance == "camera",
        "use_auto_wb": p.white_balance == "auto",
    }
    if p.white_balance == "custom" and p.wb_multipliers:
        m = list(p.wb_multipliers)
        kw["user_wb"] = (m + [m[1]] * 4)[:4]        # libraw хоче рівно чотири
        kw["use_camera_wb"] = kw["use_auto_wb"] = False
    if p.exposure:
        # libraw бере множник, а не стопи
        kw["exp_shift"] = float(2.0 ** p.exposure)
        kw["exp_preserve_highlights"] = 1.0
    if p.highlight_mode:
        kw["highlight_mode"] = int(p.highlight_mode)
    if p.noise_thr:
        kw["noise_thr"] = float(p.noise_thr)
    if p.saturation != 1.0:
        kw["user_sat"] = None                       # libraw: не наш шлях
    if p.demosaic:
        alg = getattr(rawpy.DemosaicAlgorithm, p.demosaic.upper(), None)
        if alg is not None:
            kw["demosaic_algorithm"] = alg
    return kw


# ---------------------------------------------------------------------------
# будь-який вхід: кроп, поворот, крива
# ---------------------------------------------------------------------------

def _curve_lut(p: DevelopParams) -> np.ndarray | None:
    """LUT на 1024 точки з contrast і curve. None, якщо нічого не задано."""
    if not p.contrast and not p.curve:
        return None
    x = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
    y = x.copy()
    if p.contrast:
        # Контраст як опукла комбінація тотожності й фіксованої кривої.
        # Так монотонність гарантована ПОБУДОВОЮ: обидві функції зростають,
        # ваги в [0..1], отже й результат зростає. Перша спроба була
        # аналітичною S-подібною, і при contrast=-0.8 вона давала спадну
        # ділянку — тобто інверсію тонів посеред кадру.
        k = float(np.clip(p.contrast, -1.0, 1.0))
        if k > 0:
            s_curve = y * y * (3.0 - 2.0 * y)                  # smoothstep
            y = (1.0 - k) * y + k * s_curve
        else:
            # обернена до smoothstep, теж зростає на [0..1]
            inv = 0.5 - np.sin(np.arcsin(np.clip(1.0 - 2.0 * y, -1, 1)) / 3.0)
            y = (1.0 + k) * y + (-k) * inv
        y = np.clip(y, 0.0, 1.0)
    if p.curve:
        pts = sorted((float(a), float(b)) for a, b in p.curve)
        xs = np.array([0.0] + [a for a, _ in pts] + [1.0], np.float32)
        ys = np.array([0.0] + [b for _, b in pts] + [1.0], np.float32)
        keep = np.concatenate([[True], np.diff(xs) > 1e-6])
        y = np.interp(y, xs[keep], ys[keep]).astype(np.float32)
    return np.clip(y, 0.0, 1.0)


def apply_pixels(img: np.ndarray, p: DevelopParams) -> np.ndarray:
    """Кроп, поворот і крива. Порядок саме такий.

    Кроп перший: далі все рахується по кадру, який людина бачить, — і
    ширина обличчя для radius_for (§6.3) теж.
    """
    out = img
    if p.crop and len(p.crop) == 4:
        h, w = out.shape[:2]
        x0, y0, x1, y1 = p.crop
        cx0, cy0 = int(np.clip(x0, 0, 1) * w), int(np.clip(y0, 0, 1) * h)
        cx1, cy1 = int(np.clip(x1, 0, 1) * w), int(np.clip(y1, 0, 1) * h)
        if cx1 - cx0 >= 16 and cy1 - cy0 >= 16:
            out = np.ascontiguousarray(out[cy0:cy1, cx0:cx1])
    if p.rotate:
        h, w = out.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), float(p.rotate), 1.0)
        out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    lut = _curve_lut(p)
    if lut is not None:
        idx = np.clip(out * (len(lut) - 1), 0, len(lut) - 1)
        out = lut[idx.astype(np.int32)]
    return out
