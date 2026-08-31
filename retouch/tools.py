"""Дрібні інструменти: судини в білку, зуби, матування, тон шкіри.

Усі чотири класичні, жодної моделі. Вони стали можливими не тому, що
з'явився алгоритм, а тому, що з'явилась КАРТА КЛАСІВ: face-parsing каже,
де око, де рот, де шкіра, і після цього кожен інструмент — це десять
рядків локальної корекції в межах свого класу.

Спільна форма в усіх: `(img, cls, params) -> (result, coverage)`.
Coverage — та сама альфа, що й у лікування, тож кожен віддається окремим
шаром і вимикається окремо (§1).

Спільне обмеження: корекція НЕ виходить за свій клас. Це не побажання —
на вії й ланцюжок ми вже наступали двічі, і обидва рази через те, що
альфа розповзалась ширше, ніж маска. Тут вона множиться на клас в
останню чергу, і тестом перевіряється, що поза класом рівно нуль.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .freqsep import luma
from .masks import CELEBA_CLASSES

_INV = {v: k for k, v in CELEBA_CLASSES.items()}


def _region(cls: np.ndarray, names) -> np.ndarray:
    idx = [_INV[n] for n in names if n in _INV]
    return np.isin(cls, idx) if idx else np.zeros(cls.shape, bool)


def _soft(mask: np.ndarray, feather: float) -> np.ndarray:
    """М'яка альфа, що НЕ виходить за маску.

    Розмиття розповзається в обидва боки, тому наприкінці множимо на
    вихідну маску: перехід усередині лишається гладким, за межу не
    виходить нічого.
    """
    m = mask.astype(np.float32)
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather, borderType=cv2.BORDER_REPLICATE)
        m *= mask.astype(np.float32)
    return np.clip(m, 0.0, 1.0)


def _blend(img, corrected, alpha):
    a = alpha[..., None]
    return (img * (1 - a) + corrected * a).astype(np.float32), alpha


# ---------------------------------------------------------------------------

@dataclass
class EyeVesselParams:
    strength: float = 0.7
    """Наскільки прибирати червоне 0..1."""

    sclera_percentile: float = 55.0
    """Білок — світліша частина ока. Райдужку й зіницю не чіпаємо: поріг
    береться як процентиль яскравості В МЕЖАХ класу ока, а не глобально,
    інакше на темних очах у білок потрапить半 райдужки."""

    feather: float = 1.5
    """Розмиття краю ділянки в px. Мале: білок невеликий, і сильне
    розмиття витекло б на райдужку."""


def eye_vessels(img: np.ndarray, cls: np.ndarray,
                p: EyeVesselParams | None = None):
    """Прибрати червоні судини в білку.

    Не відбілювання: яскравість не чіпаємо взагалі, знижуємо лише
    надлишок червоного над зеленим і синім. Відбілене око видно одразу,
    і це саме те, за що ретуш лають.
    """
    p = p or EyeVesselParams()
    eye = _region(cls, ("l_eye", "r_eye"))
    if not eye.any():
        return img, np.zeros(img.shape[:2], np.float32)
    y = luma(img)
    thr = float(np.percentile(y[eye], p.sclera_percentile))
    sclera = eye & (y >= thr)

    out = img.copy()
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    neutral = np.minimum(g, np.maximum(b, g))
    excess = np.clip(r - neutral, 0, None)
    out[:, :, 2] = r - excess * float(p.strength)
    return _blend(img, out, _soft(sclera, p.feather))


@dataclass
class TeethParams:
    strength: float = 0.6
    """Загальна сила 0..1. Множник альфи, тобто непрозорість шару."""

    yellow: float = 0.8
    """Скільки жовтизни прибрати. Жовтизна — це нестача синього.

    Мета — ЗМЕНШИТИ жовтизну, а не дійти до нейтралі. Заміряно на
    реальному кадрі: у оригіналі синій нижчий за середнє з решти на
    0.15, з дефолтом виходить 0.06 (зуби лишаються теплими), а при
    strength=0.9 разом з yellow=0.9 — 0.00, тобто рівно сірі. Саме це
    й читається як вставні."""

    brighten: float = 0.05
    """Підняття яскравості. Мале навмисно: білі зуби виглядають вставними."""

    teeth_percentile: float = 60.0
    """Зуби — світліша частина рота; губи й ясна темніші."""

    feather: float = 1.5
    """Розмиття краю ділянки в px. Мале: між зубами й яснами перехід
    різкий, і розмивати його немає чого."""


def teeth(img: np.ndarray, cls: np.ndarray, p: TeethParams | None = None):
    """Прибрати жовтизну зубів. Клас `mouth` у CelebA — це саме проміжок
    між губами, тобто зуби й ясна; губи лежать в u_lip/l_lip."""
    p = p or TeethParams()
    mouth = _region(cls, ("mouth",))
    if not mouth.any():
        return img, np.zeros(img.shape[:2], np.float32)
    y = luma(img)
    thr = float(np.percentile(y[mouth], p.teeth_percentile))
    area = mouth & (y >= thr)

    out = img.copy()
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    lack = np.clip(np.maximum(g, r) - b, 0, None)          # жовтизна
    out[:, :, 0] = b + lack * float(p.yellow)
    if p.brighten:
        out = np.clip(out + p.brighten, 0, 1)
    return _blend(img, np.clip(out, 0, 1), _soft(area, p.feather) * p.strength)


@dataclass
class MattifyParams:
    strength: float = 0.6
    """Сила 0..1. На 1.0 відблиск зникає повністю, а з ним і відчуття
    вологої шкіри — воно потрібне не завжди, але потрібне."""

    percentile: float = 92.0
    """Відблиск — верхні відсотки яскравості В МЕЖАХ шкіри."""

    radius: float = 24.0
    """Масштаб, на якому шукається «нормальна» яскравість довкола."""

    feather: float = 6.0
    """Розмиття краю відблиску в px. Різкий край дав би видиму пляму
    там, де корекція обривається."""


def mattify(img: np.ndarray, cls: np.ndarray, skin_names=("skin", "nose", "neck"),
            p: MattifyParams | None = None):
    """Прибрати жирний блиск: стиснути найсвітліше до навколишнього тону.

    Не затемнення шкіри. Відблиск — це локальний викид яскравості над
    сусідами, тож і прибирається він відносно сусідів, а не абсолютно:
    інакше світла ділянка щоки постраждала б разом із блиском.
    """
    p = p or MattifyParams()
    skin = _region(cls, skin_names)
    if not skin.any():
        return img, np.zeros(img.shape[:2], np.float32)
    y = luma(img)
    around = cv2.GaussianBlur(y, (0, 0), p.radius, borderType=cv2.BORDER_REPLICATE)
    thr = float(np.percentile(y[skin], p.percentile))
    hot = skin & (y > thr)
    excess = np.clip(y - around, 0, None)

    out = img.copy()
    drop = (excess * float(p.strength))[..., None]
    out = np.clip(out - drop, 0, 1)
    return _blend(img, out, _soft(hot, p.feather))


@dataclass
class SkinToneParams:
    strength: float = 0.5
    """Сила 0..1. Високі значення роблять шкіру одноколірною, а з нею
    зникає рум'янець — тобто саме те, що робить обличчя живим."""

    radius: float = 64.0
    """Масштаб, на якому вирівнюється хроматичність."""

    limit: float = 0.06
    """Стеля зсуву каналу, щоб не з'їхав загальний колір обличчя."""

    feather: float = 12.0
    """Розмиття краю маски шкіри в px."""


def skin_tone(img: np.ndarray, cls: np.ndarray,
              skin_names=("skin", "nose", "neck"),
              p: SkinToneParams | None = None):
    """Вирівняти плями кольору на шкірі — почервоніння, жовтизну.

    Працюємо з ХРОМАТИЧНІСТЮ, не з яскравістю: об'єм обличчя живе в
    яскравості, і чіпати його тут не можна — для цього є D&B.
    """
    p = p or SkinToneParams()
    skin = _region(cls, skin_names)
    if not skin.any():
        return img, np.zeros(img.shape[:2], np.float32)
    y = np.maximum(luma(img), 1e-4)
    chroma = img / y[..., None]                     # колір без яскравості
    smooth = cv2.GaussianBlur(chroma, (0, 0), p.radius,
                              borderType=cv2.BORDER_REPLICATE)
    delta = np.clip(smooth - chroma, -p.limit, p.limit) * float(p.strength)
    out = np.clip((chroma + delta) * y[..., None], 0, 1)
    return _blend(img, out, _soft(skin, p.feather))


TOOLS = {
    "eye_vessels": (eye_vessels, EyeVesselParams),
    "teeth": (teeth, TeethParams),
    "mattify": (mattify, MattifyParams),
    "skin_tone": (skin_tone, SkinToneParams),
}
