"""Збірка результату у вигляді шарів, а не готового пікселя.

Це і є компроміс між пакетністю та контролем: скрипт відпрацював сам,
але віддав не "красиве фото", а шар корекції з альфою. У Photoshop
кладеш його поверх оригіналу, крутиш opacity, домальовуєш маску,
вимикаєш там, де автоматика перестаралася.

Математика точна. Під час лікування ми рахували
    result = base*(1-a) + layer*a
тому шар відновлюється як
    layer = (result - base*(1-a)) / a   при a > 0
і накладання шару на оригінал у Photoshop дає піксель у піксель те,
що порахував конвеєр.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import imageio


# Чим заповнювати RGB там, де альфа рівно нуль. На складання це не
# впливає ніяк (множиться на нуль), зате PNG стискає рівну заливку в
# ніщо. Сіре, а не чорне: якщо шар відкрити переглядачем, що ігнорує
# альфу, нейтральне тло читається як «тут нічого немає», а чорне — як
# «тут щось стерли».
EMPTY_RGB = 0.5


def extract_layer(base: np.ndarray, result: np.ndarray,
                  coverage: np.ndarray, eps: float = 1e-3
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Повертає (rgb, alpha) шару корекції.

    Під НУЛЬОВОЮ альфою RGB заповнюється константою, а не базою. Це не
    косметика: лікування торкається 0.07% кадру, а шар писався повним
    кадром 6264x4180, у якому 99.93% пікселів — точна копія фотографії
    під повністю прозорою альфою. Заміряно на реальному кадрі: 145.3 МБ
    проти 0.4 МБ, тобто 388 разів. На зйомці з 44 кадрів це десятки
    гігабайтів різниці.

    Складання не змінюється взагалі: там, де альфа нуль, RGB множиться
    на нуль. Тому інваріант `base*(1-a) + layer*a == result` лишається
    точним, і тест це перевіряє.

    Смуга `0 < a <= eps` — окремий випадок: ділення на крихітну альфу
    роздуває шум, тому там, як і раніше, лишається база. Підставити туди
    константу означало б внести помилку `a*|base - 0.5|`, тобто до 65
    квантів на межі eps.
    """
    a = np.clip(coverage, 0, 1)
    safe = np.maximum(a, eps)[..., None]
    rgb = (result - base * (1 - a[..., None])) / safe
    rgb = np.where(a[..., None] > eps, rgb, base)
    rgb = np.where(a[..., None] > 0, rgb, EMPTY_RGB)
    return np.clip(rgb, 0, 1), a


def write_stack(out_dir: str | Path, stem: str, base: np.ndarray,
                layers: dict[str, tuple[np.ndarray, np.ndarray]],
                result: np.ndarray, dtype: np.dtype,
                masks: dict[str, np.ndarray] | None = None) -> list[Path]:
    """Пише базу, кожен шар з альфою, зведений результат і маски."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    p = out_dir / f"{stem}_00_base.tif"
    imageio.write(p, base, dtype)
    written.append(p)

    for i, (name, (rgb, alpha)) in enumerate(layers.items(), start=1):
        p = out_dir / f"{stem}_{i:02d}_{name}.png"
        imageio.write(p, rgb, np.dtype("uint16"), alpha=alpha)
        written.append(p)

    for name, m in (masks or {}).items():
        p = out_dir / f"{stem}_mask_{name}.png"
        imageio.write(p, np.dstack([m.astype(np.float32)] * 3), np.dtype("uint8"))
        written.append(p)

    p = out_dir / f"{stem}_99_flat.tif"
    imageio.write(p, result, dtype)
    written.append(p)
    return written
