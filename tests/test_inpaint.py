"""Тести видалення об'єктів. Головні інваріанти:

  1. текстура ВИДАЛЕНОГО об'єкта не повертається в дірку;
  2. сусідні області не скасовують правки одна одної;
  3. поза маскою кадр не змінюється;
  4. глибина дірки міряється чесно: волосина — не те саме, що пляма.

Ваг LaMa у репозиторії немає і не буде (ліцензія, spec.md §10), тому
модель тут — заглушка з передбачуваною поведінкою. Це навіть краще:
заглушка кладе в дірку РІВНИЙ колір, і будь-яка текстура в результаті
може взятися лише з коду, а не з моделі.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retouch.inpaint import (InpaintParams, inpaint_region,  # noqa: E402
                             mask_stats, telea_warning)

FILL = (0.0, 0.8, 0.8)


class StubModel:
    """Заливає маску рівним кольором. Нічого не вигадує."""

    def _run(self, rgb, mask, pad_to):
        out = rgb.copy()
        out[mask > 0.5] = FILL
        return out


def _scene(h=2400, w=1600, seed=3):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 0.55, np.float32)
    img += cv2.GaussianBlur(rng.normal(0, 1, (h, w, 3)).astype(np.float32),
                            (0, 0), 1.2) * 0.02
    return np.clip(img, 0, 1)


def test_removed_object_leaves_no_ghost():
    """Велика дірка не повертає текстуру того, що ми прибрали.

    §7 бере низьку частоту з моделі, а високу з оригіналу — і це вірно
    рівно доти, доки оригінал є ФОНОМ. Усередині великої маски оригінал
    це те, що видаляють, і його текстура поверталася привидом: заміряно
    розкид 0.116 у дірці замість 0.010.
    """
    h, w = 2400, 1600
    img = _scene(h, w)
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (500, 300), (1100, 2100), 1, -1)
    # «об'єкт» смугастий: якщо його текстура протече — це видно одразу
    stripes = (((np.arange(h) // 12) % 2).astype(np.float32) * 0.35 + 0.3)
    img[mask > 0] = np.broadcast_to(stripes[:, None, None], (h, w, 3))[mask > 0]

    res, _cov = inpaint_region(img, mask.copy(), StubModel(),
                               InpaintParams(max_size=1024))
    inside = mask > 0
    # заглушка поклала рівний колір, тож розкид У МЕЖАХ КАНАЛУ може
    # взятися лише з дописаної після моделі текстури
    spread = max(float(res[:, :, c][inside].std()) for c in range(3))
    print(f"  розкид у дірці: {spread:.4f} (привид дав би ~0.12)")
    assert spread < 0.03, f"текстура видаленого об'єкта повернулась: {spread:.4f}"


def test_regions_do_not_erase_each_other():
    """Друга область не вписує поверх першої прямокутник з оригіналу."""
    h, w = 900, 900
    img = _scene(h, w)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (330, 450), 12, 1, -1)
    cv2.circle(mask, (560, 450), 12, 1, -1)   # ближче, ніж 2*context

    res, cov = inpaint_region(img, mask.copy(), StubModel(),
                              InpaintParams(context=128))
    da = abs(float(res[450, 330, 0] - img[450, 330, 0]))
    db = abs(float(res[450, 560, 0] - img[450, 560, 0]))
    print(f"  зміна в центрі A={da:.4f}, B={db:.4f}")
    assert da > 0.05 and db > 0.05, "одну з правок стерто другим кропом"
    assert cov[450, 330] > 0.9 and cov[450, 560] > 0.9


def test_outside_mask_untouched():
    """Далеко від маски кадр не зачеплено."""
    h, w = 900, 900
    img = _scene(h, w)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (450, 450), 20, 1, -1)
    res, cov = inpaint_region(img, mask.copy(), StubModel(), InpaintParams())
    clean = cov < 1e-6
    d = float(np.abs(res - img).max(axis=2)[clean].max())
    print(f"  максимальна зміна поза дотиками: {d:.6f}")
    assert d < 1e-6


def test_depth_not_bbox():
    """Глибина дірки, а не її габарит.

    Волосина через пів кадру має величезний bbox і ширину сім пікселів —
    Telea з нею впорається. Кругла пляма того самого габариту — ні.
    """
    h, w = 2400, 1600
    hair = np.zeros((h, w), np.uint8)
    cv2.line(hair, (100, 100), (1400, 2200), 1, 7)
    blob = np.zeros((h, w), np.uint8)
    cv2.circle(blob, (800, 1200), 200, 1, -1)

    d_hair = mask_stats(hair)["depth"]
    d_blob = mask_stats(blob)["depth"]
    print(f"  волосина: глибина {d_hair:.0f} px -> "
          f"{'попередження' if telea_warning(hair) else 'ок'}")
    print(f"  пляма r=200: глибина {d_blob:.0f} px -> "
          f"{'попередження' if telea_warning(blob) else 'ок'}")
    assert telea_warning(hair) is None, "на волосину сварились дарма"
    assert telea_warning(blob) is not None, "про велику дірку промовчали"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        print(f"\n{name}")
        try:
            fn()
            print("  OK")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL: {e}")
    print(f"\n{'усе зелене' if not fails else f'провалено: {fails}'}")
    raise SystemExit(1 if fails else 0)
