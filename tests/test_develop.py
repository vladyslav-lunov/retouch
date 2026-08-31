"""Тести проявлення. Головні інваріанти:

  1. нульові параметри — точна тотожність;
  2. крива МОНОТОННА за будь-якого контрасту (інверсія тонів = зіпсований кадр);
  3. кроп задається в частках і переноситься між кадрами різного розміру;
  4. параметри, що мають сенс лише для RAW, на TIFF чесно повідомляються.

Пункт 2 — не формальність. Перша реалізація контрасту була аналітичною
S-подібною, і при contrast=-0.8 у неї з'являлась спадна ділянка:
серединні тони інвертувалися. На картинці це виглядає як брак, а не як
менший контраст.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retouch.develop import (DevelopParams, _curve_lut,  # noqa: E402
                             apply_pixels, rawpy_kwargs)


def _ramp(h=64, w=256):
    return np.tile(np.linspace(0, 1, w, dtype=np.float32)[None, :, None], (h, 1, 3))


def test_identity_is_exact():
    img = _ramp()
    out = apply_pixels(img, DevelopParams())
    print(f"  нульові параметри: біт-у-біт {np.array_equal(out, img)}")
    assert np.array_equal(out, img), "порожнє проявлення змінило кадр"
    assert _curve_lut(DevelopParams()) is None


def test_contrast_curve_is_monotonic():
    """Найважливіше: жодного спадного відрізка, інакше тони інвертуються."""
    bad = []
    for c in np.linspace(-1.0, 1.0, 41):
        if abs(c) < 1e-9:
            continue
        lut = _curve_lut(DevelopParams(contrast=float(c)))
        d = float(np.diff(lut).min())
        if d < -1e-6:
            bad.append((round(float(c), 2), round(d, 6)))
    print(f"  перевірено 40 значень, немонотонних: {len(bad)}")
    assert not bad, f"крива інвертує тони при contrast={bad}"


def test_contrast_direction():
    plus = _curve_lut(DevelopParams(contrast=0.6))
    minus = _curve_lut(DevelopParams(contrast=-0.6))
    print(f"  +0.6: 0.25 -> {plus[256]:.3f}   -0.6: 0.25 -> {minus[256]:.3f}")
    assert plus[256] < 0.25 < minus[256], "знак контрасту переплутано"
    assert plus[768] > 0.75 > minus[768]


def test_point_curve_hits_its_points():
    lut = _curve_lut(DevelopParams(curve=((0.25, 0.15), (0.75, 0.85))))
    print(f"  0.25 -> {lut[256]:.3f}, 0.75 -> {lut[768]:.3f}")
    assert abs(lut[256] - 0.15) < 0.01 and abs(lut[768] - 0.85) < 0.01
    assert np.all(np.diff(lut) >= -1e-6), "точкова крива вийшла немонотонною"


def test_crop_is_fractional():
    """Кроп у частках, щоб пресет переносився між кадрами різного розміру."""
    p = DevelopParams(crop=(0.25, 0.25, 0.75, 0.75))
    for h, w in ((1000, 2000), (500, 1000), (333, 777)):
        out = apply_pixels(np.zeros((h, w, 3), np.float32), p)
        assert abs(out.shape[1] - w // 2) <= 1 and abs(out.shape[0] - h // 2) <= 1, \
            f"{w}x{h} -> {out.shape[1]}x{out.shape[0]}"
    print("  кроп 0.25..0.75 дає половину на всіх трьох розмірах")


def test_raw_only_params_are_reported():
    """Те, що працює лише при декодуванні RAW, має бути перелічене."""
    p = DevelopParams(exposure=1.0, white_balance="auto", contrast=0.3,
                      crop=(0.1, 0.1, 0.9, 0.9))
    only = p.raw_only()
    print(f"  лише для RAW: {sorted(only)}")
    assert set(only) == {"exposure", "white_balance"}
    assert "contrast" not in only and "crop" not in only
    assert p.touches_pixels(), "кроп і контраст працюють з будь-яким входом"


def test_exposure_becomes_a_multiplier():
    """libraw бере множник, а не стопи — переплутати легко."""
    kw = rawpy_kwargs(DevelopParams(exposure=1.0))
    print(f"  +1 стоп -> exp_shift={kw['exp_shift']}")
    assert abs(kw["exp_shift"] - 2.0) < 1e-6
    kw2 = rawpy_kwargs(DevelopParams(exposure=-1.0))
    assert abs(kw2["exp_shift"] - 0.5) < 1e-6


def test_defaults_stay_reproducible():
    """Дефолти декодування явні: від них залежить HF, а від неї поріг (§4)."""
    kw = rawpy_kwargs(DevelopParams())
    print(f"  no_auto_bright={kw['no_auto_bright']}, gamma={kw['gamma']}, "
          f"camera_wb={kw['use_camera_wb']}")
    assert kw["no_auto_bright"] is True, "автояскравість зробить кадри серії різними"
    assert kw["output_bps"] == 16
    assert kw["use_camera_wb"] is True


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
