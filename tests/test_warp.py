"""Тести пластики. Головні інваріанти:

  1. нульове поле — точна тотожність, кадр не чіпається взагалі;
  2. поза деформацією кадр біт-у-біт той самий;
  3. сила множить поле лінійно (це аналог непрозорості шару);
  4. поле переживає запис і читання;
  5. заморозка справді захищає область;
  6. знак не переплутано: тягну вправо — їде вправо.

Пункт 2 слабший, ніж інваріант 4 для лікування, і це не недогляд:
деформація — перший етап, який ПЕРЕСЕМПЛЮЄ кадр, тож зберегти піксель
можна лише там, куди вона не дотяглася.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retouch.warp import Field, WarpParams  # noqa: E402

H, W = 800, 1000


def _scene(seed: int = 3):
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 0.5, np.float32)
    img += cv2.GaussianBlur(rng.normal(0, 1, (H, W, 3)).astype(np.float32),
                            (0, 0), 1.0) * 0.03
    cv2.circle(img, (500, 400), 18, (0.0, 0.0, 1.0), -1)
    return np.clip(img, 0, 1)


def _marker(a):
    m = (a[:, :, 2] > 0.8) & (a[:, :, 0] < 0.3)
    ys, xs = np.nonzero(m)
    return (float(xs.mean()), float(ys.mean()), int(m.sum())) if len(xs) else (0, 0, 0)


def test_identity_is_exact_noop():
    img = _scene()
    f = Field((H, W))
    out = f.apply(img)
    print(f"  поле не торкане: touched={f.touched}")
    assert np.array_equal(out, img), "нульове поле змінило кадр"


def test_far_from_warp_bit_identical():
    img = _scene()
    f = Field((H, W))
    f.push(500, 400, 150, 60, 0)
    out = f.apply(img)
    far = out[:200, :200]
    print(f"  кут: біт-у-біт {np.array_equal(far, img[:200, :200])}")
    assert np.array_equal(far, img[:200, :200]), "деформація дотяглася туди, де її нема"


def test_push_direction():
    """Знак — найлегше місце помилитися: у полі лежить, ЗВІДКИ брати."""
    img = _scene()
    f = Field((H, W))
    f.push(500, 400, 160, 60, 0)
    x0, _y0, _a = _marker(img)
    x1, _y1, _b = _marker(f.apply(img))
    print(f"  мітка {x0:.0f} -> {x1:.0f} (зсув {x1 - x0:+.0f} px)")
    assert x1 - x0 > 20, "тягнули вправо, а поїхало не туди"


def test_strength_is_linear():
    img = _scene()
    f = Field((H, W))
    f.push(500, 400, 160, 60, 0)
    x0, _, _ = _marker(img)
    shifts = []
    for k in (0.0, 0.5, 1.0):
        x, _, _ = _marker(f.apply(img, WarpParams(strength=k)))
        shifts.append(x - x0)
    print(f"  зсуви при 0 / 0.5 / 1: {shifts[0]:.1f} / {shifts[1]:.1f} / {shifts[2]:.1f}")
    assert abs(shifts[0]) < 1e-6, "нульова сила щось зсунула"
    assert abs(shifts[1] - shifts[2] / 2) < 0.15 * abs(shifts[2]), "сила не лінійна"


def test_bloat_and_pucker():
    img = _scene()
    _, _, area0 = _marker(img)
    big = Field((H, W)); big.bloat(500, 400, 200, -0.6)
    small = Field((H, W)); small.bloat(500, 400, 200, 0.6)
    _, _, a_big = _marker(big.apply(img))
    _, _, a_small = _marker(small.apply(img))
    print(f"  площа мітки: стягнуто {a_small}, було {area0}, роздуто {a_big}")
    assert a_small < area0 < a_big, "роздування і стягування переплутані"


def test_freeze_protects():
    img = _scene()
    f = Field((H, W))
    f.push(500, 400, 300, 60, 0)
    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, (500, 400), 120, 1, -1)
    f.freeze(mask)
    out = f.apply(img)
    x0, _, _ = _marker(img)
    x1, _, _ = _marker(out)
    print(f"  під заморозкою мітка зсунулась на {x1 - x0:+.1f} px")
    assert abs(x1 - x0) < 4, "заморозка не втримала область"


def test_field_survives_save_load():
    f = Field((2400, 1800))
    f.push(900, 1200, 300, 40, -25)
    f.twirl(1200, 1600, 250, 0.3)
    with tempfile.TemporaryDirectory() as d:
        p = f.save(Path(d) / "field.png")
        g = Field.load(p)
    rng = float(np.abs(f.d).max())
    err = float(np.abs(g.d - f.d).max())
    print(f"  {p.name}")
    print(f"  похибка {err:.6f} при розмаху {rng:.4f} = {err / rng * 65535:.1f} кванта")
    assert g.scale == f.scale and g.full == f.full, "метадані поля загублено"
    assert err / max(rng, 1e-9) < 1e-4, "поле не пережило запис"


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
