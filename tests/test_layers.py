"""Тести шарів корекції. Головні інваріанти:

  1. base*(1-a) + layer*a == result — головна обіцянка §1;
  2. де альфа нульова, шар несе базу, а не сміття;
  3. write_stack кладе базу, шари, маски й зведений результат;
  4. накладання ДВОХ шарів по черзі теж сходиться.

Пункт 4 важливий окремо: шкіра й видалення пишуться двома шарами, і
кожен рахувався від свого попередника. Якщо порядок наплутати, у
Photoshop (чи в нашому UI) вийде не те, що показав конвеєр.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retouch.layers import extract_layer, write_stack  # noqa: E402


def _scene(h=120, w=160, seed=3):
    rng = np.random.default_rng(seed)
    return np.clip(rng.random((h, w, 3)).astype(np.float32) * 0.6 + 0.2, 0, 1)


def test_layer_reconstructs_result():
    base = _scene()
    rng = np.random.default_rng(7)
    cov = np.zeros(base.shape[:2], np.float32)
    cov[40:80, 50:110] = rng.random((40, 60)).astype(np.float32)
    target = np.clip(base + 0.15, 0, 1)
    result = base * (1 - cov[..., None]) + target * cov[..., None]
    rgb, a = extract_layer(base, result, cov)
    recon = base * (1 - a[..., None]) + rgb * a[..., None]
    err = float(np.abs(recon - result).max())
    print(f"  похибка реконструкції: {err:.2e}")
    assert err < 2e-3, "шар не збігається з результатом"


def test_zero_alpha_carries_base():
    base = _scene()
    cov = np.zeros(base.shape[:2], np.float32)
    rgb, a = extract_layer(base, base.copy(), cov)
    print(f"  альфа скрізь {a.max():.1f}; шар == база: {np.allclose(rgb, base)}")
    assert a.max() == 0.0
    assert np.allclose(rgb, base), "де альфа нуль, у шарі має бути база"


def test_alpha_is_clipped_to_unit():
    base = _scene()
    cov = np.full(base.shape[:2], 3.0, np.float32)      # навмисно за межею
    _rgb, a = extract_layer(base, base, cov)
    print(f"  вхідна альфа 3.0 -> {a.max():.1f}")
    assert a.max() <= 1.0


def test_two_layers_compose():
    """Шкіра, потім видалення: кожен від свого попередника."""
    base = _scene()
    cov1 = np.zeros(base.shape[:2], np.float32); cov1[20:60, 20:60] = 0.8
    step1 = base * (1 - cov1[..., None]) + np.clip(base + 0.2, 0, 1) * cov1[..., None]
    cov2 = np.zeros(base.shape[:2], np.float32); cov2[40:90, 40:90] = 1.0
    step2 = step1 * (1 - cov2[..., None]) + np.clip(step1 - 0.3, 0, 1) * cov2[..., None]

    l1 = extract_layer(base, step1, cov1)
    l2 = extract_layer(step1, step2, cov2)
    cur = base
    for rgb, a in (l1, l2):
        cur = cur * (1 - a[..., None]) + rgb * a[..., None]
    err = float(np.abs(cur - step2).max())
    print(f"  два шари поспіль: похибка {err:.2e}")
    assert err < 2e-3, "послідовне накладання двох шарів розійшлося"


def test_write_stack_produces_expected_files():
    base = _scene()
    result = np.clip(base + 0.05, 0, 1)
    cov = np.zeros(base.shape[:2], np.float32); cov[30:70, 30:70] = 1.0
    layers = {"skin": extract_layer(base, result, cov)}
    masks = {"skin": (cov > 0).astype(np.uint8)}
    with tempfile.TemporaryDirectory() as t:
        files = write_stack(Path(t), "T", base, layers, result,
                            np.dtype("uint16"), masks)
        names = sorted(p.name for p in files)
        print(f"  {', '.join(names)}")
        assert any(n.endswith("_00_base.tif") for n in names)
        assert any(n.endswith("_01_skin.png") for n in names)
        assert any("_mask_skin" in n for n in names)
        assert any(n.endswith("_99_flat.tif") for n in names)
        lay = cv2.imread(str(Path(t) / "T_01_skin.png"), cv2.IMREAD_UNCHANGED)
        assert lay.shape[2] == 4, "шар записано без альфи"
        assert lay.dtype == np.uint16, "шар записано не в 16 біт"


def test_written_layer_reconstructs_on_disk():
    """Те, що бачить користувач: файли, а не масиви в пам'яті."""
    base = _scene(200, 260)
    cov = np.zeros(base.shape[:2], np.float32); cov[50:150, 60:200] = 0.7
    # Результат складається САМЕ через coverage. Поза нею він мусить
    # дорівнювати базі — це інваріант 4, і шар інакше просто не може
    # виразити зміну: там альфа нульова.
    target = np.clip(base * 0.9 + 0.05, 0, 1)
    result = base * (1 - cov[..., None]) + target * cov[..., None]
    with tempfile.TemporaryDirectory() as t:
        write_stack(Path(t), "T", base, {"skin": extract_layer(base, result, cov)},
                    result, np.dtype("uint16"), {})
        rd = lambda n: cv2.imread(str(Path(t) / n), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535
        b, f = rd("T_00_base.tif"), rd("T_99_flat.tif")
        lay = rd("T_01_skin.png")
        recon = b * (1 - lay[:, :, 3][..., None]) + lay[:, :, :3] * lay[:, :, 3][..., None]
        err = float(np.abs(recon - f).max()) * 65535
        print(f"  на файлах: {err:.1f} кванта 16 біт")
        assert err < 8, f"на диску шар не сходиться: {err:.1f} кванта"


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
