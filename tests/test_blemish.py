"""Тести ядра. Головні інваріанти:

  1. детектор знаходить більшість відомих дефектів
  2. лікування прибирає контраст у місці дефекту
  3. низька частота НЕ змінюється (тон і об'єм цілі)
  4. далеко від дефектів кадр не зачеплено взагалі
  5. шар корекції реконструює результат піксель у піксель
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retouch.blemish import DetectParams, detect_blemishes, heal_blemishes  # noqa: E402
from retouch.freqsep import freq_merge, freq_split, luma  # noqa: E402
from retouch.layers import extract_layer  # noqa: E402
from tests.synth import make_skin  # noqa: E402

RADIUS = 6.0
PARAMS = DetectParams(threshold=0.010, min_area=6, max_area=1500)


def _pipeline(seed=7, n=12):
    img, spots = make_skin(n_spots=n, seed=seed)
    low, high = freq_split(img, RADIUS)
    lbl, blobs = detect_blemishes(high, None, PARAMS)
    high2, cov = heal_blemishes(high, lbl, blobs, None, search_radius=80)
    return img, spots, low, high, high2, lbl, blobs, cov, freq_merge(low, high2)


def test_detects_most_spots():
    _, spots, _, _, _, lbl, blobs, _, _ = _pipeline()
    found = sum(1 for (x, y, _) in spots if lbl[y, x] > 0)
    print(f"  знайдено {found}/{len(spots)} плям, компонент: {len(blobs)}")
    assert found >= 0.8 * len(spots), f"детектор пропустив забагато: {found}/{len(spots)}"


def test_healing_reduces_contrast():
    _, spots, _, high, high2, _, _, _, _ = _pipeline()
    before = np.array([abs(luma(high)[y, x]) for x, y, _ in spots])
    after = np.array([abs(luma(high2)[y, x]) for x, y, _ in spots])
    print(f"  |HF| у центрах плям: {before.mean():.4f} -> {after.mean():.4f}")
    assert after.mean() < 0.35 * before.mean(), "лікування майже не спрацювало"


def test_low_frequency_untouched():
    img, _, low, _, _, _, _, _, result = _pipeline()
    low2, _ = freq_split(result, RADIUS)
    d = np.abs(low2 - low).max()
    print(f"  максимальна зміна низької частоти: {d:.5f}")
    assert d < 0.02, "низька частота поїхала — тон і об'єм не збережені"


def test_untouched_areas_are_identical():
    img, _, _, _, _, _, _, cov, result = _pipeline()
    clean = cov < 1e-6
    d = np.abs(result - img).max(axis=2)[clean].max()
    print(f"  максимальна зміна поза дотиками: {d:.6f}")
    assert d < 1e-6, "конвеєр змінив пікселі, яких не мав чіпати"


def test_layer_reconstructs_result():
    img, _, _, _, _, _, _, cov, result = _pipeline()
    rgb, a = extract_layer(img, result, cov)
    recon = img * (1 - a[..., None]) + rgb * a[..., None]
    d = np.abs(recon - result).max()
    print(f"  похибка реконструкції шару: {d:.6f}")
    assert d < 2e-3, "шар не збігається з результатом — у Photoshop буде інше"


def test_stability_across_seeds():
    for seed in (1, 3, 11, 42):
        _, spots, _, _, _, lbl, _, _, _ = _pipeline(seed=seed, n=10)
        found = sum(1 for (x, y, _) in spots if lbl[y, x] > 0)
        print(f"  seed={seed}: {found}/{len(spots)}")
        assert found >= 0.7 * len(spots)


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
