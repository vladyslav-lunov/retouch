"""Тести ядра. Головні інваріанти:

  1. детектор знаходить більшість відомих дефектів
  2. лікування прибирає контраст у місці дефекту
  3. низька частота НЕ змінюється (тон і об'єм цілі)
  4. далеко від дефектів кадр не зачеплено взагалі
  5. шар корекції реконструює результат піксель у піксель
  6. лікування не тримає буферів розміру кадру (бюджет 8 ГБ, spec.md §2)
  7. лікування не виходить за маску шкіри — ні на піксель
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


def test_healing_stays_inside_skin_mask():
    """Лікування не виходить за маску шкіри — жодного пікселя.

    Детекція обмежена маскою (spec.md §6.1), але альфу дотику потім ще
    розширюють dilate(margin) і blur(feather). На повіці цього вистачало,
    щоб лікування заповзло на око: заміряно 11 px за маску і 1173 px
    усередині ока при нульовій детекції там. Ерозія маски з §5 від цього
    не рятувала — вона захищає інший етап конвеєра.

    Перевіряємо на портреті-макеті, а не на плоскому клапті: на клапті
    маска покриває весь кадр, і виходити просто нема куди.
    """
    from retouch.masks import build_skin_mask
    from retouch.freqsep import radius_for
    from tests.synth import make_face

    # Обличчя 900 px — близько до реального кропа голови. На дрібнішому
    # макеті детектор не знаходить нічого (див. нижче про radius_for), і
    # тест ставав би зеленим просто тому, що лікувати нема чого.
    img, _, truth = make_face(h=1500, w=1150, face_w=900, n_spots=20, seed=3)
    skin, _ = build_skin_mask(img)
    _, high = freq_split(img, radius_for(img.shape, skin))
    lbl, blobs = detect_blemishes(high, skin, DetectParams())
    _, cov = heal_blemishes(high, lbl, blobs, skin)

    outside = int((cov[skin == 0] > 0).sum())
    touched = int((cov > 0).sum())
    hit = {k: int(((cov > 0) & truth[k]).sum())
           for k in ("l_eye", "r_eye", "l_brow", "r_brow", "u_lip", "l_lip", "hair")}
    print(f"  плям {len(blobs)}, дотиків {touched} px, з них поза маскою: {outside}")
    print(f"  у зонах виключення: {', '.join(f'{k}={v}' for k, v in hit.items())}")
    # без цієї перевірки тест зеленів би на порожньому результаті
    assert len(blobs) >= 10 and touched > 1000, (
        f"нема чого перевіряти: {len(blobs)} плям, {touched} px дотиків")
    assert outside == 0, f"лікування вийшло за маску шкіри на {outside} px"
    assert not any(hit.values()), f"лікування залізло в зони виключення: {hit}"


def test_healing_memory_stays_local():
    """Лікування не тримає карт РОЗМІРУ КАДРУ.

    Пошук донора рахує вартість вікнами. Спокуса — порахувати карту на
    весь кадр і закешувати її за розміром вікна; на портреті розмірів
    вікна десятки, і кеш перетворюється на десятки × розмір кадру:
    0.6 ГБ на 3 Мп, 5.4 ГБ на 24 Мп. У 8 ГБ зі spec.md §2 це не лізе.

    Міряємо в ОКРЕМОМУ процесі: ru_maxrss — високий водяний знак, і в
    спільному процесі його вже підняли б попередні тести.
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]
    child = r"""
import resource, sys
sys.path.insert(0, %r)
from retouch.blemish import DetectParams, detect_blemishes, heal_blemishes
from retouch.freqsep import freq_split
from tests.synth import make_skin_mp

unit = 1 if sys.platform == "darwin" else 1024
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit / 2**20

img, _ = make_skin_mp(3.0)
_, high = freq_split(img, 6.0)
del img
lbl, blobs = detect_blemishes(high, None, DetectParams())
sizes = {((b["bbox"][2] + 6) | 1, (b["bbox"][3] + 6) | 1) for b in blobs}
before = peak()
heal_blemishes(high, lbl, blobs, None, search_radius=90)
print(f"{len(blobs)} {len(sizes)} {high.size // 3} {before:.0f} {peak():.0f}")
""" % str(root)

    out = subprocess.run([sys.executable, "-c", child], capture_output=True,
                         text=True, cwd=str(root))
    assert out.returncode == 0, f"дочірній процес впав:\n{out.stderr[-1500:]}"
    n, n_sizes, px, before, after = out.stdout.split()
    grew = float(after) - float(before)
    naive = int(n_sizes) * int(px) * 4 / 2**20
    print(f"  {n} плям, {n_sizes} розмірів вікна на {int(px) / 1e6:.1f} Мп")
    print(f"  лікування додало {grew:.0f} МБ (повнокадровий кеш дав би ~{naive:.0f} МБ)")
    assert grew < 250, f"лікування з'їло {grew:.0f} МБ — карти знову розміру кадру"


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
