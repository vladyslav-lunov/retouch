"""Тести частотного розкладання. Головні інваріанти:

  1. low + high == img з точністю float32 — інакше все решта не має сенсу;
  2. high знакова і центрована на нулі, а не зсунута на 0.5;
  3. радіус виводиться з ширини ОБЛИЧЧЯ, а не з мегапікселів: кроп і
     повний кадр з тим самим обличчям мають дати той самий радіус;
  4. radius_for тримається в межах lo..hi.

Пункт 3 — суть §6.3, і саме там відоме розходження: розмах маски це не
ширина обличчя. Тест фіксує ПОВЕДІНКУ, яка є, і окремо міряє похибку,
щоб її було видно, а не щоб вона мовчки жила далі.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retouch.freqsep import freq_merge, freq_split, luma, radius_for  # noqa: E402
from tests.synth import make_face, make_skin  # noqa: E402


def test_split_and_merge_are_inverse():
    img, _ = make_skin(h=256, w=256, seed=3)
    low, high = freq_split(img, 6.0)
    back = freq_merge(low, high)
    err = float(np.abs(back - img).max())
    print(f"  максимальна похибка: {err:.2e} (float32 дає ~1e-7)")
    assert err < 1e-5, "розкладання незворотне"


def test_high_is_signed_and_centred():
    img, _ = make_skin(h=256, w=256, seed=3)
    _low, high = freq_split(img, 6.0)
    m = float(high.mean())
    print(f"  середнє HF: {m:.2e}, мін {high.min():.3f}, макс {high.max():.3f}")
    assert abs(m) < 1e-3, "HF зсунуто — воно має бути знаковим навколо нуля"
    assert high.min() < 0 < high.max(), "у HF немає від'ємних значень"


def test_bigger_radius_moves_detail_down():
    img, _ = make_skin(h=256, w=256, seed=3)
    e = []
    for r in (2.0, 6.0, 16.0):
        _low, high = freq_split(img, r)
        e.append(float(np.abs(high).mean()))
    print(f"  |HF| при радіусах 2/6/16: {e[0]:.4f} / {e[1]:.4f} / {e[2]:.4f}")
    assert e[0] < e[1] < e[2], "більший радіус має лишати у HF більше деталі"


def test_luma_weights_sum_to_one():
    white = np.ones((4, 4, 3), np.float32)
    print(f"  luma(біле) = {float(luma(white).mean()):.6f}")
    assert abs(float(luma(white).mean()) - 1.0) < 1e-6


def test_radius_scales_with_face_not_megapixels():
    """Кроп і повний кадр з тим самим обличчям — той самий радіус."""
    # Ширина 1500 px навмисно: очікуваний радіус 7.5 лежить ПОСЕРЕДИНІ
    # діапазону lo..hi. На 400 px очікування збіглося б з нижньою межею
    # 2.0, і тест не відрізняв би обчислення від обрізання по межі.
    mask_small = np.zeros((1200, 1800), np.uint8)
    mask_small[100:1100, 100:1600] = 1               # обличчя 1500 px
    mask_big = np.zeros((3000, 4000), np.uint8)
    mask_big[500:1500, 900:2400] = 1                 # те саме 1500 px
    r1 = radius_for(mask_small.shape, mask_small)
    r2 = radius_for(mask_big.shape, mask_big)
    want = 6.0 * 1500 / 1200
    print(f"  2 Мп -> {r1:.2f}px, 12 Мп -> {r2:.2f}px, за §6.3 очікуємо {want:.2f}")
    assert abs(r1 - r2) < 1e-6, "радіус поїхав за мегапікселями, а не за обличчям"
    assert abs(r1 - want) < 0.05, "калібрування §6.3 порушено"
    assert 2.0 < r1 < 32.0, "очікування збіглося з межею — тест нічого не доводить"


def test_radius_is_clamped():
    tiny = np.zeros((50, 50), np.uint8); tiny[20:24, 20:24] = 1
    huge = np.zeros((4000, 8000), np.uint8); huge[:, :] = 1
    lo, hi = radius_for(tiny.shape, tiny), radius_for(huge.shape, huge)
    print(f"  крихітне обличчя -> {lo:.1f}px, кадр цілком -> {hi:.1f}px")
    assert lo >= 2.0 and hi <= 32.0, "радіус вийшов за межі lo..hi"


def test_known_face_width_bias_is_measured():
    """§6.3: розмах маски — не ширина обличчя. Фіксуємо величину похибки."""
    img, _s, truth = make_face(h=1500, w=1150, face_w=900, n_spots=6, seed=3)
    skin = (truth["skin"] | truth["neck"]).astype(np.uint8)
    cols = np.nonzero(skin.any(axis=0))[0]
    seen = int(cols.max() - cols.min() + 1)
    ratio = seen / 900
    print(f"  справжня ширина 900px, розмах маски {seen}px = {ratio:.0%}")
    print(f"  radius_for дає {radius_for(img.shape, skin):.2f}px "
          f"замість {6.0 * 900 / 1200:.2f}px")
    assert 0.7 < ratio < 1.35, "розходження вийшло за очікуване — §6.3 змінилась?"


def test_no_mask_falls_back_to_frame_size():
    r = radius_for((2000, 3000), None)
    print(f"  без маски: {r:.2f}px (груба здогадка з розміру кадру)")
    assert 2.0 <= r <= 32.0


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
