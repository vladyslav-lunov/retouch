"""Тести Dodge & Burn. Головні інваріанти:

  1. soft_light(base, 0.5) == base ТОЧНО — на цьому тримається весь сенс
     шару 50% сірого: його завжди можна вимкнути без сліду;
  2. корекція не виходить за маску шкіри — жодного пікселя;
  3. D&B працює з НИЗЬКОЮ частотою: текстура лишається на місці;
  4. сила множить лінійно, нульова сила — тотожність;
  5. guided filter зберігає край, а не розмиває через нього.

Пункт 3 — розмежування з лікуванням (§1): одне працює з високою
частотою, друге з низькою, і саме тому вони не конфліктують.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retouch.dodgeburn import (DodgeBurnParams, apply, coverage,  # noqa: E402
                               gray_map, guided_filter, soft_light)
from retouch.freqsep import freq_split, luma  # noqa: E402
from tests.synth import make_face  # noqa: E402


def _face(h=760, w=600, fw=470):
    img, _s, truth = make_face(h=h, w=w, face_w=fw, n_spots=8, seed=3)
    return img, (truth["skin"] | truth["neck"]).astype(np.uint8)


def test_neutral_gray_is_exact_identity():
    """Найважливіше: рівно сірий шар не робить нічого."""
    rng = np.random.default_rng(3)
    img = rng.random((80, 120, 3)).astype(np.float32)
    out = soft_light(img, np.full_like(img, 0.5))
    err = float(np.abs(out - img).max())
    print(f"  максимальна різниця: {err:.2e}")
    assert err < 1e-6, "50% сірий у Soft Light змінив кадр"


def test_soft_light_direction():
    mid = np.full((4, 4, 3), 0.5, np.float32)
    up = float(soft_light(mid, np.full_like(mid, 0.75))[0, 0, 0])
    dn = float(soft_light(mid, np.full_like(mid, 0.25))[0, 0, 0])
    print(f"  0.75 -> {up:.3f} (світліше), 0.25 -> {dn:.3f} (темніше)")
    assert dn < 0.5 < up, "напрямок Soft Light переплутано"


def test_soft_light_stays_in_range():
    rng = np.random.default_rng(7)
    for _ in range(5):
        b = rng.random((32, 32, 3)).astype(np.float32)
        s = rng.random((32, 32, 3)).astype(np.float32)
        out = soft_light(b, s)
        assert out.min() >= -1e-6 and out.max() <= 1 + 1e-6
    print("  результат лишається в [0..1] на випадкових входах")


def test_correction_never_leaves_the_mask():
    """Жодного пікселя за маскою — при будь-якому розмитті краю."""
    img, skin = _face()
    for f in (0, 12, 24, 48):
        g = gray_map(img, skin, DodgeBurnParams(strength=0.6, feather=f))
        off = int(((np.abs(g - 0.5) > 1e-4) & (skin == 0)).sum())
        print(f"  feather={f:>3}: поза маскою {off} px")
        assert off == 0, f"корекція вийшла за маску на {off} px"


def test_zero_strength_is_identity():
    img, skin = _face()
    g = gray_map(img, skin, DodgeBurnParams(strength=0.0))
    print(f"  карта: [{g.min():.6f}..{g.max():.6f}]")
    assert float(np.abs(g - 0.5).max()) < 1e-6
    assert float(np.abs(apply(img, g) - img).max()) < 1e-6


def test_strength_scales_linearly():
    img, skin = _face()
    d = []
    for k in (0.25, 0.5, 1.0):
        g = gray_map(img, skin, DodgeBurnParams(strength=k))
        d.append(float(np.abs(g - 0.5).max()))
    print(f"  максимум відхилення при 0.25/0.5/1.0: "
          f"{d[0]:.4f} / {d[1]:.4f} / {d[2]:.4f}")
    assert d[0] < d[1] < d[2]
    assert abs(d[1] - d[2] / 2) < 0.12 * d[2], "сила не лінійна"


def test_texture_is_preserved():
    """D&B — про низьку частоту. Текстура не має постраждати."""
    img, skin = _face()
    res = apply(img, gray_map(img, skin, DodgeBurnParams(strength=0.6)))
    _lo, hb = freq_split(img, 6.0)
    _lo2, ha = freq_split(res, 6.0)
    inside = skin > 0
    have = float(np.abs(luma(hb))[inside].mean())
    changed = float(np.abs(luma(ha) - luma(hb))[inside].mean())
    print(f"  |HF| на шкірі {have:.5f}, зміна {changed:.5f} "
          f"= {changed / have:.1%} наявної текстури")
    assert changed / have < 0.05, "D&B помітно з'їв текстуру"


def test_correction_evens_out_low_frequency():
    """Власне робота: розкид тону на шкірі має зменшитись."""
    img, skin = _face()
    res = apply(img, gray_map(img, skin, DodgeBurnParams(strength=0.8)))
    lo_b, _ = freq_split(img, 24.0)
    lo_a, _ = freq_split(res, 24.0)
    inside = skin > 0
    sb = float(luma(lo_b)[inside].std())
    sa = float(luma(lo_a)[inside].std())
    print(f"  розкид низької частоти: {sb:.4f} -> {sa:.4f}")
    assert sa < sb, "D&B не вирівняв тон"


def test_guided_filter_keeps_the_edge():
    """Гаусів розмив би край; guided має його втримати."""
    import cv2
    step = np.zeros((120, 120), np.float32)
    step[:, 60:] = 1.0
    noisy = step + np.random.default_rng(3).normal(0, 0.03, step.shape).astype(np.float32)
    g = guided_filter(noisy, noisy, 24, 0.005)
    blur = cv2.GaussianBlur(noisy, (0, 0), 8.0)
    edge_g = float(g[:, 70].mean() - g[:, 50].mean())
    edge_b = float(blur[:, 70].mean() - blur[:, 50].mean())
    print(f"  перепад через край: guided {edge_g:.3f}, гаусів {edge_b:.3f}")
    assert edge_g > edge_b, "guided filter розмив край не гірше за гаусів"
    flat = float(g[:, :40].std())
    print(f"  розкид на рівній ділянці після guided: {flat:.4f} (було 0.030)")
    assert flat < 0.02, "guided не згладив рівну ділянку"


def test_coverage_marks_only_changed_pixels():
    img, skin = _face()
    g = gray_map(img, skin, DodgeBurnParams(strength=0.6))
    cov = coverage(g)
    print(f"  покриття {cov.mean():.1%}, поза маскою {int((cov * (skin == 0)).sum())} px")
    assert set(np.unique(cov)) <= {0.0, 1.0}
    assert int((cov * (skin == 0)).sum()) == 0


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
