"""Тести дрібних інструментів. Головні інваріанти:

  1. корекція НЕ виходить за свій клас — жодного пікселя;
  2. нульова сила — точна тотожність;
  3. кожен робить те, що обіцяє, і в правильний бік;
  4. без потрібного класу в кадрі інструмент мовчки нічого не робить,
     а не падає;
  5. результат лишається в [0..1].

Пункт 1 повторює те, на чому вже двічі наступали: вії й ланцюжок
постраждали саме тому, що альфа розповзалась ширше за маску.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retouch.freqsep import luma  # noqa: E402
from retouch.masks import CELEBA_CLASSES  # noqa: E402
from retouch.tools import (EyeVesselParams, MattifyParams,  # noqa: E402
                           SkinToneParams, TOOLS, TeethParams, _region,
                           eye_vessels, mattify, skin_tone, teeth)

INV = {v: k for k, v in CELEBA_CLASSES.items()}
REGION = {"eye_vessels": ("l_eye", "r_eye"), "teeth": ("mouth",),
          "mattify": ("skin", "nose", "neck"), "skin_tone": ("skin", "nose", "neck")}


def _scene(h=200, w=200):
    """Кадр із намальованими класами й правдоподібним вмістом у кожному."""
    rng = np.random.default_rng(3)
    img = np.full((h, w, 3), 0.55, np.float32)
    img[:, :, 2] = 0.72                                   # тілесний
    cls = np.full((h, w), INV["background"], np.int32)

    cls[20:120, 20:180] = INV["skin"]
    cls[60:80, 90:120] = INV["nose"]
    img[60:80, 90:120] = (0.86, 0.90, 0.95)               # відблиск на носі

    cls[30:50, 40:80] = INV["l_eye"]
    img[30:50, 40:80] = (0.80, 0.80, 0.86)                # білок
    img[36:44, 52:68] = (0.20, 0.28, 0.30)                # райдужка
    img[30:34, 40:80, 2] = 0.95                           # судини: червоне

    cls[140:170, 70:130] = INV["mouth"]
    img[140:170, 70:130] = (0.42, 0.66, 0.74)             # жовтуваті зуби
    img += rng.normal(0, 0.004, img.shape).astype(np.float32)
    return np.clip(img, 0, 1), cls


def _run(name, **kw):
    fn, P = TOOLS[name]
    img, cls = _scene()
    res, cov = fn(img, cls, p=P(**kw)) if name in ("mattify", "skin_tone") \
        else fn(img, cls, P(**kw))
    return img, cls, res, cov


def test_nothing_leaves_its_class():
    for name in TOOLS:
        img, cls, res, cov = _run(name, strength=1.0)
        reg = _region(cls, REGION[name])
        off = int(((cov > 0) & ~reg).sum())
        changed_off = int((np.abs(res - img).max(axis=2) > 1e-6)[~reg].sum())
        print(f"  {name:<12} альфа поза класом {off}, змінених пікселів {changed_off}")
        assert off == 0, f"{name}: альфа вийшла за клас"
        assert changed_off == 0, f"{name}: змінено пікселі поза класом"


def test_zero_strength_is_identity():
    for name in TOOLS:
        img, _cls, res, cov = _run(name, strength=0.0)
        d = float(np.abs(res - img).max())
        print(f"  {name:<12} максимальна зміна {d:.2e}")
        assert d < 1e-6, f"{name} щось зробив при нульовій силі"


def test_results_stay_in_range():
    for name in TOOLS:
        _img, _cls, res, _cov = _run(name, strength=1.0)
        assert res.min() >= -1e-6 and res.max() <= 1 + 1e-6, name
    print("  усі чотири лишаються в [0..1]")


def test_eye_vessels_reduce_red_not_brightness():
    img, cls, res, _cov = _run("eye_vessels", strength=1.0)
    eye = _region(cls, ("l_eye", "r_eye"))
    y = luma(img)
    sclera = eye & (y >= np.percentile(y[eye], 55))
    r_before = float(img[:, :, 2][sclera].mean())
    r_after = float(res[:, :, 2][sclera].mean())
    y_before = float(luma(img)[sclera].mean())
    y_after = float(luma(res)[sclera].mean())
    print(f"  червоний {r_before:.3f} -> {r_after:.3f}, "
          f"яскравість {y_before:.3f} -> {y_after:.3f}")
    assert r_after < r_before, "судини не прибрано"
    assert y_after <= y_before + 0.01, "око відбілено, а мало лише знечервонитись"


def test_teeth_reduce_yellow_but_stay_warm():
    img, cls, res, _cov = _run("teeth")
    m = _region(cls, ("mouth",))
    y = luma(img)
    area = m & (y >= np.percentile(y[m], 60))
    blue = lambda a: float((a[:, :, 0] - (a[:, :, 1] + a[:, :, 2]) / 2)[area].mean())
    b0, b1 = blue(img), blue(res)
    print(f"  синява {b0:.4f} -> {b1:.4f} (нуль = рівно сірі)")
    assert b1 > b0, "жовтизну не прибрано"
    assert b1 < 0.0, "зуби доведено до нейтралі — це читається як вставні"


def test_mattify_darkens_only_the_hotspot():
    img, cls, res, _cov = _run("mattify", strength=1.0)
    nose = _region(cls, ("nose",))
    skin_only = _region(cls, ("skin",))
    dy_hot = float((luma(img) - luma(res))[nose].mean())
    dy_flat = float(np.abs(luma(img) - luma(res))[skin_only].mean())
    print(f"  відблиск темнішає на {dy_hot:.4f}, рівна шкіра на {dy_flat:.4f}")
    assert dy_hot > 0.002, "відблиск не прибрано"
    assert dy_flat < dy_hot, "затемнено шкіру загалом, а не блиск"


def test_skin_tone_evens_chroma_not_luma():
    img, cls, res, _cov = _run("skin_tone", strength=1.0)
    skin = _region(cls, ("skin", "nose", "neck"))
    y = np.maximum(luma(img), 1e-4)
    ch_b = (img / y[..., None])[skin].std()
    ch_a = (res / np.maximum(luma(res), 1e-4)[..., None])[skin].std()
    dy = float(np.abs(luma(img) - luma(res))[skin].mean())
    print(f"  розкид хроматичності {ch_b:.4f} -> {ch_a:.4f}, "
          f"яскравість зсунулась на {dy:.4f}")
    assert ch_a < ch_b, "хроматичність не вирівняно"
    assert dy < 0.02, "зачеплено яскравість — це робота D&B, не ця"


def test_missing_class_is_a_no_op():
    """Кадр без рота чи без очей не має валити інструмент."""
    img = np.full((60, 60, 3), 0.5, np.float32)
    cls = np.full((60, 60), INV["background"], np.int32)
    for name, (fn, P) in TOOLS.items():
        res, cov = fn(img, cls, p=P()) if name in ("mattify", "skin_tone") \
            else fn(img, cls, P())
        assert float(np.abs(res - img).max()) == 0.0 and cov.max() == 0.0, name
    print("  без потрібного класу всі чотири мовчки нічого не роблять")


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
