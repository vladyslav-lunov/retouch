"""Тести оглядового аркуша. Головні інваріанти:

  1. кропи в аркуші — рідні пікселі, без ресайзу (§1: інакше дивитись
     нема на що);
  2. кирилиця в підписах транслітерується, а не стає «???????»;
  3. поля вписування не плутаються з даними;
  4. аркуш будується і без результату, і без маски — не падає.

Пункт 2 — не косметика. Тричі за проєкт підпис перетворювався на знаки
питання, бо putText уміє лише Hershey-шрифти.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retouch.blemish import DetectParams, detect_blemishes, heal_blemishes  # noqa: E402
from retouch.freqsep import freq_merge, freq_split  # noqa: E402
from retouch.preview import contact_sheet, to_latin  # noqa: E402
from tests.synth import make_skin  # noqa: E402


def _run(h=420, w=520):
    img, _spots = make_skin(h=h, w=w, n_spots=10, seed=3)
    low, high = freq_split(img, 6.0)
    lbl, blobs = detect_blemishes(high, None, DetectParams(threshold=0.010))
    high2, cov = heal_blemishes(high, lbl, blobs, None, search_radius=60)
    return img, freq_merge(low, high2), cov, lbl, blobs


def test_transliteration_has_no_question_marks():
    cases = ["обережний — тільки помітне", "«Стандарт»… 0.85",
             "ретельний, з лімітом", "Пластика 28.4px"]
    for t in cases:
        got = to_latin(t)
        print(f"  {t}  ->  {got}")
        assert "?" not in got, f"незакрита кирилиця чи пунктуація: {got}"
        assert all(ord(c) < 128 for c in got), "у результаті лишилось не-ASCII"


def test_transliteration_keeps_ascii_intact():
    t = "R18 0.68% @1125,2231 [ok]"
    assert to_latin(t) == t, "латиницю зіпсовано"
    print(f"  {t} — без змін")


def test_crops_are_native_resolution():
    """Кроп має бути рівно crop x crop пікселів кадру, без масштабування."""
    img, res, cov, lbl, blobs = _run()
    sheet = contact_sheet(img, res, cov, None, lbl, blobs, n_crops=2, crop=180,
                          panel=200)
    print(f"  аркуш {sheet.shape[1]}x{sheet.shape[0]}")
    # три плитки по 180 + проміжки 8 + підпис — ряд кропів має бути >= 3*180
    assert sheet.shape[1] >= 3 * 180, "кропи стиснуто — дивитись нема на що"
    assert sheet.dtype == np.float32 and 0.0 <= sheet.min() and sheet.max() <= 1.0


def test_survives_without_result_or_mask():
    img, _res, _cov, lbl, blobs = _run()
    zero = np.zeros(img.shape[:2], np.float32)
    sheet = contact_sheet(img, img, zero, None, lbl, blobs, n_crops=3, panel=180)
    print(f"  без дотиків: аркуш {sheet.shape[1]}x{sheet.shape[0]} побудовано")
    assert sheet.size > 0


def test_padding_is_not_mistaken_for_data():
    """Поля вписування колись перевищували поріг детекції й малювались
    червоними смугами на весь кадр."""
    img, res, cov, lbl, blobs = _run(h=200, w=600)     # дуже широкий кадр
    sheet = contact_sheet(img, res, cov, None, lbl, blobs, n_crops=1, panel=200)
    red = ((sheet[:, :, 2] > 0.9) & (sheet[:, :, 0] < 0.3) & (sheet[:, :, 1] < 0.3))
    print(f"  червоних пікселів у аркуші: {red.mean():.3%}")
    assert red.mean() < 0.10, "поля вписування знову зафарбовані як детекція"


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
