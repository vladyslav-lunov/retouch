"""Тести введення/виведення. Головні інваріанти:

  1. розрядність вхідного файлу повертається на запис — 16-бітний TIFF
     не має деградувати до 8 біт (spec.md §4);
  2. запис і читання зберігають значення в межах кванта;
  3. альфа доїжджає до файлу як четвертий канал;
  4. відмови зрозумілі: RAW, якого нема чим прочитати, і зіпсований файл
     мають давати текст, а не трасування.

Пункт 1 — головна обіцянка §4, і досі вона не перевірялась жодним тестом.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retouch import imageio  # noqa: E402
from retouch.imageio import InputError  # noqa: E402

Q16 = 1.0 / 65535
Q8 = 1.0 / 255


def _grad(h=64, w=128):
    x = np.linspace(0, 1, w, dtype=np.float32)
    return np.dstack([np.tile(x, (h, 1)),
                      np.tile(x[::-1], (h, 1)),
                      np.full((h, w), 0.5, np.float32)])


def test_16bit_stays_16bit():
    """Найважливіше: TIFF з Camera Raw не має втратити розрядність."""
    img = _grad()
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "a.tif"
        cv2.imwrite(str(p), (img * 65535 + 0.5).astype(np.uint16))
        back, dt = imageio.read(p)
        print(f"  прочитано як {dt}, форма {back.shape}")
        assert dt == np.dtype("uint16"), f"розрядність втрачено: {dt}"
        out = Path(t) / "b.tif"
        imageio.write(out, back, dt)
        raw = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
        print(f"  записано як {raw.dtype}")
        assert raw.dtype == np.uint16, "на запису деградувало до 8 біт"


def test_8bit_stays_8bit():
    img = _grad()
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "a.png"
        cv2.imwrite(str(p), (img * 255 + 0.5).astype(np.uint8))
        back, dt = imageio.read(p)
        print(f"  8-бітний вхід -> {dt}")
        assert dt == np.dtype("uint8")


def test_roundtrip_within_one_quantum():
    """Запис-читання не має накопичувати похибку."""
    img = _grad()
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "a.tif"
        imageio.write(p, img, np.dtype("uint16"))
        back, _ = imageio.read(p)
        err = float(np.abs(back - img).max())
        print(f"  похибка обороту: {err / Q16:.2f} кванта 16 біт")
        assert err <= Q16 * 1.5, f"втрата на обороті: {err}"


def test_alpha_survives():
    img = _grad()
    a = np.linspace(0, 1, img.shape[1], dtype=np.float32)
    alpha = np.tile(a, (img.shape[0], 1))
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "layer.png"
        imageio.write(p, img, np.dtype("uint16"), alpha=alpha)
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        print(f"  {raw.shape[2]} канали, dtype {raw.dtype}")
        assert raw.shape[2] == 4, "альфа не доїхала до файлу"
        back_a = raw[:, :, 3].astype(np.float32) / 65535
        assert float(np.abs(back_a - alpha).max()) <= Q16 * 1.5


def test_grayscale_becomes_bgr():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "g.png"
        cv2.imwrite(str(p), (np.linspace(0, 255, 128)[None, :]
                             .repeat(64, 0)).astype(np.uint8))
        img, _ = imageio.read(p)
        print(f"  сірий вхід -> {img.shape}")
        assert img.ndim == 3 and img.shape[2] == 3


def test_missing_file_is_clear():
    try:
        imageio.read("/nope/definitely/absent.tif")
    except InputError as e:
        print(f"  {e}")
        assert "нема" in str(e).lower()
    else:
        raise AssertionError("на відсутній файл не поскаржилось")


def test_undecodable_is_clear():
    """Не трасування, а текст: це помилка користувача, не збій коду."""
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "broken.tif"
        p.write_bytes(b"not an image at all")
        try:
            imageio.read(p)
        except InputError as e:
            print(f"  {str(e).splitlines()[0]}")
            assert "декодувати" in str(e)
        else:
            raise AssertionError("зіпсований файл прочитався?")


def test_raw_suffixes_are_recognised():
    from retouch.rawread import RAW_SUFFIXES
    for ext in (".cr3", ".nef", ".arw", ".dng"):
        assert ext in RAW_SUFFIXES
    print(f"  розширень RAW у списку: {len(RAW_SUFFIXES)}")


def test_mask_read_is_binary_and_resized():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.png"
        m = np.zeros((50, 50), np.uint8)
        m[10:40, 10:40] = 255
        cv2.imwrite(str(p), m)
        got = imageio.read_mask(p, (100, 200))
        print(f"  {m.shape} -> {got.shape}, значення {sorted(np.unique(got))}")
        assert got.shape == (100, 200), "маску не підігнано під кадр"
        assert set(np.unique(got)) <= {0, 1}, "маска не бінарна"


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
