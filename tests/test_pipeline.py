"""Наскрізні тести конвеєра: від файлу до записаних шарів.

Тести ядра перевіряють математику на масивах у пам'яті, і саме тому
повз них проліз збій у Session.write: шлях «прочитати -> полікувати ->
зібрати шари -> записати» не був покритий узагалі, і `retouch IMG.tif
-o out` падав на None у extract_layer.

Тут перевіряється те, що бачить користувач: чи вийшли файли і чи
збігається накладання шару на базу зі зведеним результатом — уже
після квантування в 16 біт, а не в float32.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retouch.pipeline import Config, Session  # noqa: E402
from tests.synth import make_face  # noqa: E402

QUANT = 1.0 / 65535


def _fixture(d: Path) -> Path:
    img, _spots, _truth = make_face(h=1500, w=1150, face_w=900, n_spots=20, seed=3)
    p = d / "T.tif"
    assert cv2.imwrite(str(p), (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16))
    return p


def _reconstructs(d: Path, stem: str = "T") -> float:
    """Похибка `base*(1-a) + layer*a` проти зведеного файлу, у квантах."""
    rd = lambda n: cv2.imread(str(d / n), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535
    base, flat = rd(f"{stem}_00_base.tif"), rd(f"{stem}_99_flat.tif")
    cur = base
    for layer in sorted(d.glob(f"{stem}_0*_*.png")):
        lay = rd(layer.name)
        rgb, a = lay[:, :, :3], lay[:, :, 3]
        cur = cur * (1 - a[..., None]) + rgb * a[..., None]
    return float(np.abs(cur - flat).max() / QUANT)


def test_write_without_removal():
    """Найпростіший прогін. Саме він і падав."""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        sess = Session(_fixture(d), Config(force_mask=True)).load().analyze().heal()
        written = sess.write(d / "out")
        names = sorted(p.name for p in written)
        print(f"  файли: {', '.join(names)}")
        assert any(n.endswith("_00_base.tif") for n in names)
        assert any(n.endswith("_99_flat.tif") for n in names)
        err = _reconstructs(d / "out")
        print(f"  реконструкція шару: {err:.1f} кванта 16 біт")
        assert err < 8, f"шар не збігається зі зведеним: {err:.1f} кванта"


def test_write_with_removal():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        sess = Session(_fixture(d), Config(force_mask=True)).load().analyze().heal()
        mask = np.zeros(sess.img.shape[:2], np.uint8)
        cv2.circle(mask, (560, 700), 16, 1, -1)
        sess.remove(mask)
        written = sess.write(d / "out")
        names = sorted(p.name for p in written)
        print(f"  файли: {', '.join(names)}")
        assert any("_remove" in n for n in names), "шару видалення немає"
        err = _reconstructs(d / "out")
        print(f"  реконструкція двох шарів: {err:.1f} кванта 16 біт")
        assert err < 8, f"шари не збігаються зі зведеним: {err:.1f} кванта"


def test_write_with_warp():
    """Після пластики база — деформований кадр, інакше шар не зійдеться."""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        sess = Session(_fixture(d), Config(force_mask=True)).load()
        sess.warp_field().bloat(560, 700, 220, -0.35)
        sess.apply_warp().analyze().heal()
        written = sess.write(d / "out")
        names = sorted(p.name for p in written)
        print(f"  файли: {', '.join(names)}")
        assert any("_warp" in n for n in names), "поле зміщення не записано"
        err = _reconstructs(d / "out")
        print(f"  реконструкція після пластики: {err:.1f} кванта 16 біт")
        assert err < 8, f"після деформації шар не сходиться: {err:.1f} кванта"


def test_warp_refreshes_class_map():
    """Карта класів не має пережити деформацію.

    apply_warp перебудовує маску, і спокуса — покликати build_skin_mask
    напряму. Тоді Session.cls лишається від кадру ДО деформації, і будь-яке
    перемикання класів (у UI це просто галочка) ліпить маску, зсунуту
    відносно зображення: заміряно 4% площі розходження при зсуві 220 px.

    Ваг у репозиторії немає, тому кладемо в cls мітку і дивимось, чи вона
    вижила. Це перевіряє рівно те, що зламалось: чи apply_warp іде через
    _build_mask (перераховує cls) чи повз нього (лишає старий).
    """
    import numpy as np

    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        sess = Session(_fixture(d), Config(force_mask=True)).load()
        MARK = 7
        sess.cls = np.full(sess.img.shape[:2], MARK, np.int32)
        sess.warp_field().push(560, 700, 300, 180, 0)
        sess.apply_warp()
        survived = sess.cls is not None and bool(np.all(sess.cls == MARK))
        print(f"  мітка в карті класів пережила деформацію: {survived}")
        assert not survived, (
            "cls лишився від старої геометрії — apply_warp обійшов _build_mask")


def test_warp_resets_downstream():
    """Деформація скидає все, що рахувалося по старій геометрії."""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        sess = Session(_fixture(d), Config(force_mask=True)).load().analyze().heal()
        n_before = len(sess.blobs)
        sess.warp_field().push(560, 700, 260, 40, 0)
        sess.apply_warp()
        print(f"  до пластики {n_before} плям, після скидання "
              f"labels={sess.labels}, result={sess.result}")
        assert sess.labels is None and sess.result is None, (
            "після деформації лишились результати по старій геометрії")
        sess.analyze().heal()
        print(f"  після перерахунку: {len(sess.blobs)} плям")
        assert sess.result is not None


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
