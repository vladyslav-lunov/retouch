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


def test_full_stack_reconstructs_with_all_layers():
    """Уся стопка: шкіра, інструменти, D&B — і кожен зі своїм режимом.

    Номер у назві файлу — це ПОРЯДОК складання. Свого часу D&B писався з
    жорстким «03» і стикався з третім інструментом: два файли з тим самим
    індексом складати стає нíяк.
    """
    import glob

    from retouch.dodgeburn import soft_light

    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        cfg = Config(force_mask=True, dodgeburn_on=True,
                     tools=("mattify", "teeth"))
        cfg.dodgeburn.strength = 0.5
        sess = Session(_fixture(d), cfg).load().analyze().heal()
        # Карту класів кладемо руками: ваг у репозиторії немає, а без cls
        # інструменти пропускаються — і тест перевіряв би стопку з одного
        # шару, тобто саме ту ситуацію, в якій зіткнення індексів НЕ
        # відтворюється. Заради чого він і написаний.
        sess.cls = _cls_map(sess)
        sess.run_tools()
        sess.dodge_burn()
        assert len(sess.tool_layers) >= 2, (
            "інструменти не дали шарів — перевіряти нумерацію нема на чому")
        out = d / "out"
        sess.write(out)

        names = sorted(p.name for p in out.iterdir())
        print(f"  {', '.join(names)}")
        assert len(list(out.glob("T_[0-9][0-9]_*.png"))) >= 4, (
            "шарів менше, ніж етапів: щось не записалось")
        idx = [n.split("_")[1] for n in names if n[-4:] == ".png" and n[1:3].isdigit()
               or (len(n.split("_")) > 1 and n.split("_")[1].isdigit())]
        assert len(idx) == len(set(idx)), f"однакові індекси у шарах: {idx}"

        rd = lambda n: (cv2.imread(str(out / n), cv2.IMREAD_UNCHANGED)
                        .astype(np.float32) / 65535)
        stem = "T"
        cur = rd(f"{stem}_00_base.tif")
        flat = rd(f"{stem}_99_flat.tif")
        for p in sorted(out.glob(f"{stem}_[0-9][0-9]_*.png")):
            lay = rd(p.name)
            if "softlight" in p.name:
                cur = soft_light(cur, lay)
            else:
                a = lay[:, :, 3]
                cur = cur * (1 - a[..., None]) + lay[:, :, :3] * a[..., None]
        err = float(np.abs(cur - flat).max()) * 65535
        print(f"  похибка складання всієї стопки: {err:.1f} кванта")
        assert err < 12, f"стопка не сходиться: {err:.1f} кванта"


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


# ---------------------------------------------------------------------------
# повторний прогін етапу
# ---------------------------------------------------------------------------

def _cls_map(sess):
    """Правдоподібна карта класів без ваг: смуга шкіри поперек обличчя."""
    from retouch.masks import CELEBA_CLASSES
    inv = {v: k for k, v in CELEBA_CLASSES.items()}
    h, w = sess.img.shape[:2]
    cls = np.full((h, w), inv["background"], np.int32)
    cls[h // 4:3 * h // 4, w // 4:3 * w // 4] = inv["skin"]
    cls[h // 2:h // 2 + h // 20, w // 2:w // 2 + w // 12] = inv["mouth"]
    return cls


def test_dodge_burn_twice_gives_the_same_frame():
    """Повзунок у UI сунуть десять разів підряд.

    Якщо етап рахується від власного результату, друге натискання дає
    подвійну силу при тому самому числі на повзунку — тобто UI показує
    одне, а кадр інше.
    """
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        cfg = Config(force_mask=True, dodgeburn_on=True)
        sess = Session(_fixture(d), cfg).load().analyze().heal()
        first = sess.dodge_burn().result.copy()
        second = sess.dodge_burn().result
        delta = float(np.abs(first - second).max() / QUANT)
        print(f"  розбіжність між першим і другим прогоном: {delta:.2f} кванта")
        assert delta < 0.5, "D&B накопичується сам на собі"


def test_tools_twice_do_not_stack():
    """Те саме для інструментів, плюс шари не мають дублюватись."""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        cfg = Config(force_mask=True, tools=("mattify",))
        sess = Session(_fixture(d), cfg).load().analyze().heal()
        sess.cls = _cls_map(sess)
        first = sess.run_tools().result.copy()
        n1 = len(sess.tool_layers)
        second = sess.run_tools().result
        delta = float(np.abs(first - second).max() / QUANT)
        print(f"  шарів {n1} -> {len(sess.tool_layers)}, "
              f"розбіжність {delta:.2f} кванта")
        assert len(sess.tool_layers) == n1, "шари продублювались"
        assert delta < 0.5, "інструмент наклався сам на себе"


def test_turning_every_tool_off_undoes_them():
    """Знята галочка має ПРИБРАТИ ефект, а не лишити його останнім."""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        cfg = Config(force_mask=True, tools=("mattify",))
        sess = Session(_fixture(d), cfg).load().analyze().heal()
        sess.cls = _cls_map(sess)
        healed = sess.result.copy()
        sess.run_tools()
        touched = float(np.abs(sess.result - healed).max() / QUANT)
        sess.cfg.tools = ()
        sess.run_tools()
        back = float(np.abs(sess.result - healed).max() / QUANT)
        print(f"  інструмент змінив {touched:.0f} квантів, після вимкнення "
              f"лишилось {back:.2f}")
        assert touched > 1, "інструмент нічого не зробив — тест порожній"
        assert back < 0.5 and not sess.tool_layers, "ефект лишився після вимкнення"


def test_reheal_drops_stale_downstream_layers():
    """Перелікування робить карти інструментів і D&B недійсними.

    Вони рахувалися поверх іншого кадру, а база шару в layers() береться
    саме з них: лишити їх — записати шар, знятий з неіснуючого стану.
    """
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        cfg = Config(force_mask=True, tools=("mattify",), dodgeburn_on=True)
        sess = Session(_fixture(d), cfg).load().analyze().heal()
        sess.cls = _cls_map(sess)
        sess.run_tools().dodge_burn()
        print(f"  до: шарів {len(sess.tool_layers)}, D&B {sess.db_gray is not None}")
        sess.cfg.detect.threshold = 0.02
        sess.analyze().heal()
        print(f"  після перелікування: шарів {len(sess.tool_layers)}, "
              f"D&B {sess.db_gray is not None}")
        assert not sess.tool_layers and sess.db_gray is None, (
            "лишились шари, зняті з кадру до перелікування")


# ---------------------------------------------------------------------------
# що саме знайдено, а не скільки
# ---------------------------------------------------------------------------

def _cls_with_neck(sess, chain_y):
    """Карта класів зі смугою `neck` унизу — там, де на портреті груди."""
    from retouch.masks import CELEBA_CLASSES
    inv = {v: k for k, v in CELEBA_CLASSES.items()}
    h, w = sess.img.shape[:2]
    cls = np.full((h, w), inv["background"], np.int32)
    cls[:chain_y, :] = inv["skin"]
    cls[chain_y:, :] = inv["neck"]
    return cls


def test_detection_is_reported_by_class():
    """Скільки знайдено — саме по собі нічого не каже.

    154 плями на обличчі й 154 на ланцюжку в консолі виглядають однаково.
    Клас каже, ЩО знайдено, і це єдине, з чого видно підміну.
    """
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        sess = Session(_fixture(d), Config(force_mask=True)).load()
        sess.cls = _cls_with_neck(sess, sess.img.shape[0] // 2)
        sess.analyze()
        print(f"  {sess.blob_classes}")
        assert sess.blob_classes, "розподіл по класах порожній"
        assert sum(c for _n, c in sess.blob_classes) == len(sess.blobs), (
            "плями загубились між класами")
        assert sess.blob_classes == sorted(sess.blob_classes,
                                           key=lambda r: -r[1]), "не відсортовано"


def test_warning_fires_when_defects_pile_up_outside_the_face():
    """Заміряно на реальному кадрі: 39 зі 154 знахідок у класі neck — це
    ланцюжок, і лікування рве його на шматки (spec.md §15)."""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        sess = Session(_fixture(d), Config(force_mask=True)).load()
        # межу ставимо високо, щоб у «шию» потрапила помітна частка плям
        sess.cls = _cls_with_neck(sess, sess.img.shape[0] // 4)
        sess.analyze()
        share = dict(sess.blob_classes).get("neck", 0) / max(len(sess.blobs), 1)
        print(f"  у класі neck {share:.0%} знахідок; попередження: "
              f"{(sess.detect_warn or '—')[:60]}")
        assert share >= 0.15, "мало плям у neck — тест нічого не перевіряє"
        assert sess.detect_warn and "neck" in sess.detect_warn


def test_no_warning_when_the_class_is_not_treated_as_skin():
    """Клас поза набором шкіри не лікується, тож і попереджати нема про що."""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        from retouch.masks import MaskParams as MP
        cfg = Config(force_mask=True, mask=MP(skin_classes=("skin", "nose")))
        sess = Session(_fixture(d), cfg).load()
        sess.cls = _cls_with_neck(sess, sess.img.shape[0] // 4)
        sess.analyze()
        print(f"  skin_classes={cfg.mask.skin_classes}, "
              f"попередження: {sess.detect_warn}")
        assert sess.detect_warn is None, (
            "попередили про клас, який і так не лікується")


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
