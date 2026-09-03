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
    ланцюжок, і лікування рве його на шматки (spec.md §15).

    Шию тут вмикаємо явно: з дефолту її прибрано саме через це, і
    попередження існує рівно для того, хто ввімкнув її назад.
    """
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        from retouch.masks import MaskParams as MP
        cfg = Config(force_mask=True,
                     mask=MP(skin_classes=("skin", "nose", "neck")))
        sess = Session(_fixture(d), cfg).load()
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


# ---------------------------------------------------------------------------
# поріг під ціль
# ---------------------------------------------------------------------------

def _noisy(d: Path, spots: int, seed: int, noise: float) -> Path:
    """Кадр із заданою кількістю дефектів і заданою «текстурою шкіри».

    Шум тут — не прикраса: саме він відрізняє кадр, на якому поріг 0.012
    працює, від кадру, на якому той самий поріг ловить пори й переходить
    у згладжування (spec.md §6.2).
    """
    img, _s, _t = make_face(h=1200, w=900, face_w=700, n_spots=spots, seed=seed)
    if noise:
        rng = np.random.default_rng(seed)
        img = np.clip(img + rng.normal(0, noise, img.shape).astype(np.float32), 0, 1)
    p = d / f"N{seed}_{int(noise*1000)}.tif"
    cv2.imwrite(str(p), (img * 65535 + 0.5).astype(np.uint16))
    return p


def test_target_coverage_beats_a_fixed_threshold_across_frames():
    """Ціль переноситься між кадрами, поріг — ні.

    Це головний висновок калібрування на 44 реальних кадрах: з
    фіксованим 0.012 у робочу зону потрапило 15, а 14 пішли в
    згладжування. Тут те саме на двох кадрах із різною текстурою.
    """
    from retouch.blemish import DetectParams
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        frames = [_noisy(d, 12, 3, 0.0), _noisy(d, 12, 4, 0.012)]

        fixed, targeted = [], []
        for f in frames:
            s1 = Session(f, Config(force_mask=True)).load().analyze().heal()
            fixed.append(float((s1.coverage > 0).sum()) / float(s1.skin.sum()))
            cfg = Config(force_mask=True,
                         detect=DetectParams(target_coverage=0.03))
            s2 = Session(f, cfg).load().analyze().heal()
            targeted.append((float((s2.coverage > 0).sum()) / float(s2.skin.sum()),
                             s2.cfg.detect.threshold))

        spread_fixed = max(fixed) / max(min(fixed), 1e-9)
        spread_targ = max(c for c, _t in targeted) / max(
            min(c for c, _t in targeted), 1e-9)
        print(f"  фіксований поріг: торкнуто {[f'{c:.2%}' for c in fixed]}, "
              f"розкид ×{spread_fixed:.1f}")
        print(f"  під ціль 3%:      торкнуто "
              f"{[f'{c:.2%} @{t}' for c, t in targeted]}, розкид ×{spread_targ:.1f}")
        assert spread_targ < spread_fixed, (
            "ціль не зменшила розкид — сенсу в підборі немає")
        for cov, _t in targeted:
            assert cov <= 0.045, f"ціль 3% не втримана: {cov:.2%}"


def test_solver_says_so_when_the_target_is_unreachable():
    """Кадр буває просто такий. Мовчки поставити найжорсткіший поріг і
    вдати, що ціль досягнута, — це те саме мовчазне «наближено» (§1)."""
    from retouch.blemish import DetectParams
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        f = _noisy(d, 30, 7, 0.03)          # суцільна текстура
        cfg = Config(force_mask=True,
                     detect=DetectParams(target_coverage=0.0005))
        sess = Session(f, cfg).load().analyze()
        print(f"  поріг {sess.cfg.detect.threshold}, "
              f"нота: {(sess.threshold_note or '—')[:70]}")
        assert sess.threshold_note, "недосяжна ціль пройшла мовчки"
        assert sess.cfg.detect.threshold == Session.THRESHOLD_LADDER[-1]


def test_solver_leaves_the_threshold_alone_when_no_target():
    """Без цілі поводимось як раніше — жодних сюрпризів у старих пресетах."""
    from retouch.blemish import DetectParams
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        cfg = Config(force_mask=True, detect=DetectParams(threshold=0.019))
        sess = Session(_fixture(d), cfg).load().analyze()
        print(f"  поріг лишився {sess.cfg.detect.threshold}, "
              f"крива {sess.threshold_curve}")
        assert sess.cfg.detect.threshold == 0.019
        assert not sess.threshold_curve, "підбір запустився без цілі"


def test_search_radius_scales_with_the_face():
    """Питання §13 №2. Відповідь дав розкид ширини обличчя у 8 разів."""
    with tempfile.TemporaryDirectory() as t:
        sess = Session(_fixture(Path(t)), Config(force_mask=True)).load()
        got = {w: sess._search_radius(w) for w in (200, 600, 1200, 2400)}
        print(f"  обличчя -> пошук: {got}")
        assert got[1200] == Session.BASE_SEARCH, (
            "на опорному обличчі 1200 px має вийти рівно калібрувальне число")
        assert got[600] < got[1200] < got[2400], "не масштабується"
        # у частках обличчя має бути СТАЛИМ — у цьому вся суть
        share = [got[w] / w for w in (600, 1200, 2400)]
        print(f"  у частках обличчя: {[round(x, 3) for x in share]}")
        assert max(share) - min(share) < 0.01, f"частка гуляє: {share}"


def test_explicit_search_radius_stays_absolute():
    """Хто задав пікселі — мав на увазі пікселі. Мовчки перерахувати їх
    було б підміною параметра (той самий принцип, що з target_coverage)."""
    with tempfile.TemporaryDirectory() as t:
        cfg = Config(force_mask=True, search_radius=150)
        sess = Session(_fixture(Path(t)), cfg).load().analyze()
        print(f"  задано 150 -> вжито {sess.search_radius_px}")
        assert sess.search_radius_px == 150


def test_radius_comes_from_the_detector_when_there_is_one():
    """Session має брати ширину з рамки, а не з габариту маски."""
    from retouch.freqsep import radius_for
    with tempfile.TemporaryDirectory() as t:
        sess = Session(_fixture(Path(t)), Config(force_mask=True)).load()
        sess.face_w = 1200.0                     # ніби детектор щось знайшов
        sess.analyze()
        print(f"  джерело {sess.face_w_source}, радіус {sess.radius:.2f}, "
              f"пошук {sess.search_radius_px}")
        assert sess.face_w_source == "detector"
        assert abs(sess.radius - radius_for(sess.img.shape, face_w=1200.0)) < 1e-6
        assert sess.search_radius_px == Session.BASE_SEARCH


def test_clamped_radius_is_reported_as_a_limit():
    """Підлога радіуса — межа застосовності, а не деталь реалізації.

    Нижче 2 px частотка не відділяє дефект від пікселя, тобто
    калібрувальне «обличчя 1200 -> радіус 6» перестає діяти. На реальній
    зйомці таких кадрів 16 із 44, і мовчки віддати результат означало б
    удати, що він порівнянний з рештою.
    """
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        small = Session(_fixture(d), Config(force_mask=True)).load()
        small.face_w = 200.0                  # 6*200/1200 = 1.0 -> підлога
        small.analyze()
        big = Session(_fixture(d), Config(force_mask=True)).load()
        big.face_w = 1200.0
        big.analyze()
        print(f"  обличчя 200 px: радіус {small.radius}, "
              f"попередження {'є' if small.radius_warn else 'немає'}")
        print(f"  обличчя 1200 px: радіус {big.radius}, "
              f"попередження {'є' if big.radius_warn else 'немає'}")
        assert small.radius == 2.0 and small.radius_clamped
        assert small.radius_warn and "191" not in small.radius_warn
        assert "200" in small.radius_warn, "у тексті немає ширини обличчя"
        assert not big.radius_clamped and big.radius_warn is None


def test_explicit_radius_is_never_called_clamped():
    """Хто задав радіус руками — знає, що робить, і попереджати нема про що."""
    with tempfile.TemporaryDirectory() as t:
        cfg = Config(force_mask=True, hf_radius=2.0)
        sess = Session(_fixture(Path(t)), cfg).load()
        sess.face_w = 200.0
        sess.analyze()
        print(f"  задано 2.0 вручну -> clamped={sess.radius_clamped}")
        assert not sess.radius_clamped and sess.radius_warn is None


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
