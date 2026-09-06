"""Тести масок. Головні інваріанти:

  1. набір класів — параметр: та сама карта класів дає різні маски;
  2. зони виключення справді віднімаються, і дилатація їх розширює;
  3. ерозія звужує маску, а не розширює;
  4. евристика повертає бінарну маску потрібної форми;
  5. перевірка правдоподібності спрацьовує на 90% і мовчить на 20%.

Пункт 1 — те, заради чого §15 і писався: «шкіра» це рішення фотографа,
а не властивість пікселя. Якщо параметр перестане працювати, зникне
єдиний спосіб урятувати ланцюжок на грудях.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retouch.masks import (CELEBA_CLASSES, MaskParams,  # noqa: E402
                           heuristic_skin_mask, mask_from_classes)
from retouch.pipeline import Config, check_skin_mask  # noqa: E402
from tests.synth import make_face  # noqa: E402

INV = {v: k for k, v in CELEBA_CLASSES.items()}


def _cls_map(h=200, w=200):
    """Синтетична карта класів: смуги по класах, щоб рахувати площі точно."""
    c = np.zeros((h, w), np.int32)
    c[:50] = INV["hair"]
    c[50:110] = INV["skin"]
    c[110:140] = INV["neck"]
    c[140:160] = INV["l_lip"]
    c[160:] = INV["cloth"]
    return c


def test_skin_classes_are_a_parameter():
    """Та сама карта, різні набори — різні маски."""
    c = _cls_map()
    p0 = MaskParams(erode=0, exclude_dilate=0)
    full = mask_from_classes(c, MaskParams(erode=0, exclude_dilate=0,
                                           skin_classes=("skin", "neck")))
    face = mask_from_classes(c, MaskParams(erode=0, exclude_dilate=0,
                                           skin_classes=("skin",)))
    print(f"  skin+neck: {full.mean():.1%}, лише skin: {face.mean():.1%}")
    assert full.mean() > face.mean(), "зняття класу не зменшило маску"
    assert abs(full.mean() - 0.45) < 0.02, "skin(60)+neck(30) з 200 рядків = 45%"
    assert abs(face.mean() - 0.30) < 0.02


def test_excluded_classes_are_subtracted():
    c = _cls_map()
    m = mask_from_classes(c, MaskParams(erode=0, exclude_dilate=0,
                                        skin_classes=("skin", "neck", "l_lip"),
                                        exclude_classes=("l_lip",)))
    lip = c == INV["l_lip"]
    print(f"  губи в масці: {m[lip].sum()} px з {int(lip.sum())}")
    assert m[lip].sum() == 0, "зона виключення лишилась у масці"


def test_exclude_dilate_widens_exclusion():
    c = _cls_map()
    a = mask_from_classes(c, MaskParams(erode=0, exclude_dilate=0,
                                        skin_classes=("skin", "neck"),
                                        exclude_classes=("l_lip",)))
    b = mask_from_classes(c, MaskParams(erode=0, exclude_dilate=6,
                                        skin_classes=("skin", "neck"),
                                        exclude_classes=("l_lip",)))
    print(f"  без дилатації {a.mean():.1%}, з дилатацією {b.mean():.1%}")
    assert b.mean() < a.mean(), "exclude_dilate не розширив зону виключення"


def test_erode_shrinks():
    c = _cls_map()
    a = mask_from_classes(c, MaskParams(erode=0, exclude_dilate=0))
    b = mask_from_classes(c, MaskParams(erode=8, exclude_dilate=0))
    print(f"  erode=0: {a.mean():.1%}, erode=8: {b.mean():.1%}")
    assert b.mean() < a.mean(), "ерозія не звузила маску"
    assert b.sum() > 0, "ерозія з'їла все"


def test_unknown_class_name_is_ignored_not_fatal():
    """Пресет може прийти з чужою назвою класу — це не привід падати."""
    c = _cls_map()
    m = mask_from_classes(c, MaskParams(erode=0, exclude_dilate=0,
                                        skin_classes=("skin", "вигаданий")))
    print(f"  з невідомим класом у наборі: маска {m.mean():.1%}")
    assert m.mean() > 0, "через невідому назву втрачено всю маску"


def test_heuristic_returns_binary_mask():
    img, _s, _t = make_face(h=400, w=300, face_w=220, n_spots=4, seed=3)
    m = heuristic_skin_mask(img)
    print(f"  форма {m.shape}, значення {sorted(np.unique(m))}, покриття {m.mean():.1%}")
    assert m.shape == img.shape[:2]
    assert set(np.unique(m)) <= {0, 1}, "маска не бінарна"
    assert m.dtype == np.uint8


def test_sanity_check_fires_on_implausible_mask():
    cfg = Config()
    assert check_skin_mask(0.91, cfg, "heuristic") is not None, "на 91% промовчали"
    assert check_skin_mask(0.20, cfg, "face-parsing") is None, "на 20% сварились дарма"
    warn = check_skin_mask(0.91, cfg, "heuristic")
    print(f"  {warn.splitlines()[0]}")
    assert "91%" in warn and "60%" in warn, "у тексті немає ані факту, ані межі"


def test_sanity_threshold_is_configurable():
    cfg = Config(max_skin_fraction=0.95)
    print(f"  межа {cfg.max_skin_fraction:.0%}: на 91% -> "
          f"{'мовчить' if check_skin_mask(0.91, cfg, 'x') is None else 'сварить'}")
    assert check_skin_mask(0.91, cfg, "x") is None


def test_detector_miss_is_reported_not_hidden():
    """Провал YuNet скидає нас на розбір ПОВНОГО кадру, а §5 називає його
    непридатним: 1.4% шкіри і 28% «капелюха» там, де капелюха немає.

    Раніше в звіті лишалось «face-parsing+yunet» — тобто звіт про кроп,
    якого не було. Ваг у репозиторії немає, тож перевіряємо саме логіку
    імені й скидання рамок, підмінивши розбір заглушкою.
    """
    from retouch.masks import FaceParser

    class Stub(FaceParser):
        def __init__(self):                    # без ONNX
            pass

        def _parse_whole(self, img):
            return np.zeros(img.shape[:2], np.int32)

    img = np.zeros((80, 60, 3), np.float32)
    fp = Stub()
    fp.last_faces = [(0, 0, 40, 40)]           # ніби лишилось із минулого разу
    fp.parse(img, detector=None)
    print(f"  без детектора last_faces={fp.last_faces}")
    assert fp.last_faces == [], (
        "рамка з попереднього кадру пережила розбір — ширина обличчя "
        "рахувалася б по чужому кадру")


def test_source_name_says_what_happened():
    """Ім'я джерела — це звіт про те, що сталося, а не переказ налаштувань.

    Різниця змістова: «+yunet» означає, що кадр кропнуто по обличчю, а це
    умова роботи BiSeNet (§15.1). Якщо детектор промазав, розбирався
    ПОВНИЙ кадр — інший шлях, з іншою надійністю, і називати його тим
    самим іменем не можна.
    """
    import tempfile
    import cv2
    from pathlib import Path as P
    from retouch import masks as masks_mod
    from retouch.pipeline import Config, Session
    from tests.synth import make_face

    class Stub:
        """Замість ONNX: віддає карту класів і те, що «знайшов» детектор."""
        found: list = []

        def __init__(self, _path):
            self.last_faces = []

        def parse(self, img, detector=None, margin=None, **kw):
            self.last_faces = list(Stub.found)
            cls = np.zeros(img.shape[:2], np.int32)
            h, w = cls.shape
            cls[h // 4:3 * h // 4, w // 4:3 * w // 4] = 1      # skin
            return cls

    real = masks_mod.FaceParser
    masks_mod.FaceParser = Stub
    try:
        with tempfile.TemporaryDirectory() as t:
            img, _s, _tr = make_face(h=600, w=460, face_w=300, n_spots=4, seed=1)
            f = P(t) / "T.tif"
            cv2.imwrite(str(f), (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16))
            model = P(t) / "fake.onnx"
            model.write_bytes(b"x")
            det = P(t) / "det.onnx"
            det.write_bytes(b"x")
            cfg = lambda: Config(face_model=str(model), face_detector=str(det),
                                 force_mask=True)

            Stub.found = [(10, 10, 300, 300)]
            hit = Session(f, cfg()).load()
            Stub.found = []
            miss = Session(f, cfg()).load()

        print(f"  знайшов -> {hit.skin_source}, обличчя {hit.face_w}")
        print(f"  промазав -> {miss.skin_source}, обличчя {miss.face_w}")
        assert hit.skin_source == "face-parsing+yunet" and hit.face_w == 300
        assert miss.skin_source != hit.skin_source, (
            "промах детектора називається так само, як влучання")
        assert "МИМО" in miss.skin_source and miss.face_w is None
    finally:
        masks_mod.FaceParser = real


# ---------------------------------------------------------------------------
# кроп обличчя і кілька облич
# ---------------------------------------------------------------------------

def test_face_crop_is_square():
    """BiSeNet бере 512x512, тож кроп розтягується до квадрата.

    Портретна рамка (сторони 0.69) стискала обличчя по горизонталі на
    третину, і модель починала вигадувати: на родинному кадрі позначала
    «шкірою» сукню в горошок, а лікування чесно ретушувало горошини.
    """
    from retouch.masks import face_crop_box
    img = np.zeros((4000, 6000, 3), np.float32)
    for box in ((3000, 1500, 500, 700), (100, 80, 400, 400), (2000, 200, 300, 900)):
        x0, y0, x1, y1 = face_crop_box(img, box)
        side = (x1 - x0, y1 - y0)
        print(f"  обличчя {box[2]}x{box[3]} -> кроп {side[0]}x{side[1]} "
              f"(ар {side[0]/side[1]:.2f})")
        assert abs(side[0] - side[1]) <= 1, f"кроп не квадратний: {side}"


def test_face_at_the_edge_is_shifted_not_squashed():
    """Обличчя біля краю має дістати квадрат, зсунутий усередину, а не
    прямокутник — інакше повертається те саме спотворення."""
    from retouch.masks import face_crop_box
    img = np.zeros((3000, 3000, 3), np.float32)
    x0, y0, x1, y1 = face_crop_box(img, (5, 5, 400, 400))
    print(f"  обличчя в кутку -> кроп ({x0},{y0})-({x1},{y1}) "
          f"{x1-x0}x{y1-y0}")
    assert x0 == 0 and y0 == 0
    assert abs((x1 - x0) - (y1 - y0)) <= 1, "у кутку кроп перестав бути квадратом"


def test_face_crop_keeps_hair_and_chin():
    """Рамка YuNet вужча за голову — розмір веде БІЛЬША сторона обличчя."""
    import inspect
    from retouch.masks import face_crop_box
    # Запас беремо з САМОЇ функції: вписати число сюди означало б завести
    # другий дефолт, а на цьому в цьому ж файлі вже наступили.
    m = inspect.signature(face_crop_box).parameters["margin"].default
    img = np.zeros((4000, 4000, 3), np.float32)
    x0, y0, x1, y1 = face_crop_box(img, (1000, 1000, 300, 900))
    side = x1 - x0
    print(f"  обличчя 300x900, запас {m} -> кроп {side} px "
          f"(мало б бути >= {int(2 * m * 900)})")
    assert side >= int(2 * m * 900) - 2, "кроп веде вузька сторона — голова обріжеться"


def test_second_face_does_not_overwrite_the_first():
    """Кропи сусідніх облич перетинаються. У перетині має вигравати те
    обличчя, чий кроп центрований на ньому самому."""
    from retouch import masks as mm

    calls = []

    class Stub(mm.FaceParser):
        def __init__(self):
            self.last_faces = []

        def _parse_whole(self, img):
            calls.append(img.shape[:2])
            # кожне «обличчя» фарбує свій кроп власним номером класу
            return np.full(img.shape[:2], len(calls), np.int32)

    img = np.zeros((2000, 3000, 3), np.float32)
    faces = [(1000, 800, 400, 400), (1200, 800, 300, 300)]   # перетинаються
    fp = Stub()
    out = fp._merge(img, faces) if hasattr(fp, "_merge") else None
    if out is None:                       # merge живе всередині parse
        orig = mm.detect_faces
        mm.detect_faces = lambda *a, **k: faces
        try:
            out = fp.parse(img, detector="x", max_faces=2, min_face=10)
        finally:
            mm.detect_faces = orig
    first, second = (out == 1).sum(), (out == 2).sum()
    print(f"  проходів моделі: {len(calls)}; пікселів від #1 {first}, від #2 {second}")
    assert len(calls) == 2, "друге обличчя не розібрано"
    assert first > 0 and second > 0, "одне з облич не потрапило в карту"
    # перетин має належати першому: воно більше й розібране раніше
    x0, y0, x1, y1 = mm.face_crop_box(img, faces[0])
    assert (out[y0:y1, x0:x1] == 2).sum() == 0, (
        "друге обличчя перезаписало територію першого")


def test_tiny_faces_are_skipped():
    """Обличчя вужче за min_face модель розбирає навмання — це шум у
    масці, а не ще одна людина."""
    from retouch import masks as mm

    class Stub(mm.FaceParser):
        def __init__(self):
            self.last_faces = []
            self.n = 0

        def _parse_whole(self, img):
            self.n += 1
            return np.full(img.shape[:2], self.n, np.int32)

    img = np.zeros((2000, 3000, 3), np.float32)
    faces = [(1000, 800, 400, 400), (200, 200, 40, 40)]
    orig = mm.detect_faces
    mm.detect_faces = lambda *a, **k: faces
    try:
        fp = Stub()
        fp.parse(img, detector="x", max_faces=4, min_face=120)
    finally:
        mm.detect_faces = orig
    print(f"  проходів моделі: {fp.n} (друге обличчя 40 px мало відсіятись)")
    assert fp.n == 1


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
