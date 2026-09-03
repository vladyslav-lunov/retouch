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

        def parse(self, img, detector=None, margin=0.85):
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
