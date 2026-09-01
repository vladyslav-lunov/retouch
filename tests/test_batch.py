"""Тести пакетної обробки. Головні інваріанти:

  1. збій на одному кадрі НЕ зупиняє решту — інакше вранці буде три
     кадри з двохсот і трасування;
  2. уже оброблене пропускається, і це можна вимкнути;
  3. покадровий пресет знаходиться поруч і лягає ПОВЕРХ загального;
  4. пресет одного кадру не протікає в наступний;
  5. звіт не бреше: скільки зроблено, пропущено, зламалось.

Пункт 4 — найтонший. Config складається один раз на кадр, і якщо його
перевикористати, поріг з IMG_002.yaml лишиться на IMG_003.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retouch import batch  # noqa: E402
from retouch.pipeline import Config  # noqa: E402
from tests.synth import make_face  # noqa: E402


def _shoot(d: Path, n=3, broken=True) -> Path:
    src = d / "shoot"
    src.mkdir()
    for i, seed in enumerate(range(3, 3 + n), 1):
        img, _s, _t = make_face(h=700, w=560, face_w=430, n_spots=10, seed=seed)
        cv2.imwrite(str(src / f"IMG_{i:03d}.tif"),
                    (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16))
    if broken:
        (src / "IMG_099.tif").write_bytes(b"not an image")
    return src


def _cfg():
    return Config(force_mask=True)


def test_broken_file_does_not_stop_the_run():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        rep = batch.process(_shoot(d), d / "out", cfg_factory=_cfg)
        print(f"  {rep.text().splitlines()[0]}")
        assert len(rep.done) == 3, "хороші кадри не доїхали"
        assert len(rep.failed) == 1, "битий файл не позначено збоєм"
        assert rep.failed[0].note, "збій без пояснення"


def test_resume_skips_finished():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src = _shoot(d, broken=False)
        first = batch.process(src, d / "out", cfg_factory=_cfg)
        second = batch.process(src, d / "out", cfg_factory=_cfg)
        print(f"  перший: {len(first.done)} зроблено; "
              f"другий: {len(second.skipped)} пропущено")
        assert len(first.done) == 3 and len(second.skipped) == 3
        assert not second.done, "переробив уже зроблене"


def test_no_resume_redoes_everything():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src = _shoot(d, broken=False)
        batch.process(src, d / "out", cfg_factory=_cfg)
        again = batch.process(src, d / "out", cfg_factory=_cfg, resume=False)
        print(f"  з resume=False зроблено {len(again.done)}")
        assert len(again.done) == 3


def test_sidecar_preset_is_found_and_applied():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src = _shoot(d, broken=False)
        (src / "IMG_002.yaml").write_text(
            "detect:\n  threshold: 0.009\n", encoding="utf-8")
        seen = {}

        def cfg_factory():
            c = Config(force_mask=True)
            seen[len(seen)] = c
            return c

        rep = batch.process(src, d / "out", base_preset={"detect": {"threshold": 0.02}},
                            cfg_factory=cfg_factory)
        by_name = {i.path.name: i for i in rep.items}
        print(f"  IMG_002 пресет: '{by_name['IMG_002.tif'].preset}', "
              f"IMG_001: '{by_name['IMG_001.tif'].preset}'")
        assert by_name["IMG_002.tif"].preset == "IMG_002.yaml"
        assert not by_name["IMG_001.tif"].preset, "сайдкар знайдено там, де його нема"
        thr = [c.detect.threshold for c in seen.values()]
        print(f"  пороги по кадрах: {thr}")
        assert thr == [0.02, 0.009, 0.02], "покадровий пресет протік або не спрацював"


def test_preset_does_not_leak_between_frames():
    """Пункт 4: свіжий Config на кожен кадр."""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src = _shoot(d, broken=False)
        (src / "IMG_001.yaml").write_text(
            "strength: 0.3\n", encoding="utf-8")
        made = []
        rep = batch.process(src, d / "out",
                            cfg_factory=lambda: made.append(Config(force_mask=True)) or made[-1])
        print(f"  strength по кадрах: {[c.strength for c in made]}")
        assert made[0].strength == 0.3
        assert all(c.strength == 1.0 for c in made[1:]), "пресет протік далі"
        assert len(rep.done) == 3


def test_find_inputs_ignores_non_images():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src = _shoot(d, broken=False)
        (src / "notes.txt").write_text("x", encoding="utf-8")
        (src / "IMG_001.yaml").write_text("x: 1", encoding="utf-8")
        got = [p.name for p in batch.find_inputs(src)]
        print(f"  знайдено: {got}")
        assert all(g.endswith(".tif") for g in got), "у список потрапило зайве"
        assert len(got) == 3


def test_report_counts_are_consistent():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        rep = batch.process(_shoot(d), d / "out", cfg_factory=_cfg)
        total = len(rep.done) + len(rep.skipped) + len(rep.failed)
        print(f"  {rep.text().splitlines()[0]} (усього {len(rep.items)})")
        assert total == len(rep.items), "звіт не сходиться"
        assert all(i.seconds >= 0 for i in rep.items)


def test_xmp_is_read_per_frame():
    """У теці зі зйомки в кожного кадру свій кроп — це те, чого пресет
    на зйомку дати не може."""
    side = (Path(__file__).resolve().parent / "fixtures" / "acr_sidecar.xmp"
            ).read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src = _shoot(d, n=3, broken=False)
        for name in ("IMG_001", "IMG_003"):
            (src / f"{name}.xmp").write_text(side, encoding="utf-8")
        rep = batch.process(src, d / "out", cfg_factory=_cfg, use_xmp=True)
        sizes = {}
        for item in rep.done:
            flat = d / "out" / f"{item.path.stem}_99_flat.tif"
            im = cv2.imread(str(flat), cv2.IMREAD_UNCHANGED)
            sizes[item.path.stem] = (im.shape[1], im.shape[0])
        marks = {i.path.stem: i.preset for i in rep.items}
        print(f"  розміри: {sizes}")
        print(f"  джерела: {marks}")
    assert sizes["IMG_001"] == sizes["IMG_003"], "однакові XMP дали різні кадри"
    assert sizes["IMG_002"] != sizes["IMG_001"], (
        "кадр без сайдкара теж кроплено — XMP протік між кадрами")
    assert "xmp" in marks["IMG_001"].lower(), "у звіті не видно, звідки взято"
    assert not marks["IMG_002"], "кадру без сайдкара приписано джерело"


def test_shoot_preset_beats_xmp():
    """Стиль на зйомку має діяти й на кадрах із сайдкаром."""
    side = (Path(__file__).resolve().parent / "fixtures" / "acr_sidecar.xmp"
            ).read_text(encoding="utf-8")
    seen = []
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src = _shoot(d, n=2, broken=False)
        for f in sorted(src.glob("*.tif")):
            f.with_suffix(".xmp").write_text(side, encoding="utf-8")
        batch.process(src, d / "out", cfg_factory=_cfg, use_xmp=True,
                      base_preset={"develop": {"contrast": 0.4}},
                      on_item=lambda i, r: seen.append(i))
        # contrast з XMP = 0.12, з пресету зйомки = 0.4; має перемогти пресет
        cfg = _cfg()
        from retouch import presets as pm
        from retouch import xmp as xm
        pre, _r, _w = xm.from_image(sorted(src.glob("*.tif"))[0])
        pm.apply(cfg, pm.merge(pre, {"develop": {"contrast": 0.4}}))
        print(f"  contrast після накладання: {cfg.develop.contrast}, "
              f"кроп з XMP лишився: {bool(cfg.develop.crop)}")
    assert cfg.develop.contrast == 0.4, "пресет зйомки не переміг XMP"
    assert cfg.develop.crop, "пресет зйомки затер кроп, якого не згадував"
    assert all(i.status == "done" for i in seen), [i.note for i in seen]


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
