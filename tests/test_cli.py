"""Тести CLI. Головні інваріанти:

  1. явний прапорець виграє в YAML і в пресеті — колись було навпаки, і
     `--config c.yaml --threshold 0.02` мовчки працював з порогом з файлу;
  2. пресети накладаються в порядку, а прапорці б'ють усіх;
  3. дефолти беруться з дата-класів, а не дублюються в сигнатурі;
  4. --schema працює без вхідного файлу;
  5. відмови (RAW, відсутній файл) — текст і код 1, а не трасування.

CLI запускається окремим процесом: перевіряємо те, що справді набирає
людина, включно з кодом повернення.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retouch.blemish import DetectParams  # noqa: E402
from retouch.cli import build_config, build_parser  # noqa: E402
from retouch.pipeline import Config  # noqa: E402
from tests.synth import make_face  # noqa: E402


def _run(*args, expect=0):
    r = subprocess.run([sys.executable, "-m", "retouch.cli", *args],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == expect, (
        f"код {r.returncode}, чекали {expect}\nSTDOUT{r.stdout[-400:]}\n"
        f"STDERR{r.stderr[-400:]}")
    return r


def _args(**kw):
    """Namespace з РЕАЛЬНОГО парсера, а не переписаний тут.

    Перша версія дублювала список полів, і кожен новий прапорець валив
    тести з єдиної причини — що вони про нього не знали.
    """
    ns = build_parser().parse_args([])
    for k, v in kw.items():
        assert hasattr(ns, k), f"немає такого прапорця: {k}"
        setattr(ns, k, v)
    return ns


def _fixture(d: Path) -> Path:
    img, _s, _t = make_face(h=900, w=700, face_w=560, n_spots=12, seed=3)
    p = d / "T.tif"
    cv2.imwrite(str(p), (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16))
    return p


def test_flag_beats_yaml():
    """Найдорожча помилка: YAML мовчки з'їдав явний прапорець."""
    with tempfile.TemporaryDirectory() as t:
        y = Path(t) / "c.yaml"
        y.write_text("detect:\n  threshold: 0.012\n", encoding="utf-8")
        cfg = build_config(_args(config=str(y), threshold=0.025))
        print(f"  yaml 0.012 + прапорець 0.025 -> {cfg.detect.threshold}")
        assert cfg.detect.threshold == 0.025, "YAML переміг прапорець"
        cfg2 = build_config(_args(config=str(y)))
        assert cfg2.detect.threshold == 0.012, "без прапорця YAML має діяти"


def test_flag_beats_preset():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "p.yaml"
        p.write_text("detect:\n  threshold: 0.014\n", encoding="utf-8")
        cfg = build_config(_args(preset=[str(p)], threshold=0.03))
        print(f"  пресет 0.014 + прапорець 0.03 -> {cfg.detect.threshold}")
        assert cfg.detect.threshold == 0.03


def test_presets_stack_left_to_right():
    with tempfile.TemporaryDirectory() as t:
        a = Path(t) / "a.yaml"; a.write_text(
            "detect:\n  threshold: 0.010\n  min_area: 5\n", encoding="utf-8")
        b = Path(t) / "b.yaml"; b.write_text(
            "detect:\n  threshold: 0.020\n", encoding="utf-8")
        cfg = build_config(_args(preset=[str(a), str(b)]))
        print(f"  threshold {cfg.detect.threshold} (з другого), "
              f"min_area {cfg.detect.min_area} (з першого)")
        assert cfg.detect.threshold == 0.020, "пізніший пресет не переміг"
        assert cfg.detect.min_area == 5, "перший пресет затерто цілком"


def test_defaults_come_from_dataclasses():
    cfg = build_config(_args())
    d = DetectParams()
    print(f"  threshold {cfg.detect.threshold} == дата-клас {d.threshold}")
    assert cfg.detect.threshold == d.threshold
    assert cfg.search_radius == Config().search_radius
    assert cfg.strength == Config().strength


def test_no_skin_mask_flag():
    assert build_config(_args(no_skin_mask=True)).use_skin_mask is False
    assert build_config(_args()).use_skin_mask is True
    print("  --no-skin-mask вимикає маску, без нього вона є")


def test_schema_runs_without_input():
    r = _run("--schema")
    d = json.loads(r.stdout)
    print(f"  розділи: {list(d['sections'])}")
    assert "develop" in d["sections"] and "detect" in d["sections"]
    assert d["meta"]["why"], "у схемі немає поля why"


def test_missing_file_exits_1_with_text():
    r = _run("/nope/absent.tif", "--dry-run", expect=1)
    print(f"  stderr: {r.stderr.strip().splitlines()[-1][:60]}")
    assert "нема" in r.stderr.lower()
    assert "Traceback" not in r.stderr, "показано трасування замість тексту"


def test_raw_without_decoder_message_is_actionable():
    """RAW тепер читається; якщо ні — повідомлення має бути дієвим."""
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "x.cr3"
        p.write_bytes(b"not really a raw file")
        r = _run(str(p), "--dry-run", expect=1)
        print(f"  {r.stderr.strip().splitlines()[0][:70]}")
        assert "Traceback" not in r.stderr


def test_dry_run_writes_nothing():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        p = _fixture(d)
        out = d / "out"
        _run(str(p), "--dry-run", "-o", str(out), "--force-mask")
        print(f"  тека виводу створена: {out.exists()}")
        assert not out.exists(), "--dry-run щось записав"


def test_full_run_writes_expected_files():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        p = _fixture(d)
        out = d / "out"
        r = _run(str(p), "-o", str(out), "--force-mask")
        names = sorted(x.name for x in out.iterdir())
        print(f"  {', '.join(names)}")
        assert any(n.endswith("_00_base.tif") for n in names)
        assert any(n.endswith("_99_flat.tif") for n in names)
        assert "[heal]" in r.stdout


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
