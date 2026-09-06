"""Тести досьє на зйомку. Головні інваріанти:

  1. звіт показує РОЗКИД, а не середнє — заради цього він і пишеться;
  2. крайні кадри обираються за числами, а не за порядком у теці;
  3. вердикт «одного пресету досить» відповідає даним;
  4. один зіпсований файл не валить досьє на всю зйомку;
  5. кропи беруться з крайніх кадрів, бо пресет ламається на краях.

Пункт 1 — не косметика. spec.md §6.2.1: на 44 кадрах однієї зйомки
фіксований поріг дав від 0% до 24% торкнутої шкіри. Середнє по такій
вибірці відповідає на питання, якого ніхто не ставив.
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

from tests.synth import make_face  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import shoot_brief as sb  # noqa: E402


def _shoot(d: Path, n: int = 3, broken: bool = False) -> Path:
    src = d / "shoot"
    src.mkdir()
    for i in range(1, n + 1):
        img, _s, _t = make_face(h=700, w=560, face_w=430 + i * 40,
                                n_spots=4 + i * 4, seed=i)
        cv2.imwrite(str(src / f"IMG_{i:03d}.tif"),
                    (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16))
    if broken:
        (src / "IMG_099.tif").write_bytes(b"not an image")
    return src


def test_spread_is_min_median_max_not_a_mean():
    rows = [{"t": 0.008}, {"t": 0.012}, {"t": 0.025}, {"t": None}]
    sp = sb.spread(rows, "t")
    print(f"  {sp}")
    assert sp == {"min": 0.008, "median": 0.012, "max": 0.025, "n": 3}
    assert "mean" not in sp, "середнє ховає саме те, заради чого звіт пишеться"
    assert sb.spread(rows, "нема_такого") is None


def test_extremes_are_picked_by_numbers():
    """Крайні кадри — це де пресет ламається, а не перший і останній."""
    rows = [
        {"file": "a", "recommended_threshold": 0.012, "face_w": 900,
         "blobs": 10, "skin_frac": 0.05},
        {"file": "b", "recommended_threshold": 0.025, "face_w": 200,
         "blobs": 400, "skin_frac": 0.01},
        {"file": "c", "recommended_threshold": 0.008, "face_w": 500,
         "blobs": 50, "skin_frac": 0.09},
    ]
    ex = sb.extremes(rows)
    print(f"  {ex}")
    assert ex["найжорсткіший поріг"] == "b"
    assert ex["найм'якший поріг"] == "c"
    assert ex["найбільше обличчя"] == "a"
    assert ex["найменше обличчя"] == "b"
    assert ex["найбільше знахідок"] == "b"
    assert ex["найбільша маска"] == "c"


def test_extremes_survive_missing_values():
    """Кадр без обличчя не має валити вибір крайніх."""
    rows = [{"file": "a", "recommended_threshold": 0.012, "face_w": None,
             "blobs": 3, "skin_frac": None},
            {"file": "b", "recommended_threshold": 0.02, "face_w": 300,
             "blobs": 9, "skin_frac": 0.04}]
    ex = sb.extremes(rows)
    print(f"  {ex}")
    assert ex["найбільше обличчя"] == "b"
    assert "найм'якший поріг" in ex


def test_writes_all_four_parts():
    """Числа, проза, аркуш і кропи — кожна частина відповідає за своє."""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src = _shoot(d, 3)
        out = d / "brief"
        rc = sb.main([str(src), "-o", str(out), "--target", "0.05"])
        names = sorted(p.name for p in out.iterdir())
        print(f"  код {rc}; {', '.join(names)}")
        assert rc == 0
        assert "shoot.json" in names and "shoot.md" in names
        assert "contact.jpg" in names
        assert any(n.startswith("crop_") for n in names), "кропів 1:1 немає"


def test_verdict_matches_the_data():
    """«Одного пресету досить» має відповідати числам, а не настрою."""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        out = d / "brief"
        sb.main([str(_shoot(d, 3)), "-o", str(out), "--target", "0.05"])
        data = json.loads((out / "shoot.json").read_text(encoding="utf-8"))
        ths = data["distinct_thresholds"]
        print(f"  порогів {ths}, вердикт «досить одного»: "
              f"{data['one_preset_enough']}")
        assert data["one_preset_enough"] == (len(ths) == 1)
        assert data["spread"]["recommended_threshold"]["n"] == data["shoot"]["frames"]
        md = (out / "shoot.md").read_text(encoding="utf-8")
        assert "target_coverage" in md, "не сказано головного: писати ЦІЛЬ"


def test_one_broken_file_does_not_kill_the_shoot():
    """Одна ніч роботи не має гинути через один битий файл."""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        out = d / "brief"
        rc = sb.main([str(_shoot(d, 2, broken=True)), "-o", str(out),
                      "--target", "0.05"])
        data = json.loads((out / "shoot.json").read_text(encoding="utf-8"))
        print(f"  код {rc}; кадрів {data['shoot']['frames']}, "
              f"пропущено {[s['file'] for s in data['shoot']['skipped']]}")
        assert rc == 0 and data["shoot"]["frames"] == 2
        assert len(data["shoot"]["skipped"]) == 1


def test_empty_folder_refuses_clearly():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / "empty"
        d.mkdir()
        r = subprocess.run([sys.executable, "scripts/shoot_brief.py", str(d),
                            "-o", str(Path(t) / "b")],
                           capture_output=True, text=True, cwd=str(ROOT))
        print(f"  код {r.returncode}: {r.stderr.strip()[:50]}")
        assert r.returncode == 1 and r.stderr.strip()


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
