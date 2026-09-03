"""Тести зведення калібрування. Головні інваріанти:

  1. розкид рахується по КАДРАХ, а не змішується з розкидом по порогах;
  2. кадри з різним розміром обличчя не усереднюються разом;
  3. таблиця чесно каже, що кадрів мало, коли їх мало;
  4. `--from-json` дає те саме, що повний прогін.

Пункт 1 — не педантизм. Уся цінність цієї таблиці в колонці «розкид»:
якщо на одному кадрі поріг дає 2% шкіри, а на іншому 22%, то одного
дефолту не існує, і це головний висновок. Помилка в цій колонці
перетворює висновок на протилежний, а помітити її нічим.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import calibrate  # noqa: E402

THS = [0.008, 0.012, 0.018]


def _frame(name, face_w, per_threshold, skin_frac=0.02, radius=4.0):
    """Кадр у тому вигляді, в якому його кладе measure()."""
    return {
        "file": name, "w": 4000, "h": 6000, "skin_source": "face-parsing+yunet",
        "skin_frac": skin_frac, "radius_px": radius, "face_w": face_w,
        "sweep": [{"threshold": t, "blobs": b, "touched_of_skin": s,
                   "touched_of_frame": s / 50, "suspect_share": susp,
                   "by_class": {}}
                  for t, (b, s, susp) in zip(THS, per_threshold)],
    }


def _report(frames, out):
    # verbose=False: інакше кожен тест друкує повний звіт, і в наборі
    # не видно власного виводу тестів.
    return calibrate.build_report(frames, THS, [], {"skin_classes": "skin,nose"},
                                  out, verbose=False)


def test_spread_is_across_frames_not_thresholds():
    """Розкид на порозі — це min і max ПО КАДРАХ саме на цьому порозі."""
    frames = [
        _frame("a.CR3", 800, [(100, 0.20, 0.0), (50, 0.10, 0.0), (10, 0.02, 0.0)]),
        _frame("b.CR3", 820, [(300, 0.02, 0.0), (150, 0.01, 0.0), (30, 0.005, 0.0)]),
    ]
    with tempfile.TemporaryDirectory() as t:
        out = _report(frames, t)
    row = next(r for r in out["aggregate"] if r["threshold"] == 0.012)
    print(f"  0.012: медіана {row['touched_of_skin_median']}, "
          f"розкид {row['touched_of_skin_min']}–{row['touched_of_skin_max']}, "
          f"плям {row['blobs_min']}–{row['blobs_max']}")
    assert row["touched_of_skin_min"] == 0.01 and row["touched_of_skin_max"] == 0.10
    assert row["blobs_min"] == 50 and row["blobs_max"] == 150, (
        "у розкид плям потрапили інші пороги")


def test_frames_are_split_by_face_size():
    """Обличчя 200 px і 900 px — різні задачі, а не розкид одної."""
    frames = [
        _frame("small.CR3", 200, [(5, 0.05, 0.0), (3, 0.02, 0.0), (1, 0.01, 0.0)]),
        _frame("mid.CR3", 500, [(200, 0.20, 0.0), (120, 0.09, 0.0), (30, 0.04, 0.0)]),
        _frame("big.CR3", 900, [(280, 0.11, 0.0), (110, 0.04, 0.0), (36, 0.01, 0.0)]),
    ]
    with tempfile.TemporaryDirectory() as t:
        out = _report(frames, t)
    seg = {b["bucket"]: b for b in out["by_face_size"]}
    print("  " + "; ".join(f"{k}: n={v['n']}, обличчя {v['face_w']}"
                           for k, v in seg.items()))
    assert len(seg) == 3, f"кошики злилися: {list(seg)}"
    for b in out["by_face_size"]:
        assert b["n"] == 1, f"{b['bucket']} зібрав {b['n']} кадрів замість 1"
    small = next(b for b in out["by_face_size"] if "дрібне" in b["bucket"])
    assert small["sweep"][0]["blobs_median"] == 5, "дрібне змішалось з рештою"


def test_report_says_when_there_are_too_few_frames():
    """§6.2 вимагає 20-30. Три кадри — це не калібрування, і таблиця
    не має вдавати, ніби це воно."""
    frames = [_frame(f"{i}.CR3", 800, [(10, 0.02, 0.0)] * 3) for i in range(3)]
    with tempfile.TemporaryDirectory() as t:
        _report(frames, t)
        md = (Path(t) / "calib.md").read_text(encoding="utf-8")
    print(f"  {[l for l in md.splitlines() if 'МАЛО' in l]}")
    assert "ЦЬОГО МАЛО" in md, "мовчить про те, що вибірки не досить"

    many = [_frame(f"{i}.CR3", 800, [(10, 0.02, 0.0)] * 3) for i in range(22)]
    with tempfile.TemporaryDirectory() as t:
        _report(many, t)
        md = (Path(t) / "calib.md").read_text(encoding="utf-8")
    assert "ЦЬОГО МАЛО" not in md, "лається на достатню вибірку"


def test_suspect_share_max_is_reported_not_hidden():
    """Один кадр з браком не має ховатися за медіаною нуль."""
    frames = [_frame(f"ok{i}.CR3", 800, [(10, 0.02, 0.0)] * 3) for i in range(5)]
    frames.append(_frame("bad.CR3", 800, [(10, 0.02, 0.6)] * 3))
    with tempfile.TemporaryDirectory() as t:
        out = _report(frames, t)
    row = out["aggregate"][0]
    print(f"  медіана {row['suspect_median']}, максимум {row['suspect_max']}")
    assert row["suspect_median"] == 0.0
    assert row["suspect_max"] == 0.6, "поганий кадр зник за медіаною"


def test_from_json_gives_the_same_tables():
    """Перебудова зі збереженого JSON має збігатися з прогоном."""
    frames = [
        _frame("a.CR3", 400, [(200, 0.20, 0.0), (120, 0.09, 0.0), (30, 0.04, 0.0)]),
        _frame("b.CR3", 900, [(280, 0.11, 0.0), (110, 0.04, 0.0), (36, 0.01, 0.0)]),
    ]
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        first = _report(frames, d / "one")
        (d / "one" / "calib.json").exists()
        import argparse
        rc = calibrate.main(["--from-json", str(d / "one"), "-o", str(d / "two")])
        second = json.loads((d / "two" / "calib.json").read_text(encoding="utf-8"))
    print(f"  код {rc}; агрегатів {len(first['aggregate'])} -> "
          f"{len(second['aggregate'])}")
    assert rc == 0
    assert first["aggregate"] == second["aggregate"], "перебудова дала інше"
    assert first["by_face_size"] == second["by_face_size"]


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
