"""Тести пресетів. Головні інваріанти:

  1. пресет ЧАСТКОВИЙ: не згадане не чіпається;
  2. пресети накладаються вглиб, а не поверхнево;
  3. незнайомий ключ не валить роботу, але й не мовчить;
  4. кожне поле схеми має опис — інакше агент не зрозуміє параметр;
  5. пресет переживає запис і читання.

Пункт 4 — не косметика. spec.md §1.2 вимагає семантичності: агент має
обґрунтувати своє рішення в полі `why`, а обґрунтувати параметр, значення
якого йому не пояснили, він не може. Тест ловить нове поле, додане без
опису, у той самий день, коли його додали.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retouch import presets  # noqa: E402
from retouch.pipeline import Config  # noqa: E402


def test_preset_is_partial():
    """Не згадане лишається як було."""
    cfg = Config()
    before = (cfg.detect.min_area, cfg.strength, cfg.search_radius)
    presets.apply(cfg, {"detect": {"threshold": 0.02}})
    after = (cfg.detect.min_area, cfg.strength, cfg.search_radius)
    print(f"  threshold {cfg.detect.threshold}, решта: {before} -> {after}")
    assert cfg.detect.threshold == 0.02
    assert before == after, "пресет зачепив те, чого не згадував"


def test_merge_is_deep():
    """Стиль зйомки + уточнення кадру мають скластися, а не затерти одне одного."""
    shoot = {"detect": {"threshold": 0.014}, "mask": {"erode": 8}}
    frame = {"detect": {"min_area": 6}}
    m = presets.merge(shoot, frame)
    print(f"  detect після злиття: {m['detect']}")
    assert m["detect"] == {"threshold": 0.014, "min_area": 6}, "поверхневе злиття"
    assert m["mask"] == {"erode": 8}, "розділ, якого не було в другому, загубився"


def test_later_preset_wins():
    a = {"detect": {"threshold": 0.010}}
    b = {"detect": {"threshold": 0.020}}
    assert presets.merge(a, b)["detect"]["threshold"] == 0.020
    print("  пізніший виграє: 0.010 -> 0.020")


def test_unknown_keys_warn_but_do_not_break():
    """Пресет від агента чи старішої версії не має втратити все через одну стрічку."""
    cfg = Config()
    notes = presets.apply(cfg, {"detect": {"threshold": 0.02, "нема_такого": 1},
                                "вигаданий_розділ": {"x": 1}})
    print(f"  зауважень: {len(notes)} -> {notes}")
    assert cfg.detect.threshold == 0.02, "через зайвий ключ втрачено справжній"
    assert len(notes) == 2, "про невідомі ключі промовчали"


def test_metadata_is_not_a_parameter():
    """name/why/for — це метадані, вони не мають потрапляти в Config."""
    cfg = Config()
    notes = presets.apply(cfg, {"name": "тест", "why": "бо так", "for": "кадр 1"})
    print(f"  зауважень на метадані: {notes or 'немає'}")
    assert not notes, "метадані сприйнято як параметри"


def test_tuple_fields_survive_yaml():
    """YAML не знає кортежів, а дата-класи їх тримають."""
    cfg = Config()
    presets.apply(cfg, {"mask": {"skin_classes": ["skin", "nose"]}})
    print(f"  skin_classes = {cfg.mask.skin_classes} ({type(cfg.mask.skin_classes).__name__})")
    assert cfg.mask.skin_classes == ("skin", "nose")


def test_every_schema_field_is_documented():
    """Поле без опису агент не зможе обґрунтувати (spec.md §1.2)."""
    s = presets.schema()
    missing = [f"{sec}.{k}" for sec, fields in s["sections"].items()
               for k, v in fields.items() if not v["doc"].strip()]
    total = sum(len(f) for f in s["sections"].values())
    print(f"  полів у схемі: {total}, без опису: {len(missing)}")
    assert not missing, f"без опису: {missing}"


def test_schema_matches_dataclasses():
    """Схема будується з коду, тож не може від нього відстати."""
    s = presets.schema()
    assert "threshold" in s["sections"]["detect"]
    assert s["sections"]["detect"]["threshold"]["default"] == Config().detect.threshold
    assert "skin" in s["vocabulary"]["mask.skin_classes / mask.exclude_classes"]
    print(f"  detect.threshold типово {s['sections']['detect']['threshold']['default']}, "
          f"словник класів: {len(s['vocabulary']['mask.skin_classes / mask.exclude_classes'])}")


def test_save_load_roundtrip():
    data = {"name": "тест", "why": "перевірка запису",
            "detect": {"threshold": 0.017}, "mask": {"skin_classes": ["skin"]}}
    with tempfile.TemporaryDirectory() as t:
        p = presets.save(Path(t) / "p.yaml", data)
        back = presets.load(p)
    print(f"  {p.name}: {back}")
    assert back == data, "пресет не пережив запис"


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
