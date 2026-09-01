"""Тести пресетів. Головні інваріанти:

  1. пресет ЧАСТКОВИЙ: не згадане не чіпається;
  2. пресети накладаються вглиб, а не поверхнево;
  3. незнайомий ключ не валить роботу, але й не мовчить;
  4. кожне поле схеми має опис — інакше агент не зрозуміє параметр;
  5. пресет переживає запис і читання;
  6. розділ `tools` приймає ту форму, яку показує схема.

Пункт 6 з'явився з живої помилки: схема оголошувала розділи «tools.teeth»,
а `apply()` їх не приймала. Обидва очевидні прочитання схеми мовчки не
спрацьовували, причому друге — найгірше з можливих: інструмент таки
запускався, але з дефолтами, тож зовні виглядало як застосований пресет.

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


# ---------------------------------------------------------------------------
# розділ tools
# ---------------------------------------------------------------------------

def test_tools_accepts_every_form_the_schema_implies():
    """Список, словник із параметрами, `true` — усі три працюють."""
    for data, want_names, want_params in [
        ({"tools": ["teeth", "mattify"]}, ("teeth", "mattify"), {}),
        ({"tools": {"teeth": {"strength": 0.4}}}, ("teeth",),
         {"teeth": {"strength": 0.4}}),
        ({"tools": {"teeth": True, "mattify": {"radius": 30}}},
         ("teeth", "mattify"), {"mattify": {"radius": 30}}),
    ]:
        cfg = Config()
        notes = presets.apply(cfg, data)
        print(f"  {str(data)[:44]:46} -> {cfg.tools} {cfg.tool_params}")
        assert not notes, f"чиста форма дала зауваження: {notes}"
        assert cfg.tools == want_names, f"{cfg.tools} != {want_names}"
        assert cfg.tool_params == want_params


def test_tools_order_is_application_order():
    """Порядок ключів у файлі — це порядок застосування, а не абищо."""
    cfg = Config()
    presets.apply(cfg, {"tools": {"mattify": {}, "teeth": {}, "eye_vessels": {}}})
    print(f"  {cfg.tools}")
    assert cfg.tools == ("mattify", "teeth", "eye_vessels")


def test_tool_can_be_switched_off_by_a_later_preset():
    """Пресет кадру має вміти зняти інструмент, увімкнений на зйомці."""
    cfg = Config()
    presets.apply(cfg, {"tools": {"teeth": {"strength": 0.4}, "mattify": {}}})
    presets.apply(cfg, {"tools": {"mattify": False}})
    print(f"  після вимкнення: {cfg.tools}, параметри {cfg.tool_params}")
    assert cfg.tools == ("teeth",), cfg.tools
    assert cfg.tool_params["teeth"] == {"strength": 0.4}, "зачепило чужі параметри"


def test_typo_in_tools_is_reported_not_swallowed():
    """Мовчазний неправильний результат гірший за помилку (§1)."""
    for data, expect in [({"tools": {"nosuch": {}}}, "невідомий інструмент"),
                         ({"tools": {"teeth": {"nope": 1}}}, "невідомий параметр")]:
        cfg = Config()
        notes = presets.apply(cfg, data)
        print(f"  {str(data)[:40]:42} -> {notes}")
        assert notes and expect in notes[0], f"проковтнуло: {data}"
    assert Config().tools == (), "дефолт не має вмикати інструментів"


def test_merge_goes_two_levels_deep():
    """`tools` вкладений двічі — зупинка на першому рівні губить зйомку."""
    shoot = {"tools": {"teeth": {"strength": 0.3, "yellow": 0.7}}}
    frame = {"tools": {"teeth": {"yellow": 0.5}, "mattify": {}}}
    out = presets.merge(shoot, frame)
    print(f"  {out}")
    assert out["tools"]["teeth"] == {"strength": 0.3, "yellow": 0.5}, \
        "правка одного параметра затерла інструмент цілком"
    assert "mattify" in out["tools"]


def test_merge_replaces_lists_whole():
    """Крива й набір класів — неподільні: злити дві криві не можна."""
    out = presets.merge({"mask": {"skin_classes": ["skin", "nose", "neck"]}},
                        {"mask": {"skin_classes": ["skin"]}})
    print(f"  {out['mask']['skin_classes']}")
    assert out["mask"]["skin_classes"] == ["skin"]


def test_schema_example_applies_without_a_single_note():
    """Приклад у схемі — це те, що агент скопіює першим.

    Якщо він не застосовується начисто, схема вчить писати непридатні
    пресети, і дізнаємось ми про це з кадрів, а не з тестів.
    """
    cfg = Config()
    notes = presets.apply(cfg, presets.schema()["example"])
    print(f"  зауважень: {notes}; tools={cfg.tools}, db={cfg.dodgeburn_on}")
    assert not notes, f"приклад зі схеми не застосовується: {notes}"
    assert cfg.tools and cfg.dodgeburn_on, "приклад нічого не вмикає — він марний"


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
