"""Тести читання XMP. Головні інваріанти:

  1. у пресет потрапляє ТІЛЬКИ те, що ми справді застосовуємо;
  2. ярус не завищується: те, що не точно, не називається точним;
  3. звіт покриває кожен прочитаний параметр — мовчазних немає;
  4. сміття на вході дає зрозумілу помилку, а не трасування;
  5. отриманий пресет застосовується до Config без зауважень.

Пункт 1 — головний. Найгірший можливий результат цієї гілки не помилка
в числі, а кадр, який ВИГЛЯДАЄ проявленим як в ACR, хоча половину
параметрів ми не вміємо (PLAN.md §5). Тому перевіряється не лише те, що
пресет містить, а й те, чого він НЕ містить.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retouch import xmp  # noqa: E402
from retouch.pipeline import Config  # noqa: E402
from retouch import presets  # noqa: E402

HERE = Path(__file__).resolve().parent
SIDECAR = HERE / "fixtures" / "acr_sidecar.xmp"

HEAD = ('<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about=""'
        ' xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"'
        ' xmlns:tiff="http://ns.adobe.com/tiff/1.0/"')
TAIL = "</rdf:Description></rdf:RDF></x:xmpmeta>"


def _xmp(attrs: dict, body: str = "") -> str:
    a = " ".join(f'crs:{k}="{v}"' for k, v in attrs.items())
    return f"{HEAD} {a}>{body}{TAIL}"


def _write(d: Path, text: str, name: str = "T.xmp") -> Path:
    p = d / name
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------

def test_reads_attributes_and_child_elements():
    """ACR пише і так, і так — залежно від версії й типу значення."""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        body = ("<crs:Exposure2012>+0.75</crs:Exposure2012>"
                "<crs:ToneCurvePV2012><rdf:Seq>"
                "<rdf:li>0, 0</rdf:li><rdf:li>128, 150</rdf:li>"
                "<rdf:li>255, 255</rdf:li></rdf:Seq></crs:ToneCurvePV2012>")
        raw = xmp.read(_write(d, _xmp({"Contrast2012": "+20"}, body)))
        print(f"  атрибут Contrast2012={raw.get('Contrast2012')}, "
              f"елемент Exposure2012={raw.get('Exposure2012')}, "
              f"Seq={raw.get('ToneCurvePV2012')}")
        assert raw["Contrast2012"] == "+20", "атрибут не прочитано"
        assert raw["Exposure2012"] == "+0.75", "дочірній елемент не прочитано"
        assert isinstance(raw["ToneCurvePV2012"], list), "rdf:Seq не прочитано"


def test_crop_keeps_acr_fractions_as_is():
    """Рамка ACR у частках — це рівно наш DevelopParams.crop."""
    raw = xmp.read(SIDECAR)
    pre, rep = xmp.to_preset(raw)
    crop = pre["develop"]["crop"]
    print(f"  {crop}")
    assert crop == [0.125, 0.083333, 0.875, 0.916667], crop


def test_crop_without_angle_is_exact_with_angle_is_not():
    """Ярус не завищується.

    Без кута рамка переноситься один в один. З кутом — ні: в ACR вона
    лежить у ВИРІВНЯНОМУ кадрі, а ми ріжемо до повороту. Назвати це
    точним означало б пообіцяти те, чого немає.
    """
    box = {"HasCrop": "True", "CropLeft": "0.1", "CropTop": "0.1",
           "CropRight": "0.9", "CropBottom": "0.9"}
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        _, straight = xmp.to_preset(xmp.read(_write(d, _xmp(box), "a.xmp")))
        _, tilted = xmp.to_preset(xmp.read(
            _write(d, _xmp({**box, "CropAngle": "-2.5"}), "b.xmp")))
    print(f"  без кута: {'crop' in straight.exact}; "
          f"з кутом: exact={'crop' in tilted.exact}, "
          f"approx={'crop' in tilted.approx}")
    assert "crop" in straight.exact, "рівна рамка мала бути точною"
    assert "crop" not in tilted.exact and "crop" in tilted.approx, (
        "рамку з кутом названо точною — це обіцянка, якої ми не тримаємо")


def test_tone_curve_is_scaled_and_linear_is_dropped():
    """0..255 -> 0..1, а пряма — це «нічого не робити»."""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        seq = lambda pts: ("<crs:ToneCurvePV2012><rdf:Seq>" + "".join(
            f"<rdf:li>{x}, {y}</rdf:li>" for x, y in pts)
            + "</rdf:Seq></crs:ToneCurvePV2012>")
        custom, _ = xmp.to_preset(xmp.read(_write(
            d, _xmp({}, seq([(0, 0), (64, 48), (128, 140), (255, 255)])), "c.xmp")))
        linear, _ = xmp.to_preset(xmp.read(_write(
            d, _xmp({}, seq([(0, 0), (128, 128), (255, 255)])), "l.xmp")))
    curve = custom["develop"]["curve"]
    print(f"  крива: {[[round(x, 3), round(y, 3)] for x, y in curve]}")
    print(f"  пряма дала develop={linear.get('develop')}")
    assert curve[0] == [0.0, 0.0] and curve[-1] == [1.0, 1.0]
    assert abs(curve[1][0] - 64 / 255) < 1e-9, "не поділили на 255"
    assert "curve" not in linear.get("develop", {}), (
        "пряму записано кривою — зайва інтерполяція там, де правки не було")


def test_nothing_from_tier_c_leaks_into_the_preset():
    """Головний інваріант: у пресеті лише те, що ми справді робимо."""
    raw = xmp.read(SIDECAR)
    pre, rep = xmp.to_preset(raw)
    dev = pre.get("develop", {})
    print(f"  у пресеті: {sorted(dev)}")
    print(f"  не застосовано: {sorted(rep.ignored)}")
    forbidden = {"clarity", "dehaze", "texture", "vibrance", "sharpness",
                 "highlights", "shadows", "whites", "blacks",
                 "temperature", "tint"}
    for key in dev:
        assert key.lower() not in forbidden, f"{key} потрапив у пресет"
    assert rep.ignored, "звіт порожній — ярус C кудись подівся"
    assert "Clarity2012" in rep.ignored, "Clarity не позначено як незастосовану"


def test_every_nonzero_parameter_is_mentioned_in_the_report():
    """Ненульовий параметр не має пройти мовчки.

    Перевіряється ТЕКСТ звіту, а не набір ключів: частина параметрів
    згадується гуртом (Highlights/Shadows/Blacks — одним рядком про
    параметричну криву, бо причина в них спільна), і вимагати окремий
    рядок на кожен означало б плодити повтори. Фотографу важливо, що він
    про них дізнався, а не під яким заголовком.

    Нульові не рахуємо: нуль в ACR — це «не чіпав».
    """
    raw = xmp.read(SIDECAR)
    _pre, rep = xmp.to_preset(raw)
    text = " ".join(rep.lines())
    housekeeping = {"Version", "ProcessVersion", "ToneCurveName2012",
                    # геометрія звітується під іменами crop/rotate
                    "HasCrop", "CropTop", "CropLeft", "CropRight",
                    "CropBottom", "CropAngle",
                    # свідомо приглушені при As Shot — див. окремий тест
                    "Temperature", "Tint"}
    # Нейтральне значення не в усіх параметрів нуль: у Orientation це 1
    # («як знято»), і вимагати про нього рядок у звіті означало б писати
    # «нічого не змінено» на кожному кадрі.
    neutral = {"Orientation": 1.0}
    silent = []
    for k, v in raw.items():
        if k in housekeeping or isinstance(v, list):
            continue
        try:
            if float(str(v).lstrip("+")) == neutral.get(k, 0.0):
                continue
        except ValueError:
            pass
        if k not in text:
            silent.append(k)
    print(f"  прочитано {len(raw)}, рядків у звіті {len(rep.lines())}, "
          f"мовчки: {silent}")
    assert not silent, f"параметри пройшли мовчки: {silent}"
    for k in ("Highlights2012", "Shadows2012", "Blacks2012"):
        assert k in text, f"{k} згадано ніде, навіть гуртом"


def test_preset_from_xmp_applies_clean():
    """Пресет з XMP — звичайний пресет: іде в Config без зауважень."""
    pre, _rep = xmp.to_preset(xmp.read(SIDECAR))
    cfg = Config()
    notes = presets.apply(cfg, pre)
    print(f"  зауважень {notes}; crop={cfg.develop.crop}, "
          f"contrast={cfg.develop.contrast}, точок кривої {len(cfg.develop.curve)}")
    assert not notes, f"пресет з XMP не застосовується: {notes}"
    assert cfg.develop.crop and cfg.develop.curve


def test_why_says_where_the_numbers_came_from():
    """Пресет з XMP і пресет від агента лежать в одній теці (§1.2)."""
    pre, _ = xmp.to_preset(xmp.read(SIDECAR))
    why = pre["why"]
    print(f"  {why[:110]}…")
    assert "XMP" in why and "Наближено" in why and "НЕ застосовано" in why


def test_sidecar_is_found_both_ways():
    """`IMG.xmp` і `IMG.CR3.xmp` — обидва написання трапляються."""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        (d / "A.CR3").write_bytes(b"x")
        (d / "A.xmp").write_text(_xmp({}), encoding="utf-8")
        (d / "B.CR3").write_bytes(b"x")
        (d / "B.CR3.xmp").write_text(_xmp({}), encoding="utf-8")
        (d / "C.CR3").write_bytes(b"x")
        got = [xmp.find_sidecar(d / f"{n}.CR3") for n in "ABC"]
        print(f"  {[p.name if p else None for p in got]}")
        assert got[0] and got[0].name == "A.xmp"
        assert got[1] and got[1].name == "B.CR3.xmp"
        assert got[2] is None, "знайшло сайдкар там, де його немає"


def test_garbage_gives_a_readable_error():
    """Зрозуміла відмова, а не трасування (§1)."""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        for name, data in [("bad.xmp", "<не xml"), ("empty.xmp", "")]:
            try:
                xmp.read(_write(d, data, name))
                raise AssertionError(f"{name}: помилки не було")
            except xmp.XmpError as e:
                print(f"  {name}: {str(e)[:60]}")
        try:
            xmp.read(d / "немає.xmp")
            raise AssertionError("неіснуючий файл не дав помилки")
        except xmp.XmpError as e:
            print(f"  немає.xmp: {str(e)[:60]}")

        big = d / "big.xmp"
        big.write_bytes(b"<x/>" + b" " * (xmp.MAX_BYTES + 1))
        try:
            xmp.read(big)
            raise AssertionError("велетенський файл прочитався")
        except xmp.XmpError as e:
            print(f"  big.xmp: {str(e)[:60]}")


def test_as_shot_does_not_report_temperature_as_lost():
    """ACR пише температуру завжди; при As Shot це опис, а не вказівка.

    Фальшива тривога тут дорожча за пропущену: від звіту, у якому
    половина рядків не про справу, починають гортати не читаючи.
    """
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        shot = {"WhiteBalance": "As Shot", "Temperature": "5550", "Tint": "+9"}
        _, as_shot = xmp.to_preset(xmp.read(_write(d, _xmp(shot), "s.xmp")))
        _, custom = xmp.to_preset(xmp.read(_write(
            d, _xmp({**shot, "WhiteBalance": "Custom"}), "u.xmp")))
    print(f"  As Shot -> {sorted(as_shot.ignored)}")
    print(f"  Custom  -> {sorted(custom.ignored)}")
    assert "Temperature" not in as_shot.ignored, "фальшива тривога при As Shot"
    assert "Temperature" in custom.ignored, "своя температура пройшла мовчки"


def test_embedded_block_is_found_in_a_binary():
    """У TIFF/JPEG пакет лежить усередині контейнера."""
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "X.tif"
        p.write_bytes(b"II*\x00" + b"\x00" * 64
                      + _xmp({"Exposure2012": "-0.5"}).encode()
                      + b"\xff" * 32)
        raw = xmp.read_embedded(p)
        print(f"  знайдено ключів: {len(raw)}, Exposure2012={raw.get('Exposure2012')}")
        assert raw.get("Exposure2012") == "-0.5"


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
