"""XMP: прочитати, що фотограф вирішив у Camera Raw, і чесно сказати,
що з цього ми вміємо.

Це НЕ спроба відтворити ACR. spec.md §4 уже дав аргумент проти власного
демозаїка — місяці роботи заради гіршого результату; той самий аргумент
стосується профілів камер, Clarity й локальних корекцій. Тут інша
постановка: **міст до проявника, який у фотографа вже є**.

Три яруси, і плутати їх не можна (PLAN.md §1).

**Ярус A — точно.** Геометрія: кроп, кут, орієнтація. Це арифметика,
наближень немає. І це найкорисніше з усього: якщо кадр кроплений в ACR,
конвеєр зараз про це не знає і рахує маску та радіус по ПОВНОМУ кадру,
включно зі шматком, який у фінал не потрапить.

**Ярус B — наближено, з підписом.** Експозиція, тон-крива, насиченість,
контраст. Форма збігається, значення — ні.

**Ярус C — читаємо, показуємо, НЕ вдаємо.** Clarity, Dehaze, Texture,
Highlights/Shadows/Whites/Blacks, профілі об'єктивів, локальні корекції.
Підробити їх «схожим ефектом» гірше, ніж не робити нічого: результат
буде іншим, а виглядатиме як застосований.

Найбільший ризик тут — не помилка в коді, а мовчазне «наближено»
(PLAN.md §5). Тому звіт повертається завжди і показується завжди, а
`to_preset` кладе походження кожного значення в поле `why`.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"
TIFF = "http://ns.adobe.com/tiff/1.0/"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

# Стеля розміру. Сайдкар ACR — це кілька кілобайт; мегабайтний файл або
# не XMP, або спроба згодувати парсеру щось, чого ми не просили.
MAX_BYTES = 4 << 20


class XmpError(Exception):
    """XMP непридатний. Окремий тип, щоб CLI показав текст, а не трасування."""


# ---------------------------------------------------------------------------
# читання
# ---------------------------------------------------------------------------

def find_sidecar(image_path: str | Path) -> Path | None:
    """Сайдкар поруч із кадром.

    ACR пише `IMG.xmp` (без розширення кадру) для RAW і вбудовує блок у
    TIFF/JPEG. Lightroom подекуди лишає `IMG.CR3.xmp`. Перевіряємо обидва
    написання й регістр — на APFS він може бути будь-який.
    """
    p = Path(image_path)
    for cand in (p.with_suffix(".xmp"), p.with_suffix(".XMP"),
                 p.parent / (p.name + ".xmp"), p.parent / (p.name + ".XMP")):
        if cand.exists() and cand.is_file():
            return cand
    return None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _ns(tag: str) -> str:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") else ""


def read(path: str | Path) -> dict:
    """Плоский словник властивостей XMP: `{"Exposure2012": "+0.35", ...}`.

    Простір імен зрізаємо, але тримаємо лише crs: і tiff: — решта (dc:,
    exif:, photoshop:) до проявлення стосунку не має, а тягнути її в
    словник означає плутати ключі.

    ACR пише властивості і атрибутами на rdf:Description, і дочірніми
    елементами — залежно від версії й від того, чи значення просте.
    Читаємо обидва: файли з різних років лежать в одній теці.
    """
    p = Path(path)
    if not p.exists():
        raise XmpError(f"XMP немає: {p}")
    size = p.stat().st_size
    if size > MAX_BYTES:
        raise XmpError(f"{p.name}: {size / 1e6:.1f} МБ — це не сайдкар ACR")
    try:
        root = ET.fromstring(p.read_bytes())
    except ET.ParseError as e:
        raise XmpError(f"{p.name}: не читається як XML — {e}") from None
    return _harvest(root)


def read_embedded(path: str | Path) -> dict:
    """XMP-блок, вбудований у TIFF/JPEG.

    Шукаємо пакет за розділювачами, а не розбираємо контейнер: TIFF-тег
    700 і JPEG APP1 — це два різні формати навколо того самого XML, а
    сам XML в обох випадках лежить між `<?xpacket ...>` і `</x:xmpmeta>`.
    Дешевше знайти його байтами, ніж тягнути повний розбір контейнера.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    data = p.read_bytes()
    i = data.find(b"<x:xmpmeta")
    if i < 0:
        return {}
    j = data.find(b"</x:xmpmeta>", i)
    if j < 0:
        return {}
    try:
        return _harvest(ET.fromstring(data[i:j + 12]))
    except ET.ParseError:
        return {}


def _harvest(root: ET.Element) -> dict:
    out: dict = {}
    for el in root.iter():
        for k, v in el.attrib.items():
            if _ns(k) in (CRS, TIFF):
                out[_local(k)] = v
        if _ns(el.tag) in (CRS, TIFF):
            seq = _read_seq(el)
            if seq is not None:
                out[_local(el.tag)] = seq
            elif el.text and el.text.strip():
                out[_local(el.tag)] = el.text.strip()
    return out


def _read_seq(el: ET.Element) -> list[str] | None:
    """rdf:Seq/rdf:Bag -> список рядків. Так лежить тон-крива."""
    for child in el:
        if _local(child.tag) in ("Seq", "Bag", "Alt"):
            return [(li.text or "").strip() for li in child]
    return None


# ---------------------------------------------------------------------------
# звіт
# ---------------------------------------------------------------------------

@dataclass
class Report:
    """Що зробили з кожним параметром. Не журнал, а частина результату:
    мовчазне «наближено» — головний ризик цієї гілки (PLAN.md §5)."""

    exact: dict = field(default_factory=dict)
    """Ярус A: застосовано точно."""

    approx: dict = field(default_factory=dict)
    """Ярус B: застосовано наближено. Значення — чим саме наблизили."""

    ignored: dict = field(default_factory=dict)
    """Ярус C: прочитано, НЕ застосовано. Значення — чому."""

    def lines(self) -> list[str]:
        out = []
        for k, v in self.exact.items():
            out.append(f"  = {k}: {v}")
        for k, v in self.approx.items():
            out.append(f"  ≈ {k}: {v}")
        for k, v in self.ignored.items():
            out.append(f"  × {k}: {v}")
        return out

    def summary(self) -> str:
        return (f"XMP: точно {len(self.exact)}, наближено {len(self.approx)}, "
                f"не застосовано {len(self.ignored)}")


# ---------------------------------------------------------------------------
# у пресет
# ---------------------------------------------------------------------------

def _num(raw: dict, key: str):
    v = raw.get(key)
    if v is None or isinstance(v, list):
        return None
    try:
        return float(str(v).strip().lstrip("+"))
    except ValueError:
        return None


def _flag(raw: dict, key: str) -> bool:
    return str(raw.get(key, "")).strip().lower() in ("true", "1")


# Ярус C: те, що ACR робить своїми алгоритмами. Значення — чому не
# вдаємо. Формулювання коротке навмисно: воно йде в звіт і в UI.
TIER_C = {
    "Clarity2012": "власний алгоритм Adobe, схожого ефекту не існує",
    "Dehaze": "власний алгоритм Adobe",
    "Texture": "власний алгоритм Adobe",
    "Highlights2012": "параметрична крива ACR; наша крива інша за формою",
    "Shadows2012": "параметрична крива ACR",
    "Whites2012": "параметрична крива ACR",
    "Blacks2012": "параметрична крива ACR",
    "Vibrance": "нелінійна і щадить тілесні тони — підробка зіпсує саме шкіру",
    "Sharpness": "інший алгоритм; різкість тут узагалі не наша справа (§12)",
    "LuminanceSmoothing": "шумозаглушення ACR і wavelet у libraw — різні речі",
    "ColorNoiseReduction": "те саме",
    "LensProfileEnable": "профілі об'єктивів не читаємо (PLAN.md §4)",
    "Upright": "автовирівнювання ACR",
    "UprightVersion": "автовирівнювання ACR",
    "PostCropVignetteAmount": "віньєтка ACR",
    "GrainAmount": "зерно ACR",
    "Temperature": "потрібна матриця камери з профілю DNG; без неї це вгадування",
    "Tint": "те саме, що Temperature",
}

LOCAL_CORRECTIONS = ("MaskGroupBasedCorrections", "CircularGradientBasedCorrections",
                     "GradientBasedCorrections", "PaintBasedCorrections",
                     "RetouchAreas", "RetouchInfo")


def to_preset(raw: dict, name: str = "з XMP") -> tuple[dict, Report]:
    """XMP -> пресет нашого формату плюс звіт.

    Пресет частковий, як і будь-який інший (§1.2): що в XMP не задано —
    того в пресеті немає, і дефолт дата-класу лишається чинним.
    """
    rep = Report()
    dev: dict = {}
    skip_c: set = set()

    # --- ярус A: геометрія ---------------------------------------------
    ang = _num(raw, "CropAngle") or 0.0
    if _flag(raw, "HasCrop"):
        box = [_num(raw, "CropLeft"), _num(raw, "CropTop"),
               _num(raw, "CropRight"), _num(raw, "CropBottom")]
        if all(v is not None for v in box):
            x0, y0, x1, y1 = box
            if x1 > x0 and y1 > y0:
                dev["crop"] = [x0, y0, x1, y1]
                where = (f"{x0:.4f}, {y0:.4f}, {x1:.4f}, {y1:.4f} "
                         f"({(x1 - x0) * 100:.0f}%×{(y1 - y0) * 100:.0f}% кадру)")
                if ang:
                    # Без кута рамка ACR — це рівно наш DevelopParams.crop
                    # у частках, перерахунку немає. З кутом — ні: в ACR
                    # рамка живе у ВИРІВНЯНОМУ кадрі, а ми ріжемо до
                    # повороту. Назвати це ярусом A означало б пообіцяти
                    # точність, якої тут немає (PLAN.md §5).
                    rep.approx["crop"] = (
                        f"{where}; рамка задана разом з кутом {ang:+g}°, "
                        f"а в ACR вона лежить у вирівняному кадрі — ми ж "
                        f"ріжемо ДО повороту, тож країв це не збіжить")
                else:
                    rep.exact["crop"] = where
            else:
                rep.ignored["crop"] = f"порожня рамка {box}"

    if ang:
        # ПЕРЕВІР: знак кута. В ACR CropAngle — це поворот РАМКИ відносно
        # кадру, а наш rotate повертає КАДР; тобто знаки мають бути
        # протилежні. Виведено з означення, а не звірено з файлом ACR —
        # таких у нас немає. До перевірки на реальному файлі це припущення.
        dev["rotate"] = -ang
        rep.approx["rotate"] = (f"{-ang:+g}° з CropAngle {ang:+g}°; "
                                f"знак виведено з означення, на файлі з ACR "
                                f"НЕ звірено (ПЕРЕВІР)")

    orient = _num(raw, "Orientation")
    if orient and int(orient) != 1:
        rep.ignored["Orientation"] = (
            f"{int(orient)}: орієнтацію застосовує декодер при читанні, "
            f"а не пресет")

    # --- ярус A/B: баланс білого ---------------------------------------
    wb = str(raw.get("WhiteBalance", "")).strip()
    if wb:
        if wb.lower() in ("as shot", "asshot", "as-shot"):
            dev["white_balance"] = "camera"
            rep.exact["WhiteBalance"] = "As Shot -> камерний ББ"
            # ACR пише температуру завжди, зокрема й при As Shot: там це
            # ОПИС того, що зняла камера, а не вказівка. Показувати її
            # серед «не застосовано» — фальшива тривога, а від фальшивих
            # тривог звіт починають гортати не читаючи.
            skip_c |= {"Temperature", "Tint"}
        elif wb.lower() == "auto":
            dev["white_balance"] = "auto"
            rep.approx["WhiteBalance"] = "Auto -> автоББ libraw (алгоритм інший)"
        else:
            rep.ignored["WhiteBalance"] = (
                f"{wb}: користувацька температура без матриці камери "
                f"перерахунку не піддається")

    # --- ярус B: тон ----------------------------------------------------
    ev = _num(raw, "Exposure2012")
    if ev:
        dev["exposure"] = ev
        rep.approx["Exposure2012"] = (
            f"{ev:+g} EV -> exp_shift 2^{ev:g}; в ACR це не чистий підсил, "
            f"а тон-мап зі згортанням світлів")

    con = _num(raw, "Contrast2012")
    if con:
        dev["contrast"] = max(-1.0, min(1.0, con / 100.0))
        rep.approx["Contrast2012"] = (
            f"{con:+g} -> contrast {dev['contrast']:+.2f}; форма кривої інша")

    sat = _num(raw, "Saturation")
    if sat:
        dev["saturation"] = max(0.0, 1.0 + sat / 100.0)
        rep.approx["Saturation"] = f"{sat:+g} -> множник {dev['saturation']:.2f}"

    curve = _tone_curve(raw)
    if curve:
        dev["curve"] = curve
        rep.approx["ToneCurvePV2012"] = (
            f"{len(curve)} точок; інтерполяція в ACR своя, форма збігається")
        para = [k for k in ("Highlights2012", "Shadows2012",
                            "Whites2012", "Blacks2012") if _num(raw, k)]
        if para:
            # Найпідступніше місце всього яруса B: точкова крива в ACR
            # лежить ПОВЕРХ параметричної, і взяти лише її — значить
            # застосувати половину тонової правки, зовні не відрізнивши.
            rep.ignored["параметрична крива"] = (
                f"{', '.join(para)} задані, але не застосовані — "
                f"точкова крива в ACR лежить ПОВЕРХ них, тож тон вийде "
                f"інший, а не «майже той самий»")
            skip_c |= set(para)     # уже сказано разом, не повторювати поокремо

    # --- ярус C ---------------------------------------------------------
    for key, why in TIER_C.items():
        if key in rep.ignored or key in skip_c:
            continue
        v = raw.get(key)
        if v in (None, "", "0", "+0", "0.0") or _num(raw, key) == 0:
            continue
        rep.ignored[key] = f"{v}: {why}"
    for key in LOCAL_CORRECTIONS:
        if raw.get(key):
            n = len(raw[key]) if isinstance(raw[key], list) else 1
            rep.ignored[key] = f"{n} шт.: локальні корекції не читаємо"

    preset: dict = {"name": name, "why": _why(rep)}
    if dev:
        preset["develop"] = dev
    return preset, rep


def _tone_curve(raw: dict) -> list | None:
    """ToneCurvePV2012 -> список точок у [0..1].

    ACR тримає точки рядками «x, y» в діапазоні 0..255. Порожня крива і
    крива з двох кінцевих точок — це «Linear», тобто нічого не робити;
    класти таку в пресет означало б виконувати зайву роботу і вносити
    похибку інтерполяції там, де правки не було.
    """
    pts = raw.get("ToneCurvePV2012")
    if not isinstance(pts, list):
        return None
    out = []
    for s in pts:
        m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", str(s))
        if m:
            out.append([float(m.group(1)) / 255.0, float(m.group(2)) / 255.0])
    if len(out) < 3:
        return None
    if all(abs(x - y) < 1e-6 for x, y in out):
        return None                      # пряма — це «Linear»
    return out


def _why(rep: Report) -> str:
    """Походження значень — у поле `why`, як у будь-якого пресету (§1.2).

    Пресет з XMP і пресет від агента лежать в одній теці й виглядають
    однаково. Різниця в тому, ЧОМУ там ці числа, і якщо її не записати,
    через тиждень не відрізнити.
    """
    bits = ["Прочитано з XMP Camera Raw."]
    if rep.exact:
        bits.append("Точно: " + ", ".join(rep.exact) + ".")
    if rep.approx:
        bits.append("Наближено (значення НЕ тотожні ACR): "
                    + ", ".join(rep.approx) + ".")
    if rep.ignored:
        bits.append("НЕ застосовано: " + ", ".join(rep.ignored) + ".")
    return " ".join(bits)


def from_image(image_path: str | Path, name: str | None = None):
    """Найкоротший шлях: кадр -> (пресет, звіт, звідки взято).

    Сайдкар має пріоритет над вбудованим блоком: ACR оновлює саме його,
    а вбудований лишається таким, яким був на момент експорту.
    """
    p = Path(image_path)
    side = find_sidecar(p)
    if side is not None:
        raw = read(side)
        src = str(side)
    else:
        raw = read_embedded(p) if p.is_file() else {}
        src = f"{p.name} (вбудований блок)" if raw else ""
    if not raw:
        return None, Report(), ""
    preset, rep = to_preset(raw, name or f"XMP: {p.stem}")
    return preset, rep, src
