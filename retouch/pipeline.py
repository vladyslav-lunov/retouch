"""Оркестрація. Кожен етап звітує, кожен проміжний результат можна записати.

Це головна вимога до проєкту: не чорна скринька. Після прогону з
--debug у папці лежать усі проміжні шари, і видно, ЩО саме модель
вважала шкірою, ЩО порахувала дефектом і ЩО в підсумку змінила.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from . import imageio, layers as layers_mod
from .blemish import DetectParams, detect_blemishes, heal_blemishes
from .freqsep import face_width, freq_merge, freq_split, radius_for
from .masks import MaskParams, build_skin_mask
from .develop import DevelopParams, apply_pixels
from .dodgeburn import DodgeBurnParams, apply as db_apply, coverage as db_cov, gray_map
from .tools import TOOLS
from .warp import Field, WarpParams


class MaskSanityError(Exception):
    """Маска шкіри вийшла неправдоподібною. Окремий тип, щоб CLI показав
    пораду, а не трасування: це не збій коду, а непридатний вхід."""


def check_skin_mask(frac: float, cfg: "Config", source: str) -> str | None:
    """Текст попередження, якщо маска накрила пів кадру. Інакше None.

    Евристика по YCrCb — це тест на КОЛІР, а не на обличчя. На вуличному
    портреті, де бежевий камінь, бежеве пальто і шкіра лежать в одному
    діапазоні, вона віддає 90+% кадру. Далі валиться все: radius_for
    бачить "обличчя" завширшки з кадр і завищує радіус у рази, детектор
    сипле тисячами спрацювань по волоссю й тканині, а лікування ріже
    волосини на шматки й затирає ланцюжок.

    Заміряно на реальному кадрі 26 Мп: покриття 91%, радіус 20.8px замість
    правильних 4.7, 4558 "дефектів". Мовчати про таке не можна — конвеєр
    за визначенням не приймає рішень мовчки (spec.md §1), а в пакетному
    прогоні на ніч попередження в скролбеку однаково ніхто не побачить.

    Стеля навмисно щедра (0.6) і навмисно тупа: частка маски не відрізняє
    "маска бреше" від "це щільний кроп обличчя, де справді все шкіра".
    Розрізняти їх нема чим, тому помиляємось у бік зупинки: зайва зупинка
    коштує одного прапорця, пропущений випадок — зіпсованого волосся.
    """
    if frac <= cfg.max_skin_fraction:
        return None
    return (
        f"маска шкіри накрила {frac:.0%} кадру (межа {cfg.max_skin_fraction:.0%}).\n"
        f"Джерело — {source}: це діапазон у YCrCb, тобто тест на КОЛІР, а не\n"
        f"на обличчя. Коли фон, одяг чи русяве волосся лежать у тому ж\n"
        f"діапазоні, маска віддає майже весь кадр, і лікування йде по\n"
        f"волоссю, тканині та контурах очей і губ.\n"
        f"\n"
        f"Кроп голови це НЕ лікує — перевірено: на щільному кропі самого\n"
        f"обличчя маска лишилась 90%, бо русяве волосся за кольором теж\n"
        f"«шкіра». Єдиний надійний шлях — face-parsing:\n"
        f"  --face-model models/face_parsing.onnx   spec.md §5\n"
        f"\n"
        f"Прогнати свідомо, розуміючи, що вийде:\n"
        f"  --force-mask     якщо в кадрі справді майже все шкіра\n"
        f"  --no-skin-mask   обробити весь кадр без маски взагалі")


@dataclass
class Config:
    hf_radius: float | None = None
    """None = порахувати з роздільності через radius_for()."""

    max_skin_fraction: float = 0.6
    """Стеля правдоподібності маски. Вище — зупиняємось і просимо втрутитись.
    Щедро: на реальному кропі голови маска дає 15-25%, на макеті 21%."""

    force_mask: bool = False
    """Не зупинятися на неправдоподібній масці."""

    raw_decoder: str | None = None
    """Примусовий декодер RAW: "rawpy" або "imageio". None = як вийде."""

    warp: WarpParams = field(default_factory=WarpParams)
    """Пластика. Саме поле живе в Session, тут лише як його застосовувати."""

    develop: DevelopParams = field(default_factory=DevelopParams)
    """Проявлення: етапи 1-15 конвеєра (spec.md §16). Іде ПЕРШИМ — усе
    подальше рахується по кадру, який людина бачить."""

    dodgeburn: DodgeBurnParams = field(default_factory=DodgeBurnParams)
    """Вирівнювання низької частоти. Вимкнено, поки strength не задано
    явно: воно змінює тон і об'єм, а це не те, що робиться мовчки."""

    dodgeburn_on: bool = False
    """Чи виконувати D&B. Окремим прапорцем, а не «strength > 0», щоб
    пресет міг задати силу наперед, не вмикаючи етап."""

    tools: tuple[str, ...] = ()
    """Дрібні інструменти в порядку застосування: eye_vessels, teeth,
    mattify, skin_tone. Кожен віддається окремим шаром і вимикається
    окремо. Потребують карти класів, тобто face-parsing."""

    tool_params: dict = field(default_factory=dict)
    """Параметри інструментів: {"teeth": {"strength": 0.5}, ...}.
    Не задано — беруться дефолти дата-класів."""

    detect: DetectParams = field(default_factory=DetectParams)
    mask: MaskParams = field(default_factory=MaskParams)
    search_radius: int | None = None
    """Як далеко від плями шукати донора, px. None = порахувати з ширини
    обличчя (90 px на обличчя 1200 px, той самий ідіом, що в hf_radius).
    Явне число лишається абсолютним. Масштабувати обов'язково: на 44
    реальних кадрах обличчя гуляло від 191 до 1582 px, і фіксовані 90 px
    означали то 47% ширини обличчя, то 5.7% (spec.md §6.3)."""

    strength: float = 1.0
    """Сила лікування 0..1. Множник альфи дотику, тобто прямий аналог
    непрозорості шару."""

    limit: int | None = None
    """Лікувати лише N найконтрастніших плям. None = усі. Плями
    відсортовані за спаданням контрасту, тож це «прибери найпомітніше,
    решту зроблю руками»."""
    face_model: str | None = None
    """ONNX face-parsing (BiSeNet). Без нього маска евристична, а на
    вуличному кадрі це непридатно — див. §5."""

    face_detector: str | None = None
    """YuNet для кропа голови перед face-parsing. Без нього повний кадр
    моделі не по зубах — див. masks.FaceParser.parse."""
    lama_model: str | None = None
    """ONNX LaMa для видалення об'єктів. Без нього запасний Telea, який
    годиться лише для дрібного (§7)."""

    use_skin_mask: bool = True
    """Обмежувати роботу маскою шкіри. Вимикати лише для тестів: без маски
    детектор працює по всьому кадру, включно з волоссям і тканиною."""


class Stage:
    """Етап із заміром часу.

    sink — необов'язковий приймач подій. У терміналі його немає і все
    друкується як раніше; UI підставляє свій і бачить ті самі етапи, не
    перехоплюючи stdout.
    """

    def __init__(self, name: str, sink=None):
        self.name, self.t0, self.sink = name, time.time(), sink
        print(f"[{name}] ...", flush=True)
        if sink:
            sink({"stage": name, "state": "start"})

    def done(self, note: str = "") -> None:
        sec = time.time() - self.t0
        print(f"[{self.name}] {sec:.2f}s {note}", flush=True)
        if self.sink:
            self.sink({"stage": self.name, "state": "done",
                       "sec": round(sec, 2), "note": note})


def run(
    image_path: str | Path,
    out_dir: str | Path,
    cfg: Config | None = None,
    remove_mask_path: str | Path | None = None,
    debug: bool = False,
    preview: bool = False,
) -> dict:
    cfg = cfg or Config()
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    stem = image_path.stem

    # Послідовність етапів живе в Session — щоб CLI і UI ганяли той самий
    # код, а не два схожі. Різниця лише в реакції на криву маску: тут
    # зупиняємось, у UI показуємо попередження і питаємо людину.
    sess = Session(image_path, cfg).load()
    if sess.warn and not cfg.force_mask:
        raise MaskSanityError(sess.warn)
    if sess.warn:
        print(f"[skin-mask] УВАГА: {sess.warn.splitlines()[0]} — йду далі "
              f"через --force-mask", flush=True)

    sess.analyze().heal()
    sess.run_tools()
    if cfg.dodgeburn_on:
        sess.dodge_burn()

    img, dtype, skin, source = sess.img, sess.dtype, sess.skin, sess.skin_source
    h, w = img.shape[:2]
    radius, lbl, blobs = sess.radius, sess.labels, sess.blobs
    low, high, high2 = sess.low, sess.high, sess.high2
    healed, heal_cov = sess.result, sess.coverage

    if not debug:
        # spec.md §9: на 50 Мп кожен такий буфер — 600 МБ, а їх тут чотири.
        # З --debug вони ще потрібні для дампу, тому звільняємо тільки тут.
        # Аркушу вони не потрібні: він працює з img, result, coverage і lbl.
        sess.low = sess.high = sess.high2 = None
        del low, high, high2

    out_layers: dict[str, tuple[np.ndarray, np.ndarray]] = sess.layers()
    result = healed

    # --- видалення об'єктів ---------------------------------------------
    if remove_mask_path:
        s = Stage("inpaint")
        rm = imageio.read_mask(remove_mask_path, (h, w))
        if cfg.lama_model and Path(cfg.lama_model).exists():
            from .inpaint import LamaInpainter, inpaint_region
            model = LamaInpainter(cfg.lama_model)
            result, cov = inpaint_region(result, rm, model)
            note = "lama"
        else:
            from .inpaint import inpaint_classic, telea_warning
            warn_t = telea_warning(rm)
            if warn_t:
                print(f"[inpaint] УВАГА: {warn_t}", flush=True)
            before = result
            result = inpaint_classic(result, rm)
            cov = cv2.GaussianBlur(rm.astype(np.float32), (0, 0), 3.0)
            result = before * (1 - cov[..., None]) + result * cov[..., None]
            note = "telea (моделі немає)"
        out_layers["remove"] = layers_mod.extract_layer(healed, result, cov)
        s.done(note)

    # --- запис ----------------------------------------------------------
    s = Stage("write")
    masks = {"skin": skin} if skin is not None else {}
    written = layers_mod.write_stack(out_dir, stem, img, out_layers,
                                     result, dtype, masks)
    # Сіра карта D&B пишеться окремо: це НЕ звичайний шар з альфою, його
    # треба класти в Soft Light, і режим стоїть у самій назві файлу.
    # Покласти в Normal — отримати сіру пляму на пів кадру.
    if sess.db_gray is not None:
        # Номер — ПІСЛЯ всіх звичайних шарів, бо номер і є порядком
        # складання. Жорстке «03» стикалось з третім інструментом, і
        # виходило два файли з однаковим індексом: збирати їх у якомусь
        # порядку стає неможливо.
        pg = out_dir / f"{stem}_{len(out_layers) + 1:02d}_dodgeburn_softlight.png"
        imageio.write(pg, np.dstack([sess.db_gray] * 3), np.dtype("uint16"))
        written.append(pg)
    s.done(f"{len(written)} файлів")

    if preview:
        s = Stage("preview")
        from .preview import contact_sheet
        sheet = contact_sheet(img, result, heal_cov, skin, lbl, blobs)
        p = out_dir / f"{stem}_preview.png"
        imageio.write(p, sheet, np.dtype("uint8"))
        written.append(p)
        s.done(str(p))

    if debug:
        _dump_debug(out_dir / f"{stem}_debug", img, low, high, high2,
                    lbl, heal_cov, result)

    return {"result": result, "blobs": blobs, "skin_source": source,
            "files": written, "radius": radius}


class Session:
    """Один кадр, відкритий для інтерактивної роботи.

    CLI проганяє конвеєр раз і виходить. UI навпаки: той самий кадр
    переганяється по кілька разів — інший поріг, підправлена руками маска,
    знята галочка з плями. Перечитувати файл і рахувати частотку щоразу
    безглуздо, тому стан кадру живе тут, а не в UI: за конвенцією проєкту
    послідовність етапів лишається в pipeline.py.

    Ціна — пам'ять: img, low і high тримаються весь сеанс. На 26 Мп це
    близько 0.9 ГБ, і саме тому сеанс один і кадр один (spec.md §2).

    На відміну від run(), неправдоподібна маска тут НЕ зупиняє роботу:
    UI показує попередження і дає вирішити людині — власне заради цього
    він і потрібен. Рішення все одно не мовчазне (§1).
    """

    def __init__(self, image_path: str | Path, cfg: Config | None = None):
        self.path = Path(image_path)
        self.cfg = cfg or Config()
        self.img = None
        self.dtype = None
        self.raw_decoder = None
        self.skin = None
        self.skin_auto = None      # маска, як її порахувала система
        self.skin_source = "off"
        self.warn = None
        self.low = self.high = None
        self.radius = None
        self.labels = None
        self.blobs: list[dict] = []
        self.high2 = self.coverage = self.result = None
        self.remove_cov = self.remove_base = None
        self.telea_warn = None
        self.img_src = None
        self._field = None
        self.cls = None
        self.develop_ignored: list[str] = []
        self.db_gray = self.db_base = self.db_coverage = None
        self.tool_layers: list = []
        self.blob_classes: list = []
        self.detect_warn: str | None = None
        self.threshold_curve: list = []
        self.threshold_note: str | None = None
        self.faces: list = []
        self.face_w: float | None = None
        self.face_w_source: str = "guess"
        self.search_radius_px: int = 0

    # --- етапи ---------------------------------------------------------
    def load(self, sink=None) -> "Session":
        s = Stage("read", sink)
        self.img, self.dtype = imageio.read(self.path, self.cfg.raw_decoder,
                                            develop=self.cfg.develop)
        self.raw_decoder = imageio.last_raw_decoder
        h, w = self.img.shape[:2]
        s.done(f"{w}x{h} {self.dtype}"
               + (f" raw={self.raw_decoder}" if self.raw_decoder else ""))

        dp = self.cfg.develop
        # Параметри, що мають сенс лише під час декодування RAW. Якщо на
        # вході TIFF — вони не спрацювали, і мовчати про це не можна (§1).
        self.develop_ignored = [] if self.raw_decoder else list(dp.raw_only())
        if dp.touches_pixels():
            s = Stage("develop", sink)
            self.img = apply_pixels(self.img, dp)
            h, w = self.img.shape[:2]
            bits = []
            if dp.crop:
                bits.append(f"кроп -> {w}x{h}")
            if dp.rotate:
                bits.append(f"поворот {dp.rotate:+g}°")
            if dp.contrast:
                bits.append(f"контраст {dp.contrast:+g}")
            if dp.curve:
                bits.append(f"крива {len(dp.curve)} точок")
            s.done(", ".join(bits))
        if self.develop_ignored:
            print(f"[develop] УВАГА: не для TIFF, проігноровано: "
                  f"{', '.join(self.develop_ignored)}", flush=True)

        # оригінал для пластики — уже ПРОЯВЛЕНИЙ кадр: деформувати треба
        # те, що людина бачить, а не те, що віддав декодер
        self.img_src = self.img

        if self.cfg.use_skin_mask:
            s = Stage("skin-mask", sink)
            self.skin, self.skin_source = self._build_mask()
            self.skin_auto = self.skin.copy()
            frac = float(self.skin.mean())
            s.done(f"джерело={self.skin_source} покриття={frac:.1%}")
            self.warn = check_skin_mask(frac, self.cfg, self.skin_source)
        else:
            self.skin, self.skin_source, self.warn = None, "off", None
        return self

    def _build_mask(self):
        """Маска + запам'ятана карта класів.

        Карту класів тримаємо, бо вона НЕ залежить від того, що ми потім
        назвемо шкірою. UI дає перебирати набори класів мишею, і ганяти
        заради кожної галочки модель по 4 секунди безглуздо.
        """
        from .masks import FaceParser, heuristic_skin_mask, mask_from_classes
        mp = self.cfg.face_model
        if mp and Path(mp).exists():
            try:
                det = (str(self.cfg.face_detector)
                       if self.cfg.face_detector and Path(self.cfg.face_detector).exists()
                       else None)
                fp = FaceParser(mp)
                self.cls = fp.parse(self.img, det)
                self.faces = list(fp.last_faces)
                self.face_w = float(self.faces[0][2]) if self.faces else None
                return (mask_from_classes(self.cls, self.cfg.mask),
                        "face-parsing" + ("+yunet" if det else ""))
            except Exception as exc:                      # noqa: BLE001
                print(f"[masks] face-parsing не спрацював ({exc}), беру евристику")
        self.cls = None
        self.faces, self.face_w = [], None
        return heuristic_skin_mask(self.img, self.cfg.mask), "heuristic"

    def remask(self) -> bool:
        """Перебрати маску з наявної карти класів, без запуску моделі.
        Повертає False, якщо карти немає (евристика) — тоді треба load()."""
        if self.cls is None:
            return False
        from .masks import mask_from_classes
        self.skin_auto = mask_from_classes(self.cls, self.cfg.mask)
        self.skin = self.skin_auto.copy()
        self.warn = check_skin_mask(float(self.skin.mean()), self.cfg,
                                    self.skin_source)
        self.labels, self.blobs = None, []
        self.high2 = self.coverage = self.result = None
        return True

    def class_stats(self) -> list[dict]:
        """Частка кадру по кожному класу — щоб UI показав, що взагалі є."""
        from .masks import CELEBA_CLASSES
        if self.cls is None:
            return []
        out = []
        for idx, name in CELEBA_CLASSES.items():
            frac = float((self.cls == idx).mean())
            if frac > 1e-5:
                out.append({"name": name, "frac": round(frac, 5)})
        return sorted(out, key=lambda r: -r["frac"])

    def warp_field(self) -> Field:
        """Поле зміщення цього кадру, створюється на першу вимогу."""
        if self._field is None:
            self._field = Field(self.img.shape[:2], self.cfg.warp.scale)
        return self._field

    def apply_warp(self, sink=None) -> "Session":
        """Деформувати кадр і скинути все, що з нього росло.

        Пластика йде ПЕРЕД лікуванням шкіри: спершу форма, потім текстура.
        Інакше ретуш робилася б по пікселях, які деформація потім
        пересемплить, а шар корекції з'їхав би відносно бази.

        Деформуємо ЗАВЖДИ від оригіналу (img_src), а не від попереднього
        результату: інакше кожне ворушіння повзунка сили накладало б
        ресемпл на ресемпл і кадр повільно мився б.
        """
        f = self._field
        if f is None or not f.touched:
            self.img = self.img_src
        else:
            s = Stage("warp", sink)
            self.img = f.apply(self.img_src, self.cfg.warp)
            st = f.stats()
            s.done(f"макс {st['max_px']}px, поле {st['field'][0]}x{st['field'][1]}")
        # усе нижче по конвеєру рахувалося по старій геометрії
        self.low = self.high = None
        self.labels, self.blobs = None, []
        self.high2 = self.coverage = self.result = None
        self.remove_cov = self.remove_base = None
        if self.cfg.use_skin_mask:
            # Саме _build_mask, а не build_skin_mask: воно перераховує ще й
            # карту класів. Інакше self.cls описував би геометрію ДО
            # деформації, і перемикання класів у UI ліпило б маску, зсунуту
            # відносно кадру — заміряно 4% площі розходження.
            #
            # Показуємо етапом: на 26 Мп це секунди мовчання, а мовчазних
            # пауз у цьому конвеєрі не має бути (§1).
            ms = Stage("skin-mask", sink)
            self.skin, self.skin_source = self._build_mask()
            ms.done(f"джерело={self.skin_source} "
                    f"покриття={self.skin.mean():.1%} (після пластики)")
            self.skin_auto = self.skin.copy()
            self.warn = check_skin_mask(float(self.skin.mean()), self.cfg,
                                        self.skin_source)
        return self

    def set_skin_mask(self, mask) -> None:
        """Підмінити маску (ручні правки в UI). Скидає все, що з неї росте."""
        self.skin = None if mask is None else (mask > 0).astype(np.uint8)
        self.labels, self.blobs = None, []
        self.high2 = self.coverage = self.result = None

    def analyze(self, sink=None) -> "Session":
        """Частотка + детекція. Частотку рахуємо лише коли змінився радіус."""
        fw, self.face_w_source = face_width(self.img.shape, self.skin, self.face_w)
        self.search_radius_px = self._search_radius(fw)
        radius = self.cfg.hf_radius or radius_for(self.img.shape, self.skin,
                                                  face_w=self.face_w)
        if self.low is None or radius != self.radius:
            s = Stage("freq-split", sink)
            self.low, self.high = freq_split(self.img, radius)
            self.radius = radius
            s.done(f"radius={radius:.1f}px")

        if self.cfg.detect.target_coverage is not None:
            self.solve_threshold(self.cfg.detect.target_coverage, sink)

        s = Stage("detect", sink)
        self.labels, self.blobs = detect_blemishes(self.high, self.skin,
                                                   self.cfg.detect)
        self.blob_classes = self._blob_classes()
        note = f"знайдено {len(self.blobs)} плям"
        if self.blob_classes:
            note += " (" + ", ".join(f"{n} {c}" for n, c in
                                     self.blob_classes[:3]) + ")"
        s.done(note)
        self.detect_warn = self._check_blob_classes()
        if self.detect_warn:
            print(f"[detect] УВАГА: {self.detect_warn}", flush=True)
        return self

    # Драбина порогів для підбору. Не бісекція: крива корисна сама по
    # собі — її видно в звіті, і по ній зрозуміло, наскільки кадр
    # чутливий до порога взагалі.
    THRESHOLD_LADDER = (0.008, 0.010, 0.012, 0.014, 0.018, 0.022, 0.028, 0.035)

    def solve_threshold(self, target: float, sink=None) -> float:
        """Підібрати поріг під ЦІЛЬОВУ частку торкнутої шкіри.

        Навіщо взагалі: поріг контрасту між кадрами не переноситься. На
        44 реальних кадрах однієї людини фіксовані 0.012 дали від 0% до
        24% торкнутої шкіри — у робочу зону потрапило 15 кадрів із 44, а
        14 пішли в згладжування (spec.md §6.2). Частка торкнутого —
        навпаки, означає те саме на будь-якому кадрі, бо це відповідь на
        питання «скільки шкіри ми переписали».

        Рахуємо чесно, лікуванням, а не оцінкою по площі плям: альфа
        дотику ширша за саму пляму приблизно вчетверо, і на цій різниці
        ціль перестала б означати те, що написано в її назві.
        """
        st = Stage("solve-threshold", sink)
        skin_px = float(self.skin.sum()) if self.skin is not None else 0.0
        if skin_px <= 0:
            st.done("маски немає — підбирати нема під що")
            return self.cfg.detect.threshold

        curve, chosen = [], None
        for t in self.THRESHOLD_LADDER:
            p = replace(self.cfg.detect, threshold=t, target_coverage=None)
            lbl, blobs = detect_blemishes(self.high, self.skin, p)
            _h2, cov = heal_blemishes(self.high, lbl, blobs, self.skin,
                                      search_radius=self.search_radius_px,
                                      strength=self.cfg.strength)
            frac = float((cov > 0).sum()) / skin_px
            curve.append({"threshold": t, "blobs": len(blobs),
                          "touched_of_skin": round(frac, 5)})
            if frac <= target:
                chosen = t
                break              # драбина зростає — далі буде лише менше

        self.threshold_curve = curve
        if chosen is None:
            # Навіть найжорсткіший поріг дає більше цілі. Це не привід
            # мовчки поставити його: кадр просто такий, і сказати про це
            # чесніше, ніж вдати, що ціль досягнута.
            chosen = self.THRESHOLD_LADDER[-1]
            self.threshold_note = (
                f"ціль {target:.1%} не досягнута навіть на {chosen}: "
                f"вийшло {curve[-1]['touched_of_skin']:.1%}. Кадр із дуже "
                f"вираженою текстурою — дивись на кроп 1:1, перш ніж вірити "
                f"результату")
            print(f"[solve-threshold] УВАГА: {self.threshold_note}", flush=True)
        else:
            self.threshold_note = None
        self.cfg.detect = replace(self.cfg.detect, threshold=chosen)
        st.done(f"ціль {target:.1%} -> поріг {chosen} "
                f"({curve[-1]['touched_of_skin']:.2%} шкіри, "
                f"{len(curve)} проб)")
        return chosen

    # Обличчя, під яке відкалібровані search_radius і radius_for.
    BASE_FACE = 1200.0
    BASE_SEARCH = 90

    def _search_radius(self, face_w: float) -> int:
        """Радіус пошуку донора в пікселях.

        Масштабується з обличчям, і це не витончення. Заміряно на 44
        реальних кадрах: ширина обличчя гуляє від 191 до 1582 px, тобто
        у 8 разів. Фіксовані 90 px — це 47% ширини обличчя на дрібному
        кадрі й 5.7% на великому; у першому випадку донор для підборіддя
        законно береться з чола, у другому — з сусідньої пори.

        Питання було відкрите з §13 (питання 2). Розкид у 8 разів на
        реальній зйомці на нього й відповідає.

        `cfg.search_radius = None` означає «порахуй сам» — той самий
        ідіом, що в `hf_radius`. Явне число лишається абсолютним: людина,
        яка його задала, мала на увазі пікселі, і мовчки перерахувати їх
        було б підміною параметра.
        """
        if self.cfg.search_radius is not None:
            return int(self.cfg.search_radius)
        return int(np.clip(round(self.BASE_SEARCH * face_w / self.BASE_FACE),
                           24, 400))

    def _blob_classes(self) -> list[tuple[str, int]]:
        """Розподіл знайдених плям по класах face-parsing.

        Скільки плям знайдено — саме по собі нічого не каже: 154 на
        обличчі і 154 на ланцюжку виглядають в консолі однаково. Клас
        каже, ЩО саме знайдено, і це єдине, з чого видно підміну.
        """
        from .masks import CELEBA_CLASSES
        if self.cls is None or not self.blobs:
            return []
        h, w = self.cls.shape[:2]
        cnt: dict[str, int] = {}
        for b in self.blobs:
            x, y = b["center"]
            xi = min(w - 1, max(0, int(x)))
            yi = min(h - 1, max(0, int(y)))
            name = CELEBA_CLASSES.get(int(self.cls[yi, xi]), "?")
            cnt[name] = cnt.get(name, 0) + 1
        return sorted(((n, c) for n, c in cnt.items()), key=lambda r: -r[1])

    # Класи, у яких «дефект» майже завжди виявляється не дефектом. Шия й
    # груди — це ланцюжок, комір і тінь від підборіддя; вухо — сережка.
    # Список короткий навмисно: це не фільтр, а привід перепитати.
    SUSPECT_CLASSES = ("neck", "neck_l", "l_ear", "r_ear", "ear_r", "cloth")

    def _check_blob_classes(self, share: float = 0.15) -> str | None:
        """Попередити, коли забагато «дефектів» лежить не на обличчі.

        Заміряно на реальному кадрі: з класом neck у наборі 21% усіх
        знахідок — це ланцюжок на грудях, і лікування рве його на шматки
        (spec.md §15). Детектор тут ні до чого і ускладнювати його не
        треба: питання знімається набором класів, тобто галочкою.
        """
        n = len(self.blobs)
        if not self.blob_classes or n < 20:
            return None
        skin = set(self.cfg.mask.skin_classes)
        bad = [(name, c) for name, c in self.blob_classes
               if name in self.SUSPECT_CLASSES and name in skin
               and c / n >= share]
        if not bad:
            return None
        which = ", ".join(f"{c} з {n} у класі «{name}»" for name, c in bad)
        return (f"{which}. Там зазвичай не дефекти, а прикраси, комір і "
                f"тіні — лікування їх ПОРВЕ. Прибери клас із набору "
                f"(--preset з mask.skin_classes, або галочка на вкладці "
                f"«Маска») і перегони.")

    def heal(self, keep_ids=None, sink=None) -> "Session":
        """keep_ids — які плями лікувати. None = усі (з урахуванням limit).

        Фільтруємо СПИСОК плям, а не labels: labels лишаються повними, щоб
        пошук донора й далі обходив усі дефекти, а не лише вибрані.
        """
        if self.labels is None or self.high is None:
            # Найчастіше сюди приходять після зміни маски: remask() свідомо
            # скидає детекцію, бо вона рахувалася по іншій масці. Лікувати
            # за старими мітками не можна — вони показують на плями, яких у
            # новій масці вже немає.
            raise RuntimeError("немає детекції — спершу analyze() "
                               "(у UI це «Перегнати»)")
        todo = self.blobs
        if keep_ids is not None:
            ids = set(keep_ids)
            todo = [b for b in self.blobs if b["id"] in ids]

        s = Stage("heal", sink)
        self.high2, self.coverage = heal_blemishes(
            self.high, self.labels, todo, self.skin,
            search_radius=self.search_radius_px,
            strength=self.cfg.strength,
            limit=None if keep_ids is not None else self.cfg.limit)
        self.result = freq_merge(self.low, self.high2)
        # Усе, що йшло ПІСЛЯ лікування, рахувалося поверх іншого кадру.
        # Лишити його — значить показати шари, зняті з неіснуючого стану:
        # база шару в layers() вказувала б на масив до перелікування.
        self.tool_layers = []
        self.db_gray = self.db_base = self.db_coverage = None
        s.done(f"торкнулися {self.coverage.mean():.3%} кадру")
        return self

    def remove(self, mask, sink=None) -> "Session":
        """Видалення об'єктів по білій масці. Працює поверх ПОТОЧНОГО
        результату, тобто після лікування шкіри — як і в run()."""
        s = Stage("inpaint", sink)
        base = self.result if self.result is not None else self.img
        self.telea_warn = None
        if self.cfg.lama_model and Path(self.cfg.lama_model).exists():
            from .inpaint import LamaInpainter, inpaint_region
            out, cov = inpaint_region(base, mask, LamaInpainter(self.cfg.lama_model))
            note = "lama"
        else:
            from .inpaint import inpaint_classic, telea_warning
            self.telea_warn = telea_warning(mask)
            cov = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3.0)
            filled = inpaint_classic(base, mask)
            out = base * (1 - cov[..., None]) + filled * cov[..., None]
            note = "telea (моделі немає)"
        self.remove_base = base
        self.remove_cov = cov
        self.result = out
        s.done(note)
        return self

    def dodge_burn(self, sink=None) -> "Session":
        """Вирівняти низьку частоту. Іде ПІСЛЯ лікування шкіри.

        Порядок саме такий, бо D&B вирівнює тон, а свіжа пляма — це
        локальна нерівність тону. Якби D&B ішов першим, він намагався б
        вирівняти те, що лікування за секунду прибере зовсім, і витратив
        би на це частину своєї сили.
        """
        # Повторний прогін (UI сунув повзунок) має рахуватися від кадру
        # ДО D&B, а не від уже вирівняного: інакше сила накопичується і
        # друге натискання дає не те, що показує повзунок.
        base = (self.db_base if self.db_base is not None else
                self.result if self.result is not None else self.img)
        s = Stage("dodge-burn", sink)
        self.db_gray = gray_map(base, self.skin, self.cfg.dodgeburn)
        self.db_base = base
        self.db_coverage = db_cov(self.db_gray)
        self.result = db_apply(base, self.db_gray)
        s.done(f"сила {self.cfg.dodgeburn.strength}, "
               f"торкнулися {self.db_coverage.mean():.1%} кадру")
        return self

    def run_tools(self, sink=None) -> "Session":
        """Дрібні інструменти по черзі. Кожен — окремий шар.

        Потрібна карта класів: без неї ми не знаємо, де око, а де рот, і
        робити тут нічого. Мовчки не робимо нічого замість того, щоб
        застосувати навмання (§1).
        """
        # Повторний прогін починається з кадру ДО інструментів. Разом з
        # ним скидається D&B: він рахувався поверх старого набору, і
        # лишити його — значить показувати вирівнювання чужого кадру.
        if self.tool_layers:
            self.result = self.tool_layers[0][1]
            self.tool_layers = []
            self.db_gray = self.db_base = self.db_coverage = None
        if not self.cfg.tools:
            return self
        if self.cls is None:
            print("[tools] пропущено: немає карти класів, потрібен --face-model",
                  flush=True)
            return self
        base = self.result if self.result is not None else self.img
        for name in self.cfg.tools:
            if name not in TOOLS:
                print(f"[tools] невідомий інструмент: {name}", flush=True)
                continue
            fn, P = TOOLS[name]
            kw = dict(self.cfg.tool_params.get(name) or {})
            s = Stage(f"tool:{name}", sink)
            prev = base
            base, cov = (fn(base, self.cls, p=P(**kw))
                         if name in ("mattify", "skin_tone")
                         else fn(base, self.cls, P(**kw)))
            if cov.max() > 0:
                self.tool_layers.append((name, prev, base.copy(), cov))
            s.done(f"торкнулися {(cov > 0).mean():.2%} кадру")
        self.result = base
        return self

    def layers(self) -> dict:
        out: dict[str, tuple] = {}
        # База для шару шкіри — кадр ДО видалення об'єктів, якщо воно було.
        # getattr із дефолтом тут не годиться: атрибут існує і дорівнює
        # None, тож дефолт не спрацьовує і в extract_layer їде None.
        # База шару шкіри — стан ОДРАЗУ після лікування, до всього, що
        # йде далі. Інструменти й D&B рахувались уже від нього.
        healed = (self.tool_layers[0][1] if self.tool_layers else
                  self.db_base if self.db_base is not None else
                  self.remove_base if self.remove_base is not None else self.result)
        if self.coverage is not None and self.coverage.max() > 0:
            out["skin"] = layers_mod.extract_layer(self.img, healed, self.coverage)
        for i, (name, prev, after, cov) in enumerate(self.tool_layers):
            out[name] = layers_mod.extract_layer(prev, after, cov)
        rc = self.remove_cov
        if rc is not None and rc.max() > 0:
            out["remove"] = layers_mod.extract_layer(healed, self.result, rc)
        return out

    def write(self, out_dir, sink=None) -> list[Path]:
        s = Stage("write", sink)
        masks = {"skin": self.skin} if self.skin is not None else {}
        # База — кадр ПІСЛЯ пластики, а не вихідний файл. Інакше
        # base*(1-a) + layer*a не зійдеться: лікування рахувалося вже по
        # деформованій геометрії. Оригінал нікуди не дівається — він
        # лишається у вхідному файлі, а як саме його зігнуто, записано
        # окремо полем зміщення.
        written = layers_mod.write_stack(out_dir, self.path.stem, self.img,
                                         self.layers(), self.result,
                                         self.dtype, masks)
        if self.db_gray is not None:
            # Окремим файлом, з режимом у назві й номером ПІСЛЯ всіх
            # звичайних шарів: номер — це порядок складання.
            p = Path(out_dir) / (f"{self.path.stem}_{len(self.layers()) + 1:02d}"
                                 f"_dodgeburn_softlight.png")
            imageio.write(p, np.dstack([self.db_gray] * 3), np.dtype("uint16"))
            written.append(p)
        if self._field is not None and self._field.touched:
            written.append(self._field.save(
                Path(out_dir) / f"{self.path.stem}_warp.png"))
        s.done(f"{len(written)} файлів")
        return written


def detect_only(image_path: str | Path, cfg: Config | None = None) -> dict:
    """--dry-run: порахувати дефекти й нічого не писати.

    Живе тут, а не в cli.py: читання -> маска -> частотка -> детекція — це
    оркестрація, а вона за конвенцією проєкту тільки в pipeline.py.
    """
    cfg = cfg or Config()
    img, _ = imageio.read(image_path, cfg.raw_decoder)
    if cfg.use_skin_mask:
        skin, source = build_skin_mask(img, cfg.face_model, cfg.mask,
                                       cfg.face_detector)
    else:
        skin, source = None, "off"
    warn = None if skin is None else check_skin_mask(float(skin.mean()), cfg, source)
    radius = cfg.hf_radius or radius_for(img.shape, skin)
    _, high = freq_split(img, radius)
    _, blobs = detect_blemishes(high, skin, cfg.detect)
    return {"blobs": blobs, "radius": radius, "skin_source": source, "warn": warn}


def _dump_debug(d: Path, img, low, high, high2, lbl, cov, result) -> None:
    d.mkdir(parents=True, exist_ok=True)
    w8 = lambda n, a: cv2.imwrite(  # noqa: E731
        str(d / n), np.clip(a * 255, 0, 255).astype(np.uint8))
    w8("01_low.png", low)
    w8("02_high.png", high + 0.5)
    overlay = img.copy()
    overlay[lbl > 0] = (0, 0, 1)
    w8("03_detected.png", overlay)
    w8("04_high_healed.png", high2 + 0.5)
    w8("05_coverage.png", np.dstack([cov] * 3))
    w8("06_diff_x4.png", (result - img) * 4 + 0.5)
    print(f"[debug] {d}")
