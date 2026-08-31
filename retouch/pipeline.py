"""Оркестрація. Кожен етап звітує, кожен проміжний результат можна записати.

Це головна вимога до проєкту: не чорна скринька. Після прогону з
--debug у папці лежать усі проміжні шари, і видно, ЩО саме модель
вважала шкірою, ЩО порахувала дефектом і ЩО в підсумку змінила.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import imageio, layers as layers_mod
from .blemish import DetectParams, detect_blemishes, heal_blemishes
from .freqsep import freq_merge, freq_split, radius_for
from .masks import MaskParams, build_skin_mask
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

    detect: DetectParams = field(default_factory=DetectParams)
    mask: MaskParams = field(default_factory=MaskParams)
    search_radius: int = 90
    strength: float = 1.0
    limit: int | None = None
    face_model: str | None = None
    face_detector: str | None = None
    """YuNet для кропа голови перед face-parsing. Без нього повний кадр
    моделі не по зубах — див. masks.FaceParser.parse."""
    lama_model: str | None = None
    use_skin_mask: bool = True


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

    # --- етапи ---------------------------------------------------------
    def load(self, sink=None) -> "Session":
        s = Stage("read", sink)
        self.img, self.dtype = imageio.read(self.path, self.cfg.raw_decoder)
        # оригінал тримаємо окремо: пластика завжди деформує ЙОГО
        self.img_src = self.img
        self.raw_decoder = imageio.last_raw_decoder
        h, w = self.img.shape[:2]
        s.done(f"{w}x{h} {self.dtype}"
               + (f" raw={self.raw_decoder}" if self.raw_decoder else ""))

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
                self.cls = FaceParser(mp).parse(self.img, det)
                return (mask_from_classes(self.cls, self.cfg.mask),
                        "face-parsing" + ("+yunet" if det else ""))
            except Exception as exc:                      # noqa: BLE001
                print(f"[masks] face-parsing не спрацював ({exc}), беру евристику")
        self.cls = None
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
            self.skin, self.skin_source = build_skin_mask(
                self.img, self.cfg.face_model, self.cfg.mask,
                self.cfg.face_detector)
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
        radius = self.cfg.hf_radius or radius_for(self.img.shape, self.skin)
        if self.low is None or radius != self.radius:
            s = Stage("freq-split", sink)
            self.low, self.high = freq_split(self.img, radius)
            self.radius = radius
            s.done(f"radius={radius:.1f}px")

        s = Stage("detect", sink)
        self.labels, self.blobs = detect_blemishes(self.high, self.skin,
                                                   self.cfg.detect)
        s.done(f"знайдено {len(self.blobs)} плям")
        return self

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
            search_radius=self.cfg.search_radius,
            strength=self.cfg.strength,
            limit=None if keep_ids is not None else self.cfg.limit)
        self.result = freq_merge(self.low, self.high2)
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

    def layers(self) -> dict:
        out: dict[str, tuple] = {}
        # База для шару шкіри — кадр ДО видалення об'єктів, якщо воно було.
        # getattr із дефолтом тут не годиться: атрибут існує і дорівнює
        # None, тож дефолт не спрацьовує і в extract_layer їде None.
        healed = self.remove_base if self.remove_base is not None else self.result
        if self.coverage is not None and self.coverage.max() > 0:
            out["skin"] = layers_mod.extract_layer(self.img, healed, self.coverage)
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
