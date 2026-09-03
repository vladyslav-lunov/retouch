"""Локальний веб-інтерфейс. Тільки stdlib: http.server, без фреймворків.

Чому браузер, а не вікно: уся цінність цього інструмента — подивитися на
результат у масштабі 1:1. Зменшений портрет 24 Мп не показує роботи по
текстурі взагалі, а панорамування й зум по великому зображенню браузер
уміє з коробки, і робить це краще, ніж я написав би на Tk.

Чому це НЕ суперечить §12. Заборонений там інтерактивний канвас — це
живе редагування з перерахунком на кожен рух миші, і причина заборони в
§2 суто продуктивнісна: ONNX на CPU дає «натиснув і чекаєш». Тут модель
рівно така: натиснув «Прогнати» — чекаєш — дивишся. Малювання маски теж
нічого не перераховує на льоту: пензлем правиться МАСКА, а конвеєр
переганяється окремою кнопкою.

Повний кадр у браузер не віддається ніколи. Оглядовий план приходить
зменшеним, а кропи 1:1 сервер ріже на запит і шле маленькими PNG.
Інакше на 26 Мп ми б клали і браузер, і 8 ГБ пам'яті (spec.md §2).
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

from .blemish import DetectParams
from .imageio import RAW_SUFFIXES, InputError
from .masks import MaskParams
from . import presets as presets_mod
from .pipeline import Config, Session
from .warp import WarpParams

STATIC = Path(__file__).resolve().parent / "static"
SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


# ---------------------------------------------------------------------------
# стан
# ---------------------------------------------------------------------------

class App:
    """Один відкритий кадр і одна фонова задача. Свідомо однокористувацький
    і однозадачний: це локальний інструмент на 8 ГБ, а не сервіс."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sess: Session | None = None
        self.busy = False
        self.stages: list[dict] = []
        self.error: str | None = None
        self.sweep: list[dict] | None = None
        self.keep_ids: set[int] | None = None
        self.painted: np.ndarray | None = None   # ручні правки маски шкіри
        self.remove_mask: np.ndarray | None = None   # що видаляти
        self.proxy: np.ndarray | None = None         # зменшена копія для пластики
        self.written: list[str] = []
        # Розділи, яких немає серед повзунків зліва: проявлення, D&B,
        # інструменти. Тримаємо їх пресетом, а не двадцятьма полями
        # форми, бо це той самий словник, що пише агент і що лягає у
        # файл — один формат на UI, CLI і агента (spec.md §1.2).
        self.preset: dict = {}
        self.preset_notes: list[str] = []
        self.preset_name: str = ""

    # --- прогрес --------------------------------------------------------
    def sink(self, ev: dict) -> None:
        with self.lock:
            if ev.get("state") == "start":
                self.stages.append({"stage": ev["stage"], "sec": None, "note": ""})
            else:
                for st in reversed(self.stages):
                    if st["stage"] == ev["stage"] and st["sec"] is None:
                        st["sec"], st["note"] = ev["sec"], ev.get("note", "")
                        break

    def state(self) -> dict:
        s = self.sess
        d = {
            "busy": self.busy, "stages": self.stages, "error": self.error,
            "sweep": self.sweep, "written": self.written, "loaded": s is not None,
            # Пресет — стан ЗАСТОСУНКУ, а не кадру: агент присилає його
            # наперед, і вкладка «Пресети» читається до відкриття файлу.
            "preset": self.preset, "preset_notes": self.preset_notes,
            "preset_name": self.preset_name,
        }
        if s is None:
            return d
        h, w = (s.img.shape[:2] if s.img is not None else (0, 0))
        d.update({
            "path": str(s.path), "name": s.path.name, "w": w, "h": h,
            "mp": round(w * h / 1e6, 1),
            "skin_source": s.skin_source, "warn": s.warn,
            "raw_decoder": getattr(s, "raw_decoder", None),
            "skin_frac": (None if s.skin is None else round(float(s.skin.mean()), 4)),
            "radius": (None if s.radius is None else round(s.radius, 2)),
            "n_blobs": len(s.blobs),
            "healed": (None if s.coverage is None
                       else round(float((s.coverage > 0).mean()), 5)),
            "has_result": s.result is not None,
            "painted": self.painted is not None,
            "remove_px": (0 if self.remove_mask is None
                          else int((self.remove_mask > 0).sum())),
            "remove_depth": self._remove_depth(),
            "telea_warn": getattr(s, "telea_warn", None),
            "warp": (s._field.stats() if s._field is not None and s._field.touched
                     else None),
            "classes": s.class_stats(),
            "blob_classes": [{"name": n, "n": c} for n, c in
                             (getattr(s, "blob_classes", None) or [])],
            "detect_warn": getattr(s, "detect_warn", None),
            "face_w": s.face_w,
            "face_w_source": s.face_w_source,
            "n_faces": len(s.faces),
            "threshold_curve": getattr(s, "threshold_curve", []),
            "threshold_note": getattr(s, "threshold_note", None),
            "skin_classes": list(s.cfg.mask.skin_classes),
            "has_cls": s.cls is not None,
            "keep": (None if self.keep_ids is None else sorted(self.keep_ids)),
            "develop_ignored": list(getattr(s, "develop_ignored", []) or []),
            "tool_layers": [t[0] for t in s.tool_layers],
            "tool_touched": {t[0]: round(float((t[3] > 0).mean()), 5)
                             for t in s.tool_layers},
            "db": (None if s.db_gray is None else
                   {"touched": round(float(s.db_coverage.mean()), 5),
                    "strength": s.cfg.dodgeburn.strength}),
            "params": {
                "threshold": s.cfg.detect.threshold, "radius": s.cfg.hf_radius,
                "target_coverage": s.cfg.detect.target_coverage,
                "min_area": s.cfg.detect.min_area, "max_area": s.cfg.detect.max_area,
                "max_elongation": s.cfg.detect.max_elongation,
                "strength": s.cfg.strength, "limit": s.cfg.limit,
                "search_radius": s.cfg.search_radius,
                "search_radius_px": s.search_radius_px,
                "mask_erode": s.cfg.mask.erode,
                "use_skin_mask": s.cfg.use_skin_mask,
            },
        })
        return d

    def _remove_depth(self) -> float:
        if self.remove_mask is None or not self.remove_mask.any():
            return 0.0
        from .inpaint import mask_stats
        return round(mask_stats(self.remove_mask)["depth"], 1)

    # --- задачі ---------------------------------------------------------
    def job(self, fn) -> None:
        def wrap():
            try:
                fn()
            except Exception as e:      # noqa: BLE001 — показуємо людині, не падаємо
                self.error = (str(e) if isinstance(e, InputError)
                              else f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
            finally:
                self.busy = False
        self.busy, self.error, self.stages = True, None, []
        threading.Thread(target=wrap, daemon=True).start()


APP = App()


def cfg_from(d: dict, preset: dict | None = None) -> Config:
    """Параметри з форми плюс пресет поверх. Дефолти — з дата-класів.

    Порядок саме такий: спершу повзунки, потім пресет. Повзунків мало і
    вони покривають лікування; усе інше — проявлення, D&B, інструменти —
    приходить пресетом, і руками там правиться той самий словник.
    """
    def num(key, cast, default=None):
        v = d.get(key, "")
        if v in ("", None):
            return default
        try:
            return cast(v)
        except (TypeError, ValueError):
            return default

    dp = DetectParams()
    detect = DetectParams(
        scales=dp.scales,
        threshold=num("threshold", float, dp.threshold),
        target_coverage=num("target_coverage", float, dp.target_coverage),
        min_area=num("min_area", int, dp.min_area),
        max_area=num("max_area", int, dp.max_area),
        max_elongation=num("max_elongation", float, dp.max_elongation),
        darks=bool(d.get("darks", dp.darks)),
        lights=bool(d.get("lights", dp.lights)),
    )
    mp = MaskParams()
    c = Config(
        hf_radius=num("radius", float, None),
        detect=detect,
        mask=MaskParams(erode=num("mask_erode", int, mp.erode),
                        feather=mp.feather, exclude_dilate=mp.exclude_dilate,
                        skin_classes=tuple(d.get("skin_classes") or mp.skin_classes),
                        exclude_classes=mp.exclude_classes),
        search_radius=num("search_radius", int, Config().search_radius),
        strength=num("strength", float, Config().strength),
        limit=num("limit", int, None),
        face_model=d.get("face_model") or None,
        face_detector=d.get("face_detector") or None,
        lama_model=d.get("lama_model") or None,
        use_skin_mask=bool(d.get("use_skin_mask", True)),
        raw_decoder=(d.get("raw_decoder") or None),
        force_mask=True,      # у UI попередження показується, а не зупиняє
    )
    if preset:
        APP.preset_notes = presets_mod.apply(c, preset)
    return c


# ---------------------------------------------------------------------------
# моделі
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[tuple, str] = {}


def model_role(path: Path) -> str | None:
    """Що це за модель — за ПІДПИСОМ, а не за назвою файлу.

    Назви нічого не гарантують: ваги перекладають, перейменовують і
    плутають. Підпис однозначний:
      два входи (image + mask)        -> lama
      виходи bbox_*/kps_*             -> детектор облич (YuNet)
      один вхід [*, 3, H, W], 3 виходи -> face-parsing (BiSeNet)

    Сесію тримаємо в кеші за (шлях, розмір, mtime): відкриття LaMa на
    198 МБ коштує секунди, а список моделей UI просить часто.
    """
    key = (str(path), path.stat().st_size, int(path.stat().st_mtime))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key] or None
    role = ""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        ins, outs = sess.get_inputs(), sess.get_outputs()
        names = {o.name for o in outs}
        if len(ins) == 2:
            role = "lama"
        elif any(n.startswith(("bbox_", "kps_")) for n in names):
            role = "detector"
        else:
            sh = ins[0].shape
            if len(sh) == 4 and sh[1] == 3:
                role = "face"
    except Exception:                                    # noqa: BLE001
        role = ""
    _MODEL_CACHE[key] = role
    return role or None


def scan_models(d: str | Path = "models") -> dict:
    """Що лежить у теці моделей і на що воно годиться."""
    d = Path(d)
    out: dict[str, list] = {"face": [], "detector": [], "lama": [], "unknown": []}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.onnx")):
        role = model_role(p) or "unknown"
        out[role].append({"path": str(p), "name": p.name,
                          "mb": round(p.stat().st_size / 2**20, 1)})
    return out


# ---------------------------------------------------------------------------
# картинки
# ---------------------------------------------------------------------------

def _png(arr: np.ndarray) -> bytes:
    """float32 [0..1] -> 8-бітний PNG. Для показу; файли на диску 16-бітні."""
    u8 = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".png", u8)
    if not ok:
        raise RuntimeError("не закодувалося в PNG")
    return buf.tobytes()


def _fit(a: np.ndarray, maxw: int) -> np.ndarray:
    h, w = a.shape[:2]
    if w <= maxw:
        return a
    s = maxw / w
    return cv2.resize(a, (maxw, max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def overview(sess: Session, kind: str, maxw: int) -> np.ndarray:
    """Загальний план. Тут зменшення доречне: цей вид відповідає на
    питання «чи не полізло воно кудись», а не «як лягла текстура»."""
    if kind == "before":
        return _fit(sess.img, maxw)
    if kind == "after":
        return _fit(sess.result if sess.result is not None else sess.img, maxw)
    if kind == "diff":
        if sess.result is None:
            return _fit(np.full_like(sess.img, 0.5), maxw)
        return _fit(np.clip((sess.result - sess.img) * 4 + 0.5, 0, 1), maxw)
    if kind == "mask":
        if sess.skin is None:
            return _fit(np.zeros_like(sess.img), maxw)
        m = _fit(sess.skin.astype(np.float32), maxw)
        return np.dstack([m] * 3)
    if kind == "coverage":
        if sess.coverage is None:
            return _fit(np.zeros_like(sess.img), maxw)
        return np.dstack([_fit(sess.coverage, maxw)] * 3)
    if kind == "detected":
        base = _fit(sess.img, maxw).copy()
        if sess.labels is not None:
            # INTER_AREA дає частку площі плями в пікселі; поріг низький,
            # бо після зменшення в десять разів дефект інакше зникає
            hit = _fit((sess.labels > 0).astype(np.float32), maxw)
            base[hit > 0.02] = (0.15, 0.15, 1.0)
        return base
    raise ValueError(f"невідомий вид: {kind}")


def crop(sess: Session, kind: str, cx: int, cy: int, size: int) -> np.ndarray:
    """Вирізка 1:1 навколо точки. Саме цей вид — єдина підстава судити
    про якість ретуші, тому він завжди в рідних пікселях, без ресайзу."""
    h, w = sess.img.shape[:2]
    size = max(32, min(size, 1200))
    x0 = int(np.clip(cx - size // 2, 0, max(0, w - size)))
    y0 = int(np.clip(cy - size // 2, 0, max(0, h - size)))
    sl = (slice(y0, min(h, y0 + size)), slice(x0, min(w, x0 + size)))
    before = sess.img[sl]
    if kind == "before" or sess.result is None:
        return before
    after = sess.result[sl]
    if kind == "after":
        return after
    if kind == "diff":
        return np.clip((after - before) * 4 + 0.5, 0, 1)
    if kind == "coverage":
        return np.dstack([sess.coverage[sl]] * 3)
    return before


def list_presets(root: str = "presets") -> list[dict]:
    """Пресети з теки, з `why` у списку.

    `why` показуємо одразу, а не за кліком: коли агент дав десять
    варіантів, у числах вони виглядають однаково, і вибрати можна лише
    за причиною (spec.md §1.2).
    """
    d = Path(root).expanduser()
    out = []
    if not d.exists():
        return out
    for f in sorted(d.rglob("*.y*ml")) + sorted(d.rglob("*.json")):
        row = {"path": str(f), "file": f.name,
               "dir": str(f.parent.relative_to(d)) if f.parent != d else ""}
        try:
            data = presets_mod.load(f)
            row["name"] = str(data.get("name") or f.stem)
            row["why"] = str(data.get("why") or "").strip()
            row["for"] = str(data.get("for") or "").strip()
            row["keys"] = sorted(k for k in data
                                 if k not in ("name", "why", "for"))
        except presets_mod.PresetError as e:
            row["name"], row["error"] = f.stem, str(e)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# дії
# ---------------------------------------------------------------------------

def apply_painted(sess: Session, painted: np.ndarray | None) -> None:
    """Накласти ручні правки на автоматичну маску.

    painted — int8 у масштабі кадру: +1 домалювали, -1 стерли, 0 не чіпали.
    Тримаємо саме правки, а не готову маску: тоді зміна параметрів
    перераховує автоматичну частину, а ручна лишається.
    """
    if sess.skin_auto is None:
        return
    m = sess.skin_auto.copy()
    if painted is not None:
        m[painted > 0] = 1
        m[painted < 0] = 0
    sess.skin = m


def do_open(path: str, params: dict) -> None:
    APP.sess = Session(path, cfg_from(params, APP.preset)).load()
    APP.painted, APP.keep_ids, APP.sweep, APP.written = None, None, None, []
    APP.proxy = None
    APP.sess.analyze(APP.sink)


def do_rerun(params: dict, keep_ids=None) -> None:
    """Перегнати відкритий кадр з іншими параметрами."""
    sess = APP.sess
    old_radius = sess.cfg.hf_radius
    sess.cfg = cfg_from(params, APP.preset)
    if sess.cfg.hf_radius != old_radius:
        sess.low = sess.high = None      # радіус змінився — частотку наново
    apply_painted(sess, APP.painted)
    sess.analyze(APP.sink)
    if keep_ids is not None:
        APP.keep_ids = set(keep_ids)
    sess.remove_cov = sess.remove_base = None
    sess.heal(APP.keep_ids, APP.sink)


def do_heal(keep_ids=None) -> None:
    APP.keep_ids = None if keep_ids is None else set(keep_ids)
    APP.sess.remove_cov = APP.sess.remove_base = None
    APP.sess.heal(APP.keep_ids, APP.sink)


def do_sweep(params: dict, thresholds: list[float]) -> None:
    """Прогнати той самий кадр кількома порогами (spec.md §6.2).

    Частотка рахується один раз на всі пороги — вона від порога не
    залежить, а на 26 Мп це 12 секунд, які нема сенсу платити чотири рази.
    """
    sess = APP.sess
    rows = []
    for t in thresholds:
        p = dict(params, threshold=t)
        sess.cfg = cfg_from(p, APP.preset)
        apply_painted(sess, APP.painted)
        sess.analyze(APP.sink)
        sess.heal(None, APP.sink)
        rows.append({
            "threshold": t, "n_blobs": len(sess.blobs),
            "radius": round(sess.radius, 2),
            "touched": round(float((sess.coverage > 0).mean()), 5),
        })
        APP.sweep = list(rows)

    # Повертаємо кадр до порога, який стоїть у формі. Інакше сеанс мовчки
    # лишався б на ОСТАННЬОМУ з проміряних (зазвичай найжорсткішому), і
    # наступне «Зберегти» записало б не те, що людина бачила в панелі.
    sess.cfg = cfg_from(params, APP.preset)
    apply_painted(sess, APP.painted)
    sess.analyze(APP.sink)
    sess.heal(APP.keep_ids, APP.sink)


PROXY_W = 900


def proxy_of(sess) -> np.ndarray:
    """Зменшена копія ОРИГІНАЛУ для інтерактивної пластики.

    Саме оригіналу, а не поточного кадру: інакше кожен мазок лягав би на
    вже деформовану копію й спотворення накопичувалося б удвічі.
    """
    if APP.proxy is None:
        src = sess.img_src
        h, w = src.shape[:2]
        pw = min(PROXY_W, w)
        APP.proxy = np.ascontiguousarray(
            cv2.resize(src, (pw, max(1, int(h * pw / w))),
                       interpolation=cv2.INTER_AREA))
    return APP.proxy


def do_warp_apply(strength: float) -> None:
    APP.sess.cfg.warp.strength = float(strength)
    APP.sess.apply_warp(APP.sink).analyze(APP.sink)
    APP.sess.heal(APP.keep_ids, APP.sink)


def do_remove() -> None:
    if APP.remove_mask is None or not APP.remove_mask.any():
        raise RuntimeError("маска видалення порожня — намалюй, що прибрати")
    APP.sess.remove(APP.remove_mask, APP.sink)


def do_develop(params: dict) -> None:
    """Перечитати кадр із новим проявленням.

    Саме перечитати, а не перерахувати: проявлення живе в `load()`, бо
    половина його параметрів — це параметри ДЕКОДЕРА RAW, і застосувати
    їх після декодування нічим. Кроп і поворот міняють геометрію, тому
    все, що прив'язане до пікселів старого кадру, скидається: правлена
    маска, поле пластики й маска видалення вказували б не туди.
    """
    sess = APP.sess
    path = sess.path
    APP.painted = APP.remove_mask = APP.proxy = None
    APP.keep_ids = APP.sweep = None
    APP.written = []
    APP.sess = Session(path, cfg_from(params, APP.preset)).load(APP.sink)
    APP.sess.analyze(APP.sink)


def do_stage(stage: str, params: dict) -> None:
    """Дорахувати один етап поверх уже полікованого кадру.

    Окремою кнопкою, а не всередині «Перегнати»: лікування на 26 Мп
    коштує хвилини, а посунути силу D&B хочеться десять разів підряд.
    Обидва етапи ідемпотентні — рахуються від кадру ДО себе, тож
    повторне натискання дає те, що показує повзунок, а не суму.
    """
    sess = APP.sess
    if stage not in ("tools", "dodgeburn"):
        # Спершу назва, потім стан: інакше друкарська помилка в назві
        # відповідала б «спершу Перегнати», і шукали б її не там.
        raise ValueError(f"невідомий етап: {stage}")
    sess.cfg = cfg_from(params, APP.preset)
    if sess.result is None:
        raise RuntimeError("спершу «Перегнати» — етап іде поверх лікування")
    if stage == "tools":
        sess.run_tools(APP.sink)
        if sess.cfg.dodgeburn_on:
            sess.dodge_burn(APP.sink)      # інструменти скинули стару карту
    elif stage == "dodgeburn":
        if sess.cfg.dodgeburn_on:
            sess.dodge_burn(APP.sink)
        else:
            # вимкнули галочку — прибрати результат, а не лишити його
            if sess.db_base is not None:
                sess.result = sess.db_base
            sess.db_gray = sess.db_base = sess.db_coverage = None


def do_write(out_dir: str, preview: bool) -> None:
    sess = APP.sess
    written = sess.write(out_dir, APP.sink)
    if preview:
        from .preview import contact_sheet
        from . import imageio as iio
        sheet = contact_sheet(sess.img, sess.result, sess.coverage,
                              sess.skin, sess.labels, sess.blobs)
        p = Path(out_dir) / f"{sess.path.stem}_preview.png"
        iio.write(p, sheet, np.dtype("uint8"))
        written.append(p)
    APP.written = [str(x) for x in written]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "retouch-lab"

    def log_message(self, fmt, *args):     # тиша: прогрес і так друкує Stage
        pass

    # --- відповіді ------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass               # користувач гортав і скасував запит — нормально

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8") or "{}")

    # --- маршрути -------------------------------------------------------
    def do_GET(self) -> None:                                # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(),
                                  "text/html; charset=utf-8")
            if u.path == "/api/state":
                return self._json(APP.state())
            if u.path == "/api/schema":
                # Та сама схема, що йде агентові через `--schema`. Форма
                # нових вкладок будується з неї, а не пишеться руками:
                # інакше поле в дата-класі й поле в UI розходяться на
                # першій же зміні.
                return self._json(presets_mod.schema())
            if u.path == "/api/presets":
                return self._json(list_presets(q.get("dir", "presets")))
            if u.path == "/api/models":
                return self._json(scan_models(q.get("dir", "models")))
            if u.path == "/api/browse":
                return self._json(self._browse(q.get("dir", "")))
            if u.path == "/api/blobs":
                s = APP.sess
                if s is None:
                    return self._json([])
                return self._json([
                    {"id": b["id"], "x": round(b["center"][0]),
                     "y": round(b["center"][1]), "area": b["area"],
                     "contrast": round(b["contrast"], 5)}
                    for b in s.blobs[:4000]])
            if u.path == "/api/view":
                s = APP.sess
                if s is None or s.img is None:
                    return self._json({"error": "кадр не відкрито"}, 409)
                arr = overview(s, q.get("kind", "before"), int(q.get("w", 1400)))
                return self._send(200, _png(arr), "image/png")
            if u.path == "/api/warp/preview":
                s = APP.sess
                if s is None or s.img is None:
                    return self._json({"error": "кадр не відкрито"}, 409)
                pr = proxy_of(s)
                f = s._field
                k = float(q.get("strength", 1.0))
                out = pr if f is None else f.apply_to(pr, WarpParams(strength=k))
                return self._send(200, _png(out), "image/png")
            if u.path == "/api/crop":
                s = APP.sess
                if s is None or s.img is None:
                    return self._json({"error": "кадр не відкрито"}, 409)
                arr = crop(s, q.get("kind", "before"), int(float(q.get("x", 0))),
                           int(float(q.get("y", 0))), int(q.get("size", 320)))
                return self._send(200, _png(arr), "image/png")
            return self._json({"error": "немає такого"}, 404)
        except InputError as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:                               # noqa: BLE001
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self) -> None:                               # noqa: N802
        u = urlparse(self.path)
        try:
            d = self._body()
            if APP.busy and u.path != "/api/state":
                return self._json({"error": "зайнято, зачекай"}, 409)

            if u.path == "/api/open":
                p = Path(d.get("path", "")).expanduser()
                APP.job(lambda: do_open(str(p), d.get("params", {})))
                return self._json({"ok": True})
            if u.path == "/api/rerun":
                if APP.sess is None:
                    return self._json({"error": "спершу відкрий кадр"}, 409)
                APP.job(lambda: do_rerun(d.get("params", {}), d.get("keep")))
                return self._json({"ok": True})
            if u.path == "/api/heal":
                if APP.sess is None:
                    return self._json({"error": "спершу відкрий кадр"}, 409)
                APP.job(lambda: do_heal(d.get("keep")))
                return self._json({"ok": True})
            if u.path == "/api/sweep":
                if APP.sess is None:
                    return self._json({"error": "спершу відкрий кадр"}, 409)
                th = d.get("thresholds") or [0.008, 0.012, 0.018, 0.025]
                APP.sweep = None
                APP.job(lambda: do_sweep(d.get("params", {}), [float(x) for x in th]))
                return self._json({"ok": True})
            if u.path == "/api/paint":
                return self._json(self._paint(d))
            if u.path == "/api/classes":
                # перебір набору класів: карта класів уже є, модель не потрібна
                s = APP.sess
                if s is None:
                    return self._json({"error": "спершу відкрий кадр"}, 409)
                from .masks import MaskParams as MP
                m = s.cfg.mask
                s.cfg.mask = MP(erode=m.erode, feather=m.feather,
                                exclude_dilate=m.exclude_dilate,
                                skin_classes=tuple(d.get("skin_classes") or ()),
                                exclude_classes=m.exclude_classes)
                if not s.remask():
                    return self._json({"error": "карти класів немає — "
                                       "це евристична маска, не модель"}, 409)
                APP.painted = None
                # Вибір класів кладемо В ПРЕСЕТ, а не лише в поточний
                # Config. Інакше «Перегнати» будує Config із форми
                # наново, набір повертається до дефолтного, і виходить
                # розбіжність: маска в пам'яті вже без neck, а конфіг
                # каже, що neck там є. Далі будь-яка перебудова маски
                # (пластика, проявлення) мовчки повертає його назад.
                # А заразом це правильно по суті: «що вважати шкірою» —
                # рішення фотографа під кадр, тобто пресет (spec.md §15).
                APP.preset = presets_mod.merge(
                    APP.preset,
                    {"mask": {"skin_classes": list(s.cfg.mask.skin_classes)}})
                return self._json({"ok": True,
                                   "preset": APP.preset,
                                   "skin_frac": round(float(s.skin.mean()), 4)})
            if u.path == "/api/warp":
                return self._json(self._warp(d))
            if u.path == "/api/warp/apply":
                if APP.sess is None:
                    return self._json({"error": "спершу відкрий кадр"}, 409)
                APP.job(lambda: do_warp_apply(d.get("strength", 1.0)))
                return self._json({"ok": True})
            if u.path == "/api/remove":
                if APP.sess is None:
                    return self._json({"error": "спершу відкрий кадр"}, 409)
                APP.job(do_remove)
                return self._json({"ok": True})
            if u.path == "/api/preset":
                return self._json(self._preset(d))
            if u.path == "/api/xmp":
                return self._json(self._xmp(d))
            if u.path == "/api/preset/save":
                data = dict(d.get("data") or APP.preset)
                if d.get("name"):
                    data = {"name": d["name"], **data}
                if d.get("why"):
                    data = {**data, "why": d["why"]}
                out = Path(d.get("path") or "presets").expanduser()
                if out.is_dir() or not out.suffix:
                    out = out / f"{d.get('file') or 'preset'}.yaml"
                return self._json({"ok": True,
                                   "path": str(presets_mod.save(out, data))})
            if u.path == "/api/develop":
                if APP.sess is None:
                    return self._json({"error": "спершу відкрий кадр"}, 409)
                APP.job(lambda: do_develop(d.get("params", {})))
                return self._json({"ok": True})
            if u.path == "/api/stage":
                if APP.sess is None:
                    return self._json({"error": "спершу відкрий кадр"}, 409)
                st = d.get("stage", "")
                APP.job(lambda: do_stage(st, d.get("params", {})))
                return self._json({"ok": True})
            if u.path == "/api/write":
                if APP.sess is None or APP.sess.result is None:
                    return self._json({"error": "нема чого писати"}, 409)
                APP.job(lambda: do_write(d.get("out") or "out",
                                         bool(d.get("preview", True))))
                return self._json({"ok": True})
            return self._json({"error": "немає такого"}, 404)
        except Exception as e:                               # noqa: BLE001
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _preset(self, d: dict) -> dict:
        """Прочитати, накласти або замінити пресет UI.

        Накладаємо, а не замінюємо, коли прийшов файл: у фотографа є
        пресет на зйомку і уточнення на кадр, і саме їх послідовність —
        сенс формату (spec.md §1.2). Замінити цілком просить `replace`.
        """
        if d.get("path"):
            try:
                data = presets_mod.load(Path(d["path"]).expanduser())
            except presets_mod.PresetError as e:
                return {"error": str(e)}
            APP.preset_name = str(data.get("name") or Path(d["path"]).stem)
        else:
            data = d.get("data") or {}
            APP.preset_name = str(data.get("name") or APP.preset_name)
        APP.preset = (dict(data) if d.get("replace")
                      else presets_mod.merge(APP.preset, data))
        if d.get("clear"):
            APP.preset, APP.preset_name = {}, ""
        # Прогнати крізь Config, щоб зауваження показалися ЗАРАЗ, а не
        # мовчки спливли при наступному прогоні.
        APP.preset_notes = presets_mod.apply(Config(), APP.preset)
        return {"ok": True, "preset": APP.preset, "notes": APP.preset_notes,
                "name": APP.preset_name}

    def _xmp(self, d: dict) -> dict:
        """Прочитати налаштування Camera Raw і накласти те, що вміємо.

        Звіт повертається завжди і повністю: у цій гілці мовчазне
        «наближено» — головний ризик (PLAN.md §5), тому «×» видно поруч
        з «=» і «≈», а не ховається за «докладніше».
        """
        from . import xmp as xmp_mod

        src = d.get("path") or (str(APP.sess.path) if APP.sess else "")
        if not src:
            return {"error": "нема кадру і не задано шляху до XMP"}
        try:
            if str(src).lower().endswith(".xmp"):
                pre, rep = xmp_mod.to_preset(xmp_mod.read(src))
                where = str(src)
            else:
                pre, rep, where = xmp_mod.from_image(src)
        except xmp_mod.XmpError as e:
            return {"error": str(e)}
        if pre is None:
            return {"error": f"поруч із {Path(src).name} немає ні сайдкара, "
                             f"ні вбудованого блоку XMP"}
        APP.preset = presets_mod.merge(APP.preset, pre)
        APP.preset_name = str(pre.get("name") or APP.preset_name)
        APP.preset_notes = presets_mod.apply(Config(), APP.preset)
        return {"ok": True, "preset": APP.preset, "source": where,
                "notes": APP.preset_notes, "summary": rep.summary(),
                "exact": rep.exact, "approx": rep.approx, "ignored": rep.ignored}

    def _warp(self, d: dict) -> dict:
        """Один мазок пластики. Координати приходять у пікселях КАДРУ."""
        s = APP.sess
        if s is None or s.img is None:
            return {"error": "кадр не відкрито"}
        f = s.warp_field()
        if d.get("clear"):
            f.clear()
            return {"ok": True, "warp": None}
        tool = d.get("tool", "push")
        x, y = float(d.get("x", 0)), float(d.get("y", 0))
        r = max(4.0, float(d.get("radius", 100)))
        k = float(d.get("strength", 1.0))
        if tool == "push":
            f.push(x, y, r, float(d.get("mx", 0)), float(d.get("my", 0)), k)
        elif tool == "bloat":
            f.bloat(x, y, r, -abs(float(d.get("amount", 0.4))), k)
        elif tool == "pucker":
            f.bloat(x, y, r, abs(float(d.get("amount", 0.4))), k)
        elif tool == "twirl":
            f.twirl(x, y, r, float(d.get("angle", 0.3)), k)
        else:
            return {"error": f"невідомий інструмент: {tool}"}
        return {"ok": True, "warp": f.stats()}

    # --- допоміжне ------------------------------------------------------
    def _paint(self, d: dict) -> dict:
        """Пензель із браузера. Приходить PNG у зменшеному масштабі:
        гнати повнорозмірну маску туди-сюди на 26 Мп немає потреби —
        пензлем правлять великі області, а не окремі пікселі."""
        s = APP.sess
        if s is None or s.img is None:
            return {"error": "кадр не відкрито"}
        target = d.get("target", "skin")
        if d.get("clear"):
            if target == "remove":
                APP.remove_mask = None
                return {"ok": True, "remove_px": 0}
            APP.painted = None
            apply_painted(s, None)
            return {"ok": True, "painted": False}
        raw = base64.b64decode(d["png"].split(",", 1)[-1])
        buf = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if buf is None or buf.ndim != 3 or buf.shape[2] < 4:
            return {"error": "очікував RGBA-PNG від пензля"}
        h, w = s.img.shape[:2]
        big = cv2.resize(buf, (w, h), interpolation=cv2.INTER_NEAREST)
        a, r = big[:, :, 3], big[:, :, 2]
        if target == "remove":
            # для видалення півтонів не буває: пікселя або немає, або він є
            APP.remove_mask = ((a > 127) & (r > 127)).astype(np.uint8)
            from .inpaint import mask_stats
            st = mask_stats(APP.remove_mask)
            return {"ok": True, "remove_px": st["area"], "depth": st["depth"]}

        painted = np.zeros((h, w), np.int8)
        painted[(a > 127) & (r > 127)] = 1      # червоне = домалювати шкіру
        painted[(a > 127) & (r <= 127)] = -1    # синє = стерти
        APP.painted = painted
        apply_painted(s, painted)
        return {"ok": True, "painted": True,
                "skin_frac": round(float(s.skin.mean()), 4)}

    def _browse(self, d: str) -> dict:
        base = Path(d).expanduser() if d else Path.home()
        if not base.is_dir():
            base = base.parent if base.parent.is_dir() else Path.home()
        dirs, files = [], []
        try:
            for p in sorted(base.iterdir(), key=lambda x: x.name.lower()):
                if p.name.startswith("."):
                    continue
                if p.is_dir():
                    dirs.append(p.name)
                elif p.suffix.lower() in SUFFIXES | RAW_SUFFIXES:
                    files.append({"name": p.name,
                                  "raw": p.suffix.lower() in RAW_SUFFIXES,
                                  "mb": round(p.stat().st_size / 2**20, 1)})
        except PermissionError:
            pass
        return {"dir": str(base), "up": str(base.parent), "dirs": dirs, "files": files}


def serve(port: int = 8765, open_browser: bool = True) -> None:
    # Прогріваємо кеш моделей одразу: перше сканування відкриває LaMa на
    # 198 МБ і коштує секунд десять. Краще заплатити їх, поки людина
    # відкриває браузер, ніж коли вона вже клацнула по списку.
    threading.Thread(target=scan_models, daemon=True).start()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"retouch-lab UI: {url}\nCtrl+C — зупинити")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nзупинено")
    finally:
        httpd.server_close()


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="retouch-ui",
                                 description="локальний інтерфейс retouch-lab")
    ap.add_argument("input", nargs="?", help="одразу відкрити цей файл")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    if a.input:
        try:
            do_open(str(Path(a.input).expanduser()), {})
        except InputError as e:
            print(e)
    serve(a.port, not a.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
