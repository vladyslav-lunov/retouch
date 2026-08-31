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
            "skin_classes": list(s.cfg.mask.skin_classes),
            "has_cls": s.cls is not None,
            "keep": (None if self.keep_ids is None else sorted(self.keep_ids)),
            "params": {
                "threshold": s.cfg.detect.threshold, "radius": s.cfg.hf_radius,
                "min_area": s.cfg.detect.min_area, "max_area": s.cfg.detect.max_area,
                "max_elongation": s.cfg.detect.max_elongation,
                "strength": s.cfg.strength, "limit": s.cfg.limit,
                "search_radius": s.cfg.search_radius,
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


def cfg_from(d: dict) -> Config:
    """Параметри з форми. Дефолти беруться з дата-класів, як і в CLI."""
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
    return c


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
    APP.sess = Session(path, cfg_from(params)).load()
    APP.painted, APP.keep_ids, APP.sweep, APP.written = None, None, None, []
    APP.proxy = None
    APP.sess.analyze(APP.sink)


def do_rerun(params: dict, keep_ids=None) -> None:
    """Перегнати відкритий кадр з іншими параметрами."""
    sess = APP.sess
    old_radius = sess.cfg.hf_radius
    sess.cfg = cfg_from(params)
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
        sess.cfg = cfg_from(p)
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
    sess.cfg = cfg_from(params)
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
                return self._json({"ok": True,
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
            if u.path == "/api/write":
                if APP.sess is None or APP.sess.result is None:
                    return self._json({"error": "нема чого писати"}, 409)
                APP.job(lambda: do_write(d.get("out") or "out",
                                         bool(d.get("preview", True))))
                return self._json({"ok": True})
            return self._json({"error": "немає такого"}, 404)
        except Exception as e:                               # noqa: BLE001
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

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
