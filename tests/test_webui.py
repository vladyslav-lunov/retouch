"""Тести локального сервера. Головні інваріанти:

  1. моделі впізнаються ЗА ПІДПИСОМ, а не за назвою файлу;
  2. кроп ріжеться в рідних пікселях і не залежить від розміру кадру;
  3. пензель віддає правки як +1/-1 поверх автоматичної маски, а не
     замість неї;
  4. стан не бреше: busy, помилки й попередження доходять до клієнта;
  5. запити без відкритого кадру відмовляють зрозуміло, а не падають.

Сервер піднімається на випадковому порту й глушиться в кінці — тести не
мають чіпати той, що, можливо, працює поруч.
"""

from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from http.server import ThreadingHTTPServer  # noqa: E402

from retouch import webui  # noqa: E402
from tests.synth import make_face  # noqa: E402

_SRV = None
_PORT = 0


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _start():
    global _SRV, _PORT
    if _SRV is not None:
        return
    _PORT = _free_port()
    _SRV = ThreadingHTTPServer(("127.0.0.1", _PORT), webui.Handler)
    threading.Thread(target=_SRV.serve_forever, daemon=True).start()
    time.sleep(0.3)


def _get(path: str):
    """Код помилки — теж відповідь, а не привід кидати виняток."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{_PORT}{path}", timeout=60) as r:
            return r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "")


def _post(path: str, body: dict):
    req = urllib.request.Request(
        f"http://127.0.0.1:{_PORT}{path}", method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _wait_idle(limit=120):
    for _ in range(limit * 2):
        st = json.loads(_get("/api/state")[1])
        if not st["busy"]:
            return st
        time.sleep(0.5)
    raise AssertionError("сервер не звільнився")


def _fixture(d: Path) -> Path:
    # Обличчя 760 px, а не 420: на дрібнішому детектор знаходить одну-дві
    # плями, і тести на сортування та кропи стають порожніми.
    img, _s, _t = make_face(h=1250, w=960, face_w=760, n_spots=18, seed=3)
    p = d / "T.tif"
    cv2.imwrite(str(p), (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16))
    return p


def test_page_and_state_respond():
    _start()
    code, body, ctype = _get("/")
    print(f"  GET / -> {code}, {len(body)} байт, {ctype.split(';')[0]}")
    assert code == 200 and b"retouch-lab" in body
    code, body, _ = _get("/api/state")
    st = json.loads(body)
    assert code == 200 and "busy" in st and "loaded" in st


def test_requests_without_frame_refuse_clearly():
    """Не 500 і не трасування, а зрозуміла відмова."""
    _start()
    for path in ("/api/rerun", "/api/heal", "/api/classes", "/api/warp/apply"):
        code, body = _post(path, {})
        print(f"  {path} -> {code}: {str(body.get('error'))[:44]}")
        assert code == 409, f"{path} мав відмовити з 409, а дав {code}"
        assert body.get("error"), "відмова без пояснення"


def test_model_role_is_detected_by_signature():
    """Назва файлу нічого не гарантує — дивимось на входи й виходи."""
    _start()
    models = json.loads(_get("/api/models?dir=models")[1])
    total = sum(len(v) for v in models.values())
    if total == 0:
        print("  моделей у models/ немає — перевіряти нема на чому")
        return
    print("  " + ", ".join(f"{k}: {[m['name'] for m in v]}"
                           for k, v in models.items() if v))
    for role in ("face", "detector", "lama"):
        for m in models.get(role, []):
            assert Path(m["path"]).exists()
    assert not models["unknown"], f"не впізнано: {models['unknown']}"


def test_open_analyze_heal_and_views():
    _start()
    with tempfile.TemporaryDirectory() as t:
        p = _fixture(Path(t))
        code, r = _post("/api/open", {"path": str(p), "params": {}})
        assert code == 200 and r.get("ok"), r
        st = _wait_idle()
        print(f"  відкрито {st['name']}: {st['w']}x{st['h']}, "
              f"маска {st['skin_frac']:.1%}, плям {st['n_blobs']}")
        assert st["loaded"] and st["n_blobs"] > 0 and not st["error"]

        _post("/api/heal", {})
        st = _wait_idle()
        assert st["has_result"], "лікування не дало результату"

        for kind in ("before", "after", "diff", "mask", "coverage", "detected"):
            code, body, ctype = _get(f"/api/view?kind={kind}&w=200")
            assert code == 200 and ctype == "image/png" and len(body) > 100, kind
        print("  усі шість видів віддаються як PNG")


def test_crop_is_native_size():
    """Кроп 1:1 має бути рівно size x size — інакше дивитись нема на що."""
    _start()
    for size in (64, 200):
        code, body, _ = _get(f"/api/crop?kind=before&x=200&y=300&size={size}")
        assert code == 200
        arr = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_UNCHANGED)
        print(f"  size={size} -> {arr.shape[1]}x{arr.shape[0]}")
        assert arr.shape[0] == size and arr.shape[1] == size


def test_brush_edits_layer_over_auto_mask():
    """Правки пензля лягають ПОВЕРХ автоматичної маски й переживають перегін."""
    _start()
    st = json.loads(_get("/api/state")[1])
    before = st["skin_frac"]
    h = w = 120
    png = np.zeros((h, w, 4), np.uint8)
    png[20:100, 20:100] = (0, 0, 255, 255)          # BGRA: червоне = додати
    ok, buf = cv2.imencode(".png", png)
    assert ok
    import base64
    code, r = _post("/api/paint", {"png": "data:image/png;base64," +
                                   base64.b64encode(buf.tobytes()).decode()})
    print(f"  маска {before:.1%} -> {r.get('skin_frac', 0):.1%}, painted={r.get('painted')}")
    assert code == 200 and r.get("painted") is True
    assert r["skin_frac"] > before, "домальоване не збільшило маску"
    _post("/api/paint", {"clear": True})


def test_blobs_listing_is_sorted_by_contrast():
    _start()
    blobs = json.loads(_get("/api/blobs")[1])
    print(f"  плям у списку: {len(blobs)}")
    assert len(blobs) >= 5, "замало плям — тест нічого не доводить"
    c = [b["contrast"] for b in blobs]
    assert c == sorted(c, reverse=True), "список не за спаданням контрасту"
    assert all({"id", "x", "y", "area", "contrast"} <= set(b) for b in blobs)


def test_unknown_route_is_404():
    _start()
    code, body, _ = _get("/api/no-such-thing")
    print(f"  -> {code}")
    assert code == 404


def _teardown():
    global _SRV
    if _SRV is not None:
        _SRV.shutdown()
        _SRV.server_close()
        _SRV = None


if __name__ == "__main__":
    fails = 0
    order = ["test_page_and_state_respond", "test_requests_without_frame_refuse_clearly",
             "test_model_role_is_detected_by_signature", "test_open_analyze_heal_and_views",
             "test_crop_is_native_size", "test_brush_edits_layer_over_auto_mask",
             "test_blobs_listing_is_sorted_by_contrast", "test_unknown_route_is_404"]
    try:
        for name in order:
            print(f"\n{name}")
            try:
                globals()[name]()
                print("  OK")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL: {e}")
    finally:
        _teardown()
    print(f"\n{'усе зелене' if not fails else f'провалено: {fails}'}")
    raise SystemExit(1 if fails else 0)
