"""Тести локального сервера. Головні інваріанти:

  1. моделі впізнаються ЗА ПІДПИСОМ, а не за назвою файлу;
  2. кроп ріжеться в рідних пікселях і не залежить від розміру кадру;
  3. пензель віддає правки як +1/-1 поверх автоматичної маски, а не
     замість неї;
  4. стан не бреше: busy, помилки й попередження доходять до клієнта;
  5. запити без відкритого кадру відмовляють зрозуміло, а не падають;
  6. форма нових вкладок будується зі СХЕМИ, тому схема має віддаватися
     цілою й у тій самій формі, що читає агент;
  7. пресет у UI — той самий словник, що в файлі: накладається, а не
     замінює, і зауваження доходять одразу, а не при наступному прогоні.

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

        # Формат не однаковий, і це навмисно. Маска, покриття й карта
        # знахідок — це ДВА РІВНІ, і JPEG розмив би межу, по якій на них
        # і дивляться. Фотографічні види йдуть JPEG: прев'ю пластики
        # важило 1.3 МБ у PNG і перечитувалось після кожного мазка.
        for kind in ("mask", "coverage", "detected"):
            code, body, ctype = _get(f"/api/view?kind={kind}&w=200")
            assert code == 200 and ctype == "image/png" and len(body) > 100, kind
        sizes = {}
        for kind in ("before", "after", "diff"):
            code, body, ctype = _get(f"/api/view?kind={kind}&w=760")
            assert code == 200 and ctype == "image/jpeg" and len(body) > 100, kind
            sizes[kind] = len(body)
        # Ширини навмисно різні: розбіжність в один піксель між
        # складанням і зменшенням ховалась саме там, де округлення
        # збігалося. Вид «різниця» від цього падав із 500.
        for w in (200, 513, 760, 959):
            for kind in ("before", "after", "diff"):
                code, body, _ct = _get(f"/api/view?kind={kind}&w={w}")
                assert code == 200 and len(body) > 100, (kind, w, code)
        print(f"  двійкові види PNG, фотографічні JPEG: "
              f"{ {k: v // 1024 for k, v in sizes.items()} } КБ")
        assert max(sizes.values()) < 600_000, (
            "фотографічний вид знову важить як PNG — канвас від цього й "
            "підвисав")


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


# ---------------------------------------------------------------------------
# схема, пресети, нові етапи
# ---------------------------------------------------------------------------

def test_schema_endpoint_feeds_the_form():
    """Вкладки «Проявлення», «Світлотінь» і «Інструменти» малюються з цього.

    Порожній розділ означає порожню вкладку, і помітили б ми це лише
    відкривши браузер — тому перевіряємо тут.
    """
    _start()
    code, body, _ = _get("/api/schema")
    sc = json.loads(body)
    need = ["develop", "dodgeburn", "tools.teeth", "tools.mattify",
            "tools.eye_vessels", "tools.skin_tone"]
    print("  розділи: " + ", ".join(f"{k}({len(sc['sections'][k])})" for k in need))
    assert code == 200
    for k in need:
        assert sc["sections"].get(k), f"розділ {k} порожній — вкладка буде порожня"
        for name, meta in sc["sections"][k].items():
            assert meta["doc"].strip(), f"{k}.{name} без опису — поле буде без підпису"
    assert sc.get("example"), "приклад для агента зник зі схеми"


def test_preset_stacks_and_reports_notes():
    """Накладається, а не замінює; помилку показує ЗАРАЗ."""
    _start()
    _post("/api/preset", {"clear": True})
    _post("/api/preset", {"data": {"tools": {"teeth": {"strength": 0.4}}}})
    code, r = _post("/api/preset", {"data": {"tools": {"mattify": {}},
                                             "dodgeburn_on": True}})
    print(f"  після двох накладань: {r['preset']}")
    assert code == 200
    assert r["preset"]["tools"]["teeth"] == {"strength": 0.4}, "друге затерло перше"
    assert "mattify" in r["preset"]["tools"] and r["preset"]["dodgeburn_on"]

    _, bad = _post("/api/preset", {"data": {"tools": {"teeth": {"нема": 1}}}})
    print(f"  зауваження: {bad['notes']}")
    assert bad["notes"], "невідомий параметр проковтнуло мовчки"

    _, cleared = _post("/api/preset", {"clear": True})
    assert cleared["preset"] == {}, "очищення не спрацювало"


def test_preset_list_shows_why():
    """Вибрати з десяти пресетів агента можна лише за причиною."""
    _start()
    rows = json.loads(_get("/api/presets?dir=presets")[1])
    if not rows:
        print("  у presets/ порожньо — перевіряти нема на чому")
        return
    print("  " + "; ".join(f"{r['name']}: {len(r.get('why',''))} симв." for r in rows))
    for r in rows:
        assert r.get("name"), "пресет без назви — у списку буде порожній рядок"
        assert "error" in r or r.get("keys") is not None


def test_preset_save_roundtrip():
    """Записаний з UI пресет має читатися назад тим самим модулем."""
    _start()
    from retouch import presets as pm
    with tempfile.TemporaryDirectory() as t:
        _post("/api/preset", {"clear": True})
        _post("/api/preset", {"data": {"detect": {"threshold": 0.019}}})
        code, r = _post("/api/preset/save",
                        {"path": t, "file": "ui", "why": "перевірка запису"})
        back = pm.load(r["path"])
        print(f"  {Path(r['path']).name}: {back}")
        assert code == 200 and back["detect"]["threshold"] == 0.019
        assert back["why"] == "перевірка запису", "причина не дійшла до файлу"
        _post("/api/preset", {"clear": True})


def test_develop_tab_rereads_the_frame():
    """Кроп міняє геометрію, тому проявлення — це перечитування файлу."""
    _start()
    with tempfile.TemporaryDirectory() as t:
        p = _fixture(Path(t))
        _post("/api/open", {"path": str(p), "params": {}})
        st = _wait_idle()
        w0, h0 = st["w"], st["h"]
        _post("/api/preset", {"clear": True})
        _post("/api/preset", {"data": {"develop": {"crop": [0.1, 0.1, 0.8, 0.8]}}})
        code, r = _post("/api/develop", {"params": {}})
        st = _wait_idle()
        print(f"  {w0}x{h0} -> {st['w']}x{st['h']}, помилка: {st['error']}")
        assert code == 200 and r.get("ok") and not st["error"]
        assert (st["w"], st["h"]) != (w0, h0), "кроп не застосувався"
        # crop — це (x0, y0, x1, y1) у частках, тобто рамка, а не розмір
        assert st["w"] == int(w0 * 0.8) - int(w0 * 0.1), st["w"]
        assert st["h"] == int(h0 * 0.8) - int(h0 * 0.1), st["h"]
        _post("/api/preset", {"clear": True})


def test_dodgeburn_stage_runs_and_can_be_switched_off():
    """Етап рахується поверх лікування і знімається галочкою назад."""
    _start()
    with tempfile.TemporaryDirectory() as t:
        p = _fixture(Path(t))
        _post("/api/open", {"path": str(p), "params": {}})
        _wait_idle()
        _post("/api/rerun", {"params": {}})
        _wait_idle()
        _post("/api/preset", {"clear": True})
        _post("/api/preset", {"data": {"dodgeburn_on": True,
                                       "dodgeburn": {"strength": 0.5}}})
        _post("/api/stage", {"stage": "dodgeburn", "params": {}})
        st = _wait_idle()
        print(f"  увімкнено: {st['db']}, помилка {st['error']}")
        assert st["db"] and st["db"]["touched"] > 0, "D&B нічого не торкнувся"

        _post("/api/preset", {"clear": True})
        _post("/api/stage", {"stage": "dodgeburn", "params": {}})
        st = _wait_idle()
        print(f"  вимкнено: {st['db']}")
        assert st["db"] is None, "знята галочка лишила результат у кадрі"


def test_tools_stage_without_class_map_says_so():
    """Без face-parsing інструменти пропускаються — але стан це показує."""
    _start()
    with tempfile.TemporaryDirectory() as t:
        p = _fixture(Path(t))
        _post("/api/open", {"path": str(p), "params": {}})
        _wait_idle()
        _post("/api/rerun", {"params": {}})
        st = _wait_idle()
        _post("/api/preset", {"clear": True})
        _post("/api/preset", {"data": {"tools": {"mattify": {}}}})
        _post("/api/stage", {"stage": "tools", "params": {}})
        st = _wait_idle()
        print(f"  has_cls={st['has_cls']}, шарів {st['tool_layers']}, "
              f"помилка {st['error']}")
        assert not st["error"], "без карти класів має пропустити, а не впасти"
        assert st["has_cls"] is False and st["tool_layers"] == []
        _post("/api/preset", {"clear": True})


def test_unknown_stage_is_reported():
    """Друкарська помилка в назві етапу не має тихо нічого не робити."""
    _start()
    with tempfile.TemporaryDirectory() as t:
        p = _fixture(Path(t))
        _post("/api/open", {"path": str(p), "params": {}})
        _wait_idle()
        _post("/api/stage", {"stage": "нема-такого", "params": {}})
        st = _wait_idle()
        print(f"  помилка: {st['error']}")
        assert st["error"] and "нема-такого" in st["error"]


def test_xmp_import_merges_and_reports():
    """Кнопка «Взяти з XMP»: накладає те, що вміємо, і показує решту."""
    _start()
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        p = _fixture(d)
        side = Path(__file__).resolve().parent / "fixtures" / "acr_sidecar.xmp"
        (d / "T.xmp").write_text(side.read_text(encoding="utf-8"), encoding="utf-8")
        _post("/api/open", {"path": str(p), "params": {}})
        _wait_idle()
        _post("/api/preset", {"clear": True})
        code, r = _post("/api/xmp", {})
        print(f"  {r.get('summary')}")
        print(f"  у пресет: {sorted((r.get('preset') or {}).get('develop', {}))}")
        assert code == 200 and not r.get("error"), r
        assert r["preset"]["develop"]["crop"], "кроп не дійшов"
        assert r["approx"] and r["ignored"], (
            "звіт без «наближено» або без «не застосовано» — "
            "саме вони й роблять його чесним")
        assert not r["notes"], f"пресет з XMP дав зауваження: {r['notes']}"
        _post("/api/preset", {"clear": True})


def test_xmp_without_sidecar_says_so():
    """Немає сайдкара — зрозуміла відмова, а не порожній успіх."""
    _start()
    with tempfile.TemporaryDirectory() as t:
        p = _fixture(Path(t))
        _post("/api/open", {"path": str(p), "params": {}})
        _wait_idle()
        code, r = _post("/api/xmp", {})
        print(f"  -> {code}: {str(r.get('error'))[:60]}")
        assert r.get("error") and not r.get("ok")


def test_class_choice_survives_rerun():
    """«Що вважати шкірою» — рішення під кадр, і воно має пережити прогін.

    Було так: /api/classes міняв Config, а «Перегнати» будував Config із
    форми наново — набір мовчки повертався до дефолтного. Маска в
    пам'яті лишалась однією, конфіг казав інше, і будь-яка наступна
    перебудова маски повертала знятий клас назад.
    """
    _start()
    with tempfile.TemporaryDirectory() as t:
        p = _fixture(Path(t))
        _post("/api/preset", {"clear": True})
        _post("/api/open", {"path": str(p), "params": {}})
        _wait_idle()
        # Ваг у репозиторії немає, тож карту класів кладемо самі — інакше
        # has_cls=False, тест мовчки нічого не перевіряє й лишається
        # зеленим. На цьому в цьому наборі вже наступали двічі.
        from retouch.masks import CELEBA_CLASSES
        inv = {v: k for k, v in CELEBA_CLASSES.items()}
        sess = webui.APP.sess
        h, w = sess.img.shape[:2]
        cls = np.full((h, w), inv["background"], np.int32)
        cls[: h // 2, :] = inv["skin"]
        cls[h // 2:, :] = inv["neck"]        # там, де на портреті груди
        sess.cls = cls
        st = json.loads(_get("/api/state")[1])
        assert st["has_cls"], "карта класів не дійшла до стану"
        _post("/api/classes", {"skin_classes": ["skin", "nose"]})
        _post("/api/rerun", {"params": {}})
        st = _wait_idle()
        print(f"  після «Перегнати»: {st['skin_classes']}, "
              f"у пресеті {st['preset'].get('mask')}, "
              f"маска {st['skin_frac']:.1%}, знайдене {st['blob_classes']}")
        assert st["skin_classes"] == ["skin", "nose"], "набір класів скинувся"
        assert st["preset"]["mask"]["skin_classes"] == ["skin", "nose"], (
            "вибір не потрапив у пресет — його не можна ні зберегти, ні "
            "повторити на наступному кадрі")
        assert not any(c["name"] == "neck" for c in st["blob_classes"]), (
            "знайдене лишилось у знятому класі — маску не перебудовано")
        _post("/api/preset", {"clear": True})


def test_shoot_listing_marks_written_frames():
    """«Готово» в панелі і «готово» в пакеті мають означати одне й те саме.

    Ознака одна на обох — зведений файл, за яким `batch.already_done`
    вирішує, чи кадр дійшов до кінця. Два різні означення розійшлися б
    на першому ж перерваному прогоні.
    """
    _start()
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src = d / "shoot"
        src.mkdir()
        for n in ("A", "B"):
            img, _s, _tr = make_face(h=600, w=460, face_w=340, n_spots=6, seed=1)
            cv2.imwrite(str(src / f"{n}.tif"),
                        (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16))
        out = d / "out"
        out.mkdir()
        rows = json.loads(_get(f"/api/session?dir={src}&out={out}")[1])
        print(f"  до запису: {[(r['file'], r['done']) for r in rows]}")
        assert len(rows) == 2 and not any(r["done"] for r in rows)

        (out / "A_99_flat.tif").write_bytes(b"x")     # ніби кадр записано
        rows = json.loads(_get(f"/api/session?dir={src}&out={out}")[1])
        print(f"  після: {[(r['file'], r['done']) for r in rows]}")
        assert dict((r["file"], r["done"]) for r in rows) == {"A.tif": True,
                                                              "B.tif": False}


def test_batch_from_ui_uses_the_same_engine_and_reports_each_frame():
    """UI не має власного циклу по кадрах.

    Якби мав, «прогнати один кадр руками» і «прогнати теку на ніч»
    давали б різний результат, і дізнались би ми про це вранці.
    """
    _start()
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src = d / "shoot"
        src.mkdir()
        for n in ("A", "B"):
            img, _s, _tr = make_face(h=600, w=460, face_w=340, n_spots=6, seed=2)
            cv2.imwrite(str(src / f"{n}.tif"),
                        (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16))
        out = d / "out"
        code, r = _post("/api/session/batch",
                        {"dir": str(src), "out": str(out), "params": {}})
        assert code == 200 and r.get("ok"), r
        st = _wait_idle()
        print(f"  {[(b['file'], b['status']) for b in st['batch']]}")
        print(f"  {st['written']}")
        assert len(st["batch"]) == 2
        assert all(b["status"] == "done" for b in st["batch"]), st["batch"]
        assert (out / "A_99_flat.tif").exists()

        # продовження: другий прогін має пропустити те, що вже записано
        _post("/api/session/batch", {"dir": str(src), "out": str(out), "params": {}})
        st = _wait_idle()
        print(f"  повторно: {[(b['file'], b['status']) for b in st['batch']]}")
        assert all(b["status"] == "skipped" for b in st["batch"])


def test_batch_without_a_folder_refuses():
    """Порожня тека — зрозуміла відмова, а не мовчазний «успіх» ні над чим.

    Теку скидаємо явно: попередні тести її задають, і без скидання тест
    просто нічого не перевіряв би, лишаючись зеленим.
    """
    _start()
    saved = webui.APP.shoot_dir
    webui.APP.shoot_dir = ""
    try:
        code, r = _post("/api/session/batch", {"params": {}})
        print(f"  -> {code}: {r.get('error')}")
        assert code == 409 and r.get("error")
    finally:
        webui.APP.shoot_dir = saved


def test_variants_compare_presets_on_one_frame():
    """Друга половина роботи з агентом: обрати з десяти пресетів."""
    _start()
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        p = _fixture(d)
        a = d / "a.yaml"
        a.write_text("name: м'яко\nwhy: обережно\ndetect: {threshold: 0.03}\n",
                     encoding="utf-8")
        b = d / "b.yaml"
        b.write_text("name: жорстко\nwhy: щільно\ndetect: {threshold: 0.006}\n",
                     encoding="utf-8")
        _post("/api/open", {"path": str(p), "params": {}})
        _wait_idle()
        code, r = _post("/api/variants", {"presets": [str(a), str(b)],
                                          "x": 480, "y": 620, "size": 200,
                                          "params": {}})
        assert code == 200 and r.get("ok"), r
        st = _wait_idle()
        vs = st["variants"]
        print(f"  {[(v['name'], v['blobs']) for v in vs]}")
        assert len(vs) == 3, "оригінал і два варіанти — три картки"
        assert vs[0]["name"] == "ОРИГІНАЛ" and vs[0]["blobs"] is None
        assert vs[1]["why"] and vs[2]["why"], "причина не дійшла до картки"
        assert vs[2]["blobs"] > vs[1]["blobs"], (
            "жорсткіший поріг дав не більше знахідок — порівнювати нема чого")
        for i in range(3):
            code, body, ctype = _get(f"/api/variant?i={i}")
            assert code == 200 and ctype.startswith("image/"), (code, ctype)
        code, _b, _c = _get("/api/variant?i=99")
        assert code == 404, "неіснуючий варіант мав дати 404"


def test_variants_say_when_develop_is_not_applied():
    """Проявлення живе в load(). Мовчки його проігнорувати означало б
    показати варіант, якого пресет не описує."""
    _start()
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        p = _fixture(d)
        v = d / "v.yaml"
        v.write_text("name: з проявленням\ndevelop: {contrast: 0.3}\n",
                     encoding="utf-8")
        _post("/api/open", {"path": str(p), "params": {}})
        _wait_idle()
        _post("/api/variants", {"presets": [str(v)], "x": 480, "y": 620,
                                "size": 160, "params": {}})
        st = _wait_idle()
        notes = st["variants"][1]["notes"]
        print(f"  {notes}")
        assert any("develop" in n for n in notes), (
            "розділ develop проігноровано мовчки")


def test_variants_refuse_without_presets_or_frame():
    _start()
    saved, webui.APP.sess = webui.APP.sess, None
    try:
        code, r = _post("/api/variants", {"presets": ["x.yaml"]})
        print(f"  без кадру -> {code}: {r.get('error')}")
        assert code == 409
    finally:
        webui.APP.sess = saved
    if webui.APP.sess is not None:
        code, r = _post("/api/variants", {"presets": []})
        print(f"  без пресетів -> {code}: {r.get('error')}")
        assert code == 409


# ---------------------------------------------------------------------------
# інтерфейс: що можна ховати, а що ні
# ---------------------------------------------------------------------------

def _page() -> str:
    _start()
    return _get("/")[1].decode("utf-8")


def test_warnings_are_outside_the_foldable_area():
    """Пояснення згортаються, ПОПЕРЕДЖЕННЯ — ніколи.

    §1 будується на тому, що конвеєр нічого не робить мовчки. Сховати
    «39 зі 154 знахідок на шиї» за кнопкою означало б це порушити, тому
    гарантія структурна, а не за домовленістю: попередження живуть у
    #msg, згортання торкається виключно #pane.
    """
    html = _page()
    assert "#msg" in html and "#pane" in html
    i_msg, i_pane = html.index('id="msg"'), html.index('id="pane"')
    assert i_msg < i_pane, "попередження мають стояти ПЕРЕД панеллю"
    fold = html[html.index("function foldHints"):]
    fold = fold[:fold.index("\n}")]
    print(f"  foldHints обмежений: {'#pane' in fold}, "
          f"чіпає .note: {'.note' in fold}")
    assert "$('#pane')" in fold, "згортання не обмежене панеллю"
    assert ".note" not in fold, "згортання дотягнулось до попереджень"


def test_every_tab_has_a_hint():
    """Іконка без підказки — ребус: підписів на всіх не вистачає ширини,
    тож пояснення має бути хоч на наведення."""
    import re
    html = _page()
    tabs = re.findall(r'<button data-t="([a-z]+)"([^>]*)>', html)
    missing = [t for t, attrs in tabs if "data-tip" not in attrs]
    print(f"  вкладок {len(tabs)}, без підказки: {missing}")
    assert tabs, "вкладок не знайдено — розмітка змінилась"
    assert not missing, f"вкладки без підказки: {missing}"


def test_side_panel_controls_explain_themselves():
    """Повзунки, значення яких неочевидне, мають пояснення на наведення."""
    import re
    html = _page()
    side = html[html.index('<div class="side">'):html.index('<div class="main">')]
    need = ("target_coverage", "threshold", "radius", "search_radius",
            "face_model", "face_detector", "raw_decoder")
    missing = []
    for cid in need:
        row = side.rfind("<div class=\"row\"", 0, side.index(f'id="{cid}"'))
        chunk = side[row:side.index(f'id="{cid}"')]
        if "data-tip" not in chunk:
            missing.append(cid)
    print(f"  перевірено {len(need)} контролів, без пояснення: {missing}")
    assert not missing, f"без пояснення: {missing}"


def test_every_schema_field_has_a_human_name():
    """Друга мова на ті самі поля — для фотографа, а не для агента.

    Перший зовнішній тест на живій людині дав саме цю відповідь:
    «виглядає технічно, для програмістів». У панелі стояли
    `wb_multipliers`, `noise_thr`, `§6.2` — мова коду, не мова
    фотографії. Тест ловить поле, додане без людської назви, у той
    самий день, коли його додали.
    """
    _start()
    sc = json.loads(_get("/api/schema")[1])
    missing, no_level = [], []
    for sec, fields in sc["sections"].items():
        for name, meta in fields.items():
            if meta.get("label", name) == name:
                missing.append(f"{sec}.{name}")
            if meta.get("level") not in ("basic", "advanced"):
                no_level.append(f"{sec}.{name}")
    total = sum(len(f) for f in sc["sections"].values())
    basic = sum(1 for f in sc["sections"].values()
                for m in f.values() if m.get("level") == "basic")
    print(f"  полів {total}, з них щоденних {basic}; без назви: {missing}")
    assert not missing, f"без людської назви: {missing}"
    assert not no_level, f"без рівня: {no_level}"
    assert 10 <= basic <= total // 2, (
        f"щоденних полів {basic} з {total} — режим «Просто» або порожній, "
        f"або нічим не відрізняється від «Експерт»")


def test_technical_description_survives_next_to_the_human_one():
    """Людська назва ДОДАЄТЬСЯ, а не замінює: агентові потрібен саме
    технічний опис, і §1 забороняє його втрачати."""
    _start()
    sc = json.loads(_get("/api/schema")[1])
    m = sc["sections"]["dodgeburn"]["eps"]
    print(f"  label={m['label']!r}\n  doc={m['doc'][:60]!r}")
    assert m["label"] != "eps" and m["doc"], "технічний опис зник"
    assert "guided" in m["doc"], "у doc уже не технічний текст"


def test_ui_does_not_quote_the_spec_at_the_photographer():
    """«§6.2» у панелі — це документація для розробника, що витекла в
    інтерфейс. У режимі «Експерт» вона доречна, у видимому тексті — ні."""
    html = _page()
    body = html[html.index('<div class="wrap">'):]
    import re
    refs = re.findall(r'§\d[\d.]*', body)
    print(f"  згадок специфікації у видимій розмітці: {len(refs)} {set(refs)}")
    assert len(refs) <= 3, f"специфікація протекла в панель: {refs}"


def test_class_names_are_not_the_dataset_ones():
    """`l_lip`, `u_lip`, `neck_l` — імена класів чужого датасету.

    Фотограф має бачити «губа верхня». Машинне ім'я лишається в
    «Експерт» і в пресеті: саме його треба писати в YAML.
    """
    from retouch.labels import CLASS_UA, class_name
    from retouch.masks import CELEBA_CLASSES
    missing = sorted(set(CELEBA_CLASSES.values()) - set(CLASS_UA))
    print(f"  класів у моделі {len(set(CELEBA_CLASSES.values()))}, "
          f"без перекладу: {missing}")
    assert not missing, f"класи без людської назви: {missing}"
    assert class_name("u_lip") == "губа верхня"
    assert class_name("нема_такого") == "нема_такого", (
        "невідомий клас має лишатись як є, а не перекладатись навмання")


def test_state_carries_both_names_for_classes():
    """Обидві мови поруч: панель бере ua, пресет і «Експерт» — name."""
    _start()
    with tempfile.TemporaryDirectory() as t:
        p = _fixture(Path(t))
        _post("/api/open", {"path": str(p), "params": {}})
        st = _wait_idle()
        for row in st.get("classes", []):
            assert "name" in row and "ua" in row, row
        if st.get("classes"):
            print(f"  {[(c['name'], c['ua']) for c in st['classes'][:3]]}")
        else:
            print("  карти класів немає (без ваг) — перевірено лише словник")


if __name__ == "__main__":
    fails = 0
    # Порядок — той, у якому тести написані: вони ділять один сервер і
    # один відкритий кадр, тож «відкрити» має йти раніше за «полікувати».
    # Але СПИСКОМ імен його тримати не можна: доданий тест мовчки не
    # запускався б, і набір лишався б зеленим, нічого не перевіривши.
    # Рівно на цьому вже наступили в test_cli.
    order = [n for n, f in list(globals().items())
             if n.startswith("test_") and callable(f)]
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
