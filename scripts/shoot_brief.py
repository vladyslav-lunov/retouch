"""Досьє на ЗЙОМКУ для агента: що спільне, що різне, і де межа пресету.

`brief.py` описує один кадр. Але пресет пишеться не на кадр, а на
зйомку — і головне питання агента інше: **чи існує один пресет на всі
кадри, чи їх треба кілька, і за якою ознакою ділити**.

Заміряно (spec.md §6.2.1): поріг контрасту між кадрами не переноситься
взагалі — на 44 кадрах однієї зйомки фіксовані 0.012 дали від 0% до 24%
торкнутої шкіри. Переноситься ЦІЛЬ. Тому досьє на зйомку показує не
середні значення, а РОЗКИД: середнє тут приховує саме те, заради чого
досьє й читають.

    python3 scripts/shoot_brief.py photos/ -o brief \\
        --face-model models/resnet18.onnx --face-detector models/yunet.onnx

Виходить:
    shoot.json       усі числа по кадрах і зведення по зйомці
    shoot.md         те саме прозою, з інструкцією, що з цим робити
    contact.jpg      контрольний аркуш: усі кадри дрібно
    NNN_*.jpg        кропи 1:1 з КРАЙНІХ кадрів, а не з випадкових

Крайніх, а не випадкових: пресет ламається на краях діапазону, і саме
там його треба дивитись. Кадр із медіанними числами нічого не перевіряє.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from retouch.batch import find_inputs
from retouch.imageio import InputError
from retouch.masks import CELEBA_CLASSES
from retouch.pipeline import Config, Session

TARGET = 0.03


def frame_row(path: Path, cfg_factory) -> dict:
    """Числа одного кадру. Дешевше за повний прогін: без лікування на
    повну і без запису — лише те, з чого пишеться пресет."""
    sess = Session(path, cfg_factory()).load()
    row: dict = {"file": path.name}
    h, w = sess.img.shape[:2]
    row.update({"w": w, "h": h, "mp": round(w * h / 1e6, 1),
                "raw_decoder": sess.raw_decoder,
                "skin_source": sess.skin_source,
                "skin_frac": round(float(sess.skin.mean()), 5)
                if sess.skin is not None else None})
    sess.analyze()
    row.update({
        "faces": len(sess.faces),
        "face_widths": [int(b[2]) for b in sess.faces],
        "face_w": sess.face_w,
        "radius_px": round(sess.radius, 2),
        "radius_clamped": bool(sess.radius_clamped),
        "search_radius_px": sess.search_radius_px,
        "recommended_threshold": sess.cfg.detect.threshold,
        "threshold_curve": sess.threshold_curve,
        "unreachable": sess.threshold_note,
        "blobs": len(sess.blobs),
        "blobs_by_class": dict(sess.blob_classes),
        "detect_warn": sess.detect_warn,
        "faces_note": sess.faces_note,
    })
    lum = (0.114 * sess.img[:, :, 0] + 0.587 * sess.img[:, :, 1]
           + 0.299 * sess.img[:, :, 2])
    row["luma_median"] = round(float(np.median(lum)), 4)
    row["clipped_high"] = round(float((lum > 0.99).mean()), 5)
    row["clipped_low"] = round(float((lum < 0.01).mean()), 5)
    if sess.skin is not None and sess.skin.any():
        row["skin_luma_median"] = round(float(np.median(lum[sess.skin > 0])), 4)
    return row, sess


def spread(rows: list[dict], key: str) -> dict | None:
    """Мін-медіана-макс по кадрах. Саме розкид, а не середнє: одне число
    на зйомку відповідає на питання, якого ніхто не ставив."""
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return {"min": min(vals), "median": statistics.median(vals),
            "max": max(vals), "n": len(vals)}


def contact_sheet(thumbs: list[tuple[str, np.ndarray]], cols: int = 6,
                  cell: int = 320) -> np.ndarray:
    """Контрольний аркуш. Дрібно і навмисно: він про те, ЯКІ це кадри й
    чим вони відрізняються між собою, а не про якість ретуші."""
    from retouch.preview import to_latin
    rows = (len(thumbs) + cols - 1) // cols
    bar = 22
    sheet = np.full((rows * (cell + bar), cols * cell, 3), 0.12, np.float32)
    for i, (name, img) in enumerate(thumbs):
        r, c = divmod(i, cols)
        y0, x0 = r * (cell + bar) + bar, c * cell
        ih, iw = img.shape[:2]
        k = min(cell / iw, cell / ih)
        t = cv2.resize(img, (max(1, int(iw * k)), max(1, int(ih * k))),
                       interpolation=cv2.INTER_AREA)
        sheet[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
        cv2.putText(sheet, to_latin(name), (x0 + 4, y0 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0.85, 0.85, 0.85), 1,
                    cv2.LINE_AA)
    return np.clip(sheet, 0, 1)


def extremes(rows: list[dict]) -> dict[str, str]:
    """Кадри, на яких пресет ламається найшвидше."""
    pick: dict[str, str] = {}

    def by(key, rev, label):
        have = [r for r in rows if r.get(key) is not None]
        if have:
            pick[label] = sorted(have, key=lambda r: r[key], reverse=rev)[0]["file"]

    by("recommended_threshold", True, "найжорсткіший поріг")
    by("recommended_threshold", False, "найм'якший поріг")
    by("face_w", True, "найбільше обличчя")
    by("face_w", False, "найменше обличчя")
    by("blobs", True, "найбільше знахідок")
    by("skin_frac", True, "найбільша маска")
    return pick


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="досьє на зйомку для агента")
    ap.add_argument("input", help="тека з кадрами")
    ap.add_argument("-o", "--out", default="brief")
    ap.add_argument("--face-model", default=None)
    ap.add_argument("--face-detector", default=None)
    ap.add_argument("--raw-decoder", default=None, choices=("rawpy", "imageio"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--target", type=float, default=TARGET,
                    help=f"ціль по частці торкнутої шкіри (типово {TARGET})")
    ap.add_argument("--crops", type=int, default=6,
                    help="скільки крайніх кадрів показати в 1:1")
    a = ap.parse_args(argv)

    files = find_inputs(a.input)
    if a.limit:
        files = files[:a.limit]
    if not files:
        print("порожня тека", file=sys.stderr)
        return 1

    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)

    def cfg_factory():
        from retouch.blemish import DetectParams
        return Config(face_model=a.face_model, face_detector=a.face_detector,
                      raw_decoder=a.raw_decoder, force_mask=True,
                      detect=DetectParams(target_coverage=a.target))

    rows, thumbs, skipped = [], [], []
    keep: dict[str, np.ndarray] = {}
    want = set()
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}", flush=True)
        try:
            row, sess = frame_row(f, cfg_factory)
        except (InputError, Exception) as e:            # noqa: BLE001
            print(f"  пропущено: {type(e).__name__}: {e}", flush=True)
            skipped.append({"file": f.name, "why": f"{type(e).__name__}: {e}"})
            continue
        rows.append(row)
        k = min(1.0, 400 / sess.img.shape[1])
        thumbs.append((f.stem, cv2.resize(sess.img, None, fx=k, fy=k,
                                          interpolation=cv2.INTER_AREA)))
        # Повний кадр тримати не можна — 44 по 300 МБ не влізуть у 8 ГБ
        # (spec.md §2). Тому кроп із кожного беремо ОДРАЗУ, а вибираємо
        # серед них потім: дешевше зберегти 44 вирізки, ніж 44 кадри.
        if sess.blobs:
            b = sess.blobs[0]
            x, y = int(b["center"][0]), int(b["center"][1])
            hh, ww = sess.img.shape[:2]
            x0 = int(np.clip(x - 210, 0, max(0, ww - 420)))
            y0 = int(np.clip(y - 210, 0, max(0, hh - 420)))
            keep[f.name] = sess.img[y0:y0 + 420, x0:x0 + 420].copy()
        del sess

    if not rows:
        print("жодного придатного кадру", file=sys.stderr)
        return 1

    ex = extremes(rows)
    for label, name in list(ex.items())[:a.crops]:
        if name in keep:
            p = d / f"crop_{name.split('.')[0]}.jpg"
            cv2.imwrite(str(p), (np.clip(keep[name], 0, 1) * 255).astype(np.uint8),
                        [cv2.IMWRITE_JPEG_QUALITY, 94])

    sheet = contact_sheet(thumbs)
    cv2.imwrite(str(d / "contact.jpg"),
                (sheet * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 88])

    ths = sorted({r["recommended_threshold"] for r in rows})
    multi = [r for r in rows if r["faces"] > 1]
    clamped = [r for r in rows if r["radius_clamped"]]
    unreachable = [r for r in rows if r.get("unreachable")]

    out = {
        "shoot": {"dir": str(a.input), "frames": len(rows),
                  "skipped": skipped, "target_coverage": a.target},
        "spread": {k: spread(rows, k) for k in
                   ("recommended_threshold", "face_w", "skin_frac", "blobs",
                    "radius_px", "luma_median", "skin_luma_median")},
        "one_preset_enough": len(ths) == 1,
        "distinct_thresholds": ths,
        "multi_face_frames": [r["file"] for r in multi],
        "radius_clamped_frames": [r["file"] for r in clamped],
        "target_unreachable_frames": [r["file"] for r in unreachable],
        "extremes": ex,
        "frames": rows,
        "how_to_use": {
            "головне": "пиши ЦІЛЬ (detect.target_coverage), а не поріг: "
                       "поріг між кадрами не переноситься (spec.md §6.2.1)",
            "скільки пресетів": "дивись spread.recommended_threshold. Якщо "
                                "min і max відрізняються в рази — одного "
                                "пресету на зйомку не існує, і ділити треба "
                                "за тим, що РОЗХОДИТЬСЯ, а не за номером кадру",
            "чому": "поле why обов'язкове. Десять пресетів у числах "
                    "виглядають однаково, і вибрати з них можна лише за "
                    "причиною (spec.md §1.2)",
            "crop_*.jpg": "кропи 1:1 з КРАЙНІХ кадрів. Про ретуш судити "
                          "тільки по них — на contact.jpg її не видно взагалі",
            "contact.jpg": "про що зйомка й чим кадри відрізняються: світло, "
                           "план, скільки людей",
            "схема": "python3 -m retouch.cli --schema",
        },
    }
    (d / "shoot.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    L = [f"# Досьє на зйомку: {len(rows)} кадрів", "",
         f"тека `{a.input}`, ціль {a.target:.0%} торкнутої шкіри", ""]
    if skipped:
        L += [f"пропущено {len(skipped)}: "
              + ", ".join(s["file"] for s in skipped), ""]
    L += ["## Що розходиться по зйомці", "",
          "| що | мін | медіана | макс |", "|---|---|---|---|"]
    names = {"recommended_threshold": "поріг під ціль", "face_w": "обличчя, px",
             "skin_frac": "маска, частка кадру", "blobs": "знахідок",
             "radius_px": "радіус частотки", "luma_median": "яскравість, медіана",
             "skin_luma_median": "яскравість шкіри"}
    for k, label in names.items():
        sp = out["spread"].get(k)
        if sp:
            f = (lambda v: f"{v:.4g}")
            L.append(f"| {label} | {f(sp['min'])} | {f(sp['median'])} | "
                     f"{f(sp['max'])} |")
    L += ["", "## Скільки пресетів потрібно", ""]
    if out["one_preset_enough"]:
        L.append(f"Усі кадри просять той самий поріг `{ths[0]}` — одного "
                 f"пресету досить.")
    else:
        L.append(f"Кадри просять **{len(ths)} різних порогів** "
                 f"({', '.join(str(t) for t in ths)}). Одного пресету на "
                 f"зйомку не існує. Але це не привід писати {len(ths)} "
                 f"пресетів: правильніше задати ЦІЛЬ "
                 f"`detect.target_coverage: {a.target}` одним пресетом, і "
                 f"поріг підбереться під кожен кадр сам.")
    if multi:
        L += ["", f"**Групових кадрів: {len(multi)} з {len(rows)}.** "
              f"Там працює `mask.max_faces`; за замовчуванням 4."]
    if clamped:
        L += ["", f"**Радіус упирається в підлогу на {len(clamped)} кадрах** "
              f"(обличчя вужче за 400 px). Результат там не порівнянний "
              f"з рештою — spec.md §6.3."]
    if unreachable:
        L += ["", f"**Ціль недосяжна на {len(unreachable)} кадрах**: навіть "
              f"найжорсткіший поріг лишає більше. Це кадри з дуже вираженою "
              f"текстурою, дивись їх у 1:1 окремо."]
    L += ["", "## Крайні кадри", "",
          "Пресет ламається на краях діапазону, тому дивитись треба саме їх:",
          ""]
    for label, name in ex.items():
        L.append(f"- **{label}**: `{name}`")
    L += ["", "## Кадри", "",
          "| кадр | облич | обличчя px | маска | поріг | знахідок |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['file']} | {r['faces']} | "
                 f"{r['face_w'] or '—'} | {(r['skin_frac'] or 0):.2%} | "
                 f"{r['recommended_threshold']} | {r['blobs']} |")
    L += ["", "## Як цим користуватись", ""]
    for k, v in out["how_to_use"].items():
        L.append(f"- **{k}**: {v}")
    (d / "shoot.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("\n" + "\n".join(L[:40]))
    print(f"\nзаписано в {d}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
