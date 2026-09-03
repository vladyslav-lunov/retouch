"""Досьє на кадр для агента: що подивитись і що заміряно.

Агент не може відкрити CR3 і не може судити про ретуш по зменшеній
картинці — §1 прямо каже, що якість видно лише в масштабі 1:1. Тому
досьє складається з трьох частин, і кожна відповідає за своє:

  оглядовий JPEG   — про ПРОЯВЛЕННЯ: експозиція, ББ, колір. Це видно
                     на зменшеному, і саме це агент оцінює оком;
  кропи 1:1        — про РЕТУШ: текстура, плями, зерно. На зменшеному
                     їх немає за визначенням;
  brief.json/md    — про те, чого не видно взагалі: гістограма, розподіл
                     класів, скільки плям на якому порозі.

Розділення не формальне. Поріг детекції по картинці підібрати неможливо
— його треба МІРЯТИ, і саме тому в досьє є міні-таблиця по порогах.

    python3 scripts/brief.py IMG.CR3 -o brief
    python3 scripts/brief.py IMG.CR3 -o brief --face-model models/resnet18.onnx \
        --face-detector models/yunet.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from retouch import imageio
from retouch.blemish import DetectParams, detect_blemishes, heal_blemishes
from retouch.freqsep import freq_split, radius_for
from retouch.masks import CELEBA_CLASSES, MaskParams, mask_from_classes
from retouch.pipeline import Config, Session

SWEEP = (0.008, 0.012, 0.018, 0.025)


def tone_stats(img: np.ndarray) -> dict:
    """Гістограма в цифрах: де впирається, де середина.

    Агент не бачить гістограми на зменшеному JPEG — там усе виглядає
    прийнятно. Відсічені тіні й вибиті світла видно лише в числах.
    """
    out = {}
    for i, ch in enumerate(("b", "g", "r")):
        v = img[:, :, i]
        out[ch] = {
            "median": round(float(np.median(v)), 4),
            "p01": round(float(np.percentile(v, 1)), 4),
            "p99": round(float(np.percentile(v, 99)), 4),
            "clipped_low": round(float((v <= 0.002).mean()), 5),
            "clipped_high": round(float((v >= 0.998).mean()), 5),
        }
    lum = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]
    out["luma"] = {"median": round(float(np.median(lum)), 4),
                   "mean": round(float(lum.mean()), 4),
                   "p01": round(float(np.percentile(lum, 1)), 4),
                   "p99": round(float(np.percentile(lum, 99)), 4)}
    return out


def crops_for(sess, n: int = 3, size: int = 420) -> list[tuple[str, np.ndarray]]:
    """Кропи 1:1 там, де найбільше роботи — саме їх агент має розглядати."""
    h, w = sess.img.shape[:2]
    picks, out = [], []
    for b in sess.blobs:
        if len(picks) >= n:
            break
        x, y = int(b["center"][0]), int(b["center"][1])
        if any(abs(x - px) < size and abs(y - py) < size for px, py in picks):
            continue
        picks.append((x, y))
    for i, (x, y) in enumerate(picks, 1):
        x0 = int(np.clip(x - size // 2, 0, max(0, w - size)))
        y0 = int(np.clip(y - size // 2, 0, max(0, h - size)))
        out.append((f"crop{i}_{x}x{y}", sess.img[y0:y0 + size, x0:x0 + size]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="досьє на кадр для агента")
    ap.add_argument("input")
    ap.add_argument("-o", "--out", default="brief")
    ap.add_argument("--face-model", default=None)
    ap.add_argument("--face-detector", default=None)
    ap.add_argument("--raw-decoder", default=None, choices=("rawpy", "imageio"))
    ap.add_argument("--width", type=int, default=1400, help="ширина оглядового JPEG")
    a = ap.parse_args(argv)

    cfg = Config(face_model=a.face_model, face_detector=a.face_detector,
                 raw_decoder=a.raw_decoder, force_mask=True)
    sess = Session(a.input, cfg).load().analyze()
    img = sess.img
    h, w = img.shape[:2]
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    stem = Path(a.input).stem

    # --- картинки ---------------------------------------------------------
    def jpg(name, arr, q=92):
        p = d / f"{stem}_{name}.jpg"
        u8 = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
        cv2.imwrite(str(p), u8, [cv2.IMWRITE_JPEG_QUALITY, q])
        return p.name

    k = min(1.0, a.width / w)
    overview = cv2.resize(img, (int(w * k), int(h * k)),
                          interpolation=cv2.INTER_AREA) if k < 1 else img
    files = {"overview": jpg("overview", overview)}
    for name, arr in crops_for(sess):
        files[name] = jpg(name, arr)

    # --- виміри -----------------------------------------------------------
    # Який поріг ЦЕЙ кадр просить під розумну ціль. Агентові це
    # потрібніше за саму криву: поріг між кадрами не переноситься, і
    # писати в пресет треба число, підібране під кадр (spec.md §6.2).
    recommended = None
    try:
        recommended = sess.solve_threshold(0.03)
    except Exception as e:                                    # noqa: BLE001
        print(f"[brief] підбір порога не вийшов: {e}", flush=True)

    sweep = []
    for t in SWEEP:
        lbl, blobs = detect_blemishes(sess.high, sess.skin, DetectParams(threshold=t))
        _, cov = heal_blemishes(sess.high, lbl, blobs, sess.skin,
                                search_radius=sess.search_radius_px)
        sweep.append({"threshold": t, "blobs": len(blobs),
                      "touched": round(float((cov > 0).mean()), 5)})

    classes = {}
    if sess.cls is not None:
        for idx, name in CELEBA_CLASSES.items():
            frac = float((sess.cls == idx).mean())
            if frac > 1e-4:
                classes[name] = round(frac, 4)

    # --- що фотограф уже вирішив у проявнику -------------------------------
    # Агенту це потрібніше за будь-який наш вимір: він має пропонувати
    # ПОВЕРХ рішень фотографа, а не замість них. Тут же видно, чого ми з
    # цих рішень не вміємо — щоб агент не будував на них своє.
    xmp_block = None
    try:
        from retouch import xmp as xmp_mod
        pre, rep, where = xmp_mod.from_image(a.input)
        if pre is not None:
            xmp_block = {
                "source": where,
                "applied_exactly": rep.exact,
                "applied_approximately": rep.approx,
                "read_but_not_applied": rep.ignored,
                "preset": pre.get("develop", {}),
            }
    except Exception as e:                                    # noqa: BLE001
        xmp_block = {"error": f"{type(e).__name__}: {e}"}

    brief = {
        "frame": {"file": Path(a.input).name, "w": w, "h": h,
                  "mp": round(w * h / 1e6, 1),
                  "raw_decoder": sess.raw_decoder,
                  "bit_depth": str(sess.dtype)},
        "tone": tone_stats(img),
        "mask": {"source": sess.skin_source,
                 "coverage": round(float(sess.skin.mean()), 4) if sess.skin is not None else None,
                 "skin_classes": list(cfg.mask.skin_classes),
                 "classes_in_frame": classes},
        "detection": {"radius_px": round(sess.radius, 2),
                      "blobs_at_default": len(sess.blobs),
                      # Скільки знайдено — саме по собі нічого не каже:
                      # 154 на обличчі й 154 на ланцюжку виглядають
                      # однаково. Клас каже, ЩО знайдено.
                      "blobs_by_class": dict(sess.blob_classes),
                      "warning": sess.detect_warn,
                      "recommended_threshold": recommended,
                      "recommended_for_coverage": 0.03,
                      "threshold_note": sess.threshold_note,
                      "sweep": sweep},
        "camera_raw": xmp_block,
        "warnings": [x for x in [sess.warn] if x],
        "files": files,
        "how_to_read": {
            "overview": "про проявлення: експозиція, ББ, колір. Оцінювати оком.",
            "crops": "про ретуш: текстура і плями в масштабі 1:1. "
                     "На зменшеному їх не видно — §1.",
            "sweep": "поріг детекції підбирають ЗА ЦИМИ ЧИСЛАМИ, а не по картинці.",
            "camera_raw": "що фотограф уже зробив в ACR. Пропонуй ПОВЕРХ цього. "
                          "read_but_not_applied — те, чого ми не вміємо: "
                          "не спирайся на нього, кадр цього не отримав.",
        },
    }
    (d / f"{stem}_brief.json").write_text(
        json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- те саме текстом, щоб агент читав без розбору JSON ------------------
    L = [f"# Досьє: {Path(a.input).name}", "",
         f"{w}x{h} ({brief['frame']['mp']} Мп), декодер {sess.raw_decoder or '—'}, "
         f"розрядність {sess.dtype}", ""]
    t = brief["tone"]["luma"]
    L += ["## Тон", "",
          f"- яскравість: медіана {t['median']}, середня {t['mean']}, "
          f"p01 {t['p01']}, p99 {t['p99']}"]
    for ch in ("r", "g", "b"):
        c = brief["tone"][ch]
        L.append(f"- {ch}: медіана {c['median']}, зрізано в тінях "
                 f"{c['clipped_low']:.2%}, у світлах {c['clipped_high']:.2%}")
    L += ["", "## Маска", "",
          f"- джерело: {sess.skin_source}, покриття {brief['mask']['coverage']:.1%}",
          f"- як шкіра: {', '.join(brief['mask']['skin_classes'])}"]
    if classes:
        L.append("- класи в кадрі: " +
                 ", ".join(f"{k} {v:.1%}" for k, v in
                           sorted(classes.items(), key=lambda i: -i[1])))
    L += ["", "## Детекція", "",
          f"- радіус частотки {brief['detection']['radius_px']} px"]
    if sess.blob_classes:
        L.append("- знайдене по класах: " +
                 ", ".join(f"{n} {c}" for n, c in sess.blob_classes))
    if sess.detect_warn:
        L.append(f"- **УВАГА**: {sess.detect_warn}")
    if recommended:
        L.append(f"- **поріг під ціль 3% шкіри: `{recommended}`** — пиши в "
                 f"пресет саме його, дефолт 0.012 між кадрами не переноситься")
    if sess.threshold_note:
        L.append(f"- **УВАГА**: {sess.threshold_note}")
    L += ["", "| поріг | плям | дотиків |", "|---|---|---|"]
    for r in sweep:
        L.append(f"| {r['threshold']} | {r['blobs']} | {r['touched']:.3%} |")
    if xmp_block and "error" not in xmp_block:
        L += ["", "## Camera Raw", "", f"джерело: `{xmp_block['source']}`", ""]
        for sign, key in (("=", "applied_exactly"), ("≈", "applied_approximately"),
                          ("×", "read_but_not_applied")):
            for k_, v in xmp_block[key].items():
                L.append(f"- `{sign}` **{k_}**: {v}")
        L += ["", "Пропонуй ПОВЕРХ цього. Рядки з `×` кадр НЕ отримав — "
              "не будуй на них своє рішення."]
    if brief["warnings"]:
        L += ["", "## Попередження", ""] + [f"- {x.splitlines()[0]}" for x in brief["warnings"]]
    L += ["", "## Файли", ""]
    for k_, v in files.items():
        L.append(f"- `{v}` — {'загальний план' if k_ == 'overview' else 'кроп 1:1'}")
    L += ["", "## Як це читати", ""]
    for k_, v in brief["how_to_read"].items():
        L.append(f"- **{k_}**: {v}")
    (d / f"{stem}_brief.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"досьє в {d}/")
    for f in sorted(d.glob(f"{stem}_*")):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
