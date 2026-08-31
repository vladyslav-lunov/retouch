"""Прогнати кілька пресетів по одному кадру і показати їх поруч.

Друга половина роботи з агентом. Агент дає десять пресетів — обрати з
них по YAML неможливо, у числах вони виглядають однаково. Потрібно
побачити, і побачити ПРАВИЛЬНО: загальний план для проявлення, кропи 1:1
для ретуші (§1). Тому аркуш робиться в тому самому масштабі, що й у
preview.py, а не «щоб влізло».

Кадр читається і розкладається ОДИН раз на всі варіанти: частотка від
пресету не залежить, а на 26 Мп вона коштує секунди.

    python3 scripts/variants.py IMG.tif presets/agent/*.yaml -o variants
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from retouch import presets as presets_mod
from retouch.pipeline import Config, Session
from retouch.preview import to_latin

F = cv2.FONT_HERSHEY_SIMPLEX
BG = 26


def cap(tile: np.ndarray, lines: list[str], col=(240, 240, 240)) -> np.ndarray:
    h = 20 + 18 * len(lines)
    o = np.full((tile.shape[0] + h, tile.shape[1], 3), BG, np.uint8)
    o[h:] = tile
    for i, t in enumerate(lines):
        # напис на картинці — латиницею; повна назва лишається у звіті
        cv2.putText(o, to_latin(t), (7, 17 + i * 18), F, 0.45,
                    col if i == 0 else (150, 150, 150), 1, cv2.LINE_AA)
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="порівняти пресети на одному кадрі")
    ap.add_argument("input")
    ap.add_argument("presets", nargs="+")
    ap.add_argument("-o", "--out", default="variants")
    ap.add_argument("--face-model", default=None)
    ap.add_argument("--face-detector", default=None)
    ap.add_argument("--crop", type=int, default=400, help="розмір кропа 1:1")
    ap.add_argument("--panel", type=int, default=420, help="ширина загального плану")
    a = ap.parse_args(argv)

    base_cfg = Config(face_model=a.face_model, face_detector=a.face_detector,
                      force_mask=True)
    sess = Session(a.input, base_cfg).load()
    h, w = sess.img.shape[:2]
    u8 = (np.clip(sess.img, 0, 1) * 255 + 0.5).astype(np.uint8)

    rows_over, rows_crop, report = [], [], []
    # Місце для кропа обираємо ОДИН раз — по найконтрастнішій плямі з
    # дефолтних параметрів. Інакше варіанти показували б різні ділянки,
    # і порівняти їх було б неможливо.
    sess.analyze()
    spot = (w // 2, h // 2)
    if sess.blobs:
        b = sess.blobs[0]
        spot = (int(b["center"][0]), int(b["center"][1]))
    cx0 = int(np.clip(spot[0] - a.crop // 2, 0, max(0, w - a.crop)))
    cy0 = int(np.clip(spot[1] - a.crop // 2, 0, max(0, h - a.crop)))
    sl = (slice(cy0, cy0 + a.crop), slice(cx0, cx0 + a.crop))

    fit = lambda x, W: cv2.resize(x, (W, max(1, int(x.shape[0] * W / x.shape[1]))),
                                  interpolation=cv2.INTER_AREA)   # noqa: E731
    rows_over.append(cap(fit(u8, a.panel), ["ОРИГІНАЛ"]))
    rows_crop.append(cap(u8[sl].copy(), [f"ОРИГІНАЛ 1:1  @{spot[0]},{spot[1]}"]))

    for path in a.presets:
        data = presets_mod.load(path)
        cfg = Config(face_model=a.face_model, face_detector=a.face_detector,
                     force_mask=True)
        notes = presets_mod.apply(cfg, data)
        sess.cfg = cfg
        if sess.cls is not None:
            sess.remask()
        sess.analyze().heal()
        res = (np.clip(sess.result, 0, 1) * 255 + 0.5).astype(np.uint8)
        name = data.get("name") or Path(path).stem
        touched = float((sess.coverage > 0).mean())
        info = [name[:44], f"{len(sess.blobs)} плям · {touched:.2%} кадру"]
        rows_over.append(cap(fit(res, a.panel), info, (120, 220, 120)))
        rows_crop.append(cap(res[sl].copy(), info, (120, 220, 120)))
        report.append({"file": Path(path).name, "name": name,
                       "why": (data.get("why") or "").strip(),
                       "blobs": len(sess.blobs), "touched": touched,
                       "notes": notes})

    def strip(tiles):
        H = max(t.shape[0] for t in tiles)
        pad = [np.vstack([t, np.full((H - t.shape[0], t.shape[1], 3), BG, np.uint8)])
               if t.shape[0] < H else t for t in tiles]
        gap = np.full((H, 8, 3), BG, np.uint8)
        return np.hstack([y for t in pad for y in (t, gap)][:-1])

    over, crop = strip(rows_over), strip(rows_crop)
    W = max(over.shape[1], crop.shape[1])
    widen = lambda x: (x if x.shape[1] == W else                       # noqa: E731
                       np.hstack([x, np.full((x.shape[0], W - x.shape[1], 3), BG, np.uint8)]))
    sheet = np.vstack([widen(over), np.full((10, W, 3), BG, np.uint8), widen(crop)])

    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    stem = Path(a.input).stem
    p = d / f"{stem}_variants.png"
    cv2.imwrite(str(p), sheet)

    lines = [f"# Варіанти: {Path(a.input).name}", "",
             "Верхній ряд — загальний план (проявлення). Нижній — 1:1 (ретуш).",
             "Судити про ретуш можна ЛИШЕ за нижнім рядом.", ""]
    for r in report:
        lines += [f"## {r['name']}", "",
                  f"`{r['file']}` — {r['blobs']} плям, торкнулися {r['touched']:.2%}", ""]
        if r["why"]:
            lines += ["> " + l for l in r["why"].splitlines()] + [""]
        if r["notes"]:
            lines += [f"- зауваження: {n}" for n in r["notes"]] + [""]
    (d / f"{stem}_variants.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"аркуш: {p}")
    print(f"звіт:  {d / f'{stem}_variants.md'}")
    for r in report:
        print(f"  {r['name'][:40]:<42}{r['blobs']:>5} плям  {r['touched']:>7.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
