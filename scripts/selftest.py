"""Табель конвеєра на макеті-портреті: що знайшов, що пропустив, куди поліз.

Тести кажуть «зелено/червоно». Цей скрипт каже НАСКІЛЬКИ добре, і саме
його треба дивитися, коли крутиш пороги. На макеті є істина: відомо, де
кожен дефект і де кожна зона виключення, — тож повноту й хибні
спрацювання можна не вгадувати, а рахувати.

    python3 scripts/selftest.py
    python3 scripts/selftest.py --face-w 1200 --threshold 0.018
    python3 scripts/selftest.py --sweep          # таблиця по порогах, як §6.2

ВАЖЛИВО: це макет, а не фото. Числа тут годяться, щоб порівняти ДВІ
версії коду між собою. Калібрувати за ними пороги для реальних зйомок
не можна — див. spec.md §6.2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from retouch.blemish import DetectParams, detect_blemishes, heal_blemishes
from retouch.freqsep import freq_split, radius_for
from retouch.masks import build_skin_mask
from tests.synth import make_face

EXCLUDED = ("l_eye", "r_eye", "l_brow", "r_brow", "u_lip", "l_lip", "hair", "background")


def run_once(img, spots, truth, threshold=None, search_radius=90):
    skin, source = build_skin_mask(img)
    radius = radius_for(img.shape, skin)
    _, high = freq_split(img, radius)
    dp = DetectParams() if threshold is None else DetectParams(threshold=threshold)
    lbl, blobs = detect_blemishes(high, skin, dp)
    _, cov = heal_blemishes(high, lbl, blobs, skin, search_radius=search_radius)

    found = sum(1 for x, y, _ in spots if lbl[y, x] > 0)
    healed = sum(1 for x, y, _ in spots if cov[y, x] > 0)
    return {
        "skin_source": source, "radius": radius, "mask": skin,
        "labels": lbl, "blobs": blobs, "cov": cov,
        "found": found, "healed": healed,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="табель конвеєра на макеті")
    ap.add_argument("--height", type=int, default=1500)
    ap.add_argument("--width", type=int, default=1150)
    ap.add_argument("--face-w", type=int, default=900)
    ap.add_argument("--spots", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--search-radius", type=int, default=90)
    ap.add_argument("--sweep", action="store_true", help="таблиця по порогах")
    a = ap.parse_args(argv)

    img, spots, truth = make_face(h=a.height, w=a.width, face_w=a.face_w,
                                  n_spots=a.spots, seed=a.seed)
    skin_truth = truth["skin"] | truth["neck"]

    if a.sweep:
        print(f"макет {a.width}x{a.height}, обличчя {a.face_w}px, "
              f"{len(spots)} дефектів\n")
        print(f"{'поріг':>7}{'компонент':>11}{'влучань':>10}{'зайвих':>9}"
              f"{'у зонах викл.':>15}")
        for t in (0.008, 0.012, 0.018, 0.025):
            r = run_once(img, spots, truth, threshold=t, search_radius=a.search_radius)
            excl = sum(int(((r["cov"] > 0) & truth[k]).sum()) for k in EXCLUDED)
            print(f"{t:>7.3f}{len(r['blobs']):>11}{r['found']:>7}/{len(spots)}"
                  f"{len(r['blobs']) - r['found']:>9}{excl:>15}")
        print("\nПорівнюй колонки між ВЕРСІЯМИ КОДУ, а не з таблицею §6.2:")
        print("та знята на плоскому клапті, ця — на макеті обличчя.")
        return 0

    r = run_once(img, spots, truth, a.threshold, a.search_radius)
    skin, cov, lbl, blobs = r["mask"], r["cov"], r["labels"], r["blobs"]

    print(f"макет {a.width}x{a.height}, обличчя {a.face_w}px, "
          f"{len(spots)} відомих дефектів")
    print(f"маска: {r['skin_source']}, радіус частотки {r['radius']:.1f}px\n")

    seen_w = 0
    cols = np.nonzero(skin.any(axis=0))[0]
    if len(cols):
        seen_w = int(cols.max() - cols.min() + 1)
    print("МАСКА ШКІРИ")
    tp = int((skin.astype(bool) & skin_truth).sum())
    fp = int((skin.astype(bool) & ~skin_truth).sum())
    print(f"  точність {tp / max(tp + fp, 1):>6.1%}   повнота {skin[skin_truth].mean():>6.1%}")
    for k in ("l_brow", "u_lip", "hair", "background"):
        print(f"  захопила {k:<12}{skin[truth[k]].mean():>7.1%}")
    print(f"  ширину обличчя бачить як {seen_w}px замість {a.face_w}px "
          f"({seen_w / a.face_w:.0%}, spec.md §6.3)\n")

    print("ДЕТЕКЦІЯ")
    print(f"  компонент {len(blobs)}, з них справжніх дефектів "
          f"{r['found']}/{len(spots)} = {r['found'] / len(spots):.0%}")
    print(f"  зайвих спрацювань: {len(blobs) - r['found']}\n")

    print("ЛІКУВАННЯ")
    print(f"  вилікувано відомих дефектів {r['healed']}/{len(spots)}")
    print(f"  торкнулися {int((cov > 0).sum())} px = {(cov > 0).mean():.3%} кадру")
    outside = int((cov[skin == 0] > 0).sum())
    print(f"  поза маскою шкіри: {outside} px" + ("  <-- ІНВАРІАНТ 7" if outside else "  ок"))
    bad = {k: int(((cov > 0) & truth[k]).sum()) for k in EXCLUDED}
    bad = {k: v for k, v in bad.items() if v}
    print(f"  у зонах виключення: {bad or 'ніде'}"
          + ("  <-- ІНВАРІАНТ 7" if bad else ""))

    near_eye = sum(1 for b in blobs
                   if truth["l_eye"][int(b["center"][1]), int(b["center"][0])]
                   or truth["r_eye"][int(b["center"][1]), int(b["center"][0])]
                   or _near(truth, b, 30))
    print(f"\n  спрацювань біля очей/брів/губ: {near_eye} з {len(blobs)}")
    print("  (контур ока — сильний край одразу за маскою; це те, заради чого")
    print("   в §5 стоїть face-parsing замість евристики)")
    return 0


def _near(truth, b, rad: int) -> bool:
    """Чи лежить центр плями ближче ніж rad px до зони виключення."""
    import cv2
    key = "_dil_cache"
    if not hasattr(_near, key):
        excl = np.zeros_like(truth["l_eye"])
        for k in ("l_eye", "r_eye", "l_brow", "r_brow", "u_lip", "l_lip"):
            excl |= truth[k]
        setattr(_near, key, cv2.dilate(excl.astype(np.uint8),
                                       np.ones((3, 3), np.uint8), iterations=rad))
    return bool(getattr(_near, key)[int(b["center"][1]), int(b["center"][0])])


if __name__ == "__main__":
    raise SystemExit(main())
