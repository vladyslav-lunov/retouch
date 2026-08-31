"""Записати синтетичний портрет-макет у 16-бітний TIFF.

Потрібен, щоб конвеєр було на чому ганяти руками, поки нема реальних
кадрів. Це МАКЕТ: він перевіряє логіку (чи знайшло пляму, чи не полізло
в брови), а не якість ретуші. Пороги на ньому не калібрують — див.
spec.md §6.2.

    python3 scripts/make_fixture.py -o fixtures
    python3 scripts/make_fixture.py -o fixtures --face-w 900 --spots 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from tests.synth import make_face


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="синтетичний портрет для ручних прогонів")
    ap.add_argument("-o", "--out", default="fixtures", help="куди писати")
    ap.add_argument("--height", type=int, default=2400)
    ap.add_argument("--width", type=int, default=1800)
    ap.add_argument("--face-w", type=int, default=980, help="ширина обличчя в px")
    ap.add_argument("--spots", type=int, default=28)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--name", default="PORTRAIT")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    img, spots, truth = make_face(h=a.height, w=a.width, face_w=a.face_w,
                                  n_spots=a.spots, seed=a.seed)

    # 16 біт — саме той вхід, під який зроблено конвеєр (spec.md §4)
    tif = out / f"{a.name}.tif"
    assert cv2.imwrite(str(tif), (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16))

    # маска-істина шкіри — щоб було з чим звіряти те, що нарахує masks.py
    skin = (truth["skin"] | truth["neck"]).astype(np.uint8) * 255
    cv2.imwrite(str(out / f"{a.name}_truth_skin.png"), skin)

    # маска для --remove-mask: дві "родимки", які треба прибрати цілком
    remove = np.zeros(img.shape[:2], np.uint8)
    for x, y, r in spots[:2]:
        cv2.circle(remove, (x, y), max(6, r + 3), 255, -1)
    cv2.imwrite(str(out / f"{a.name}_remove.png"), remove)

    # координати дефектів — щоб перевіряти детектор об'єктивно
    (out / f"{a.name}_spots.txt").write_text(
        "# x y r\n" + "\n".join(f"{x} {y} {r}" for x, y, r in spots),
        encoding="utf-8")

    print(f"{tif}  {a.width}x{a.height}, 16 біт, обличчя {a.face_w}px")
    print(f"{len(spots)} дефектів, координати в {a.name}_spots.txt")
    print(f"істина по шкірі: {a.name}_truth_skin.png")
    print(f"маска для видалення: {a.name}_remove.png (2 плями)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
