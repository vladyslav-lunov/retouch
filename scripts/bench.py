"""Замір бюджету з spec.md §9: скільки часу і скільки пам'яті.

Синтетика замість реального кадру — щоб число можна було відтворити
на будь-якій машині й порівняти до/після зміни. Реальні файли міряються
тим самим кодом: --image PORTRAIT.tif.

    python3 scripts/bench.py                 # 24 Мп синтетика
    python3 scripts/bench.py --mp 50
    python3 scripts/bench.py --image p.tif

Пікова пам'ять — ru_maxrss процесу, тобто ВЕСЬ процес, а не тільки етап.
Це навмисно: на 8 ГБ важить саме пік процесу, а не акуратність окремої
функції.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from retouch.blemish import DetectParams, detect_blemishes, heal_blemishes
from retouch.freqsep import freq_merge, freq_split
from retouch.masks import build_skin_mask

# ru_maxrss: байти на macOS, кілобайти на Linux
_RSS_UNIT = 1 if sys.platform == "darwin" else 1024


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_UNIT / 2**20


class Step:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *exc):
        print(f"  {self.name:<22} {time.time() - self.t:6.2f}s   пік {peak_mb():7.0f} МБ")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="бюджет часу і пам'яті, spec.md §9")
    ap.add_argument("--image", help="реальний файл замість синтетики")
    ap.add_argument("--mp", type=float, default=24.0, help="мегапікселі синтетики")
    ap.add_argument("--radius", type=float, default=6.0)
    ap.add_argument("--search-radius", type=int, default=90)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--skin-mask", action="store_true",
                    help="увімкнути маску шкіри (на синтетиці зазвичай зайве)")
    a = ap.parse_args(argv)

    print(f"старт, пік процесу {peak_mb():.0f} МБ")

    if a.image:
        from retouch import imageio
        with Step("читання"):
            img, _ = imageio.read(a.image)
        h, w = img.shape[:2]
    else:
        from tests.synth import make_skin_mp
        with Step(f"синтетика {a.mp:g} Мп"):
            img, (w, h) = make_skin_mp(a.mp)

    print(f"кадр {w}x{h} = {w * h / 1e6:.1f} Мп, буфер {img.nbytes / 2**20:.0f} МБ")

    skin = None
    if a.image or a.skin_mask:
        with Step("маска шкіри"):
            skin, src = build_skin_mask(img)
        print(f"    джерело={src}, покриття {skin.mean():.1%}")

    with Step("частотка"):
        low, high = freq_split(img, a.radius)

    dp = DetectParams() if a.threshold is None else DetectParams(threshold=a.threshold)
    with Step("детекція"):
        lbl, blobs = detect_blemishes(high, skin, dp)
    sizes = {((b["bbox"][2] + 6) | 1, (b["bbox"][3] + 6) | 1) for b in blobs}
    print(f"    {len(blobs)} плям, {len(sizes)} різних розмірів вікна")
    print(f"    стара реалізація тримала б {len(sizes) * w * h * 4 / 2**30:.1f} ГБ карт")

    with Step("лікування"):
        high2, cov = heal_blemishes(high, lbl, blobs, skin,
                                    search_radius=a.search_radius)
    print(f"    торкнулися {cov.mean():.3%} кадру")

    with Step("зведення"):
        result = freq_merge(low, high2)

    print(f"\nпік процесу: {peak_mb():.0f} МБ   (результат {result.mean():.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
