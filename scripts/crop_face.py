"""Вирізати голову з кадру — щоб було на чому ганяти конвеєр сьогодні.

Це ІНСТРУМЕНТ РОЗРОБКИ, а не частина конвеєра, і лежить у scripts/
навмисно. Евристична маска шкіри на повному вуличному кадрі непридатна
(spec.md §5), конвеєр там чесно зупиняється і радить подати кроп голови
— оцим кропом він і подається, поки не підключено face-parsing.

Каскад Хаара тут не претендує на роль детектора облич у проєкті: він
грубий, на профіль і повернуту голову не спрацює. Але він уже лежить в
opencv, нових залежностей не тягне, і для «вирізати голову й подивитися,
чи працює ретуш» його вистачає. Коли з'явиться BiSeNet, цей скрипт стане
непотрібним.

    python3 scripts/crop_face.py IMG.tif
    python3 scripts/crop_face.py IMG.tif --margin 1.2 -o crops
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from retouch import imageio


def find_faces(img: np.ndarray, downscale: int = 6) -> list[tuple[int, int, int, int]]:
    """Обличчя у координатах повного кадру. Шукаємо на зменшеній копії:
    на 26 Мп каскад по повному розміру рахувався б хвилини без користі."""
    h, w = img.shape[:2]
    small = cv2.resize((np.clip(img, 0, 1) * 255).astype(np.uint8),
                       (max(1, w // downscale), max(1, h // downscale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    found = cv2.CascadeClassifier(xml).detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    return [(int(x * downscale), int(y * downscale),
             int(fw * downscale), int(fh * downscale)) for x, y, fw, fh in found]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="кроп голови для ручних прогонів")
    ap.add_argument("input")
    ap.add_argument("-o", "--out", default=None,
                    help="тека (типово — поруч із вхідним файлом)")
    ap.add_argument("--margin", type=float, default=0.9,
                    help="запас навколо обличчя в частках його ширини "
                         "(типово 0.9: влазить волосся і шия)")
    ap.add_argument("--all", action="store_true", help="усі обличчя, не лише найбільше")
    a = ap.parse_args(argv)

    src = Path(a.input)
    img, dtype = imageio.read(src)
    faces = find_faces(img)
    if not faces:
        print("облич не знайдено. Каскад бере лише фронтальні — якщо голова "
              "повернута, вирізай руками.", file=sys.stderr)
        return 1

    faces.sort(key=lambda f: -f[2])
    if not a.all:
        faces = faces[:1]

    out_dir = Path(a.out) if a.out else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]

    for i, (x, y, fw, fh) in enumerate(faces, 1):
        m = int(fw * a.margin)
        x0, y0 = max(0, x - m), max(0, y - m)
        # знизу запасу більше: шия і плечі теж шкіра, і донор часто там
        x1, y1 = min(w, x + fw + m), min(h, y + fh + int(m * 1.6))
        crop = img[y0:y1, x0:x1]

        suffix = f"_face{i}" if len(faces) > 1 else "_face"
        dst = out_dir / f"{src.stem}{suffix}.tif"
        imageio.write(dst, crop, dtype)
        print(f"{dst}  {crop.shape[1]}x{crop.shape[0]}, обличчя {fw}px")
        # §6.3: радіус частотки виводиться з ширини обличчя
        print(f"  за §6.3 очікуваний радіус ~{6.0 * fw / 1200:.1f}px "
              f"(перевір, що конвеєр дасть приблизно стільки)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
