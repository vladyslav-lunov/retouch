"""Перевірити ваги face-parsing ДО того, як їм довіряти.

spec.md §5 ставить `ПЕРЕВІР` на порядок класів: різні перезаливки
BiSeNet трапляються з переставленими індексами, і якщо повірити не
перевіривши, конвеєр вважатиме шкірою волосся, а губи — шкірою. Помітити
це по кінцевому результату важко: маска просто буде «дивна».

Скрипт відповідає на три питання:
  1. чи той у моделі контракт, який припускає masks.FaceParser;
  2. які класи вона взагалі віддає і скільки їх;
  3. що вона вважає чим — кольорова карта, яку можна звірити очима.

    python3 scripts/check_face_model.py models/face_parsing.onnx PORTRAIT.tif
    python3 scripts/check_face_model.py models/face.onnx IMG.CR3 -o out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from retouch import imageio
from retouch.masks import CELEBA_CLASSES, EXCLUDE_CLASSES, SKIN_CLASSES, FaceParser

# Кольори для карти класів. Осмислені, а не випадкові: шкіра тілесна,
# волосся коричневе, губи червоні — щоб переставлений індекс було видно
# одразу, без звірки з легендою.
COLORS = {
    "background": (60, 60, 60), "skin": (150, 190, 230), "l_brow": (40, 90, 140),
    "r_brow": (60, 110, 160), "l_eye": (250, 220, 120), "r_eye": (250, 180, 90),
    "eye_g": (200, 200, 200), "l_ear": (140, 170, 210), "r_ear": (120, 150, 200),
    "ear_r": (110, 130, 180), "nose": (170, 205, 240), "mouth": (90, 90, 220),
    "u_lip": (70, 70, 235), "l_lip": (50, 50, 200), "neck": (130, 175, 215),
    "neck_l": (110, 155, 195), "cloth": (120, 200, 120), "hair": (60, 85, 120),
    "hat": (200, 120, 200),
}


def describe(sess) -> dict:
    """Що модель насправді приймає й віддає."""
    i = sess.get_inputs()[0]
    o = sess.get_outputs()[0]
    return {"in_name": i.name, "in_shape": list(i.shape), "in_type": i.type,
            "out_name": o.name, "out_shape": list(o.shape), "out_type": o.type,
            "n_outputs": len(sess.get_outputs())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="перевірка ваг face-parsing")
    ap.add_argument("model", help="ONNX face-parsing")
    ap.add_argument("image", help="кадр для перевірки (TIFF/PNG/JPEG/RAW)")
    ap.add_argument("-o", "--out", default=".", help="куди покласти карту класів")
    a = ap.parse_args(argv)

    mp = Path(a.model)
    if not mp.exists():
        print(f"немає файлу моделі: {mp}", file=sys.stderr)
        return 1

    print(f"модель: {mp.name}  ({mp.stat().st_size / 2**20:.0f} МБ)")
    try:
        fp = FaceParser(mp)
    except Exception as e:                                # noqa: BLE001
        print(f"\nне відкрилась: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    d = describe(fp.sess)
    print("\nКОНТРАКТ")
    print(f"  вхід : {d['in_name']}  {d['in_shape']}  {d['in_type']}")
    print(f"  вихід: {d['out_name']}  {d['out_shape']}  {d['out_type']}"
          + (f"   (виходів усього {d['n_outputs']})" if d["n_outputs"] > 1 else ""))

    problems = []
    ish = d["in_shape"]
    if len(ish) != 4 or (isinstance(ish[1], int) and ish[1] != 3):
        problems.append(f"очікував вхід 1x3xHxW, а тут {ish}")
    if isinstance(ish[2], int) and ish[2] != FaceParser.SIZE:
        problems.append(f"masks.FaceParser жорстко масштабує до "
                        f"{FaceParser.SIZE}, а модель хоче {ish[2]}")
    osh = d["out_shape"]
    ncls = osh[1] if len(osh) == 4 and isinstance(osh[1], int) else None
    if ncls is None:
        problems.append(f"не видно кількості класів у виході {osh} "
                        f"(динамічна вісь — перевір на реальному прогоні)")
    elif ncls != len(CELEBA_CLASSES):
        problems.append(f"класів {ncls}, а CELEBA_CLASSES описує "
                        f"{len(CELEBA_CLASSES)} — таблиця не від цих ваг")

    print("\nЗАПУСК")
    img, _dt = imageio.read(a.image)
    h, w = img.shape[:2]
    print(f"  кадр {w}x{h}")
    try:
        cls = fp.parse(img)
    except Exception as e:                                # noqa: BLE001
        print(f"  ВПАЛО: {type(e).__name__}: {e}", file=sys.stderr)
        for p in problems:
            print(f"  ! {p}", file=sys.stderr)
        return 1

    found = sorted(int(v) for v in np.unique(cls))
    print(f"  віддала класи: {found}")
    if max(found) >= len(CELEBA_CLASSES):
        problems.append(f"індекс {max(found)} виходить за таблицю "
                        f"({len(CELEBA_CLASSES)} класів)")

    print("\nЩО ВОНА ВВАЖАЄ ЧИМ")
    print(f"  {'клас':<12}{'частка кадру':>14}   роль у конвеєрі")
    rows = []
    for idx, name in CELEBA_CLASSES.items():
        frac = float((cls == idx).mean())
        if frac < 1e-5:
            continue
        role = ("ШКІРА" if name in SKIN_CLASSES else
                "виключення" if name in EXCLUDE_CLASSES else "—")
        # У легенду на картинці йде латиниця: putText уміє лише
        # Hershey-шрифти, кирилиця в них перетворюється на «???».
        tag = ("SKIN" if name in SKIN_CLASSES else
               "excluded" if name in EXCLUDE_CLASSES else "-")
        rows.append((frac, name, role, tag))
        print(f"  {name:<12}{frac:>13.2%}   {role}")

    # Найпростіша перевірка на здоровий глузд: на портреті шкіри має бути
    # відчутно, і вона не має бути найбільшим класом після фону.
    skin = sum(r[0] for r in rows if r[1] in SKIN_CLASSES)
    hair = sum(r[0] for r in rows if r[1] == "hair")
    print(f"\n  разом шкіра: {skin:.1%}   волосся: {hair:.1%}")
    if skin < 0.02:
        problems.append(f"шкіри лише {skin:.1%} — або кадр не портрет, "
                        f"або індекси переставлені")

    d_out = Path(a.out)
    d_out.mkdir(parents=True, exist_ok=True)
    vis = np.zeros((h, w, 3), np.uint8)
    for idx, name in CELEBA_CLASSES.items():
        vis[cls == idx] = COLORS.get(name, (255, 0, 255))
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    blend = cv2.addWeighted(u8, 0.45, vis, 0.55, 0)
    sheet = np.hstack([_fit(u8, 700), _fit(vis, 700), _fit(blend, 700)])
    legend = np.full((len(rows) * 22 + 12, sheet.shape[1], 3), 30, np.uint8)
    for i, (frac, name, _role, tag) in enumerate(sorted(rows, reverse=True)):
        y = 20 + i * 22
        cv2.rectangle(legend, (10, y - 12), (34, y + 4), COLORS.get(name), -1)
        cv2.putText(legend, f"{name}  {frac:.1%}  {tag}", (44, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    p = d_out / f"{Path(a.image).stem}_classes.png"
    cv2.imwrite(str(p), np.vstack([sheet, legend]))
    print(f"\n  карта класів: {p}")
    print("  Звір очима: шкіра має лягти на шкіру, волосся на волосся,")
    print("  губи на губи. Якщо ні — індекси переставлені, і CELEBA_CLASSES")
    print("  у retouch/masks.py треба переписати під ЦІ ваги.")

    print()
    if problems:
        print("ПРОБЛЕМИ:")
        for p_ in problems:
            print(f"  ! {p_}")
        return 1
    print("Контракт збігається з тим, що припускає masks.FaceParser.")
    print("Порядок класів це НЕ доводить — його підтверджує лише око.")
    return 0


def _fit(a: np.ndarray, w: int) -> np.ndarray:
    h = max(1, int(a.shape[0] * w / a.shape[1]))
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    raise SystemExit(main())
