"""Пластика: поле зміщення і його застосування.

Геометрію не можна віддати шаром із альфою — Photoshop не складе назад
те, що ми пересунули. Тому носієм тут виступає САМЕ ПОЛЕ ЗМІЩЕННЯ: воно
маленьке, його видно, його можна зберегти, застосувати до іншої версії
того самого кадру і послабити множенням на число. Це і є аналог
непрозорості шару для геометрії, і саме він тримає §1 — «покажи, що
саме ти зробив» — там, де шар неможливий.

Поле тримаємо у ЗМЕНШЕНОМУ масштабі. Деформація тіла за визначенням
гладка: різкий злам у полі дав би розрив на картинці, а не результат.
На 26 Мп поле 1/8 важить 3.1 МБ замість 300, і жодної деталі це не
втрачає.

Дві речі, за які тут треба тримати:

1. Поза деформацією кадр має лишитись БІТ-У-БІТ таким самим. Нульове
   зміщення дає цілі координати, і remap на цілих координатах повертає
   той самий піксель. Перевіряється тестом.
2. Деформація — перший етап конвеєра, який ПЕРЕСЕМПЛЮЄ кадр. Тому вона
   йде ПЕРЕД лікуванням шкіри: спершу форма, потім текстура. Інакше
   ретуш робилася б по пікселях, які деформація потім розмиє, а шар
   корекції з'їхав би відносно бази.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class WarpParams:
    scale: int = 8
    """У скільки разів поле дрібніше за кадр. 8 — 3.1 МБ на 26 Мп."""

    strength: float = 1.0
    """Множник усього поля. Аналог непрозорості шару для геометрії."""

    smooth: float = 0.0
    """Додаткове згладжування поля перед застосуванням, у пікселях поля.
    Нуль — не згладжувати: мазки й так гладкі за побудовою."""


class Field:
    """Поле зміщення у зменшеному масштабі.

    Домовленість про знак: у полі лежить, ЗВІДКИ брати піксель. Тобто
    result(x) = source(x + D(x)). Мазок «тягну вправо» кладе у поле
    від'ємне зміщення по x — бо щоб пікселі поїхали вправо, брати їх
    треба зліва. Плутанина тут коштує дзеркального результату, тому
    інструменти нижче єдині, хто про цей знак знає.
    """

    def __init__(self, shape: tuple[int, int], scale: int = 8):
        h, w = shape[:2]
        self.full = (h, w)
        self.scale = max(1, int(scale))
        fh = max(2, h // self.scale)
        fw = max(2, w // self.scale)
        self.d = np.zeros((fh, fw, 2), np.float32)

    # --- службове -------------------------------------------------------
    @property
    def touched(self) -> bool:
        return bool(np.any(self.d))

    def stats(self) -> dict:
        mag = np.hypot(self.d[:, :, 0], self.d[:, :, 1]) * self.scale
        return {"max_px": round(float(mag.max()), 1),
                "mean_px": round(float(mag[mag > 0].mean()) if mag.any() else 0.0, 1),
                "touched_frac": round(float((mag > 0.5).mean()), 4),
                "field": [int(self.d.shape[1]), int(self.d.shape[0])],
                "mb": round(self.d.nbytes / 2**20, 2)}

    def clear(self) -> None:
        self.d[:] = 0

    def scaled(self, k: float) -> "Field":
        out = Field(self.full, self.scale)
        out.d = self.d * float(k)
        return out

    # --- інструменти ----------------------------------------------------
    def _brush(self, cx: float, cy: float, radius: float):
        """Маска пензля у координатах ПОЛЯ плюс сітка відстаней.

        Спад (1 - t²)² — гладкий і на краю, і в центрі: у нуль він іде з
        нульовою похідною, тож край мазка не лишає видимого кільця.
        """
        s = self.scale
        fx, fy, fr = cx / s, cy / s, max(1.0, radius / s)
        fh, fw = self.d.shape[:2]
        x0, x1 = max(0, int(fx - fr) - 1), min(fw, int(fx + fr) + 2)
        y0, y1 = max(0, int(fy - fr) - 1), min(fh, int(fy + fr) + 2)
        if x1 <= x0 or y1 <= y0:
            return None
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = xx - fx, yy - fy
        t = np.sqrt(dx * dx + dy * dy) / fr
        w = np.clip(1.0 - t * t, 0.0, 1.0) ** 2
        return (slice(y0, y1), slice(x0, x1)), w, dx, dy, fr

    def push(self, cx: float, cy: float, radius: float,
             mx: float, my: float, strength: float = 1.0) -> None:
        """Зсунути пікселі вслід за рухом миші (m — вектор руху, у px кадру)."""
        b = self._brush(cx, cy, radius)
        if b is None:
            return
        sl, w, _dx, _dy, _fr = b
        # мінус: у полі лежить, ЗВІДКИ брати
        self.d[sl][:, :, 0] -= w * (mx / self.scale) * strength
        self.d[sl][:, :, 1] -= w * (my / self.scale) * strength

    def bloat(self, cx: float, cy: float, radius: float,
              amount: float, strength: float = 1.0) -> None:
        """Роздути (amount > 0) або стягнути (amount < 0) до центру."""
        b = self._brush(cx, cy, radius)
        if b is None:
            return
        sl, w, dx, dy, _fr = b
        k = w * amount * strength * 0.5
        self.d[sl][:, :, 0] += k * dx
        self.d[sl][:, :, 1] += k * dy

    def twirl(self, cx: float, cy: float, radius: float,
              angle: float, strength: float = 1.0) -> None:
        """Закрутити навколо центру. angle у радіанах на повну силу пензля."""
        b = self._brush(cx, cy, radius)
        if b is None:
            return
        sl, w, dx, dy, _fr = b
        a = w * angle * strength
        ca, sa = np.cos(a), np.sin(a)
        self.d[sl][:, :, 0] += (ca * dx - sa * dy) - dx
        self.d[sl][:, :, 1] += (sa * dx + ca * dy) - dy

    def freeze(self, mask: np.ndarray) -> None:
        """Занулити поле там, де маска — щоб деформація не чіпала обличчя,
        коли працюєш поруч. Маска у масштабі кадру."""
        fh, fw = self.d.shape[:2]
        m = cv2.resize((mask > 0).astype(np.float32), (fw, fh),
                       interpolation=cv2.INTER_AREA)
        self.d *= (1.0 - m)[..., None]

    # --- застосування ---------------------------------------------------
    def maps(self, p: WarpParams | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Координатні карти для cv2.remap у масштабі кадру."""
        p = p or WarpParams()
        h, w = self.full
        d = self.d
        if p.smooth > 0:
            d = cv2.GaussianBlur(d, (0, 0), p.smooth)
        big = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
        big *= float(p.strength) * self.scale
        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        return xx + big[:, :, 0], yy + big[:, :, 1]

    def apply(self, img: np.ndarray, p: WarpParams | None = None) -> np.ndarray:
        """Деформувати кадр. Поза деформацією — біт-у-біт той самий кадр."""
        p = p or WarpParams()
        if not self.touched or p.strength == 0:
            return img
        mapx, mapy = self.maps(p)
        return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)

    def apply_to(self, img: np.ndarray, p: WarpParams | None = None) -> np.ndarray:
        """Те саме, але на зображенні ІНШОГО розміру — зазвичай на проксі.

        UI показує зменшену копію й деформує саме її: на 900 px це 9 мс,
        тобто можна крутити повзунок і бачити результат, а не чекати
        0.9 с повного кадру. Поле задане в масштабі кадру, тож зміщення
        треба перерахувати в масштаб проксі — інакше на зменшеній копії
        деформація вийде вдесятеро сильнішою.
        """
        p = p or WarpParams()
        h, w = img.shape[:2]
        if not self.touched or p.strength == 0:
            return img
        if (h, w) == self.full:
            return self.apply(img, p)
        d = self.d
        if p.smooth > 0:
            d = cv2.GaussianBlur(d, (0, 0), p.smooth)
        k = w / float(self.full[1])
        big = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
        big = big * (float(p.strength) * self.scale * k)
        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        return cv2.remap(img, xx + big[:, :, 0], yy + big[:, :, 1],
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # --- зберігання -----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """16-бітний PNG: R і G — зміщення, B — нуль.

        Зміщення знакове, тому кодуємо зі зсувом на половину діапазону.
        Крок кванта і межу пишемо в саму назву — інакше файл неможливо
        прочитати назад, не вгадуючи, чим його писали.
        """
        path = Path(path)
        lim = float(np.abs(self.d).max()) or 1.0
        enc = np.clip(self.d / lim * 0.5 + 0.5, 0, 1)
        h, w = enc.shape[:2]
        out = np.zeros((h, w, 3), np.uint16)
        out[:, :, 2] = (enc[:, :, 0] * 65535 + 0.5).astype(np.uint16)   # R = dx
        out[:, :, 1] = (enc[:, :, 1] * 65535 + 0.5).astype(np.uint16)   # G = dy
        name = (f"{path.stem}__s{self.scale}_lim{lim:.4f}"
                f"_{self.full[1]}x{self.full[0]}{path.suffix or '.png'}")
        dst = path.parent / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dst), out):
            raise IOError(f"поле не записалося: {dst}")
        return dst

    @classmethod
    def load(cls, path: str | Path) -> "Field":
        path = Path(path)
        meta = path.stem.split("__")[-1]
        parts = dict(scale=8, lim=1.0, w=0, h=0)
        for chunk in meta.split("_"):
            if chunk.startswith("s") and chunk[1:].isdigit():
                parts["scale"] = int(chunk[1:])
            elif chunk.startswith("lim"):
                parts["lim"] = float(chunk[3:])
            elif "x" in chunk:
                a, _, b = chunk.partition("x")
                if a.isdigit() and b.isdigit():
                    parts["w"], parts["h"] = int(a), int(b)
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"поле не читається: {path}")
        f = cls((parts["h"], parts["w"]), parts["scale"])
        dec = img.astype(np.float32) / 65535.0
        f.d = np.dstack([(dec[:, :, 2] - 0.5) * 2 * parts["lim"],
                         (dec[:, :, 1] - 0.5) * 2 * parts["lim"]]).astype(np.float32)
        return f
