"""Оглядовий аркуш: подивитися на результат, не відкриваючи Photoshop.

Конвеєр віддає шари, і це правильно, але щоб їх побачити, треба зібрати
документ. Для перевірки «чи не поїхало щось» це задорого. Аркуш —
дешевий спосіб глянути очима одразу після прогону.

Головне рішення тут — **кропи в масштабі 1:1**. Зменшений до екрана
портрет 24 Мп не показує ретуш шкіри взагалі: уся робота живе в
текстурі, а вона зникає при першому ж ресайзі. Тому зверху — загальний
план (чи не полізло в фон, чи та маска), а знизу — вирізки навколо
найконтрастніших плям у рідних пікселях, до і після. Судити про якість
можна лише за нижнім рядом.

Модуль нічого не знає про решту конвеєра: на вхід — масиви, на вихід —
масив. Складає його pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np

# Підписи латиницею навмисно: putText уміє лише Hershey-шрифти, кирилиця
# в них перетворюється на сміття.
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BG = 0.12
_PAD = 10


def _label(tile: np.ndarray, text: str, scale: float = 0.5) -> np.ndarray:
    """Підпис на темній підкладці, щоб читався і на світлому, і на темному."""
    th = int(26 * scale / 0.5)
    out = np.full((tile.shape[0] + th, tile.shape[1], 3), _BG, np.float32)
    out[th:] = tile
    cv2.putText(out, text, (6, th - 8), _FONT, scale, (0.95, 0.95, 0.95), 1, cv2.LINE_AA)
    return out


def _fit(a: np.ndarray, w: int, h: int, pad: float = _BG) -> np.ndarray:
    """Вписати кадр у w×h зі збереженням пропорцій, поля — темні.

    Одноканальний вхід приймається як є і розгортається в три канали
    ВЖЕ зменшеним: інакше на 50 Мп кожна маска коштувала б зайвих
    600 МБ тільки заради того, щоб стати картинкою 460 px.
    """
    ah, aw = a.shape[:2]
    s = min(w / aw, h / ah)
    r = cv2.resize(a.astype(np.float32, copy=False),
                   (max(1, int(aw * s)), max(1, int(ah * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
    if r.ndim == 2:
        r = np.dstack([r] * 3)
    out = np.full((h, w, 3), pad, np.float32)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def _row(tiles: list[np.ndarray]) -> np.ndarray:
    h = max(t.shape[0] for t in tiles)
    padded = []
    for t in tiles:
        if t.shape[0] < h:
            t = np.vstack([t, np.full((h - t.shape[0], t.shape[1], 3), _BG, np.float32)])
        padded.append(t)
        padded.append(np.full((h, _PAD, 3), _BG, np.float32))
    return np.hstack(padded[:-1])


def _stack(rows: list[np.ndarray]) -> np.ndarray:
    w = max(r.shape[1] for r in rows)
    out = []
    for r in rows:
        if r.shape[1] < w:
            r = np.hstack([r, np.full((r.shape[0], w - r.shape[1], 3), _BG, np.float32)])
        out.append(r)
        out.append(np.full((_PAD, w, 3), _BG, np.float32))
    return np.vstack(out[:-1])


def contact_sheet(
    img: np.ndarray,
    result: np.ndarray,
    coverage: np.ndarray,
    skin_mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    blobs: list[dict] | None = None,
    n_crops: int = 3,
    crop: int = 260,
    panel: int = 460,
) -> np.ndarray:
    """Аркуш для ока: загальний план зверху, кропи 1:1 знизу.

    img/result — float32 BGR [0..1], coverage — альфа дотиків.
    blobs потрібні лише щоб вибрати, куди дивитися: беремо найконтрастніші.
    """
    h, w = img.shape[:2]

    # --- верхній ряд: загальний план ------------------------------------
    det = _fit(img, panel, panel)
    if labels is not None:
        # Плями дрібні: після зменшення в десять разів вони просто зникнуть,
        # якщо брати найближчого сусіда. INTER_AREA дає частку площі плями
        # в пікселі, тож поріг ставимо низько — краще показати зайве, ніж
        # промовчати про знайдений дефект.
        # поля вписування — нулі, а не тло: інакше вони перевищать поріг
        # і аркуш намалює червоні смуги там, де взагалі немає кадру
        hit = _fit((labels > 0).astype(np.float32), panel, panel, pad=0.0)[:, :, 0]
        det[hit > 0.02] = (0.15, 0.15, 1.0)

    top = [
        _label(_fit(img, panel, panel), "BEFORE"),
        _label(_fit(result, panel, panel), "AFTER"),
        _label(_fit(np.clip((result - img) * 4 + 0.5, 0, 1), panel, panel), "DIFF x4"),
        _label(det, f"DETECTED ({0 if blobs is None else len(blobs)})"),
    ]
    if skin_mask is not None:
        top.append(_label(_fit(skin_mask, panel, panel), "SKIN MASK"))
    top.append(_label(_fit(coverage, panel, panel), "COVERAGE"))

    rows = [_row(top[:3]), _row(top[3:])]

    # --- нижній ряд: кропи 1:1 ------------------------------------------
    # Дивимося туди, де найбільший контраст: саме там видно, спрацювало
    # лікування чи змазало текстуру.
    picks: list[tuple[int, int]] = []
    for b in (blobs or []):
        if len(picks) >= n_crops:
            break
        x, y = int(b["center"][0]), int(b["center"][1])
        if coverage[min(h - 1, y), min(w - 1, x)] <= 0:
            continue                       # цю пляму пропустили, дивитись нема на що
        if any(abs(x - px) < crop and abs(y - py) < crop for px, py in picks):
            continue                       # не показуємо двічі те саме місце
        picks.append((x, y))

    for i, (x, y) in enumerate(picks, 1):
        x0 = int(np.clip(x - crop // 2, 0, max(0, w - crop)))
        y0 = int(np.clip(y - crop // 2, 0, max(0, h - crop)))
        sl = (slice(y0, y0 + crop), slice(x0, x0 + crop))
        before, after = img[sl], result[sl]
        d = np.clip((after - before) * 4 + 0.5, 0, 1)
        rows.append(_row([
            _label(before.copy(), f"#{i} 1:1 BEFORE  @{x},{y}"),
            _label(after.copy(), f"#{i} 1:1 AFTER"),
            _label(d, f"#{i} 1:1 DIFF x4"),
        ]))

    if not picks:
        note = np.full((40, panel * 3, 3), _BG, np.float32)
        cv2.putText(note, "no healed blemishes to crop", (8, 26), _FONT, 0.6,
                    (0.9, 0.6, 0.3), 1, cv2.LINE_AA)
        rows.append(note)

    return np.clip(_stack(rows), 0, 1)
