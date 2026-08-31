"""Детекція та лікування дефектів шкіри.

Ідея: дефект (прищ, пляма, дрібна зморшка) живе у високій частоті,
тон і об'єм — у низькій. Тому лікуємо ТІЛЬКИ високу частоту:
підміняємо текстуру в плямі текстурою чистої шкіри поруч,
а низьку частоту не чіпаємо взагалі.

Це те саме, що ретушер робить руками лікувалкою по HF-шару. Воно не
дає "замиленої плями", бо світлотінь лишається оригінальна, і не
з'їдає пори, бо донор — теж справжня шкіра, а не розмиття.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .freqsep import luma

_BIG = 1e9


# ---------------------------------------------------------------------------
# детекція
# ---------------------------------------------------------------------------

@dataclass
class DetectParams:
    scales: tuple[float, ...] = (1.5, 3.0, 6.0)
    """Сигми смугового банку в px. Кожна ловить дефекти свого розміру."""

    threshold: float = 0.012
    """Мінімальний контраст плями у [0..1]. Нижче — вважаємо текстурою."""

    min_area: int = 8
    max_area: int = 1200
    """Межі площі в px. max_area відсікає тіні й великі структури."""

    max_elongation: float = 6.0
    """Відношення сторін bbox. Витягнуте — це волосина або край, не пляма."""

    darks: bool = True
    lights: bool = False
    """Що шукаємо. Світлі дефекти (відблиски) вимкнені за замовчуванням."""


def detect_blemishes(
    high: np.ndarray,
    skin_mask: np.ndarray | None = None,
    p: DetectParams | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Багатомасштабна детекція плям на HF-шарі.

    Повертає (labels, blobs):
      labels — int32 карта зв'язних компонент (0 = фон),
      blobs  — список dict: id, bbox, area, contrast, scale, center.
    Список відсортований за спаданням контрасту: перші елементи —
    найпомітніші дефекти, тобто ті, які варто лікувати завжди.
    """
    p = p or DetectParams()
    hf = luma(high)
    h, w = hf.shape

    if skin_mask is None:
        skin = np.ones((h, w), np.uint8)
    else:
        skin = (skin_mask > 0).astype(np.uint8)

    acc = np.zeros((h, w), np.uint8)
    scale_of = np.zeros((h, w), np.float32)

    for s in p.scales:
        resp = cv2.GaussianBlur(hf, (0, 0), s, borderType=cv2.BORDER_REPLICATE)
        hit = np.zeros((h, w), np.uint8)
        if p.darks:
            hit |= (resp < -p.threshold).astype(np.uint8)
        if p.lights:
            hit |= (resp > p.threshold).astype(np.uint8)
        hit &= skin
        fresh = (hit > 0) & (acc == 0)
        scale_of[fresh] = s
        acc |= hit

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(acc, 8)

    blobs: list[dict] = []
    keep = np.zeros(n, bool)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (p.min_area <= area <= p.max_area):
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if max(bw, bh) > p.max_elongation * max(1, min(bw, bh)):
            continue
        m = labels[y:y + bh, x:x + bw] == i
        contrast = float(np.abs(hf[y:y + bh, x:x + bw][m]).mean())
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        keep[i] = True
        blobs.append({
            "id": i,
            "bbox": (x, y, bw, bh),
            "area": area,
            "contrast": contrast,
            "scale": float(scale_of[int(cy), int(cx)]),
            "center": (cx, cy),
        })

    labels = np.where(keep[labels], labels, 0).astype(np.int32)
    blobs.sort(key=lambda b: -b["contrast"])
    return labels, blobs


# ---------------------------------------------------------------------------
# лікування
# ---------------------------------------------------------------------------

def _donor_cost(hf: np.ndarray, busy: np.ndarray, valid: np.ndarray,
                win: tuple[int, int]) -> np.ndarray:
    """Карта вартості донора: середня енергія HF у вікні win навколо пікселя.

    Вікна, що зачіпають інші плями або не-шкіру, отримують _BIG.
    Мінімум карти = найчистіша ділянка текстури поблизу.
    """
    kw, kh = win
    energy = cv2.boxFilter(hf * hf, -1, (kw, kh), normalize=True,
                           borderType=cv2.BORDER_REPLICATE)
    occupied = cv2.boxFilter(busy, -1, (kw, kh), normalize=True,
                             borderType=cv2.BORDER_REPLICATE)
    ok = cv2.boxFilter(valid, -1, (kw, kh), normalize=True,
                       borderType=cv2.BORDER_REPLICATE)
    cost = energy.copy()
    cost[occupied > 1e-6] = _BIG
    cost[ok < 0.999] = _BIG
    return cost


def heal_blemishes(
    high: np.ndarray,
    labels: np.ndarray,
    blobs: list[dict],
    skin_mask: np.ndarray | None = None,
    search_radius: int = 90,
    feather: float = 1.2,
    margin: int = 3,
    strength: float = 1.0,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Підміняє HF у плямах текстурою найчистішої шкіри поблизу.

    Повертає (high_healed, coverage), де coverage — накопичена альфа
    дотиків у [0..1]. Низька частота не чіпається взагалі.

    limit — обробити лише N найконтрастніших плям (blobs уже відсортовані).
    Зручно для "зніми тільки найпомітніше, решту зроблю руками".
    """
    h, w = labels.shape
    out = high.copy()
    coverage = np.zeros((h, w), np.float32)
    hf_luma = luma(high)
    busy = (labels > 0).astype(np.float32)
    valid = (np.ones((h, w), np.float32) if skin_mask is None
             else (skin_mask > 0).astype(np.float32))

    cost_cache: dict[tuple[int, int], np.ndarray] = {}
    todo = blobs if limit is None else blobs[:limit]

    for b in todo:
        x, y, bw, bh = b["bbox"]
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(w, x + bw + margin), min(h, y + bh + margin)
        pw, ph = x1 - x0, y1 - y0
        if pw < 2 or ph < 2:
            continue

        win = (pw | 1, ph | 1)
        if win not in cost_cache:
            cost_cache[win] = _donor_cost(hf_luma, busy, valid, win)
        cost = cost_cache[win]

        cx, cy = int(round(b["center"][0])), int(round(b["center"][1]))
        sx0 = max(win[0] // 2, cx - search_radius)
        sy0 = max(win[1] // 2, cy - search_radius)
        sx1 = min(w - win[0] // 2, cx + search_radius)
        sy1 = min(h - win[1] // 2, cy + search_radius)
        if sx1 <= sx0 or sy1 <= sy0:
            continue

        region = cost[sy0:sy1, sx0:sx1]
        idx = int(np.argmin(region))
        if float(region.flat[idx]) >= _BIG:
            continue  # чистого донора поруч немає — краще не чіпати
        dy, dx = divmod(idx, region.shape[1])
        dcx, dcy = sx0 + dx, sy0 + dy

        gx0 = int(np.clip(dcx - pw // 2, 0, w - pw))
        gy0 = int(np.clip(dcy - ph // 2, 0, h - ph))

        alpha = (labels[y0:y1, x0:x1] == b["id"]).astype(np.float32)
        if margin > 0:
            alpha = cv2.dilate(alpha, np.ones((3, 3), np.uint8), iterations=margin)
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
        alpha = np.clip(alpha * strength, 0.0, 1.0)

        donor = high[gy0:gy0 + ph, gx0:gx0 + pw]
        a3 = alpha[..., None]
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - a3) + donor * a3
        coverage[y0:y1, x0:x1] = np.maximum(coverage[y0:y1, x0:x1], alpha)

    return out, coverage
