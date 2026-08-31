"""Синтетична "шкіра" з відомими дефектами — для тестів без реальних фото.

Дає градієнт тону (низька частота), зернистість пор (висока частота)
і N темних плям із заданими координатами. Це дозволяє перевіряти
детектор об'єктивно: ми точно знаємо, де дефекти.
"""

from __future__ import annotations

import cv2
import numpy as np


def make_skin(h: int = 512, w: int = 512, n_spots: int = 12,
              seed: int = 7, spot_strength: float = 0.06
              ) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Повертає (float32 BGR [0..1], список (x, y, r) дефектів)."""
    rng = np.random.default_rng(seed)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    tone = 0.62 + 0.10 * (xx / w) + 0.06 * np.sin(yy / h * 3.0)
    base = np.dstack([tone * 0.78, tone * 0.88, tone])  # BGR, тепліший в R

    pores = rng.normal(0, 1, (h, w, 3)).astype(np.float32)
    pores = cv2.GaussianBlur(pores, (0, 0), 0.9)
    pores /= (pores.std() + 1e-8)
    img = np.clip(base + pores * 0.012, 0, 1)

    spots: list[tuple[int, int, int]] = []
    m = 40
    for _ in range(n_spots):
        x = int(rng.integers(m, w - m))
        y = int(rng.integers(m, h - m))
        r = int(rng.integers(3, 9))
        spots.append((x, y, r))
        blob = np.zeros((h, w), np.float32)
        cv2.circle(blob, (x, y), r, 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), r * 0.45)
        img -= blob[..., None] * np.array([0.4, 0.7, 1.0], np.float32) * spot_strength
    return np.clip(img, 0, 1).astype(np.float32), spots


def make_skin_mp(mp: float, seed: int = 3, n_spots: int = 90
                 ) -> tuple[np.ndarray, tuple[int, int]]:
    """Те саме, але заданої мегапіксельності. Повертає (img, (w, h)).

    Потрібно там, де важить саме РОЗМІР кадру: бюджет пам'яті й часу зі
    spec.md §9. Робимо через ресайз маленького кадру, бо генерувати
    24 Мп шуму напряму довго; зерно пор додаємо ПІСЛЯ ресайзу, інакше
    інтерполяція його з'їдає і високочастотний шар виходить порожній.
    """
    h = int(round((mp * 1e6 * 2 / 3) ** 0.5))
    w = int(round(h * 1.5))
    small, _ = make_skin(h=768, w=1152, n_spots=n_spots, seed=seed)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    rng = np.random.default_rng(seed)
    grain = cv2.GaussianBlur(rng.normal(0, 1, (h, w, 3)).astype(np.float32),
                             (0, 0), 0.9)
    grain /= grain.std() + 1e-8
    return np.clip(img + grain * 0.012, 0, 1).astype(np.float32), (w, h)


# ---------------------------------------------------------------------------
# синтетичний портрет
# ---------------------------------------------------------------------------

# Класи, які вміє малювати make_face. Ті самі імена, що в masks.CELEBA_CLASSES,
# щоб карту-істину можна було звіряти з тим, що дає face-parsing.
FACE_REGIONS = ("background", "skin", "hair", "l_brow", "r_brow",
                "l_eye", "r_eye", "u_lip", "l_lip", "neck")


def make_face(h: int = 1600, w: int = 1200, face_w: int = 620,
              n_spots: int = 24, seed: int = 3, spot_strength: float = 0.055
              ) -> tuple[np.ndarray, list[tuple[int, int, int]], dict]:
    """Портрет-макет: обличчя, волосся, брови, очі, губи, шия, фон.

    Плаский клапоть шкіри з make_skin годиться для інваріантів ядра, але
    НЕ годиться для перевірки конвеєра цілком: на ньому маска шкіри
    покриває всі 100% кадру, зони виключення нема з чим переплутати, а
    radius_for бачить "обличчя" завширшки з кадр. Тут усе це є, і є
    істина: де шкіра, де волосся, де саме лежить кожна пляма.

    Повертає (img, spots, truth), де spots — список (x, y, r) дефектів,
    а truth — dict з булевими масками по FACE_REGIONS.

    Це МАКЕТ, а не фото: він перевіряє логіку (чи не полізло в брови, чи
    знайшло пляму на щоці), а не якість. Калібрувати пороги на ньому не
    можна — для цього потрібні реальні кадри, див. spec.md §6.2.
    """
    rng = np.random.default_rng(seed)
    cx, cy = w // 2, int(h * 0.46)
    fw, fh = face_w // 2, int(face_w * 0.66)

    truth = {k: np.zeros((h, w), bool) for k in FACE_REGIONS}
    img = np.zeros((h, w, 3), np.float32)
    img[:] = (0.34, 0.31, 0.29)                      # фон: холодний, не шкіра
    truth["background"][:] = True

    def fill(mask_u8: np.ndarray, bgr, region: str | None = None):
        m = mask_u8.astype(bool)
        img[m] = bgr
        if region is not None:
            truth[region][m] = True
            for other in FACE_REGIONS:
                if other != region:
                    truth[other][m] = False

    lay = lambda: np.zeros((h, w), np.uint8)          # noqa: E731

    # шия і обличчя. Тон шкіри цілиться в YCrCb-діапазон евристики
    # (masks.heuristic_skin_mask), інакше нема що тестувати.
    SKIN = (0.52, 0.66, 0.83)
    m = lay()
    cv2.rectangle(m, (cx - fw // 2, cy), (cx + fw // 2, h), 1, -1)
    fill(m, tuple(c * 0.90 for c in SKIN), "neck")

    m = lay()
    cv2.ellipse(m, (cx, cy), (fw, fh), 0, 0, 360, 1, -1)
    fill(m, SKIN, "skin")

    # волосся: шапка зверху і два пасма по боках
    m = lay()
    cv2.ellipse(m, (cx, cy - int(fh * 0.30)), (int(fw * 1.10), int(fh * 0.78)),
                0, 180, 360, 1, -1)
    cv2.ellipse(m, (cx - fw, cy), (int(fw * 0.22), int(fh * 0.80)), 0, 0, 360, 1, -1)
    cv2.ellipse(m, (cx + fw, cy), (int(fw * 0.22), int(fh * 0.80)), 0, 0, 360, 1, -1)
    fill(m, (0.10, 0.11, 0.14), "hair")

    ex, ey = int(fw * 0.42), cy - int(fh * 0.16)      # очі
    er = max(3, int(fw * 0.13))
    for side, name in ((-1, "l_eye"), (+1, "r_eye")):
        m = lay()
        cv2.ellipse(m, (cx + side * ex, ey), (er, int(er * 0.55)), 0, 0, 360, 1, -1)
        fill(m, (0.93, 0.93, 0.93), name)
        m = lay()
        cv2.circle(m, (cx + side * ex, ey), int(er * 0.48), 1, -1)
        fill(m, (0.16, 0.13, 0.11), name)

    for side, name in ((-1, "l_brow"), (+1, "r_brow")):   # брови
        m = lay()
        cv2.ellipse(m, (cx + side * ex, ey - int(fh * 0.17)),
                    (int(fw * 0.26), max(2, int(fh * 0.045))), 0, 0, 360, 1, -1)
        fill(m, (0.12, 0.14, 0.19), name)

    ly = cy + int(fh * 0.52)                              # губи
    m = lay()
    cv2.ellipse(m, (cx, ly), (int(fw * 0.30), int(fh * 0.075)), 0, 180, 360, 1, -1)
    fill(m, (0.36, 0.34, 0.62), "u_lip")
    m = lay()
    cv2.ellipse(m, (cx, ly), (int(fw * 0.30), int(fh * 0.095)), 0, 0, 180, 1, -1)
    fill(m, (0.33, 0.31, 0.60), "l_lip")

    img = cv2.GaussianBlur(img, (0, 0), max(1.0, face_w / 400))   # м'які межі

    # світлотінь: джерело зверху-ліворуч. Це низька частота, лікування
    # не має її чіпати — на цьому будується перевірка інваріанта 3.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shade = 1.0 + 0.16 * ((cx - xx) / w) + 0.14 * ((cy - yy) / h)
    nose = np.exp(-(((xx - cx) / (fw * 0.22)) ** 2 +
                    ((yy - cy - fh * 0.16) / (fh * 0.30)) ** 2))
    img *= (shade - 0.10 * nose)[..., None]

    skin_area = truth["skin"] | truth["neck"]
    pores = cv2.GaussianBlur(rng.normal(0, 1, (h, w, 3)).astype(np.float32), (0, 0), 0.9)
    pores /= pores.std() + 1e-8
    img += pores * 0.011 * skin_area[..., None]

    # плями — тільки по шкірі, і не впритул до краю обличчя: там маска
    # ерозується, і лікування туди свідомо не лізе (spec.md §5)
    inner = cv2.erode(skin_area.astype(np.uint8),
                      np.ones((3, 3), np.uint8), iterations=max(4, fw // 40))
    ys, xs = np.nonzero(inner)
    spots: list[tuple[int, int, int]] = []
    r_lo, r_hi = max(3, fw // 90), max(6, fw // 45)
    guard = int(fw * 0.10)
    for _ in range(n_spots * 40):
        if len(spots) >= n_spots:
            break
        i = int(rng.integers(0, len(xs)))
        x, y = int(xs[i]), int(ys[i])
        if any((x - px) ** 2 + (y - py) ** 2 < guard ** 2 for px, py, _ in spots):
            continue                       # не ліпимо плями одна на одну
        r = int(rng.integers(r_lo, r_hi))
        spots.append((x, y, r))
        blob = np.zeros((h, w), np.float32)
        cv2.circle(blob, (x, y), r, 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), r * 0.45)
        img -= (blob * skin_area)[..., None] * \
               np.array([0.35, 0.65, 1.0], np.float32) * spot_strength

    return np.clip(img, 0, 1).astype(np.float32), spots, truth
