"""Видалення об'єктів: LaMa через ONNX Runtime.

Головна проблема — роздільність. LaMa тренували приблизно на 512 px і
вище неї вона починає мазати. Тому працюємо не з усім кадром, а з
кропом навколо маски з контекстним запасом. Для портрета 50 Мп це
означає, що зайва волосина чи родимка обробляються в РІДНІЙ
роздільності, а не через даунскейл усього файлу.

Якщо кроп однаково більший за max_size — обробляємо зменшену копію,
але повертаємо в кадр тільки низьку частоту результату, а високу
беремо з оригіналу навколо. Так шов не видно і різкість не падає.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class InpaintParams:
    context: int = 128
    """Скільки px контексту навколо маски віддати моделі."""

    max_size: int = 1024
    """Стеля кропа. На A1502 більше ставити немає сенсу — буде довго."""

    pad_to: int = 8
    """LaMa вимагає сторони, кратні 8."""

    feather: float = 3.0
    """Розмиття альфи на вшивці результату назад у кадр."""

    dilate: int = 3
    """Розширення маски. Модель має бачити край об'єкта, а не рівно по ньому."""


class LamaInpainter:
    """Обгортка ONNX-моделі.

    Контракт (перевір на своєму файлі ваг, ЦЕ ПРИПУЩЕННЯ):
      входи : image 1x3xHxW float32 у [0..1] RGB, mask 1x1xHxW float32 {0,1}
      вихід : 1x3xHxW float32 у [0..255] RGB
    Імена входів беруться з самої сесії, тож перейменування не страшне.
    """

    def __init__(self, model_path: str | Path, providers: list[str] | None = None):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            str(model_path), providers=providers or ["CPUExecutionProvider"])
        names = [i.name for i in self.sess.get_inputs()]
        if len(names) != 2:
            raise ValueError(f"очікував 2 входи, модель дає {names}")
        self.img_name, self.mask_name = names

    def _run(self, rgb: np.ndarray, mask: np.ndarray, pad_to: int) -> np.ndarray:
        h, w = rgb.shape[:2]
        ph, pw = (-h) % pad_to, (-w) % pad_to
        if ph or pw:
            rgb = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT)
            mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)

        x = rgb.transpose(2, 0, 1)[None].astype(np.float32)
        m = mask[None, None].astype(np.float32)
        out = self.sess.run(None, {self.img_name: x, self.mask_name: m})[0]
        out = out[0].transpose(1, 2, 0)
        if out.max() > 1.5:  # модель віддає 0..255
            out = out / 255.0
        return np.clip(out[:h, :w], 0, 1)


def inpaint_region(
    img: np.ndarray,
    mask: np.ndarray,
    model: "LamaInpainter",
    p: InpaintParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """img: float32 BGR [0..1], mask: uint8 {0,1} — що видалити.

    Повертає (result, coverage). Кожна зв'язна область маски обробляється
    окремим кропом: так дві правки в різних кутах кадру не тягнуть за
    собою обробку всього файлу.
    """
    p = p or InpaintParams()
    h, w = img.shape[:2]
    result = img.copy()
    coverage = np.zeros((h, w), np.float32)

    if p.dilate > 0:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=p.dilate)

    n, labels = cv2.connectedComponents(mask.astype(np.uint8), 8)
    for i in range(1, n):
        sub = (labels == i).astype(np.uint8)
        ys, xs = np.nonzero(sub)
        if len(ys) == 0:
            continue
        y0 = max(0, ys.min() - p.context)
        y1 = min(h, ys.max() + p.context + 1)
        x0 = max(0, xs.min() - p.context)
        x1 = min(w, xs.max() + p.context + 1)

        crop = img[y0:y1, x0:x1]
        cmask = sub[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]

        scale = min(1.0, p.max_size / max(ch, cw))
        if scale < 1.0:
            small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            smask = cv2.resize(cmask, (small.shape[1], small.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        else:
            small, smask = crop, cmask

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        filled = model._run(rgb, smask.astype(np.float32), p.pad_to)
        filled = cv2.cvtColor(filled, cv2.COLOR_RGB2BGR)

        if scale < 1.0:
            filled = cv2.resize(filled, (cw, ch), interpolation=cv2.INTER_CUBIC)
            # апскейл змилив деталь: беремо з моделі тільки низьку частоту,
            # високу лишаємо з оригіналу — шов зникає, різкість не падає
            r = 2.0 / scale
            lo_f = cv2.GaussianBlur(filled, (0, 0), r)
            lo_o = cv2.GaussianBlur(crop, (0, 0), r)
            filled = np.clip(lo_f + (crop - lo_o), 0, 1)

        alpha = cv2.GaussianBlur(cmask.astype(np.float32), (0, 0), p.feather)
        alpha = np.clip(alpha, 0, 1)[..., None]
        result[y0:y1, x0:x1] = crop * (1 - alpha) + filled * alpha
        coverage[y0:y1, x0:x1] = np.maximum(coverage[y0:y1, x0:x1], alpha[..., 0])

    return result, coverage


def inpaint_classic(img: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Запасний варіант без моделі: Telea. Годиться лише для дрібного."""
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    out = cv2.inpaint(u8, (mask > 0).astype(np.uint8), radius, cv2.INPAINT_TELEA)
    return out.astype(np.float32) / 255.0
