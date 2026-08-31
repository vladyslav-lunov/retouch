"""Маски шкіри.

Два шляхи:
  1. heuristic_skin_mask — без моделей, працює одразу, YCrCb + guided filter.
     Груба, але для контрольованого портретного світла цілком робоча.
  2. FaceParser — BiSeNet, навчений на CelebAMask-HQ, через ONNX Runtime.
     Дає окремі класи: шкіра, брови, очі, губи, волосся, шия. Саме звідси
     беруться зони виключення, без яких автоматика з'їдає вії та губи.

Порядок робіт: спочатку жити на евристиці, підключити ONNX як тільки
базовий конвеєр працює. Інтерфейс однаковий, підміна — один прапорець.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# CelebAMask-HQ, 19 класів. ПЕРЕВІР порядок під конкретні ваги перед довірою:
# різні перезаливки BiSeNet трапляються з переставленими індексами.
CELEBA_CLASSES = {
    0: "background", 1: "skin", 2: "l_brow", 3: "r_brow", 4: "l_eye",
    5: "r_eye", 6: "eye_g", 7: "l_ear", 8: "r_ear", 9: "ear_r",
    10: "nose", 11: "mouth", 12: "u_lip", 13: "l_lip", 14: "neck",
    15: "neck_l", 16: "cloth", 17: "hair", 18: "hat",
}

SKIN_CLASSES = ("skin", "nose", "neck")
EXCLUDE_CLASSES = ("l_brow", "r_brow", "l_eye", "r_eye", "eye_g",
                   "mouth", "u_lip", "l_lip", "hair", "hat", "cloth",
                   "l_ear", "r_ear", "ear_r")


@dataclass
class MaskParams:
    erode: int = 6
    """Відступ від краю маски в px. Захищає контур обличчя від лікування."""

    feather: float = 4.0
    exclude_dilate: int = 5
    """Розширення зон виключення (очі, губи, брови)."""


def heuristic_skin_mask(img: np.ndarray, p: MaskParams | None = None) -> np.ndarray:
    """Маска шкіри без нейромереж. img: float32 BGR [0..1] -> uint8 {0,1}."""
    p = p or MaskParams()
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(u8, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    mask = ((cr > 133) & (cr < 180) & (cb > 77) & (cb < 127)).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if p.erode > 0:
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=p.erode)
    return mask


class FaceParser:
    """BiSeNet face-parsing через ONNX Runtime.

    Очікуваний контракт моделі (перевір на своїх вагах, ЦЕ ПРИПУЩЕННЯ):
      вхід  : 1x3x512x512, RGB, /255, нормалізація ImageNet
      вихід : 1x19x512x512 логіти, argmax по каналах -> карта класів
    """

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)
    SIZE = 512

    def __init__(self, model_path: str | Path, providers: list[str] | None = None):
        import onnxruntime as ort  # локальний імпорт: не потрібен для евристики
        self.sess = ort.InferenceSession(
            str(model_path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def parse(self, img: np.ndarray) -> np.ndarray:
        """Повертає int32 карту класів у роздільності оригіналу."""
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (self.SIZE, self.SIZE), interpolation=cv2.INTER_AREA)
        x = ((small - self.MEAN) / self.STD).transpose(2, 0, 1)[None].astype(np.float32)
        logits = self.sess.run(None, {self.input_name: x})[0]
        cls = logits[0].argmax(0).astype(np.uint8)
        return cv2.resize(cls, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    def skin_mask(self, img: np.ndarray, p: MaskParams | None = None) -> np.ndarray:
        p = p or MaskParams()
        cls = self.parse(img)
        inv = {v: k for k, v in CELEBA_CLASSES.items()}
        skin = np.isin(cls, [inv[c] for c in SKIN_CLASSES]).astype(np.uint8)
        excl = np.isin(cls, [inv[c] for c in EXCLUDE_CLASSES]).astype(np.uint8)
        if p.exclude_dilate > 0:
            excl = cv2.dilate(excl, np.ones((3, 3), np.uint8),
                              iterations=p.exclude_dilate)
        skin = skin & (1 - excl)
        if p.erode > 0:
            skin = cv2.erode(skin, np.ones((3, 3), np.uint8), iterations=p.erode)
        return skin


def build_skin_mask(img: np.ndarray, model_path: str | Path | None = None,
                    p: MaskParams | None = None) -> tuple[np.ndarray, str]:
    """Повертає (маска, назва_джерела). Падає на евристику, якщо моделі нема."""
    if model_path and Path(model_path).exists():
        try:
            return FaceParser(model_path).skin_mask(img, p), "face-parsing"
        except Exception as exc:  # noqa: BLE001 — свідомо не валимо конвеєр
            print(f"[masks] face-parsing не спрацював ({exc}), беру евристику")
    return heuristic_skin_mask(img, p), "heuristic"
