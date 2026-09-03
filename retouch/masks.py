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

# `neck` СВІДОМО не тут. На реальному кадрі 26 Мп він давав 39 зі 154
# знахідок — це ланцюжок на грудях, і лікування рвало його на шматки
# (заміряно, spec.md §6.2 і §15). Помиляємось у той самий бік, що й з
# маскою в §5: зайвий прищ на шиї коштує однієї галочки, порваний
# ланцюжок — кадру. Вмикається назад одним рядком у пресеті:
#   mask: {skin_classes: [skin, nose, neck]}
SKIN_CLASSES = ("skin", "nose")
EXCLUDE_CLASSES = ("l_brow", "r_brow", "l_eye", "r_eye", "eye_g",
                   "mouth", "u_lip", "l_lip", "hair", "hat", "cloth",
                   "l_ear", "r_ear", "ear_r")


@dataclass
class MaskParams:
    erode: int = 6
    """Відступ від краю маски в px. Захищає контур обличчя від лікування."""

    feather: float = 4.0
    """НЕ ВИКОРИСТОВУЄТЬСЯ. Лишилось із першої редакції; маска бінарна,
    розмиття краю робить не вона, а альфа дотику в blemish.heal_blemishes.
    Задавати немає сенсу — нічого не станеться."""

    exclude_dilate: int = 5
    """Розширення зон виключення (очі, губи, брови)."""

    skin_classes: tuple[str, ...] = SKIN_CLASSES
    """Які класи моделі вважати шкірою. Типово обличчя без шиї: skin, nose.
    Змінне НАВМИСНО — «шкіра» це не властивість пікселя, а рішення
    фотографа під кадр. Шия й декольте вмикаються додаванням neck; тоді
    ж туди потрапляє ланцюжок, і лікування його порве, тому дивись на
    розподіл знахідок по класах у звіті детекції (spec.md §15)."""

    exclude_classes: tuple[str, ...] = EXCLUDE_CLASSES
    """Що віднімати від шкіри після дилатації."""

    ensemble: str = "single"
    """Як поєднувати кілька моделей: single | intersect | union.
    intersect — лише те, з чим ЗГОДНІ обидві: менше маски, менше ризику.
    union — усе, що знайшла хоч одна. Розбіжність двох моделей на цьому
    кадрі — 2% площі, і майже вся вона на шиї (spec.md §5)."""


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


def detect_faces(img: np.ndarray, model_path: str | Path,
                 max_side: int = 900, score: float = 0.6) -> list[tuple[int, int, int, int]]:
    """Обличчя у координатах кадру через YuNet (cv2.FaceDetectorYN).

    YuNet навчений на обличчях приблизно 10-300 px, тож повний кадр
    спершу зменшуємо: на 26 Мп обличчя інакше має 900 px і не ловиться.
    Детектор лежить у самому OpenCV, нової залежності не з'являється.
    """
    h, w = img.shape[:2]
    k = min(1.0, max_side / max(h, w))
    small = cv2.resize((np.clip(img, 0, 1) * 255).astype(np.uint8),
                       (max(1, int(w * k)), max(1, int(h * k))),
                       interpolation=cv2.INTER_AREA)
    det = cv2.FaceDetectorYN.create(str(model_path), "",
                                    (small.shape[1], small.shape[0]), score)
    _rv, faces = det.detect(small)
    if faces is None:
        return []
    out = []
    for f in faces:
        x, y, fw, fh = (float(v) / k for v in f[:4])
        out.append((int(x), int(y), int(fw), int(fh)))
    out.sort(key=lambda b: -b[2])
    return out


def face_crop_box(img: np.ndarray, box, margin: float = 0.85) -> tuple[int, int, int, int]:
    """Рамка кропа навколо обличчя: з волоссям і шиєю, знизу з запасом."""
    h, w = img.shape[:2]
    x, y, fw, fh = box
    m = int(fw * margin)
    x0, y0 = max(0, x - m), max(0, y - m)
    x1 = min(w, x + fw + m)
    y1 = min(h, y + fh + int(m * 1.6))
    return x0, y0, x1, y1


class FaceParser:
    """BiSeNet face-parsing через ONNX Runtime.

    Очікуваний контракт моделі (перевір на своїх вагах, ЦЕ ПРИПУЩЕННЯ):
      вхід  : 1x3x512x512, RGB, /255, нормалізація ImageNet
      вихід : 1x19x512x512 логіти, argmax по каналах -> карта класів
    """

    last_faces: list = []
    """Рамки з останнього parse(). Порожньо, якщо детектора не було."""

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)
    SIZE = 512

    def __init__(self, model_path: str | Path, providers: list[str] | None = None):
        import onnxruntime as ort  # локальний імпорт: не потрібен для евристики
        self.sess = ort.InferenceSession(
            str(model_path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def _parse_whole(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (self.SIZE, self.SIZE), interpolation=cv2.INTER_AREA)
        x = ((small - self.MEAN) / self.STD).transpose(2, 0, 1)[None].astype(np.float32)
        logits = self.sess.run(None, {self.input_name: x})[0]
        cls = logits[0].argmax(0).astype(np.uint8)
        return cv2.resize(cls, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    def parse(self, img: np.ndarray, detector: str | Path | None = None,
              margin: float = 0.85) -> np.ndarray:
        """int32 карта класів у роздільності оригіналу.

        detector — шлях до YuNet. З ним кадр спершу кропиться по обличчю,
        і це не оптимізація, а умова роботи: BiSeNet навчений на
        вирівняних портретах 512x512, а _parse_whole стискає весь кадр до
        512. На кропі голови все гаразд, на повному кадрі 26 Мп обличчя
        стає завширшки ~100 px, і модель ламається — заміряно 1.4% шкіри
        і 28% «капелюха» там, де капелюха немає (spec.md §5).

        Поза кропом лишається background: те, що не потрапило в рамку
        обличчя, шкірою ми не вважаємо в будь-якому разі.
        """
        if not detector:
            return self._parse_whole(img)
        faces = detect_faces(img, detector)
        if not faces:
            return self._parse_whole(img)
        h, w = img.shape[:2]
        out = np.zeros((h, w), np.int32)
        for box in faces[:1]:            # найбільше обличчя; групові кадри — окрема розмова
            x0, y0, x1, y1 = face_crop_box(img, box, margin)
            out[y0:y1, x0:x1] = self._parse_whole(img[y0:y1, x0:x1])
        # Рамку запам'ятовуємо: з неї рахуються радіус частотки й радіус
        # пошуку донора, і взяти її звідси дешевше, ніж ганяти YuNet
        # удруге. Габарит маски для цього не годиться — він міряє не те
        # (freqsep.face_width, spec.md §6.3).
        self.last_faces = list(faces)
        return out

    def skin_mask(self, img: np.ndarray, p: MaskParams | None = None,
                  cls: np.ndarray | None = None) -> np.ndarray:
        """Маска зі списків класів у MaskParams. cls — готова карта класів,
        якщо її вже порахували (щоб не ганяти модель двічі)."""
        p = p or MaskParams()
        if cls is None:
            cls = self.parse(img)
        return mask_from_classes(cls, p)


def mask_from_classes(cls: np.ndarray, p: MaskParams | None = None) -> np.ndarray:
    """Карта класів -> бінарна маска шкіри за списками з MaskParams.

    Винесено з FaceParser навмисно: UI перебирає набори класів по кілька
    разів на секунду, і ганяти заради цього модель немає сенсу — карта
    класів не залежить від того, що ми потім назвемо шкірою.
    """
    p = p or MaskParams()
    inv = {v: k for k, v in CELEBA_CLASSES.items()}
    idx = lambda names: [inv[c] for c in names if c in inv]   # noqa: E731
    skin = np.isin(cls, idx(p.skin_classes)).astype(np.uint8)
    excl = np.isin(cls, idx(p.exclude_classes)).astype(np.uint8)
    if p.exclude_dilate > 0:
        excl = cv2.dilate(excl, np.ones((3, 3), np.uint8),
                          iterations=p.exclude_dilate)
    skin = skin & (1 - excl)
    if p.erode > 0:
        skin = cv2.erode(skin, np.ones((3, 3), np.uint8), iterations=p.erode)
    return skin


def build_skin_mask(img: np.ndarray, model_path: str | Path | None = None,
                    p: MaskParams | None = None,
                    detector: str | Path | None = None
                    ) -> tuple[np.ndarray, str]:
    """Повертає (маска, назва_джерела). Падає на евристику, якщо моделі нема."""
    if model_path and Path(model_path).exists():
        try:
            det = str(detector) if detector and Path(detector).exists() else None
            cls = FaceParser(model_path).parse(img, det)
            src = "face-parsing" + ("+yunet" if det else "")
            return mask_from_classes(cls, p), src
        except Exception as exc:  # noqa: BLE001 — свідомо не валимо конвеєр
            print(f"[masks] face-parsing не спрацював ({exc}), беру евристику")
    return heuristic_skin_mask(img, p), "heuristic"
