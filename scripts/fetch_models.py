"""Підказки, звідки брати ваги. Свідомо НЕ качає нічого автоматично:
ліцензії різні, і краще один раз прочитати їх очима.

Запуск: python3 scripts/fetch_models.py
"""

INFO = """
Потрібні дві ONNX-моделі. Обидві кладуться в models/.

1) FACE PARSING -> models/resnet18.onnx
   BiSeNet, навчений на CelebAMask-HQ (19 класів). Готовий ONNX, експорт
   не потрібен:

       curl -L -o models/resnet18.onnx \
         https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx

   resnet34.onnx там само — точніший на папері, але на заміряному кадрі
   дає те саме за втричі більший файл і довший інференс.

   Контракт і порядок класів ПЕРЕВІРЕНО на цих вагах: збігається з тим,
   що припускає retouch/masks.py. Для чужих ваг перевіряти заново:

       python3 scripts/check_face_model.py models/M.onnx PORTRAIT.tif

   УВАГА: моделі треба давати КРОП ГОЛОВИ, а не повний кадр. На повному
   кадрі 26 Мп обличчя стискається до ~100 px і модель ламається.
   Тимчасово: scripts/crop_face.py.

2) LAMA -> models/lama.onnx
   Suvorov et al., "Resolution-robust Large Mask Inpainting with Fourier
   Convolutions" (WACV 2022). Готові ONNX-експорти є в екосистемі IOPaint.

   УВАГА ПРО ЛІЦЕНЗІЮ: код Apache-2, але ваги big-lama — CC BY-NC-SA.
   Некомерційно. Для власного використання нормально, для продажу ні.

   ПЕРЕВІР контракт входів. Код припускає:
       image 1x3xHxW float32 [0..1] RGB
       mask  1x1xHxW float32 {0,1}
       вихід 1x3xHxW, 0..255 або 0..1 (визначається автоматично)
   Сторони мають бути кратні 8.

Без обох моделей конвеєр працює: маска шкіри — евристична (YCrCb),
інпейнт — cv2.inpaint (Telea). Гірше, але робоче.

Тренування власної моделі — тільки на Katana (RTX 5070, sm_120, torch cu128).
"""

if __name__ == "__main__":
    print(INFO)
