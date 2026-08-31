"""Підказки, звідки брати ваги. Свідомо НЕ качає нічого автоматично:
ліцензії різні, і краще один раз прочитати їх очима.

Запуск: python3 scripts/fetch_models.py
"""

INFO = """
Потрібні дві ONNX-моделі. Обидві кладуться в models/.

1) FACE PARSING -> models/face_parsing.onnx
   BiSeNet, навчений на CelebAMask-HQ (19 класів).
   Шукати: "face-parsing.PyTorch" / "BiSeNet CelebAMask-HQ onnx".
   Експорт з PyTorch:
       torch.onnx.export(net, torch.randn(1,3,512,512), "face_parsing.onnx",
                         input_names=["input"], output_names=["out"], opset_version=13)

   ПЕРЕВІР порядок класів: різні перезаливки трапляються з переставленими
   індексами. Прогнати одне фото, розфарбувати карту класів, звірити очима
   зі списком CELEBA_CLASSES у retouch/masks.py.

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
