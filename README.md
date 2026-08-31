# retouch-lab

Автоматика ретуші шкіри та видалення об'єктів. Вихід — **шари корекції**,
а не готовий піксель: видно, що система зробила, і кожен дотик можна
послабити або скасувати в Photoshop.

## Швидкий старт

```bash
pip install -r requirements.txt
python3 tests/test_blemish.py                    # перевірка, що ядро живе
python3 -m retouch.cli portrait.tif -o out --debug
```

Вхід — 16-бітний TIFF з Camera Raw. RAW проєкт не читає навмисно.

## Що вийде

```
out/portrait_00_base.tif     оригінал
out/portrait_01_skin.png     шар лікування шкіри (RGBA)
out/portrait_02_remove.png   шар видалення об'єктів
out/portrait_mask_skin.png   що система вважала шкірою
out/portrait_99_flat.tif     зведений результат
out/portrait_debug/          з --debug: усі проміжні етапи
```

У Photoshop: відкрити `_00_base.tif`, покласти зверху `_01_skin.png`,
крутити opacity або домальовувати маску шару.

## Корисні прапорці

| | |
|---|---|
| `--dry-run` | порахувати дефекти, нічого не писати |
| `--limit 30` | прибрати лише 30 найпомітніших |
| `--strength 0.7` | послабити лікування |
| `--threshold 0.018` | менш агресивна детекція |
| `--debug` | скинути всі проміжні шари |
| `--remove-mask m.png` | біла маска = що видалити |

## Моделі

Працює без них (евристична маска шкіри + Telea). З моделями краще:

```bash
python3 scripts/fetch_models.py     # підказки, звідки брати
python3 -m retouch.cli p.tif --face-model models/face.onnx --lama-model models/lama.onnx
```

Далі — `spec.md`.
