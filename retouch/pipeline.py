"""Оркестрація. Кожен етап звітує, кожен проміжний результат можна записати.

Це головна вимога до проєкту: не чорна скринька. Після прогону з
--debug у папці лежать усі проміжні шари, і видно, ЩО саме модель
вважала шкірою, ЩО порахувала дефектом і ЩО в підсумку змінила.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import imageio, layers as layers_mod
from .blemish import DetectParams, detect_blemishes, heal_blemishes
from .freqsep import freq_merge, freq_split, radius_for
from .masks import MaskParams, build_skin_mask


@dataclass
class Config:
    hf_radius: float | None = None
    """None = порахувати з роздільності через radius_for()."""

    detect: DetectParams = field(default_factory=DetectParams)
    mask: MaskParams = field(default_factory=MaskParams)
    search_radius: int = 90
    strength: float = 1.0
    limit: int | None = None
    face_model: str | None = None
    lama_model: str | None = None
    use_skin_mask: bool = True


class Stage:
    def __init__(self, name: str):
        self.name, self.t0 = name, time.time()
        print(f"[{name}] ...", flush=True)

    def done(self, note: str = "") -> None:
        print(f"[{self.name}] {time.time() - self.t0:.2f}s {note}", flush=True)


def run(
    image_path: str | Path,
    out_dir: str | Path,
    cfg: Config | None = None,
    remove_mask_path: str | Path | None = None,
    debug: bool = False,
) -> dict:
    cfg = cfg or Config()
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    stem = image_path.stem

    s = Stage("read")
    img, dtype = imageio.read(image_path)
    h, w = img.shape[:2]
    s.done(f"{w}x{h} {dtype}")

    # --- маска шкіри ---------------------------------------------------
    if cfg.use_skin_mask:
        s = Stage("skin-mask")
        skin, source = build_skin_mask(img, cfg.face_model, cfg.mask)
        s.done(f"джерело={source} покриття={skin.mean():.1%}")
    else:
        skin, source = None, "off"

    # --- дефекти шкіри --------------------------------------------------
    radius = cfg.hf_radius or radius_for(img.shape, skin)
    s = Stage("freq-split")
    low, high = freq_split(img, radius)
    s.done(f"radius={radius:.1f}px")

    s = Stage("detect")
    lbl, blobs = detect_blemishes(high, skin, cfg.detect)
    s.done(f"знайдено {len(blobs)} плям")

    s = Stage("heal")
    high2, heal_cov = heal_blemishes(
        high, lbl, blobs, skin, search_radius=cfg.search_radius,
        strength=cfg.strength, limit=cfg.limit)
    healed = freq_merge(low, high2)
    s.done(f"торкнулися {heal_cov.mean():.3%} кадру")

    out_layers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if heal_cov.max() > 0:
        out_layers["skin"] = layers_mod.extract_layer(img, healed, heal_cov)
    result = healed

    # --- видалення об'єктів ---------------------------------------------
    if remove_mask_path:
        s = Stage("inpaint")
        rm = imageio.read_mask(remove_mask_path, (h, w))
        if cfg.lama_model and Path(cfg.lama_model).exists():
            from .inpaint import LamaInpainter, inpaint_region
            model = LamaInpainter(cfg.lama_model)
            result, cov = inpaint_region(result, rm, model)
            note = "lama"
        else:
            from .inpaint import inpaint_classic
            before = result
            result = inpaint_classic(result, rm)
            cov = cv2.GaussianBlur(rm.astype(np.float32), (0, 0), 3.0)
            result = before * (1 - cov[..., None]) + result * cov[..., None]
            note = "telea (моделі немає)"
        out_layers["remove"] = layers_mod.extract_layer(healed, result, cov)
        s.done(note)

    # --- запис ----------------------------------------------------------
    s = Stage("write")
    masks = {"skin": skin} if skin is not None else {}
    written = layers_mod.write_stack(out_dir, stem, img, out_layers,
                                     result, dtype, masks)
    s.done(f"{len(written)} файлів")

    if debug:
        _dump_debug(out_dir / f"{stem}_debug", img, low, high, high2,
                    lbl, heal_cov, result)

    return {"result": result, "blobs": blobs, "skin_source": source,
            "files": written, "radius": radius}


def _dump_debug(d: Path, img, low, high, high2, lbl, cov, result) -> None:
    d.mkdir(parents=True, exist_ok=True)
    w8 = lambda n, a: cv2.imwrite(  # noqa: E731
        str(d / n), np.clip(a * 255, 0, 255).astype(np.uint8))
    w8("01_low.png", low)
    w8("02_high.png", high + 0.5)
    overlay = img.copy()
    overlay[lbl > 0] = (0, 0, 1)
    w8("03_detected.png", overlay)
    w8("04_high_healed.png", high2 + 0.5)
    w8("05_coverage.png", np.dstack([cov] * 3))
    w8("06_diff_x4.png", (result - img) * 4 + 0.5)
    print(f"[debug] {d}")
