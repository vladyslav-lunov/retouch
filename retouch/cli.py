"""CLI. Один файл або тека, параметри — з YAML або прапорцями."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .blemish import DetectParams
from .masks import MaskParams
from .pipeline import Config, run

SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def build_config(a: argparse.Namespace) -> Config:
    cfg = Config(
        hf_radius=a.radius,
        detect=DetectParams(threshold=a.threshold,
                            min_area=a.min_area,
                            max_area=a.max_area),
        mask=MaskParams(erode=a.mask_erode),
        strength=a.strength,
        limit=a.limit,
        face_model=a.face_model,
        lama_model=a.lama_model,
        use_skin_mask=not a.no_skin_mask,
    )
    if a.config:
        import yaml
        raw = yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) or {}
        for k, v in raw.items():
            if k == "detect":
                cfg.detect = DetectParams(**v)
            elif k == "mask":
                cfg.mask = MaskParams(**v)
            elif hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="retouch",
        description="Автоматика шкіри та видалення об'єктів. Вихід — шари.")
    ap.add_argument("input", help="файл або тека")
    ap.add_argument("-o", "--out", default="out", help="куди писати")
    ap.add_argument("--remove-mask", help="біла маска = що видалити")
    ap.add_argument("--config", help="YAML з параметрами")

    ap.add_argument("--radius", type=float, default=None,
                    help="радіус частотки в px (типово — з роздільності)")
    ap.add_argument("--threshold", type=float, default=0.012,
                    help="поріг контрасту дефекту, менше = агресивніше")
    ap.add_argument("--min-area", type=int, default=8)
    ap.add_argument("--max-area", type=int, default=1200)
    ap.add_argument("--strength", type=float, default=1.0,
                    help="сила лікування 0..1")
    ap.add_argument("--limit", type=int, default=None,
                    help="лікувати лише N найконтрастніших плям")
    ap.add_argument("--mask-erode", type=int, default=6)
    ap.add_argument("--no-skin-mask", action="store_true",
                    help="обробляти весь кадр (для тестів)")

    ap.add_argument("--face-model", default=None, help="ONNX face-parsing")
    ap.add_argument("--lama-model", default=None, help="ONNX LaMa")
    ap.add_argument("--debug", action="store_true",
                    help="скинути всі проміжні шари")
    ap.add_argument("--dry-run", action="store_true",
                    help="лише порахувати дефекти, нічого не писати")

    a = ap.parse_args(argv)
    cfg = build_config(a)

    src = Path(a.input)
    files = ([p for p in sorted(src.iterdir()) if p.suffix.lower() in SUFFIXES]
             if src.is_dir() else [src])
    if not files:
        print("нічого обробляти", file=sys.stderr)
        return 1

    for f in files:
        print(f"\n=== {f.name} ===")
        if a.dry_run:
            cfg_dry = cfg
            from . import imageio
            from .blemish import detect_blemishes
            from .freqsep import freq_split, radius_for
            from .masks import build_skin_mask
            img, _ = imageio.read(f)
            skin = (build_skin_mask(img, cfg_dry.face_model, cfg_dry.mask)[0]
                    if cfg_dry.use_skin_mask else None)
            r = cfg_dry.hf_radius or radius_for(img.shape, skin)
            _, high = freq_split(img, r)
            _, blobs = detect_blemishes(high, skin, cfg_dry.detect)
            print(f"{len(blobs)} плям, радіус {r:.1f}px")
            for b in blobs[:15]:
                print(f"  контраст {b['contrast']:.4f}  площа {b['area']:5d}"
                      f"  центр {b['center'][0]:.0f},{b['center'][1]:.0f}")
            continue
        run(f, a.out, cfg, a.remove_mask, debug=a.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
