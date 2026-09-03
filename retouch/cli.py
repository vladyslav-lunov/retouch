"""CLI. Один файл або тека, параметри — з YAML або прапорцями."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .blemish import DetectParams
from .imageio import RAW_SUFFIXES, InputError
from . import batch as batch_mod
from . import presets as presets_mod
from .masks import MaskParams
from .pipeline import Config, MaskSanityError, detect_only, run

SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Дефолти живуть у дата-класах (CLAUDE.md), сюди беруться лише для підказок.
_C, _D, _M = Config(), DetectParams(), MaskParams()

# прапорець -> (об'єкт конфігу, поле). Порядок не важливий, важливо, що
# перекриття явними прапорцями відбувається ПІСЛЯ читання YAML.
_OVERRIDES = (
    ("radius", "cfg", "hf_radius"),
    ("strength", "cfg", "strength"),
    ("limit", "cfg", "limit"),
    ("search_radius", "cfg", "search_radius"),
    ("face_model", "cfg", "face_model"),
    ("face_detector", "cfg", "face_detector"),
    ("raw_decoder", "cfg", "raw_decoder"),
    ("lama_model", "cfg", "lama_model"),
    ("threshold", "detect", "threshold"),
    ("target_coverage", "detect", "target_coverage"),
    ("min_area", "detect", "min_area"),
    ("max_area", "detect", "max_area"),
    ("max_elongation", "detect", "max_elongation"),
    ("mask_erode", "mask", "erode"),
)


def build_config(a: argparse.Namespace) -> Config:
    """YAML — база, явні прапорці перекривають її.

    Раніше було навпаки: YAML застосовувався ПІСЛЯ прапорців і мовчки їх
    з'їдав, тобто `--config c.yaml --threshold 0.02` працював з порогом
    із файлу. Щоб відрізнити "користувач задав" від "argparse підставив
    дефолт", дефолти прапорців — None, а справжні лишаються в дата-класах.
    """
    cfg = Config()
    # XMP — найперший: це те, що фотограф уже вирішив у проявнику, тобто
    # база, а не уточнення. Пресет зйомки й прапорці мають його
    # перекривати, а не навпаки.
    auto_on_dir = (a.xmp == "auto" and a.input
                   and Path(a.input).expanduser().is_dir())
    if getattr(a, "xmp", None) and not auto_on_dir:
        # На теці автопошук нічого не значить: сайдкар лежить біля КАДРУ,
        # і в пакеті його читає batch.process для кожного окремо.
        from . import xmp as xmp_mod
        try:
            src = a.xmp if a.xmp != "auto" else None
            if src:
                pre, rep = xmp_mod.to_preset(xmp_mod.read(src))
            else:
                pre, rep, src = xmp_mod.from_image(a.input)
            if pre is None:
                print("[xmp] сайдкара немає і вбудованого блоку теж",
                      file=sys.stderr)
            else:
                presets_mod.apply(cfg, pre)
                # Звіт друкується ЗАВЖДИ і повністю. Мовчазне «наближено»
                # — головний ризик цієї гілки (PLAN.md §5).
                print(f"[xmp] {src}\n[xmp] {rep.summary()}", file=sys.stderr)
                for line in rep.lines():
                    print(f"[xmp] {line}", file=sys.stderr)
        except xmp_mod.XmpError as e:
            print(f"[xmp] {e}", file=sys.stderr)

    # Пресети йдуть ПЕРЕД yaml і прапорцями: спершу стиль зйомки, потім
    # уточнення кадру, потім те, що людина набрала руками.
    for path in (a.preset or []):
        notes = presets_mod.apply(cfg, presets_mod.load(path))
        for n in notes:
            print(f"[preset] {Path(path).name}: {n}", file=sys.stderr)
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
            else:
                print(f"[cli] невідомий ключ у {a.config}: {k}", file=sys.stderr)

    targets = {"cfg": cfg, "detect": cfg.detect, "mask": cfg.mask}
    for flag, where, field in _OVERRIDES:
        v = getattr(a, flag)
        if v is not None:
            setattr(targets[where], field, v)

    if a.tools:
        cfg.tools = tuple(x.strip() for x in a.tools.split(",") if x.strip())
    if a.dodge_burn:
        cfg.dodgeburn_on = True
    if a.db_strength is not None:
        cfg.dodgeburn.strength = a.db_strength
        cfg.dodgeburn_on = True
    if a.no_skin_mask:
        cfg.use_skin_mask = False
    if a.force_mask:
        cfg.force_mask = True
    return cfg


def build_parser() -> argparse.ArgumentParser:
    """Парсер окремо від main(): тести мають брати дефолти звідси, а не
    дублювати їх у себе. Інакше кожен новий прапорець ламає тести, і
    ламає з тієї єдиної причини, що вони про нього не знали."""
    ap = argparse.ArgumentParser(
        prog="retouch",
        description="Автоматика шкіри та видалення об'єктів. Вихід — шари.")
    ap.add_argument("input", nargs="?", help="файл або тека")
    ap.add_argument("-o", "--out", default="out", help="куди писати")
    ap.add_argument("--remove-mask", help="біла маска = що видалити")
    ap.add_argument("--config", help="YAML з параметрами (прапорці мають пріоритет)")
    ap.add_argument("--preset", action="append", default=None, metavar="FILE",
                    help="пресет; можна кілька — накладаються зліва направо, "
                         "прапорці виграють в усіх")
    ap.add_argument("--xmp", nargs="?", const="auto", default=None, metavar="FILE",
                    help="прочитати налаштування Camera Raw; без значення — "
                         "шукати сайдкар поруч із кадром. Застосовується те, "
                         "що можемо, решта йде у звіт")
    ap.add_argument("--schema", action="store_true",
                    help="вивести схему пресету (для агента) і вийти")

    ap.add_argument("--radius", type=float, default=None,
                    help="радіус частотки в px (типово — з ширини обличчя)")
    ap.add_argument("--threshold", type=float, default=None,
                    help=f"поріг контрасту дефекту, менше = агресивніше "
                         f"(типово {_D.threshold}, див. spec.md §6.2). "
                         f"МІЖ КАДРАМИ НЕ ПЕРЕНОСИТЬСЯ — див. --target-coverage")
    ap.add_argument("--target-coverage", type=float, default=None,
                    help="замість порога — ціль: яка частка ШКІРИ має бути "
                         "торкнута (напр. 0.03). Поріг підбирається під кадр. "
                         "Саме це переноситься між кадрами, а не поріг")
    ap.add_argument("--min-area", type=int, default=None,
                    help=f"типово {_D.min_area}")
    ap.add_argument("--max-area", type=int, default=None,
                    help=f"типово {_D.max_area}")
    ap.add_argument("--max-elongation", type=float, default=None,
                    help=f"відношення сторін bbox: більше — це волосина "
                         f"чи край, не пляма (типово {_D.max_elongation})")
    ap.add_argument("--strength", type=float, default=None,
                    help=f"сила лікування 0..1 (типово {_C.strength})")
    ap.add_argument("--limit", type=int, default=None,
                    help="лікувати лише N найконтрастніших плям")
    ap.add_argument("--search-radius", type=int, default=None,
                    help=f"як далеко шукати донора, px (типово {_C.search_radius})")
    ap.add_argument("--mask-erode", type=int, default=None,
                    help=f"відступ від краю маски шкіри, px (типово {_M.erode})")
    ap.add_argument("--no-skin-mask", action="store_true",
                    help="обробляти весь кадр (для тестів)")
    ap.add_argument("--force-mask", action="store_true",
                    help="не зупинятися, якщо маска шкіри неправдоподібна")

    ap.add_argument("--raw-decoder", choices=("rawpy", "imageio"), default=None,
                    help="чим читати RAW; типово rawpy, якщо є, інакше ImageIO. "
                         "Поріг детекції від цього залежить — див. spec.md §4")
    ap.add_argument("--face-model", default=None, help="ONNX face-parsing")
    ap.add_argument("--face-detector", default=None,
                    help="ONNX YuNet: кроп голови перед face-parsing")
    ap.add_argument("--lama-model", default=None, help="ONNX LaMa")
    ap.add_argument("--preview", action="store_true",
                    help="оглядовий аркуш PNG: загальний план + кропи 1:1")
    ap.add_argument("--debug", action="store_true",
                    help="скинути всі проміжні шари")
    ap.add_argument("--dry-run", action="store_true",
                    help="лише порахувати дефекти, нічого не писати")
    ap.add_argument("--tools", default=None,
                    help="дрібні інструменти через кому: eye_vessels, teeth, "
                         "mattify, skin_tone. Потребують --face-model")
    ap.add_argument("--dodge-burn", action="store_true",
                    help="вирівняти низьку частоту (D&B), spec.md §11 v0.5")
    ap.add_argument("--db-strength", type=float, default=None,
                    help="сила D&B 0..1")
    ap.add_argument("--batch", action="store_true",
                    help="пакетний режим: збій на кадрі не зупиняє решту, "
                         "оброблене пропускається, поруч шукається IMG.yaml")
    ap.add_argument("--no-resume", action="store_true",
                    help="у пакетному режимі обробляти й те, що вже зроблено")

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.schema:
        import json
        print(json.dumps(presets_mod.schema(), ensure_ascii=False, indent=2))
        return 0
    try:
        cfg = build_config(a)
    except presets_mod.PresetError as e:
        print(e, file=sys.stderr)
        return 1

    if not a.input:
        ap.error("не вказано вхідний файл або теку")
    src = Path(a.input)
    files = ([p for p in sorted(src.iterdir()) if p.suffix.lower() in SUFFIXES]
             if src.is_dir() else [src])
    if not files:
        # Найчастіша причина порожньої теки — вона повна RAW. Мовчазне
        # "нічого обробляти" на теці з 200 CR3 збиває з пантелику.
        raws = ([p for p in sorted(src.iterdir())
                 if p.suffix.lower() in RAW_SUFFIXES] if src.is_dir() else [])
        if raws:
            print(f"у {src} лише RAW ({len(raws)} шт., напр. {raws[0].name}).\n"
                  f"RAW проєкт не читає навмисно — spec.md §4. Потрібен "
                  f"16-бітний TIFF з Camera Raw.", file=sys.stderr)
        else:
            print("нічого обробляти", file=sys.stderr)
        return 1

    if a.batch:
        def show(item, rep):
            n = len(rep.items)
            mark = {"done": "ок", "skipped": "проп.", "failed": "ЗБІЙ"}[item.status]
            extra = f"  [{item.preset}]" if item.preset else ""
            print(f"[{n}] {item.path.name}: {mark} {item.seconds:.1f}s"
                  f"{extra}{'  ' + item.note if item.note else ''}", flush=True)

        base = presets_mod.merge(*[presets_mod.load(p) for p in (a.preset or [])])
        # У пакеті XMP читається ПОКАДРОВО, всередині process(). Тут його
        # знімаємо: build_config шукав би сайдкар поруч із ТЕКОЮ і на
        # кожному кадрі повідомляв, що не знайшов.
        per_frame = argparse.Namespace(**{**vars(a), "xmp": None})
        rep = batch_mod.process(
            src, a.out, base_preset=base,
            cfg_factory=lambda: build_config(per_frame),
            resume=not a.no_resume, preview=a.preview, debug=a.debug,
            use_xmp=bool(a.xmp), on_item=show)
        print("\n" + rep.text())
        return 1 if rep.failed else 0

    for f in files:
        print(f"\n=== {f.name} ===")
        try:
            if a.dry_run:
                r = detect_only(f, cfg)
                print(f"{len(r['blobs'])} плям, радіус {r['radius']:.1f}px, "
                      f"маска: {r['skin_source']}")
                if r["warn"]:
                    print(f"УВАГА: {r['warn']}", file=sys.stderr)
                for b in r["blobs"][:15]:
                    print(f"  контраст {b['contrast']:.4f}  площа {b['area']:5d}"
                          f"  центр {b['center'][0]:.0f},{b['center'][1]:.0f}")
                continue
            run(f, a.out, cfg, a.remove_mask, debug=a.debug, preview=a.preview)
        except (InputError, MaskSanityError) as e:
            # Помилка користувача, не збій програми: показуємо текст, а не
            # трасування. На теці йдемо далі — один битий файл не має
            # зупиняти пакет.
            print(f"{e}", file=sys.stderr)
            if len(files) == 1:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
