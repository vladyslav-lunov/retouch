"""Пакетна обробка: пресет на теку, без нагляду.

Три речі, без яких пакетний прогін не пакетний, а просто цикл.

**Один битий кадр не зупиняє ніч.** Помилка на файлі записується у звіт,
і робота йде далі. Прокинути виняток нагору означає, що вранці буде
оброблено три кадри з двохсот і трасування.

**Продовження.** Якщо кадр уже оброблено, він пропускається. На 8 ГБ і
двох ядрах ніч — це реальний час прогону, і починати спочатку через
перерване живлення не можна.

**Покадровий пресет.** Поруч із `IMG.tif` може лежати `IMG.yaml`, і він
накладається ПОВЕРХ загального. Це те, заради чого пресети зроблено
частковими (spec.md §1.2): стиль зйомки один, уточнення на кожен кадр
своє. Агент пише і те, і те.

Паралелізму немає навмисно. Ядер два, а один кадр на 26 Мп бере до 2 ГБ
(§2) — два процеси просто вийдуть у своп і стануть повільнішими за один.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from . import presets as presets_mod
from .imageio import RAW_SUFFIXES, InputError
from .pipeline import Config, MaskSanityError, run

SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"} | RAW_SUFFIXES
PRESET_SUFFIXES = (".yaml", ".yml", ".json")


@dataclass
class Item:
    path: Path
    status: str = "pending"      # done | skipped | failed
    seconds: float = 0.0
    note: str = ""
    preset: str = ""


@dataclass
class Report:
    items: list[Item] = field(default_factory=list)

    @property
    def done(self) -> list[Item]:
        return [i for i in self.items if i.status == "done"]

    @property
    def failed(self) -> list[Item]:
        return [i for i in self.items if i.status == "failed"]

    @property
    def skipped(self) -> list[Item]:
        return [i for i in self.items if i.status == "skipped"]

    def text(self) -> str:
        L = [f"оброблено {len(self.done)}, пропущено {len(self.skipped)}, "
             f"збоїв {len(self.failed)}"]
        if self.done:
            t = sum(i.seconds for i in self.done)
            L.append(f"час {t:.0f}s, у середньому {t / len(self.done):.1f}s на кадр")
        for i in self.failed:
            L.append(f"  ЗБІЙ {i.path.name}: {i.note}")
        return "\n".join(L)


def find_inputs(src: str | Path) -> list[Path]:
    src = Path(src)
    if src.is_file():
        return [src]
    if not src.is_dir():
        return []
    return sorted(p for p in src.iterdir()
                  if p.is_file() and p.suffix.lower() in SUFFIXES)


def sidecar_for(path: Path) -> Path | None:
    """Покадровий пресет поруч із файлом: IMG.tif -> IMG.yaml."""
    for suf in PRESET_SUFFIXES:
        p = path.with_suffix(suf)
        if p.exists():
            return p
    return None


def already_done(path: Path, out_dir: Path) -> bool:
    """Ознака обробленого — зведений файл. Саме він пишеться останнім
    зі значущих, тож його наявність означає, що кадр дійшов до кінця."""
    return (out_dir / f"{path.stem}_99_flat.tif").exists()


def process(
    src: str | Path,
    out_dir: str | Path,
    base_preset: dict | None = None,
    cfg_factory=None,
    resume: bool = True,
    limit: int | None = None,
    preview: bool = False,
    debug: bool = False,
    use_xmp: bool = False,
    on_item=None,
) -> Report:
    """Прогнати теку. cfg_factory() має віддавати СВІЖИЙ Config на кожен
    кадр — інакше пресет одного кадру протече в наступний.

    use_xmp — шукати сайдкар Camera Raw до КОЖНОГО кадру. Тут це
    доречніше, ніж деінде: у теці зі зйомки в кожного кадру свій кроп і
    своя експозиція, і саме вони роблять покадрову різницю, якої пресет
    на зйомку дати не може."""
    out_dir = Path(out_dir)
    rep = Report()
    files = find_inputs(src)
    if limit:
        files = files[:limit]

    for path in files:
        item = Item(path=path)
        rep.items.append(item)
        if resume and already_done(path, out_dir):
            item.status = "skipped"
            item.note = "вже оброблено"
            if on_item:
                on_item(item, rep)
            continue

        t0 = time.time()
        try:
            cfg = cfg_factory() if cfg_factory else Config()
            # Порядок той самий, що в CLI: XMP — база (це те, що
            # фотограф уже вирішив у проявнику), далі стиль зйомки, далі
            # уточнення кадру. Перекривати XMP пресетом можна, навпаки —
            # ні, інакше стиль на теку не діяв би на кадрах із сайдкаром.
            stack: list[dict] = []
            marks: list[str] = []
            if use_xmp:
                from . import xmp as xmp_mod
                try:
                    pre, _rep, where = xmp_mod.from_image(path)
                except xmp_mod.XmpError as e:
                    pre, where = None, ""
                    item.note = f"XMP: {e}"
                if pre:
                    stack.append(pre)
                    marks.append(Path(where).name if where else "xmp")
            stack.append(base_preset or {})
            side = sidecar_for(path)
            if side:
                stack.append(presets_mod.load(side))
                marks.append(side.name)
            item.preset = " + ".join(marks)
            merged = presets_mod.merge(*stack)
            if merged:
                presets_mod.apply(cfg, merged)
            run(path, out_dir, cfg, preview=preview, debug=debug)
            item.status = "done"
        except (InputError, MaskSanityError, presets_mod.PresetError) as e:
            item.status = "failed"
            item.note = str(e).splitlines()[0]
        except Exception as e:                              # noqa: BLE001
            # Несподіване теж не має валити ніч, але у звіті має бути
            # видно, що це саме несподіване, а не відмова за правилами.
            item.status = "failed"
            item.note = f"{type(e).__name__}: {e}".splitlines()[0]
            item.note += "  (несподівано)"
            traceback.print_exc(limit=2)
        finally:
            item.seconds = time.time() - t0
        if on_item:
            on_item(item, rep)
    return rep
