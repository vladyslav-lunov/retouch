"""Пресети: часткові, накладаються один на одного, несуть причину.

Три вимоги, які відрізняють це від звичайного файлу налаштувань
(spec.md §1.2), і кожна з них — наслідок того, що пресети пише агент.

**Часткові.** Пресет задає ЛИШЕ те, що змінює. Ключа немає — параметр не
чіпаємо. Інакше стильовий пресет на зйомку і покадрове уточнення
неможливо накласти: другий затирав би все, чого не згадав.

**Накладаються.** `merge(a, b)` — b виграє поле за полем, углиб. Так
працює послідовність «дефолти -> пресет зйомки -> пресет кадру -> руки».

**Із причиною.** Поле `why` вільним текстом. Жоден формат проявника —
ні XMP, ні .pp3 — цього не має: вони зберігають ЩО, але не ЧОМУ. Коли
агент пропонує десять варіантів, без «чому» вибрати з них неможливо,
бо в числах вони виглядають однаково.

Схема для агента будується з САМИХ дата-класів (`schema()`), разом з
описами полів, які в цьому проєкті пишуться рядком під полем. Python їх
не зберігає, тому дістаємо з вихідного коду через ast — зате опис у схемі
завжди той самий, що в коді, і розійтися вони не можуть.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path
from typing import Any

# Розділи пресету -> куди вони лягають у Config. Порядок тут визначає
# порядок у схемі, тобто те, як його читатиме агент.
SECTIONS = {
    "develop": "develop",
    "detect": "detect",
    "mask": "mask",
    "warp": "warp",
}


class PresetError(Exception):
    """Пресет непридатний. Окремий тип, щоб CLI показав текст, а не трасування."""


# ---------------------------------------------------------------------------
# читання і накладання
# ---------------------------------------------------------------------------

def load(path: str | Path) -> dict:
    """Прочитати пресет. YAML або JSON — розрізняємо за вмістом, не за назвою."""
    import yaml

    p = Path(path)
    if not p.exists():
        raise PresetError(f"пресету немає: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:                                # noqa: BLE001
        raise PresetError(f"{p.name}: не читається як YAML/JSON — {e}") from None
    if not isinstance(data, dict):
        raise PresetError(f"{p.name}: очікував словник, а там {type(data).__name__}")
    return data


def merge(*presets: dict) -> dict:
    """Накласти пресети зліва направо. Пізніший виграє поле за полем.

    Углиб, а не поверхнево: якщо перший задав `detect.threshold`, а другий
    `detect.min_area`, у результаті буде обидва. Поверхневе злиття
    втратило б перший — а саме цей випадок і є «стиль зйомки плюс
    уточнення кадру».
    """
    out: dict = {}
    for p in presets:
        for k, v in (p or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = {**out[k], **v}
            else:
                out[k] = v
    return out


def apply(cfg, data: dict, strict: bool = False) -> list[str]:
    """Накласти пресет на Config НА МІСЦІ. Повертає список зауважень.

    Незнайомі ключі не валять роботу: пресет може прийти від агента, від
    старішої версії або з чужого XMP, і втратити через одну зайву стрічку
    всі інші двадцять — гірше, ніж їх застосувати. Але мовчати не можна,
    тому зауваження повертаються і показуються.
    """
    notes: list[str] = []
    for key, value in data.items():
        if key in ("name", "why", "for", "author", "created"):
            continue                                   # метадані, не параметри
        if key in SECTIONS:
            target = getattr(cfg, SECTIONS[key], None)
            if target is None:
                notes.append(f"розділ '{key}' цією версією не підтримується")
                continue
            if not isinstance(value, dict):
                notes.append(f"розділ '{key}' має бути словником")
                continue
            fields = {f.name for f in dataclasses.fields(target)}
            for k, v in value.items():
                if k not in fields:
                    notes.append(f"{key}.{k}: невідомий параметр")
                    continue
                setattr(target, k, _coerce(target, k, v))
        elif hasattr(cfg, key):
            setattr(cfg, key, v if (v := value) is None else _coerce(cfg, key, v))
        else:
            notes.append(f"{key}: невідомий ключ")
    if strict and notes:
        raise PresetError("пресет має проблеми:\n  " + "\n  ".join(notes))
    return notes


def _coerce(obj, name: str, value):
    """Привести значення до типу поля. YAML не розрізняє кортеж і список,
    а дата-класи в цьому проєкті подекуди тримають кортежі."""
    cur = getattr(obj, name, None)
    if isinstance(cur, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def save(path: str | Path, data: dict) -> Path:
    """Записати пресет. `why` виводимо блоком, щоб його було зручно читати."""
    import yaml

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False,
                                default_flow_style=False, width=76),
                 encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# схема для агента
# ---------------------------------------------------------------------------

def _field_docs(cls) -> dict[str, str]:
    """Описи полів дата-класу з вихідного коду.

    У цьому проєкті опис пишеться рядком ПІД полем. Python такі рядки не
    зберігає ніде, тому читаємо джерело через ast. Зате опис у схемі
    завжди той самий, що в коді.
    """
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return {}
    try:
        tree = ast.parse(inspect.cleandoc(src))
    except SyntaxError:
        return {}
    body = tree.body[0].body if tree.body else []
    docs, prev = {}, None
    for node in body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            prev = node.target.id
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
              and isinstance(node.value.value, str) and prev):
            docs[prev] = " ".join(node.value.value.split())
            prev = None
        else:
            prev = None
    return docs


def _describe(cls) -> dict:
    docs = _field_docs(cls)
    out = {}
    for f in dataclasses.fields(cls):
        default = f.default
        if default is dataclasses.MISSING:
            default = None if f.default_factory is dataclasses.MISSING else "(обчислюється)"
        if isinstance(default, tuple):
            default = list(default)
        out[f.name] = {"type": _type_name(f.type), "default": default,
                       "doc": docs.get(f.name, "")}
    return out


def _type_name(t) -> str:
    s = t if isinstance(t, str) else getattr(t, "__name__", str(t))
    return s.replace("| None", "or null").strip()


def schema() -> dict:
    """Машиночитна схема пресету — щоб агент писав валідний без здогадок.

    Береться з дата-класів, тобто не може розійтися з кодом. Разом з
    описами полів: агент має розуміти, що означає параметр, інакше
    обґрунтувати своє рішення в `why` він не зможе.
    """
    from .blemish import DetectParams
    from .develop import DevelopParams
    from .masks import CELEBA_CLASSES, MaskParams
    from .pipeline import Config
    from .warp import WarpParams

    cfg_fields = _describe(Config)
    for s in SECTIONS:
        cfg_fields.pop(s, None)
    return {
        "meta": {
            "name": "коротка назва пресету",
            "why": "ЧОМУ саме такі значення. Вільний текст. Головне поле "
                   "для вибору між кількома пресетами — без нього вони "
                   "виглядають однаково.",
            "for": "необов'язково: до якого кадру або зйомки це",
        },
        "rules": [
            "пресет ЧАСТКОВИЙ: задавай лише те, що змінюєш",
            "пресети накладаються зліва направо, пізніший виграє",
            "невідомі ключі не валять роботу, але потрапляють у зауваження",
        ],
        "sections": {
            "develop": _describe(DevelopParams),
            "detect": _describe(DetectParams),
            "mask": _describe(MaskParams),
            "warp": _describe(WarpParams),
            "(top-level)": cfg_fields,
        },
        "vocabulary": {
            "mask.skin_classes / mask.exclude_classes":
                sorted(set(CELEBA_CLASSES.values())),
        },
    }
