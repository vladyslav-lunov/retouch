"""Читання RAW. Свій демозаїк ми НЕ пишемо — це місяці роботи заради
гіршого результату (spec.md §4). Беремо чужий, і саме ті два, які §4
назвав як прийнятні: libraw через rawpy та Core Image / ImageIO на macOS.

Два шляхи з різними компромісами:

1. **rawpy (libraw)** — кращий за якістю саме для ЦЬОГО інструмента.
   Параметри демозаїка, баланс білого й гама задані явно, тож два
   прогони того самого файлу дають однакову високу частоту. Для §6.2,
   де поріг міряється по контрасту в HF, відтворюваність критична.
   Ставиться окремо: `pip install rawpy`.

2. **ImageIO** через ctypes — той самий рушій, що в Preview і sips.
   Нуль залежностей, працює одразу. Але Apple рендерить RAW зі своїм
   шумозаглушенням і різкістю, і саме вони живуть у високій частоті —
   тобто в тому шарі, з яким працює весь конвеєр. Керувати цим не можна.

Тому за замовчуванням беремо rawpy, якщо він є, і ImageIO, якщо нема.
Обидва повертають uint16 RGB у display-referred кодуванні з гамою —
як вимагає §4, бо в лінійному просторі частотка поводиться не так.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
from pathlib import Path

import numpy as np

# Розширення, які вважаємо RAW. Список ширший за те, що реально
# декодується: краще сказати «цей RAW не подужав такий-то декодер», ніж
# «не впізнав формат».
RAW_SUFFIXES = {".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2",
                ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw", ".raw", ".3fr",
                ".erf", ".kdc", ".mos", ".mrw", ".x3f", ".iiq"}


# ---------------------------------------------------------------------------
# rawpy / libraw
# ---------------------------------------------------------------------------

def _have_rawpy() -> bool:
    try:
        import rawpy  # noqa: F401
        return True
    except ImportError:
        return False


def _read_rawpy(path: Path) -> np.ndarray:
    """uint16 RGB. Параметри явні — щоб результат не «плив» між прогонами."""
    import rawpy

    with rawpy.imread(str(path)) as r:
        return r.postprocess(
            output_bps=16,
            use_camera_wb=True,      # ББ як його поставив фотограф, не «авто»
            no_auto_bright=True,     # без автояскравості: інакше однакові кадри
                                     # серії розійдуться по експозиції
            gamma=(2.222, 4.5),      # sRGB-подібна гама: §4 вимагає
                                     # display-referred, не лінійне
            output_color=rawpy.ColorSpace.sRGB,
        )


# ---------------------------------------------------------------------------
# ImageIO (macOS) через ctypes
# ---------------------------------------------------------------------------

_CGFloat = ctypes.c_double
_VOIDP = ctypes.c_void_p
_UTF8 = 0x08000100
_ALPHA_NONE_SKIP_LAST = 5
_BYTE_ORDER_16_LITTLE = 1 << 12


class _CGPoint(ctypes.Structure):
    _fields_ = [("x", _CGFloat), ("y", _CGFloat)]


class _CGSize(ctypes.Structure):
    _fields_ = [("width", _CGFloat), ("height", _CGFloat)]


class _CGRect(ctypes.Structure):
    _fields_ = [("origin", _CGPoint), ("size", _CGSize)]


class _Frameworks:
    """Ліниве завантаження: на не-macOS цього просто немає, і імпорт
    модуля не повинен від цього падати."""

    _fw = None

    @classmethod
    def get(cls):
        if cls._fw is None:
            if sys.platform != "darwin":
                raise RuntimeError("ImageIO є лише на macOS")
            cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
            cg = ctypes.CDLL(ctypes.util.find_library("CoreGraphics"))
            io = ctypes.CDLL(ctypes.util.find_library("ImageIO"))

            cf.CFStringCreateWithCString.restype = _VOIDP
            cf.CFStringCreateWithCString.argtypes = [_VOIDP, ctypes.c_char_p,
                                                     ctypes.c_uint32]
            cf.CFURLCreateWithFileSystemPath.restype = _VOIDP
            cf.CFURLCreateWithFileSystemPath.argtypes = [_VOIDP, _VOIDP,
                                                         ctypes.c_long, ctypes.c_bool]
            cf.CFRelease.argtypes = [_VOIDP]
            cf.CFDictionaryGetValue.restype = _VOIDP
            cf.CFDictionaryGetValue.argtypes = [_VOIDP, _VOIDP]
            cf.CFNumberGetValue.restype = ctypes.c_bool
            cf.CFNumberGetValue.argtypes = [_VOIDP, ctypes.c_long, _VOIDP]

            io.CGImageSourceCreateWithURL.restype = _VOIDP
            io.CGImageSourceCreateWithURL.argtypes = [_VOIDP, _VOIDP]
            io.CGImageSourceCreateImageAtIndex.restype = _VOIDP
            io.CGImageSourceCreateImageAtIndex.argtypes = [_VOIDP, ctypes.c_size_t,
                                                           _VOIDP]
            io.CGImageSourceCopyPropertiesAtIndex.restype = _VOIDP
            io.CGImageSourceCopyPropertiesAtIndex.argtypes = [_VOIDP, ctypes.c_size_t,
                                                              _VOIDP]

            cg.CGImageGetWidth.restype = ctypes.c_size_t
            cg.CGImageGetWidth.argtypes = [_VOIDP]
            cg.CGImageGetHeight.restype = ctypes.c_size_t
            cg.CGImageGetHeight.argtypes = [_VOIDP]
            cg.CGColorSpaceCreateWithName.restype = _VOIDP
            cg.CGColorSpaceCreateWithName.argtypes = [_VOIDP]
            cg.CGBitmapContextCreate.restype = _VOIDP
            cg.CGBitmapContextCreate.argtypes = [_VOIDP, ctypes.c_size_t,
                                                 ctypes.c_size_t, ctypes.c_size_t,
                                                 ctypes.c_size_t, _VOIDP,
                                                 ctypes.c_uint32]
            cg.CGContextDrawImage.argtypes = [_VOIDP, _CGRect, _VOIDP]
            cg.CGContextRelease.argtypes = [_VOIDP]
            cg.CGImageRelease.argtypes = [_VOIDP]
            cg.CGColorSpaceRelease.argtypes = [_VOIDP]
            cls._fw = (cf, cg, io)
        return cls._fw


def _orient(arr: np.ndarray, code: int) -> np.ndarray:
    """Застосувати EXIF-орієнтацію (1-8).

    CGImageSourceCreateImageAtIndex віддає кадр ЯК ЗНЯТО, не повертаючи
    його. Без цього портрет із поверненої камери приїздив би лежачи, і
    radius_for міряв би ширину обличчя по висоті кадру.
    """
    if code == 2:
        return arr[:, ::-1]
    if code == 3:
        return arr[::-1, ::-1]
    if code == 4:
        return arr[::-1]
    if code == 5:
        return np.swapaxes(arr, 0, 1)
    if code == 6:
        return np.swapaxes(arr, 0, 1)[:, ::-1]
    if code == 7:
        return np.swapaxes(arr, 0, 1)[::-1, ::-1]
    if code == 8:
        return np.swapaxes(arr, 0, 1)[::-1]
    return arr


def _read_imageio(path: Path) -> np.ndarray:
    cf, cg, io = _Frameworks.get()

    def cfstr(s: str):
        return cf.CFStringCreateWithCString(None, s.encode(), _UTF8)

    p = cfstr(str(path))
    url = cf.CFURLCreateWithFileSystemPath(None, p, 0, False)
    cf.CFRelease(p)
    if not url:
        raise RuntimeError("не вдалося зробити CFURL")
    src = io.CGImageSourceCreateWithURL(url, None)
    cf.CFRelease(url)
    if not src:
        raise RuntimeError("ImageIO не відкрив файл")

    try:
        code = 1
        props = io.CGImageSourceCopyPropertiesAtIndex(src, 0, None)
        if props:
            key = _VOIDP.in_dll(io, "kCGImagePropertyOrientation")
            num = cf.CFDictionaryGetValue(props, key)
            if num:
                out = ctypes.c_int()
                if cf.CFNumberGetValue(num, 9, ctypes.byref(out)):  # kCFNumberIntType
                    code = int(out.value)
            cf.CFRelease(props)

        img = io.CGImageSourceCreateImageAtIndex(src, 0, None)
        if not img:
            raise RuntimeError("ImageIO не декодував кадр (формат не підтримується?)")
    finally:
        cf.CFRelease(src)

    try:
        w, h = cg.CGImageGetWidth(img), cg.CGImageGetHeight(img)
        name = cfstr("kCGColorSpaceSRGB")
        cs = cg.CGColorSpaceCreateWithName(name)
        cf.CFRelease(name)
        # 16 біт на канал: 8-бітний контекст з'їв би розрядність, заради
        # якої весь §4 і написано
        buf = (ctypes.c_uint16 * (w * h * 4))()
        ctx = cg.CGBitmapContextCreate(
            buf, w, h, 16, w * 8, cs,
            _ALPHA_NONE_SKIP_LAST | _BYTE_ORDER_16_LITTLE)
        if not ctx:
            cg.CGColorSpaceRelease(cs)
            raise RuntimeError("не створився 16-бітний контекст")
        cg.CGContextDrawImage(ctx, _CGRect(_CGPoint(0, 0), _CGSize(w, h)), img)
        rgb = np.ctypeslib.as_array(buf).reshape(h, w, 4)[:, :, :3].copy()
        cg.CGContextRelease(ctx)
        cg.CGColorSpaceRelease(cs)
    finally:
        cg.CGImageRelease(img)

    return np.ascontiguousarray(_orient(rgb, code))


# ---------------------------------------------------------------------------
# фасад
# ---------------------------------------------------------------------------

def decoders() -> list[str]:
    """Які шляхи доступні тут і зараз, у порядку переваги."""
    out = []
    if _have_rawpy():
        out.append("rawpy")
    if sys.platform == "darwin":
        out.append("imageio")
    return out


def read_raw(path: str | Path, prefer: str | None = None) -> tuple[np.ndarray, str]:
    """RAW -> (uint16 RGB, назва декодера).

    Пробуємо по черзі: якщо rawpy спіткнувся об екзотичний формат, ще є
    шанс, що ImageIO його знає, і навпаки.
    """
    path = Path(path)
    order = decoders()
    if prefer and prefer in order:
        order = [prefer] + [d for d in order if d != prefer]
    if not order:
        raise RuntimeError(
            "нема чим читати RAW: rawpy не встановлено, а ImageIO є лише на macOS.\n"
            "    pip install rawpy")

    errors = []
    for name in order:
        try:
            arr = _read_rawpy(path) if name == "rawpy" else _read_imageio(path)
            if arr.dtype != np.uint16:
                arr = arr.astype(np.uint16)
            return arr, name
        except Exception as e:                       # noqa: BLE001
            errors.append(f"{name}: {e}")
    raise RuntimeError(f"жоден декодер не подужав {path.name}\n    " +
                       "\n    ".join(errors))
