"""Калібрування порога на СВОЇХ кадрах — те, чого вимагає spec.md §6.2.

`selftest.py` міряє на макеті, і його ж докстрока забороняє переносити
ті числа на зйомку. Тут інше: беремо теку реальних кадрів, ганяємо
кожен кількома порогами з ФІКСОВАНИМ проявленням і фіксованим набором
класів — і дивимось не на середнє, а на РОЗКИД.

    python3 scripts/calibrate.py SHOOT/ -o calib \\
        --face-model models/resnet18.onnx --face-detector models/yunet.onnx

Чому розкид, а не середнє. Головне питання не «яке число поставити
дефолтом», а «чи існує одне число на всі кадри». Якщо на одному кадрі
розумний поріг 0.010, а на іншому 0.020, то дефолт неправильний за
побудовою, і правильна відповідь — покадровий пресет від агента
(spec.md §1.2). Середнє це питання ховає, розкид — показує.

Істини на реальному кадрі немає, тож повноту порахувати нічим. Але дві
речі рахуються без істини, і обидві — про хибні спрацювання:

  **частка знахідок у підозрілих класах** (шия, одяг, вухо) — там майже
  завжди прикраса, комір або тінь, а не дефект (§15, заміряно);

  **частка у `background`** — дефект шкіри там неможливий за
  визначенням; якщо він є, потекла маска.

Обидві — нижня оцінка браку, а не точна. Але нижня оцінка, яка не
вимагає розмічати кадри руками, вартує більше за точну, якої не буде.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from retouch.batch import find_inputs
from retouch.blemish import DetectParams, detect_blemishes, heal_blemishes
from retouch.imageio import InputError
from retouch.masks import CELEBA_CLASSES, MaskParams, detect_faces
from retouch.pipeline import Config, Session

THRESHOLDS = (0.008, 0.010, 0.012, 0.014, 0.018, 0.022, 0.025)

# Класи, у яких «дефект» майже завжди не дефект. Той самий список, що в
# Session.SUSPECT_CLASSES, плюс background: там дефект шкіри неможливий.
SUSPECT = ("neck", "neck_l", "l_ear", "r_ear", "ear_r", "cloth", "hair",
           "background", "hat")


def measure(sess: Session, thresholds) -> list[dict]:
    """Одна прогонка кадру всіма порогами. Частотка рахується один раз."""
    rows = []
    skin_px = float(sess.skin.sum()) if sess.skin is not None else 0.0
    for t in thresholds:
        lbl, blobs = detect_blemishes(sess.high, sess.skin,
                                      DetectParams(threshold=t))
        _h2, cov = heal_blemishes(sess.high, lbl, blobs, sess.skin,
                                  search_radius=sess.search_radius_px)
        by_class, suspect = {}, 0
        if sess.cls is not None and blobs:
            h, w = sess.cls.shape[:2]
            for b in blobs:
                x, y = b["center"]
                name = CELEBA_CLASSES.get(
                    int(sess.cls[min(h - 1, max(0, int(y))),
                                 min(w - 1, max(0, int(x)))]), "?")
                by_class[name] = by_class.get(name, 0) + 1
                if name in SUSPECT:
                    suspect += 1
        touched = float((cov > 0).sum())
        rows.append({
            "threshold": t,
            "blobs": len(blobs),
            # Частка ШКІРИ, а не кадру: кадр включає фон, і на портреті
            # в повний зріст те саме лікування дає вчетверо менше число.
            # Порівнювати кадри між собою можна лише по шкірі.
            "touched_of_skin": round(touched / skin_px, 5) if skin_px else None,
            "touched_of_frame": round(touched / cov.size, 6),
            "suspect_share": round(suspect / len(blobs), 3) if blobs else 0.0,
            "by_class": by_class,
        })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="калібрування порога на реальних кадрах (spec.md §6.2)")
    ap.add_argument("input", nargs="?", help="тека або файл")
    ap.add_argument("-o", "--out", default="calib")
    ap.add_argument("--face-model", default=None)
    ap.add_argument("--face-detector", default=None)
    ap.add_argument("--raw-decoder", default=None, choices=("rawpy", "imageio"))
    ap.add_argument("--skin-classes", default=None,
                    help="через кому; типово — з MaskParams")
    ap.add_argument("--preset", default=None,
                    help="ФІКСОВАНЕ проявлення на всі кадри; без нього "
                         "кадри з різним проявленням непорівнянні")
    ap.add_argument("--thresholds", default=None, help="через кому")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--from-json", default=None, metavar="PATH",
                    help="перебудувати звіт із calib.json (або теки з ним) "
                         "без повторного декодування RAW")
    a = ap.parse_args(argv)
    if a.from_json:
        return _resume(a)

    ths = ([float(x) for x in a.thresholds.split(",")] if a.thresholds
           else list(THRESHOLDS))
    if not a.input:
        print("вкажи теку з кадрами або --from-json", file=sys.stderr)
        return 1
    files = find_inputs(a.input)[:a.limit] if a.limit else find_inputs(a.input)
    if not files:
        print("нічого калібрувати", file=sys.stderr)
        return 1

    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    frames, skipped = [], []

    for i, path in enumerate(files, 1):
        print(f"\n=== [{i}/{len(files)}] {path.name} ===", flush=True)
        try:
            mp = MaskParams()
            if a.skin_classes:
                mp = MaskParams(erode=mp.erode, feather=mp.feather,
                                exclude_dilate=mp.exclude_dilate,
                                exclude_classes=mp.exclude_classes,
                                skin_classes=tuple(
                                    x.strip() for x in a.skin_classes.split(",")))
            cfg = Config(face_model=a.face_model, face_detector=a.face_detector,
                         raw_decoder=a.raw_decoder, mask=mp, force_mask=True)
            if a.preset:
                from retouch import presets as pm
                pm.apply(cfg, pm.load(a.preset))
            sess = Session(path, cfg).load().analyze()
        except (InputError, Exception) as e:                  # noqa: BLE001
            print(f"  пропущено: {type(e).__name__}: {e}", flush=True)
            skipped.append({"file": path.name, "why": f"{type(e).__name__}: {e}"})
            continue

        # Кадр із неправдоподібною маскою НЕ усереднюємо разом з рештою:
        # він зіпсує таблицю тихо, а помітити це буде нíяк.
        frac = float(sess.skin.mean()) if sess.skin is not None else 0.0
        if sess.skin_source.startswith("heuristic"):
            print(f"  пропущено: маска евристична — поріг поверх такої "
                  f"маски безпідставний (§6.2)", flush=True)
            skipped.append({"file": path.name, "why": "евристична маска"})
            continue

        faces = detect_faces(sess.img, cfg.face_detector) if cfg.face_detector else []
        h, w = sess.img.shape[:2]
        row = {
            "file": path.name, "w": w, "h": h,
            "skin_source": sess.skin_source,
            "skin_frac": round(frac, 5),
            "radius_px": round(sess.radius, 2),
            "face_w": round(float(faces[0][2]), 1) if len(faces) else None,
            "sweep": measure(sess, ths),
        }
        frames.append(row)
        print(f"  маска {frac:.2%}, радіус {row['radius_px']} px, "
              f"обличчя {row['face_w']} px", flush=True)
        for r in row["sweep"]:
            print(f"    {r['threshold']:.3f}  плям {r['blobs']:5}  "
                  f"шкіри {(r['touched_of_skin'] or 0):7.3%}  "
                  f"підозрілих {r['suspect_share']:5.0%}", flush=True)

        # 44 кадри по 26 Мп послідовно: кожен тримає img, low, high, cls,
        # labels — це ~1.5 ГБ на кадр. Без явного звільнення набір не
        # доживе до кінця на 8 ГБ (spec.md §2), а помилка буде не
        # «мало пам'яті», а тихий своп на годину.
        sess.img = sess.low = sess.high = sess.high2 = None
        sess.cls = sess.skin = sess.skin_auto = sess.labels = None
        sess.coverage = sess.result = sess.img_src = None
        del sess
        gc.collect()

        # Проміжний запис після КОЖНОГО кадру: прогін на 44 кадри — це
        # двадцять хвилин, і втратити його через збій на сороковому
        # безглуздо.
        (d / "calib.partial.json").write_text(
            json.dumps({"frames": frames, "skipped": skipped},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    if not frames:
        print("\nжодного придатного кадру — калібрувати нема на чому",
              file=sys.stderr)
        (d / "calib.json").write_text(
            json.dumps({"frames": [], "skipped": skipped}, ensure_ascii=False,
                       indent=2), encoding="utf-8")
        return 1

    build_report(frames, ths, skipped, {
        "skin_classes": a.skin_classes or "(дефолт)", "preset": a.preset,
        "face_model": a.face_model, "raw_decoder": a.raw_decoder}, d)
    return 0


def build_report(frames, ths, skipped, settings, out_dir, verbose=True):
    """Звіт зі зміряного. Окремою функцією, щоб `--from-json` міг
    перебудувати таблиці без повторного декодування сорока RAW:
    один прогін — це двадцять хвилин, а розбивку хочеться міняти.
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    # --- зведення: медіана і РОЗКИД -----------------------------------
    agg = []
    for j, t in enumerate(ths):
        per = [f["sweep"][j] for f in frames]
        skin = [r["touched_of_skin"] for r in per if r["touched_of_skin"] is not None]
        susp = [r["suspect_share"] for r in per]
        blobs = [r["blobs"] for r in per]
        agg.append({
            "threshold": t,
            "blobs_median": statistics.median(blobs),
            "blobs_min": min(blobs), "blobs_max": max(blobs),
            "touched_of_skin_median": round(statistics.median(skin), 5) if skin else None,
            "touched_of_skin_min": round(min(skin), 5) if skin else None,
            "touched_of_skin_max": round(max(skin), 5) if skin else None,
            "suspect_median": round(statistics.median(susp), 3),
            "suspect_max": round(max(susp), 3),
        })

    out = {"frames": frames, "aggregate": agg, "skipped": skipped,
           "by_face_size": None,
           "n_frames": len(frames),
           "settings": settings}
    # (записується нижче, після побудови розбивки)

    verdict = ("цього досить." if len(frames) >= 20 else
               "**ЦЬОГО МАЛО** — висновки про дефолт робити зарано.")
    L = [f"# Калібрування порога: {len(frames)} кадрів", "",
         f"класи шкіри: `{settings.get('skin_classes') or '(дефолт)'}`, "
         f"проявлення: `{settings.get('preset') or '(як є)'}`", ""]
    if skipped:
        L += [f"пропущено {len(skipped)}: "
              + ", ".join(f"{s['file']} ({s['why']})" for s in skipped), ""]
    L += ["| поріг | плям (мед.) | плям (розкид) | % шкіри (мед.) | "
          "% шкіри (розкид) | підозрілих мед. | макс. |",
          "|---|---|---|---|---|---|---|"]
    for r in agg:
        L.append(
            f"| {r['threshold']:.3f} | {r['blobs_median']:.0f} | "
            f"{r['blobs_min']}–{r['blobs_max']} | "
            f"{(r['touched_of_skin_median'] or 0):.2%} | "
            f"{(r['touched_of_skin_min'] or 0):.2%}–{(r['touched_of_skin_max'] or 0):.2%} | "
            f"{r['suspect_median']:.0%} | {r['suspect_max']:.0%} |")
    # --- розбивка за розміром обличчя ---------------------------------
    # Обличчя 200 px і 900 px у кадрі 26 Мп — це РІЗНІ задачі, а не той
    # самий кадр із розкидом. На дрібному обличчі шкіри 0.1% кадру,
    # радіус частотки впирається в підлогу 2 px, і «дефект» там менший
    # за зерно. Усереднити їх разом означає отримати число, правильне
    # для нуля кадрів.
    buckets = [("дрібне (<350 px)", lambda fw: fw is not None and fw < 350),
               ("середнє (350-700)", lambda fw: fw is not None and 350 <= fw < 700),
               ("велике (>=700)", lambda fw: fw is not None and fw >= 700)]
    seg = []
    for label, pred in buckets:
        grp = [f for f in frames if pred(f.get("face_w"))]
        if not grp:
            continue
        rows = []
        for j, t in enumerate(ths):
            per = [f["sweep"][j] for f in grp]
            skin = [r["touched_of_skin"] for r in per
                    if r["touched_of_skin"] is not None]
            rows.append({
                "threshold": t,
                "n": len(grp),
                "blobs_median": statistics.median([r["blobs"] for r in per]),
                "skin_median": round(statistics.median(skin), 5) if skin else None,
                "skin_min": round(min(skin), 5) if skin else None,
                "skin_max": round(max(skin), 5) if skin else None,
                "suspect_max": round(max(r["suspect_share"] for r in per), 3),
            })
        seg.append({"bucket": label, "n": len(grp),
                    "face_w": [min(f["face_w"] for f in grp),
                               max(f["face_w"] for f in grp)],
                    "mask_median": round(statistics.median(
                        [f["skin_frac"] for f in grp]), 5),
                    "radius_median": round(statistics.median(
                        [f["radius_px"] for f in grp]), 2),
                    "sweep": rows})

    if seg:
        L += ["", "## За розміром обличчя", "",
              "Обличчя 200 px і 900 px у кадрі 26 Мп — це різні задачі, а "
              "не той самий кадр із розкидом. Усереднювати їх разом "
              "означає отримати число, правильне для нуля кадрів.", ""]
        for b in seg:
            L += [f"### {b['bucket']} — {b['n']} кадр(ів), "
                  f"обличчя {b['face_w'][0]:.0f}-{b['face_w'][1]:.0f} px",
                  "",
                  f"маска (медіана) {b['mask_median']:.2%}, "
                  f"радіус частотки {b['radius_median']} px", "",
                  "| поріг | плям (мед.) | % шкіри (мед.) | розкид | підозрілих макс. |",
                  "|---|---|---|---|---|"]
            for r in b["sweep"]:
                L.append(f"| {r['threshold']:.3f} | {r['blobs_median']:.0f} | "
                         f"{(r['skin_median'] or 0):.2%} | "
                         f"{(r['skin_min'] or 0):.2%}–{(r['skin_max'] or 0):.2%} | "
                         f"{r['suspect_max']:.0%} |")
            L.append("")

    L += ["", "## Як це читати", "",
          "**Розкид важливіший за медіану.** Якщо на одному кадрі поріг "
          "0.010 дає 2% шкіри, а на іншому 8% — одного дефолту на всі "
          "кадри не існує, і правильна відповідь не «підняти число», а "
          "покадровий пресет (spec.md §1.2).",
          "",
          "**«Підозрілі»** — частка знахідок у класах, де дефект шкіри "
          "малоймовірний: шия, одяг, вухо, волосся, фон. Це нижня оцінка "
          "браку, і рахується вона без розмітки. Ненульова медіана "
          "означає, що набір класів обрано неправильно, а не що поріг "
          "низький — спершу знімай клас, потім чіпай поріг (§15).",
          "",
          "**Повноти тут немає.** На реальному кадрі істини немає, тож "
          "пропуски порахувати нічим. Ця таблиця відповідає на «скільки "
          "зайвого», а не на «скільки знайдено».",
          "",
          f"Кадрів: {len(frames)}. spec.md §6.2 вимагає 20-30; " + verdict]
    out["by_face_size"] = seg
    (d / "calib.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    (d / "calib.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    if verbose:
        print("\n" + "\n".join(L[4:]))
        print(f"\nзаписано в {d}/")
    return out


def _resume(a) -> int:
    """Перебудувати звіт із calib.json (або calib.partial.json)."""
    src = Path(a.from_json)
    if src.is_dir():
        for name in ('calib.json', 'calib.partial.json'):
            if (src / name).exists():
                src = src / name
                break
    data = json.loads(src.read_text(encoding='utf-8'))
    frames = data.get('frames') or []
    if not frames:
        print('у файлі немає кадрів', file=sys.stderr)
        return 1
    ths = [r['threshold'] for r in frames[0]['sweep']]
    build_report(frames, ths, data.get('skipped') or [],
                 data.get('settings') or {}, a.out)
    return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
