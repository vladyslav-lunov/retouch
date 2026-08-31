"""Прогнати всі набори тестів по черзі й показати підсумок.

Кожен файл лишається самостійним (його можна запустити окремо), а це —
одна команда, щоб не тримати список у голові.

    python3 tests/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    files = sorted(p for p in (ROOT / "tests").glob("test_*.py"))
    rows, failed = [], 0
    for f in files:
        t = time.time()
        r = subprocess.run([sys.executable, str(f)], capture_output=True,
                           text=True, cwd=str(ROOT))
        dt = time.time() - t
        ok = r.returncode == 0
        n = r.stdout.count("\n  OK")
        rows.append((f.stem, ok, n, dt))
        if not ok:
            failed += 1
            print(f"\n=== {f.name} ===")
            for line in r.stdout.splitlines():
                if "FAIL" in line or line.startswith("test_"):
                    print("  " + line)
            if r.stderr.strip():
                print("  " + r.stderr.strip().splitlines()[-1])

    print(f"\n{'набір':<22}{'тестів':>8}{'час':>9}")
    print("-" * 40)
    for name, ok, n, dt in rows:
        print(f"{name:<22}{n:>8}{dt:>8.1f}s   {'ок' if ok else 'ПРОВАЛ'}")
    total = sum(n for _n, _o, n, _d in rows)
    secs = sum(d for *_x, d in rows)
    print("-" * 40)
    print(f"{'разом':<22}{total:>8}{secs:>8.1f}s")
    print("усе зелене" if not failed else f"провалено наборів: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
