"""Pick the Track A variant with the lowest scale-aligned AbsRel among those with cached scene predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    pred_root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/pred")
    best, best_val = None, float("inf")
    for p in sorted(Path("results").glob("mono_depth_*.json")):
        r = json.loads(p.read_text())
        var = r["variant"]
        if var.endswith("_gtfocal") or not (pred_root / var).exists():
            continue
        v = r["scaled"]["abs_rel"]
        if v < best_val:
            best, best_val = var, v
    if best is None:
        raise SystemExit("no cached predictions found")
    print(best)


if __name__ == "__main__":
    main()
