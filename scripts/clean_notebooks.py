"""Remove execution outputs from notebooks before publishing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"


def clean_notebook(path: Path, *, check: bool = False) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            changed = True
            if not check:
                cell["execution_count"] = None
        if cell.get("outputs"):
            changed = True
            if not check:
                cell["outputs"] = []

    if changed and not check:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean notebook execution outputs")
    parser.add_argument("--check", action="store_true", help="Fail if any notebook contains outputs or execution counts")
    args = parser.parse_args(argv)

    notebooks = sorted(NOTEBOOK_ROOT.rglob("*.ipynb"))
    if not notebooks:
        raise SystemExit(f"No notebooks found under {NOTEBOOK_ROOT}")

    dirty = [path for path in notebooks if clean_notebook(path, check=args.check)]
    for path in dirty:
        print(path.relative_to(ROOT))

    if args.check and dirty:
        raise SystemExit(f"{len(dirty)} notebook(s) contain execution outputs or counts")
    if not dirty:
        print("All notebooks are clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
