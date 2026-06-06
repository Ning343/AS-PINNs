"""Export notebooks under notebooks/ to direct Python ports.

The exported files preserve code-cell order for traceability. They are not a
substitute for the maintained package layer under src/as_pinns/.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"
OUTPUT_ROOT = ROOT / "scripts" / "notebook_ports"


def _convert_magic(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("%run "):
        target = stripped.split(maxsplit=1)[1]
        if target.endswith(".ipynb"):
            target = target[:-6] + ".py"
        target = Path(target).name.lower()
        return [
            "import runpy\n",
            "from pathlib import Path\n",
            f"globals().update(runpy.run_path(str(Path(__file__).with_name({target!r}))))\n",
        ]
    if stripped.startswith("%"):
        return [f"# Notebook magic removed: {line}"]
    return [line]


def export_notebook(path: Path) -> Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    relative = path.relative_to(NOTEBOOK_ROOT)
    output = (OUTPUT_ROOT / relative).with_suffix(".py")
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        '"""Python port generated from the original AS-PINNs notebook.\n',
        f"\nSource notebook: notebooks/{relative.as_posix()}\n",
        "\nThis file keeps the original executable cell order for traceability. ",
        "Reusable, tested project code lives under src/as_pinns/.\n",
        '"""\n\n',
    ]

    for index, cell in enumerate(payload.get("cells", []), 1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        lines.append(f"\n# %% [cell {index}]\n")
        for line in source.splitlines(True):
            for converted in _convert_magic(line):
                if converted.endswith("\n"):
                    lines.append(converted.rstrip() + "\n")
                else:
                    lines.append(converted.rstrip())
        if not lines[-1].endswith("\n"):
            lines.append("\n")

    output.write_text("".join(lines), encoding="utf-8")
    return output


def main() -> int:
    notebooks = sorted(NOTEBOOK_ROOT.rglob("*.ipynb"))
    if not notebooks:
        raise SystemExit(f"No notebooks found under {NOTEBOOK_ROOT}")
    for notebook in notebooks:
        output = export_notebook(notebook)
        print(output.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
