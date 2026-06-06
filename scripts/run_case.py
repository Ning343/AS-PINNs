"""Print or execute an AS-PINNs case run plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from as_pinns.execution import build_run_manifest, write_run_manifest
from as_pinns.training.runner import TrainingDependencyError, TrainingRunError, execute_training_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AS-PINNs case runner")
    parser.add_argument("case_id")
    parser.add_argument("--output-directory", default="outputs/_intermediate")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--run-name", help="Optional stable run directory name for --execute")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    if args.execute:
        try:
            summary = execute_training_run(
                args.case_id,
                output_directory=args.output_directory,
                run_name=args.run_name,
            )
        except (TrainingDependencyError, TrainingRunError) as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(summary.to_dict(), indent=2))
        return 0

    manifest = build_run_manifest(args.case_id, output_directory=args.output_directory)
    if args.write_manifest:
        path = write_run_manifest(manifest)
        print(path)
    else:
        print(json.dumps(manifest.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
