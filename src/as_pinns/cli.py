"""Command-line interface for lightweight AS-PINNs project checks."""

from __future__ import annotations

import argparse
import json

from .cases import get_case, list_cases
from .execution import build_run_manifest, write_run_manifest
from .reference_solutions import sample_reference
from .training_plan import build_training_plan


def _cmd_list_cases(_: argparse.Namespace) -> int:
    for case in list_cases():
        print(f"{case.case_id}\t{case.title}")
    return 0


def _cmd_case_summary(args: argparse.Namespace) -> int:
    case = get_case(args.case_id)
    payload = {
        "case_id": case.case_id,
        "title": case.title,
        "problem_type": case.problem_type,
        "domain": case.domain,
        "expected_discontinuities": case.expected_discontinuities,
        "notebook": f"notebooks/{case.primary_notebook}",
        "python_port": f"scripts/notebook_ports/{case.primary_port}",
        "outputs": case.outputs,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_run_plan(args: argparse.Namespace) -> int:
    plan = build_training_plan(args.case_id, output_directory=args.output_directory)
    print(json.dumps(plan.to_dict(), indent=2))
    if args.execute:
        raise SystemExit(
            "Full DeepXDE execution is not implemented in the lightweight CLI yet. "
            "Use scripts/notebook_ports/ for the direct notebook ports, preferably in a GPU/Linux environment."
        )
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    manifest = build_run_manifest(args.case_id, output_directory=args.output_directory)
    if args.write:
        path = write_run_manifest(manifest)
        print(path)
    else:
        print(json.dumps(manifest.to_dict(), indent=2))
    return 0


def _cmd_reference_summary(args: argparse.Namespace) -> int:
    profile = sample_reference(args.case_id, points=args.points)
    summary = {
        "case_id": args.case_id,
        "points": int(profile.x.shape[0]),
        "x_min": float(profile.x.min()),
        "x_max": float(profile.x.max()),
        "fields": {name: [float(values.min()), float(values.max())] for name, values in profile.values.items()},
    }
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AS-PINNs lightweight project utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list-cases", help="List available benchmark cases")
    list_parser.set_defaults(func=_cmd_list_cases)

    summary_parser = sub.add_parser("case-summary", help="Print one case summary")
    summary_parser.add_argument("case_id")
    summary_parser.set_defaults(func=_cmd_case_summary)

    plan_parser = sub.add_parser("run-plan", help="Print a reproducible dry-run training plan")
    plan_parser.add_argument("case_id")
    plan_parser.add_argument("--output-directory", default="outputs/_intermediate")
    plan_parser.add_argument("--execute", action="store_true", help="Reserved for full training execution")
    plan_parser.set_defaults(func=_cmd_run_plan)

    manifest_parser = sub.add_parser("run-manifest", help="Print or write a reproducible run manifest")
    manifest_parser.add_argument("case_id")
    manifest_parser.add_argument("--output-directory", default="outputs/_intermediate")
    manifest_parser.add_argument("--write", action="store_true", help="Write manifest JSON under the output directory")
    manifest_parser.set_defaults(func=_cmd_manifest)

    ref_parser = sub.add_parser("reference-summary", help="Summarize lightweight reference profiles")
    ref_parser.add_argument("case_id")
    ref_parser.add_argument("--points", type=int, default=101)
    ref_parser.set_defaults(func=_cmd_reference_summary)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
