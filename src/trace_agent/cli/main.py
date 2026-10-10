from __future__ import annotations

import argparse
from collections.abc import Sequence

from trace_agent.identity import TRACE_FULL_NAME, TRACE_NAME

COMMANDS: tuple[tuple[str, str], ...] = (
    ("prepare-data", "Validate and materialize aligned modular inputs."),
    ("train-modules", "Train native per-module diagnosis or Cox risk models."),
    ("score-modules", "Score raw features with crossfit or frozen module models."),
    ("train-integration", "Train a task-specific risk-integration model."),
    ("calibrate", "Fit validation-only prevalent probability calibration."),
    ("train-planner", "Train the task-specific action-value model."),
    ("infer", "Run sequential TRACE assessment."),
    ("consistency", "Compute attribution and medical-consistency outputs."),
    ("calibrate-consistency-scale", "Freeze validation-only longitudinal consistency scaling."),
    ("calibrate-consistency", "Select per-endpoint consistency rules on validation."),
    ("apply-consistency", "Apply frozen consistency rules without reading labels."),
    ("evaluate", "Evaluate frozen predictions and assessment paths."),
    ("run", "Execute a configured end-to-end pipeline."),
)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="YAML pipeline configuration.")
    parser.add_argument("--profile", help="Optional local/HPC YAML profile.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the resolved operation.")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
        help="Runtime logging level.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trace",
        description=(
            f"{TRACE_NAME} ({TRACE_FULL_NAME}): adaptive multi-disease "
            "classification and incident risk prediction."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", title="commands")
    for name, help_text in COMMANDS:
        child = subparsers.add_parser(name, help=help_text, description=help_text)
        _add_common_arguments(child)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    # Command dispatch is deliberately imported lazily so `trace --help`
    # does not import optional model runtimes.
    from trace_agent.cli.runner import run_command

    return int(run_command(args))


if __name__ == "__main__":
    raise SystemExit(main())
