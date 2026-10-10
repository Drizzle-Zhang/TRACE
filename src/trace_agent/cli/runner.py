from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trace_agent.cli.operations import (
    evaluate,
    prepare_data,
    run_consistency,
    run_inference,
)
from trace_agent.runtime.config import load_config
from trace_agent.runtime.logging import configure_logging
from trace_agent.training.service import calibrate, train_integration, train_planner


def _redact(value: Any, *, key: str = "") -> Any:
    normalized = key.lower()
    if any(token in normalized for token in ("password", "secret", "token", "api_key")):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(child_key): _redact(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, key=key) for item in value]
    return value


def _execute(command: str, config: Mapping[str, Any]) -> str | None:
    if command == "prepare-inputs":
        from trace_agent.data.input_preparation import prepare_inputs

        return json.dumps(prepare_inputs(config), indent=2, ensure_ascii=False)
    if command == "prepare-data":
        return str(prepare_data(config))
    if command == "train-modules":
        from trace_agent.training.modules.service import train_modules

        return str(train_modules(config))
    if command == "score-modules":
        from trace_agent.training.modules.service import score_modules

        return str(score_modules(config))
    if command == "infer":
        return str(run_inference(config).manifest_path)
    if command == "consistency":
        return str(run_consistency(config))
    if command == "calibrate-consistency-scale":
        from trace_agent.cli.consistency_scale import calibrate_consistency_scale

        return str(calibrate_consistency_scale(config))
    if command in {"calibrate-consistency", "apply-consistency"}:
        from trace_agent.cli.consistency_workflow import apply_consistency, calibrate_consistency

        return str(
            calibrate_consistency(config) if command == "calibrate-consistency" else apply_consistency(config)
        )
    if command == "evaluate":
        return str(evaluate(config))
    if command == "train-integration":
        return str(train_integration(config).manifest_path)
    if command == "calibrate":
        return str(calibrate(config).manifest_path)
    if command == "train-planner":
        return str(train_planner(config).manifest_path)
    raise ValueError(f"unsupported pipeline command: {command}")


def _run_pipeline(config: Mapping[str, Any]) -> list[str]:
    pipeline = config.get("pipeline", {})
    if not isinstance(pipeline, Mapping):
        raise TypeError("config.pipeline must be a mapping")
    steps = pipeline.get("steps", ["prepare-data", "infer"])
    if not isinstance(steps, list) or not steps or not all(isinstance(step, str) for step in steps):
        raise ValueError("pipeline.steps must be a non-empty string list")
    if "run" in steps:
        raise ValueError("pipeline.steps cannot recursively include 'run'")
    outputs: list[str] = []
    for step in steps:
        output = _execute(step, config)
        if output is not None:
            outputs.append(output)
    return outputs


def run_command(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    if not args.config:
        if args.dry_run:
            print(f"command={args.command}")
            print("config=")
            print(f"profile={args.profile or ''}")
            return 0
        raise SystemExit(f"{args.command} requires --config. " "Use --dry-run to inspect command resolution.")
    config = load_config(args.config, profile_path=args.profile)
    if args.command == "prepare-inputs":
        settings = config.setdefault("module_data", {})
        if not isinstance(settings, dict):
            raise TypeError("config.module_data must be a mapping")
        if args.input_table:
            settings["input_table_path"] = str(Path(args.input_table).expanduser().resolve())
        if args.id_column is not None:
            settings["input_id_column"] = args.id_column
            settings.pop("input_eid_column", None)
    if args.dry_run:
        print(f"command={args.command}")
        print(f"config={args.config}")
        print(f"profile={args.profile or ''}")
        print(
            json.dumps(
                _redact(config),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run":
        outputs = _run_pipeline(config)
        for output in outputs:
            print(output)
    else:
        result_path = _execute(args.command, config)
        if result_path is not None:
            print(result_path)
    return 0
