"""Collect synchronized public-task examples without changing their controllers.

These are engineering-reference episodes, not online agent demonstrations.
The default seed is a known development condition, not a held-out model test.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dexhand_agent.learning_data import SynchronizedRecorder
from dexhand_agent.run import _task_components


TASKS = ("hammer_button", "wrench_turn", "tweezers_transfer")


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf8")


def collect_task(task, seed, output, *, sample_period_s=.1, width=96, height=96,
                 partition="development", study_id=None):
    """Run the existing reference; retain incomplete/error outcomes as such."""
    output = Path(output)
    if partition not in ("development", "train", "validation", "final"):
        raise ValueError("Unknown data partition")
    if partition != "development" and not study_id:
        raise ValueError("A study_id is required for a declared learning partition")
    if output.exists():
        raise FileExistsError(output)
    runtime_type, make_evaluator, reference_type, goal = _task_components(task)
    source_paths = [Path(__file__).resolve(), ROOT / "dexhand_agent/learning_data.py",
                    ROOT / "assets/manifest.json", *sorted((ROOT / "dexhand_agent").glob("*.py"))]
    sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in source_paths}
    runtime = runtime_type(seed)
    evaluator = make_evaluator(runtime)
    runtime.on_sample = evaluator.update
    recorder = None
    error, finished = None, False
    decisions = 0
    attempted_actions = 0
    started = time.monotonic()
    try:
        model_bytes = np.empty(mujoco.mj_sizeModel(runtime.model), dtype=np.uint8)
        mujoco.mj_saveModel(runtime.model, None, model_bytes)
        recorder = SynchronizedRecorder(
            runtime, output, sample_period_s=sample_period_s, width=width, height=height,
            provenance={"task_id": task, "goal": goal, "seed": seed,
                        "actor_kind": "existing_engineering_reference",
                        "native_agent": False, "held_out": partition == "final",
                        "partition": partition, "study_id": study_id,
                        "purpose": ("synchronized-data development and compatibility check"
                                    if partition == "development" else
                                    "declared data-curation study; engineering-reference examples"),
                        "model_binary": {"file": "model.mjb", "bytes": int(model_bytes.size),
                                         "sha256": hashlib.sha256(model_bytes).hexdigest()},
                        "source_sha256": sources})
        with (output / "model.mjb").open("xb") as stream:
            stream.write(model_bytes.tobytes())
        reference = reference_type()
        for _ in range(48):
            action = reference.decide(runtime.observe())
            recorder.begin_action(action)
            attempted_actions += 1
            result = runtime.execute(action)
            recorder.end_action(result)
            decisions += 1
            if result["status"] == "finished":
                finished = True
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if recorder is None:
            raise
    finally:
        try:
            if recorder is not None:
                outcome = {**evaluator.summary(), "finished": finished, "error": error,
                           "decisions": decisions, "attempted_actions": attempted_actions,
                           "elapsed_wall_s": time.monotonic() - started}
                outcome["completed_task_success"] = bool(outcome["success"] and finished and error is None)
                outcome["acceptance_scope"] = (
                    "Original task evaluator plus accepted finish and no error; "
                    "force and penetration remain diagnostics, not additional quality gates.")
                recorder.finalize(status="completed" if finished and error is None else "failed",
                                  outcome=outcome)
        finally:
            if recorder is not None:
                recorder.close()
            runtime.close()
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-period", type=float, default=.1)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--partition", choices=("development", "train", "validation", "final"),
                        default="development")
    parser.add_argument("--study-id")
    args = parser.parse_args(argv)
    if args.partition != "development" and not args.study_id:
        parser.error("--study-id is required outside the development partition")
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = TASKS if args.task == "all" else (args.task,)
    outcomes = {}
    for task in tasks:
        outcomes[task] = collect_task(task, args.seed, args.output / task,
                                      sample_period_s=args.sample_period,
                                      width=args.width, height=args.height,
                                      partition=args.partition, study_id=args.study_id)
        _write(args.output / "summary.json", {
            "purpose": "development compatibility" if args.partition == "development" else "data-curation study",
            "partition": args.partition, "study_id": args.study_id,
            "native_agent": False, "held_out": args.partition == "final", "outcomes": outcomes})
        print(json.dumps({"task": task, "completed_task_success": outcomes[task]["completed_task_success"],
                          "error": outcomes[task]["error"]}), flush=True)
    return 0 if all(row["completed_task_success"] for row in outcomes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
