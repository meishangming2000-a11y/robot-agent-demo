"""Recompute task gates from an episode's saved measurements, without simulation."""

import argparse
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dexhand_agent.evaluate import TaskEvaluator
from dexhand_agent.tweezers_evaluate import TweezersEvaluator
from dexhand_agent.wrench_evaluate import WrenchEvaluator


def replay_episode(episode: Path) -> dict:
    """Recompute a saved episode and compare evaluator fields with exact equality.

    Episode metadata is ignored. Summaries without task_id use the legacy
    hammer evaluator; all initial measurements must come from saved metrics.
    """
    episode = Path(episode)
    stored = json.loads((episode / "summary.json").read_text(encoding="utf8"))
    task_id = stored.get("task_id", "hammer_button")
    evaluators = {
        "hammer_button": (TaskEvaluator, ("initial_height_m",)),
        "wrench_turn": (WrenchEvaluator, ("initial_height_m", "initial_nut_angle_rad")),
        "tweezers_transfer": (TweezersEvaluator,
                              ("initial_tool_height_m", "initial_object_height_m")),
    }
    if not isinstance(task_id, str) or task_id not in evaluators:
        raise ValueError(f"Unknown task_id: {task_id!r}")
    constructor, initial_metrics = evaluators[task_id]
    metrics = stored.get("metrics")
    for name in initial_metrics:
        if not isinstance(metrics, dict) or name not in metrics:
            raise ValueError(f"Missing required metric: metrics.{name}")
    evaluator = constructor(*(metrics[name] for name in initial_metrics))
    samples = episode / "samples.jsonl"
    stream = samples.open(encoding="utf8") if samples.is_file() else gzip.open(
        episode / "samples.jsonl.gz", "rt", encoding="utf8")
    with stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                evaluator.update(json.loads(line))
            except KeyError as error:
                raise ValueError(
                    f"Sample {line_number}: missing measurement {error.args[0]!r}") from error
    recomputed = evaluator.summary()
    if "task_id" in stored:
        recomputed["task_id"] = task_id
    matches = all(value == stored.get(key) for key, value in recomputed.items())
    return {"matches_saved_result": matches, **recomputed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    try:
        result = replay_episode(args.episode)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["matches_saved_result"] else 1)


if __name__ == "__main__":
    main()
