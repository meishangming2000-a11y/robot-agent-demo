"""Portable episode records; a demonstrated episode is not a learned skill."""

import hashlib
import json
from pathlib import Path


_FILES = ("metadata.json", "summary.json", "actions.json", "steps/000/request.json")
_LEGACY_TASK = "held_hammer_press_spring_button"
_TASKS = {
    _LEGACY_TASK: (("lift", "press", "withdraw"), ("button",)),
    "hammer_button": (("lift", "press", "withdraw"), ("button",)),
    "wrench_turn": (("lift", "turn", "withdraw"), ("nut",)),
    "tweezers_transfer": (("pickup", "pinch_lift", "transport", "release"), ("object", "goal")),
}


def _read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    json.dumps(value, allow_nan=False)  # Also reject numeric overflow such as 1e999.
    return value


def _validate(record):
    try:
        if not isinstance(record, dict) or type(record.get("schema_version")) is not int or record["schema_version"] != 1:
            raise ValueError("Unsupported experience schema_version")
        if not isinstance(record["id"], str) or not record["id"] or not isinstance(record["origin_backend"], str):
            raise ValueError("Experience needs an id and origin_backend")
        if record["source"]["episode_id"] != record["id"]:
            raise ValueError("Experience source episode_id does not match id")
        hashes = record["source"]["files_sha256"]
        if set(hashes) != set(_FILES) or any(
            not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
            for h in hashes.values()
        ):
            raise ValueError("Experience source needs relative file names and SHA256 hashes")
        applicability = record["applicability"]
        if not all(name in applicability for name in ("robot", "tool", "task", "observation_mode", "source_seed")):
            raise ValueError("Missing experience applicability")
        task = applicability["task"]
        if not isinstance(task, str) or task not in _TASKS:
            raise ValueError("Unsupported experience task")
        gate_names, fixtures = _TASKS[task]
        if applicability["new_tools_verified"] is not False or applicability["other_layouts_verified"] is not False:
            raise ValueError("An episode record cannot establish transfer")
        initial = record["initial_observation"]
        if not all(isinstance(initial.get(name), dict) for name in ("tool", *fixtures, "palm")):
            raise ValueError(f"Missing initial tool/{'/'.join(fixtures)}/palm reference")
        trajectory = record["trajectory"]
        if not isinstance(trajectory, list) or any(
            not isinstance(step, dict) or step.get("sequence") != i
            or not isinstance(step.get("action"), dict) or not isinstance(step.get("result"), dict)
            for i, step in enumerate(trajectory)
        ):
            raise ValueError("Experience trajectory must preserve ordered action/result entries")
        outcome = record["outcome"]
        if type(outcome["success"]) is not bool or not isinstance(outcome["violations"], list):
            raise ValueError("Malformed experience outcome")
        if not isinstance(outcome["gates"], dict) or set(outcome["gates"]) != set(gate_names):
            raise ValueError("Experience outcome gates do not match task")
        gates_passed = all(outcome["gates"][name]["passed"] is True for name in gate_names)
        if outcome["success"] and (not gates_passed or outcome["violations"]):
            raise ValueError("Experience outcome success contradicts physical gates or violations")
        demonstrated = outcome["success"] and outcome["error"] is None
        if record["status"] != ("demonstrated_episode" if demonstrated else "failed_episode"):
            raise ValueError("Experience status contradicts outcome")
        if outcome["decisions"] != len(trajectory) or not isinstance(outcome["metrics"], dict):
            raise ValueError("Experience outcome and trajectory disagree")
        if not isinstance(record["notes"], str):
            raise ValueError("Experience notes must be text")
        json.dumps(record, allow_nan=False)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed experience record") from exc
    return record


def export_experience(episode_dir):
    """Summarize saved public inputs and outcomes, retaining their file hashes."""
    directory = Path(episode_dir)
    metadata, summary, actions, request = [_read_json(directory / name) for name in _FILES]
    if not isinstance(metadata, dict) or not isinstance(summary, dict) or not isinstance(request, dict):
        raise ValueError("Episode inputs must be JSON objects")
    episode_id = metadata.get("episode_id")
    if not episode_id or summary.get("episode_id") != episode_id or request.get("episode_id") != episode_id:
        raise ValueError("Episode inputs have mismatched episode_id values")
    if not isinstance(actions, list):
        raise ValueError("Episode actions must be a list")
    task = metadata.get("task_id", _LEGACY_TASK)
    if not isinstance(task, str) or task not in _TASKS:
        raise ValueError("Unsupported experience task")
    _, fixtures = _TASKS[task]
    observation = request["observation"]
    robot = observation["robot"]
    outcome = {name: summary[name] for name in ("success", "gates", "violations", "error", "metrics", "decisions")}
    notes = ("Reference trajectory calibrated by the developer. " if metadata["backend"] == "reference" else
             "Online actor episode recorded under its declared inputs. ")
    record = {
        "schema_version": 1,
        "id": episode_id,
        "status": "demonstrated_episode" if summary["success"] and summary["error"] is None else "failed_episode",
        "origin_backend": metadata["backend"],
        "applicability": {"robot": "Panda + adapted Shadow right wrist/hand",
                          "tool": observation["tool"]["name"],
                          "task": task,
                          "goal": request["goal"], "protocol_version": summary.get("protocol_version", 1),
                          "observation_mode": metadata["observation_mode"], "source_seed": metadata["seed"],
                          "other_layouts_verified": False, "new_tools_verified": False},
        "source": {"episode_id": episode_id, "files_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in _FILES}},
        "initial_observation": {"tool": observation["tool"],
                                **{name: observation.get(name) for name in fixtures},
                                "palm": {"position_m": robot["palm_position_m"],
                                         "quat_wxyz": robot["palm_quat_wxyz"]}},
        "outcome": outcome,
        "trajectory": actions,
        "notes": notes + "One episode only; this export does not train model weights or establish transfer. "
                 "Use the initial poses as coordinate references, check fresh observations, and do not blindly replay targets.",
    }
    return _validate(record)


def load_experience(path):
    """Load a declared episode record; hashes identify sources, not authenticity."""
    return _validate(_read_json(path))
