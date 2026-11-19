"""Headless parameter-controller trials with measured, streamed provenance.

This is a training rollout, not the image-publishing run_episode protocol.
QualityStop enforces simulation quality limits; it is not a hardware force
guarantee. Importing this module never loads a physics engine or policy model.
"""

import gzip
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
PARAMETER_BOUNDS_M = (0.003, 0.003, 0.004, 0.004, 0.004)
QUALITY_CRITERIA = {
    "max_contact_normal_n": 25.0,
    "max_global_penetration_m": 0.002,
    "forbidden_robot_nut_contacts": 0,
    "abort_candidate_on_first_violation": True,
    "scope": "simulation quality criteria, not hardware safety guarantees",
}


class QualityStop(Exception):
    """Stop this candidate after retaining its first quality-violating tick."""


def _components():
    from .runtime import InvalidAction
    from .wrench import WrenchRuntime
    from .wrench_evaluate import WrenchEvaluator
    from .wrench_policy import ParameterWrenchReference, context_from_observation

    return (WrenchRuntime, WrenchEvaluator, ParameterWrenchReference,
            context_from_observation, InvalidAction)


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    return value


def _snapshot(value):
    return _json_value(value)


def _write_json(path, value):
    path.write_text(json.dumps(_json_value(value), ensure_ascii=False,
                               allow_nan=False, indent=2) + "\n", encoding="utf8")


def _installed_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _empty_physical_summary():
    return {"task_id": "wrench_turn", "protocol_version": 1, "success": False,
            "gates": {name: {"passed": False, "passed_at_s": None,
                             "required_continuous_s": duration, "best_continuous_s": 0.0}
                      for name, duration in (("lift", 2.0), ("turn", 0.5), ("withdraw", 1.0))},
            "violations": [],
            "metrics": {"samples": 0, "last_time_s": None, "initial_height_m": None,
                        "initial_nut_angle_rad": None, "max_lift_m": None,
                        "max_nut_turn_rad": None, "max_head_clearance_m": None,
                        "max_contact_force_n": None, "max_penetration_m": None,
                        "first_robot_nut_contact_s": None}}


def _score_number(value):
    if value is None:
        return 0.0
    number = float(value)
    return number if math.isfinite(number) else 0.0


def score_rollout(summary):
    """The frozen wrench-learning-protocol formula, separate from eligibility.

    Empty extrema contribute zero only here; raw summaries retain their nulls.
    Training selection must separately exclude optimization-ineligible trials.
    """
    progress = sum(max(0.0, min(1.0, _score_number(gate["best_continuous_s"])
                              / gate["required_continuous_s"]))
                   for gate in summary["gates"].values())
    turn = max(0.0, min(1.0, _score_number(summary["metrics"]["max_nut_turn_rad"])
                        / (math.pi / 4)))
    quality = summary["quality"]
    return (1000.0 * bool(summary["qualified_success"]) + 100.0 * progress + 10.0 * turn
            - 0.2 * _score_number(quality["max_contact_force_n"])
            - 500.0 * _score_number(quality["max_penetration_m"])
            - 1000.0 * bool(quality["breached"]))


def _require_finite(value):
    if isinstance(value, dict):
        for item in value.values():
            _require_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_finite(item)
    elif hasattr(value, "tolist"):
        _require_finite(value.tolist())
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite physical measurement")


def run_rollout(seed, parameters, output, max_decisions=32):
    """Run one fresh headless attempt and return its measured summary.

    Physical ``success`` remains exactly the independent evaluator's result.
    ``qualified_success`` additionally requires quality eligibility, an accepted
    finish action, no rejected controls and no runtime error. Expected quality aborts have error=None
    and termination='quality_stop'; unexpected ordinary exceptions are recorded.
    KeyboardInterrupt and other BaseExceptions propagate after cleanup.
    """
    try:
        if isinstance(parameters, (str, bytes, dict)):
            raise ValueError("parameters must be a numeric vector")
        parameters = [float(value) for value in parameters]
    except (TypeError, ValueError) as exc:
        raise ValueError("parameters must contain five finite bounded meters") from exc
    if len(parameters) != 5 or any(not math.isfinite(value) or abs(value) > bound
                                   for value, bound in zip(parameters, PARAMETER_BOUNDS_M)):
        raise ValueError("parameters must contain five finite bounded meters")
    if isinstance(max_decisions, bool) or not isinstance(max_decisions, int) or max_decisions < 1:
        raise ValueError("max_decisions must be a positive integer")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime, evaluator = None, None
    history = []
    error, abort_reason = None, None
    termination, finished, rejected_actions = "interrupted", False, 0
    quality = {"samples": 0, "max_contact_force_n": None, "max_penetration_m": None,
               "robot_nut_contact": False, "breached": False, "first_breach_time_s": None,
               "measurements_valid": True, "eligible": False}
    metadata = {"runner": "wrench_headless_parameter_rollout_v1", "task_id": "wrench_turn",
                "seed": seed, "parameters_m": parameters, "max_decisions": max_decisions,
                "mujoco": _installed_version("mujoco"), "numpy": _installed_version("numpy"),
                "observation_mode": "simulator_state", "initial_observation": None,
                "context": None, "quality_criteria": QUALITY_CRITERIA,
                "optimization_criteria": "measured quality eligible, accepted finish, zero rejected controls, no error",
                "nonfinite_json_encoding": {"nonfinite_float": "Python float repr"}}
    started = time.monotonic()

    try:
        with (output / "samples.jsonl.gz").open("wb") as raw_stream:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as stream:
                source_names = ("wrench.py", "runtime.py", "scene.py", "wrench_evaluate.py",
                                "wrench_rollout.py", "wrench_policy.py")
                metadata["source_sha256"] = {
                    f"dexhand_agent/{name}": hashlib.sha256((ROOT / "dexhand_agent" / name)
                                                           .read_bytes()).hexdigest()
                    for name in source_names if (ROOT / "dexhand_agent" / name).is_file()}
                metadata["asset_manifest_sha256"] = hashlib.sha256(
                    (ROOT / "assets/manifest.json").read_bytes()).hexdigest()
                protocol = ROOT / "experiments/wrench-learning-protocol.json"
                if protocol.is_file():
                    metadata["learning_protocol_sha256"] = hashlib.sha256(protocol.read_bytes()).hexdigest()
                runtime_type, evaluator_type, policy_type, make_context, invalid_action = _components()
                runtime = runtime_type(seed)
                evaluator = evaluator_type(runtime.initial_height_m, runtime.initial_nut_angle_rad)
                initial_observation = runtime.observe()
                metadata["initial_observation"] = _snapshot(initial_observation)
                metadata["context"] = _snapshot(make_context(initial_observation))
                metadata["physics_timestep_s"] = float(runtime.model.opt.timestep)
                metadata["sample_rate_hz"] = 1.0 / metadata["physics_timestep_s"]
                policy = policy_type(parameters)
                _write_json(output / "metadata.json", metadata)

                def retain(sample):
                    stream.write((json.dumps(_json_value(sample), ensure_ascii=False,
                                             allow_nan=False, separators=(",", ":")) + "\n").encode("utf8"))
                    quality["samples"] += 1
                    try:
                        _require_finite(sample)
                    except ValueError:
                        quality["measurements_valid"] = False
                        raise
                    for key in ("max_contact_force_n", "max_penetration_m"):
                        previous = quality[key]
                        quality[key] = sample[key] if previous is None else max(previous, sample[key])
                    quality["robot_nut_contact"] |= bool(sample["robot_nut_contact"])
                    reasons = []
                    if sample["max_contact_force_n"] > QUALITY_CRITERIA["max_contact_normal_n"]:
                        reasons.append("max_contact_force_n exceeds 25 N")
                    if sample["max_penetration_m"] > QUALITY_CRITERIA["max_global_penetration_m"]:
                        reasons.append("max_penetration_m exceeds 0.002 m")
                    if sample["robot_nut_contact"]:
                        reasons.append("robot_nut_contact")
                    if reasons:
                        quality["breached"] = True
                        quality["first_breach_time_s"] = sample["time_s"]
                    evaluator.update(sample)
                    if reasons:
                        raise QualityStop("; ".join(reasons))

                runtime.on_sample = retain
                for sequence in range(max_decisions):
                    observation = runtime.observe()
                    entry = {"sequence": sequence, "observation": _snapshot(observation),
                             "action": None, "result": {"status": "decision_interrupted"}}
                    history.append(entry)
                    action = policy.decide(observation)
                    entry["action"] = _snapshot(action)
                    entry["samples_before"] = quality["samples"]
                    entry["result"] = {"status": "interrupted", "reason": "execution did not complete"}
                    try:
                        result = runtime.execute(action)
                    except invalid_action as exc:
                        rejected_actions += 1
                        result = {"status": "rejected", "reason": str(exc), "time_s": runtime.data.time}
                        runtime.last_result = result
                    except Exception as exc:
                        entry["result"] = {"status": "interrupted", "reason": f"{type(exc).__name__}: {exc}",
                                           "time_s": runtime.data.time}
                        raise
                    finally:
                        entry["samples_after"] = quality["samples"]
                    entry["result"] = _snapshot(result)
                    if result["status"] == "finished":
                        finished = True
                        termination = "finished"
                        break
                else:
                    termination = "decision_limit"
    except QualityStop as exc:
        abort_reason = str(exc)
        termination = "quality_stop"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        termination = "error"
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception as exc:
                close_error = f"close {type(exc).__name__}: {exc}"
                error = f"{error}; {close_error}" if error else close_error
                termination = "error"
        quality["eligible"] = (quality["samples"] > 0 and quality["measurements_valid"]
                               and not quality["breached"])
        physical = evaluator.summary() if evaluator is not None else _empty_physical_summary()
        optimization_eligible = quality["eligible"] and error is None and finished and rejected_actions == 0
        summary = {**physical, "seed": seed, "parameters_m": parameters,
                   "physical_success": physical["success"], "quality": quality,
                   "optimization_eligible": optimization_eligible,
                   "qualified_success": bool(physical["success"] and optimization_eligible),
                   "finished": finished, "decision_budget_exhausted": termination == "decision_limit",
                   "decisions": len(history), "rejected_actions": rejected_actions, "termination": termination,
                   "abort_reason": abort_reason, "error": error,
                   "elapsed_wall_s": time.monotonic() - started}
        summary["score"] = score_rollout(summary)
        _write_json(output / "metadata.json", metadata)
        _write_json(output / "actions.json", history)
        _write_json(output / "summary.json", summary)
    return summary
