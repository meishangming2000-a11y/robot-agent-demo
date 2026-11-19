import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dexhand_agent import wrench_rollout as rollout
from dexhand_agent.wrench_evaluate import WrenchEvaluator


def measurement(time_s, **changes):
    sample = {
        "time_s": time_s,
        "wrench_pos_m": [0.53, -0.13, 0.27],
        "wrench_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "hand_tool_contacts": ["thumb", "index"],
        "tool_support_contacts": [],
        "head_nut_contact": False,
        "tool_nut_contact": False,
        "robot_nut_contact": False,
        "nut_angle_rad": 0.0,
        "head_clearance_m": 0.04,
        "raw_mj_head_distance_m": 0.04,
        "max_contact_force_n": 1.0,
        "max_penetration_m": 0.0001,
    }
    sample.update(changes)
    return sample


def install_fakes(monkeypatch, samples=(), actions=None, failure=None, evaluator=WrenchEvaluator):
    class InvalidAction(ValueError):
        pass

    class Runtime:
        initial_height_m = 0.24
        initial_nut_angle_rad = 0.0
        instances = []

        def __init__(self, seed):
            self.seed = seed
            self.data = SimpleNamespace(time=1.0)
            self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
            self.last_result = {"status": "ready"}
            self.samples = []
            self.on_sample = None
            self.closed = False
            self.instances.append(self)

        def observe(self):
            return {"mode": "simulator_state", "time_s": self.data.time,
                    "physics_step": round(self.data.time / 0.002),
                    "tool": {"name": "wrench", "position_m": [0.531, -0.128, 0.24],
                             "initial_height_m": 0.24},
                    "nut": {"angle_rad": 0.0, "initial_angle_rad": 0.0},
                    "robot": {"palm_position_m": [0.5, -0.1, 0.4]},
                    "contacts": measurement(self.data.time), "last_action": self.last_result}

        def execute(self, action):
            if action["operation"] == "invalid":
                raise InvalidAction("bad action")
            if action["operation"] == "finish":
                self.last_result = {"status": "finished", "time_s": self.data.time}
                return self.last_result
            for sample in samples:
                self.data.time = sample["time_s"]
                self.samples.append(sample)
                self.on_sample(sample)
            if failure is not None:
                raise failure
            self.last_result = {"status": "completed", "time_s": self.data.time}
            return self.last_result

        def close(self):
            self.closed = True

    class Policy:
        def __init__(self, parameters):
            self.actions = iter(actions or [{"operation": "hold"}, {"operation": "finish"}])

        def decide(self, observation):
            return next(self.actions)

    monkeypatch.setattr(rollout, "_components", lambda: (
        Runtime, evaluator, Policy, lambda observation: [0.1, 0.2], InvalidAction))
    return Runtime


def read_samples(output):
    with gzip.open(output / "samples.jsonl.gz", "rt", encoding="utf8") as stream:
        return [json.loads(line) for line in stream]


def test_import_does_not_import_physics_engine_or_policy():
    module = Path(rollout.__file__)
    program = (
        "import importlib.util,sys; "
        f"spec=importlib.util.spec_from_file_location('isolated_rollout', {str(module)!r}); "
        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module); "
        "assert 'mujoco' not in sys.modules; "
        "assert 'dexhand_agent.wrench' not in sys.modules; "
        "assert 'dexhand_agent.wrench_policy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", program], check=True)


def test_rollout_retains_every_sample_and_matches_independent_replay(monkeypatch, tmp_path):
    samples = [measurement(1 + index * 0.002) for index in range(1002)]
    samples += [measurement(3.004 + index * 0.002, nut_angle_rad=math.pi / 4,
                           head_nut_contact=True, tool_nut_contact=True) for index in range(252)]
    samples += [measurement(3.508 + index * 0.002, nut_angle_rad=math.pi / 4)
                for index in range(502)]
    runtime = install_fakes(monkeypatch, samples)
    output = tmp_path / "attempt"
    summary = rollout.run_rollout(7, [0] * 5, output)
    retained = read_samples(output)
    replay = WrenchEvaluator(0.24, 0.0)
    for sample in retained:
        replay.update(sample)
    assert retained == samples
    for key, value in replay.summary().items():
        assert summary[key] == value
    assert summary["success"] is True
    assert summary["qualified_success"] is True
    assert summary["quality"]["eligible"] is True
    assert summary["error"] is None
    assert runtime.instances[0].closed
    assert json.loads((output / "summary.json").read_text(encoding="utf8")) == summary
    assert set(path.name for path in output.iterdir()) == {
        "metadata.json", "summary.json", "actions.json", "samples.jsonl.gz"}
    assert (output / "samples.jsonl.gz").read_bytes()[4:8] == b"\x00" * 4
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf8"))
    assert metadata["seed"] == 7
    assert metadata["parameters_m"] == [0] * 5
    assert metadata["context"] == [0.1, 0.2]
    assert metadata["physics_timestep_s"] == 0.002
    for source in ("wrench.py", "runtime.py", "scene.py", "wrench_evaluate.py", "wrench_rollout.py"):
        assert len(metadata["source_sha256"][f"dexhand_agent/{source}"]) == 64
    assert len(metadata["asset_manifest_sha256"]) == 64
    actions = json.loads((output / "actions.json").read_text(encoding="utf8"))
    assert actions[0]["observation"] == metadata["initial_observation"]
    assert actions[1]["observation"]["last_action"]["status"] == "completed"
    assert actions[-1]["result"]["status"] == "finished"


@pytest.mark.parametrize("change,reason", [
    ({"max_contact_force_n": 25.00001}, "max_contact_force_n"),
    ({"max_penetration_m": 0.002001}, "max_penetration_m"),
    ({"robot_nut_contact": True}, "robot_nut_contact"),
])
def test_quality_guard_retains_trigger_and_interrupted_action(monkeypatch, tmp_path, change, reason):
    samples = [measurement(1.002), measurement(1.004, **change), measurement(1.006)]
    runtime = install_fakes(monkeypatch, samples)
    output = tmp_path / reason
    summary = rollout.run_rollout(0, [0] * 5, output)
    assert read_samples(output) == samples[:2]
    assert summary["metrics"]["samples"] == 2
    assert summary["qualified_success"] is False
    assert summary["quality"]["eligible"] is False
    assert reason in summary["abort_reason"]
    assert summary["termination"] == "quality_stop"
    assert summary["error"] is None
    assert summary["score"] < -900
    actions = json.loads((output / "actions.json").read_text(encoding="utf8"))
    assert len(actions) == 1
    assert actions[0]["action"]["operation"] == "hold"
    assert actions[0]["result"]["status"] == "interrupted"
    assert runtime.instances[0].closed


def test_exact_quality_boundaries_are_inclusive(monkeypatch, tmp_path):
    install_fakes(monkeypatch, [measurement(1.002, max_contact_force_n=25,
                                          max_penetration_m=0.002)])
    summary = rollout.run_rollout(0, [0] * 5, tmp_path / "boundary")
    assert summary["quality"]["eligible"] is True
    assert summary["abort_reason"] is None


def test_invalid_action_is_retained_and_does_not_end_attempt(monkeypatch, tmp_path):
    install_fakes(monkeypatch, actions=[{"operation": "invalid"}, {"operation": "finish"}])
    output = tmp_path / "reject"
    summary = rollout.run_rollout(0, [0] * 5, output)
    actions = json.loads((output / "actions.json").read_text(encoding="utf8"))
    assert [entry["result"]["status"] for entry in actions] == ["rejected", "finished"]
    assert actions[1]["observation"]["last_action"]["status"] == "rejected"
    assert summary["decisions"] == 2
    assert summary["rejected_actions"] == 1
    assert summary["error"] is None


def test_rejected_control_excludes_an_otherwise_successful_teacher(monkeypatch, tmp_path):
    class SuccessfulEvaluator(WrenchEvaluator):
        def summary(self):
            return {**super().summary(), "success": True}

    install_fakes(monkeypatch, [measurement(1.002)],
                  actions=[{"operation": "invalid"}, {"operation": "hold"}, {"operation": "finish"}],
                  evaluator=SuccessfulEvaluator)
    summary = rollout.run_rollout(0, [0] * 5, tmp_path / "rejected_teacher")
    assert summary["success"] is True
    assert summary["finished"] is True
    assert summary["quality"]["eligible"] is True
    assert summary["rejected_actions"] == 1
    assert summary["optimization_eligible"] is False
    assert summary["qualified_success"] is False


def test_quality_extrema_are_retained_even_when_evaluator_errors(monkeypatch, tmp_path):
    class FailingEvaluator(WrenchEvaluator):
        def update(self, sample):
            raise ValueError("evaluator failed")

    install_fakes(monkeypatch, [measurement(1.002, max_contact_force_n=26)],
                  evaluator=FailingEvaluator)
    summary = rollout.run_rollout(0, [0] * 5, tmp_path / "evaluator_error")
    assert summary["error"] == "ValueError: evaluator failed"
    assert summary["quality"]["max_contact_force_n"] == 26
    assert summary["quality"]["breached"] is True
    assert summary["optimization_eligible"] is False


def test_runtime_error_retains_partial_samples_and_closes(monkeypatch, tmp_path):
    runtime = install_fakes(monkeypatch, [measurement(1.002)], failure=LookupError("engine failed"))
    output = tmp_path / "error"
    summary = rollout.run_rollout(0, [0] * 5, output)
    assert summary["error"] == "LookupError: engine failed"
    assert summary["termination"] == "error"
    assert summary["qualified_success"] is False
    assert summary["quality"]["eligible"] is True
    assert summary["optimization_eligible"] is False
    assert read_samples(output) == [measurement(1.002)]
    assert runtime.instances[0].closed
    actions = json.loads((output / "actions.json").read_text(encoding="utf8"))
    assert actions[0]["result"]["status"] == "interrupted"


def test_physical_success_stays_separate_from_completion_and_quality(monkeypatch, tmp_path):
    class SuccessfulEvaluator(WrenchEvaluator):
        def summary(self):
            return {**super().summary(), "success": True}

    install_fakes(monkeypatch, [measurement(1.002)], evaluator=SuccessfulEvaluator)
    summary = rollout.run_rollout(0, [0] * 5, tmp_path / "budget", max_decisions=1)
    assert summary["success"] is True
    assert summary["physical_success"] is True
    assert summary["quality"]["eligible"] is True
    assert summary["finished"] is False
    assert summary["decision_budget_exhausted"] is True
    assert summary["optimization_eligible"] is False
    assert summary["qualified_success"] is False

    install_fakes(monkeypatch, [measurement(1.002, max_contact_force_n=26)],
                  evaluator=SuccessfulEvaluator)
    summary = rollout.run_rollout(0, [0] * 5, tmp_path / "quality")
    assert summary["success"] is True
    assert summary["physical_success"] is True
    assert summary["finished"] is False
    assert summary["decision_budget_exhausted"] is False
    assert summary["qualified_success"] is False


def test_nonfinite_sample_is_retained_as_tagged_value_and_errors(monkeypatch, tmp_path):
    install_fakes(monkeypatch, [measurement(1.002, max_contact_force_n=math.nan)])
    output = tmp_path / "nonfinite"
    summary = rollout.run_rollout(0, [0] * 5, output)
    assert summary["error"] == "ValueError: Non-finite physical measurement"
    assert summary["quality"]["measurements_valid"] is False
    assert summary["quality"]["eligible"] is False
    assert summary["qualified_success"] is False
    assert summary["metrics"]["max_contact_force_n"] is None
    assert read_samples(output)[0]["max_contact_force_n"] == {"nonfinite_float": "nan"}
    assert math.isfinite(summary["score"])


def test_constructor_error_still_retains_empty_attempt(monkeypatch, tmp_path):
    def failing_runtime(seed):
        raise RuntimeError("initialization failed")

    install_fakes(monkeypatch)
    _, evaluator, policy, context, invalid_action = rollout._components()
    monkeypatch.setattr(rollout, "_components", lambda: (
        failing_runtime, evaluator, policy, context, invalid_action))
    output = tmp_path / "constructor"
    summary = rollout.run_rollout(0, [0] * 5, output)
    assert summary["error"] == "RuntimeError: initialization failed"
    assert summary["metrics"]["initial_height_m"] is None
    assert summary["metrics"]["samples"] == 0
    assert read_samples(output) == []
    assert json.loads((output / "actions.json").read_text(encoding="utf8")) == []


def test_stream_compression_is_reproducible_across_output_names(monkeypatch, tmp_path):
    install_fakes(monkeypatch, [measurement(1.002)])
    first, second = tmp_path / "one", tmp_path / "two"
    rollout.run_rollout(0, [0] * 5, first)
    rollout.run_rollout(0, [0] * 5, second)
    assert (first / "samples.jsonl.gz").read_bytes() == (second / "samples.jsonl.gz").read_bytes()


def test_empty_attempt_retains_null_extrema_and_is_not_quality_eligible(monkeypatch, tmp_path):
    install_fakes(monkeypatch, actions=[{"operation": "finish"}])
    output = tmp_path / "empty"
    summary = rollout.run_rollout(0, [0] * 5, output)
    assert summary["metrics"]["samples"] == 0
    assert summary["metrics"]["max_contact_force_n"] is None
    assert summary["quality"]["max_contact_force_n"] is None
    assert summary["quality"]["eligible"] is False
    assert summary["score"] == 0.0
    assert read_samples(output) == []


def test_existing_output_and_invalid_parameters_are_rejected_before_runtime(monkeypatch, tmp_path):
    runtime = install_fakes(monkeypatch)
    with pytest.raises(FileExistsError):
        rollout.run_rollout(0, [0] * 5, tmp_path)
    for parameters in ([0] * 4, [math.nan] * 5, [0.004] * 5):
        output = tmp_path / "invalid"
        with pytest.raises(ValueError):
            rollout.run_rollout(0, parameters, output)
        assert not output.exists()
    assert runtime.instances == []


def test_keyboard_interrupt_propagates_after_resources_close(monkeypatch, tmp_path):
    runtime = install_fakes(monkeypatch, [measurement(1.002)], failure=KeyboardInterrupt())
    output = tmp_path / "interrupt"
    with pytest.raises(KeyboardInterrupt):
        rollout.run_rollout(0, [0] * 5, output)
    assert runtime.instances[0].closed
    assert read_samples(output) == [measurement(1.002)]


def test_score_uses_continuous_gate_progress_and_clamps_turn():
    summary = {"qualified_success": False,
               "gates": {"lift": {"best_continuous_s": 1.0, "required_continuous_s": 2.0}},
               "metrics": {"max_nut_turn_rad": math.pi},
               "quality": {"max_contact_force_n": 5.0, "max_penetration_m": 0.001,
                           "breached": False}}
    assert rollout.score_rollout(summary) == pytest.approx(58.5)
    summary["qualified_success"] = True
    summary["quality"]["breached"] = True
    assert rollout.score_rollout(summary) == pytest.approx(58.5)
