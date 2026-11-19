import json
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dexhand_agent import run
from dexhand_agent.file_backend import FileBackend
from dexhand_agent.runtime import InvalidAction


def test_cli_backend_error_is_not_a_success_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["dexhand-agent", "agent", "--output", str(tmp_path / "episode")])
    monkeypatch.setattr(run, "run_episode", lambda *a, **kw: {"success": True, "error": "backend timed out"})
    with pytest.raises(SystemExit) as result:
        run.main()
    assert result.value.code == 1


@pytest.fixture
def isolated_runtime(monkeypatch):
    # Isolate the physical engine; this checks orchestration, not manipulation.
    class Runtime:
        initial_height_m = 0.24
        data = SimpleNamespace(time=0.0)

        def __init__(self, seed):
            self.on_sample = None

        def observe(self):
            return {"mode": "simulator_state", "tool": {"name": "hammer", "position_m": [0, 0, 0]},
                    "button": {"position_m": [0, 0, 0]},
                    "robot": {"palm_position_m": [0, 0, 0], "palm_quat_wxyz": [1, 0, 0, 0]}}

        def render(self, *args, **kwargs):
            return np.zeros((480, 640, 3), dtype=np.uint8)

        def execute(self, action):
            if action["targets"]:
                raise InvalidAction("finish cannot contain motion targets")
            return {"status": "finished", "time_s": 0.0}

        def close(self):
            pass

    monkeypatch.setattr(run, "HandRuntime", Runtime)
    return Runtime


def test_rejected_finish_does_not_end_serving(monkeypatch, tmp_path, isolated_runtime):
    class Policy:
        calls = 0

        def decide(self, observation):
            self.calls += 1
            return {"operation": "finish", "targets": [{"actuator": "actuator1", "value": 0.0}] if self.calls == 1 else []}

    monkeypatch.setattr(run, "ReferencePolicy", Policy)
    result = run.run_episode(tmp_path / "episode", video=False)
    assert result["decisions"] == 2


def test_episode_timeout_retains_unanswered_request_and_failure(monkeypatch, tmp_path, isolated_runtime):
    monkeypatch.setattr(run, "FileBackend", lambda path, timeout_s: FileBackend(path, timeout_s=0.001))
    output = tmp_path / "episode"
    result = run.run_episode(output, backend="native", video=False)
    assert result["success"] is False
    assert result["decisions"] == 0
    assert "Timed out" in result["error"]
    assert json.loads((output / "summary.json").read_text(encoding="utf8")) == result
    assert (output / "error.txt").is_file()
    assert (output / "steps/000/request.json").is_file()
    assert (output / "steps/000/front.png").is_file()
    assert not (output / "steps/000/action.json").exists()
    pointer = json.loads((output / "actor/current.json").read_text(encoding="utf8"))
    assert pointer["status"] == "timeout"
    assert pointer["request_id"].endswith("-0")
    experience = json.loads((output / "experience.json").read_text(encoding="utf8"))
    assert experience["status"] == "failed_episode"
    assert "Physical success: FAIL" in (output / "report.html").read_text(encoding="utf8")


@pytest.mark.parametrize("completed", [0, 1])
def test_backend_permission_error_retains_completed_history_without_pending_action(
        monkeypatch, tmp_path, isolated_runtime, completed):
    action = {"operation": "hold", "targets": [], "duration_s": 0.1}
    result = {"status": "completed", "time_s": 0.0}
    executed = []

    def decide(request, image_paths):
        if request["sequence"] == completed:
            raise PermissionError(13, "Access is denied", "current.json", 5)
        return action

    def execute(self, command):
        executed.append(command)
        return result

    monkeypatch.setattr(run, "FileBackend", lambda path, timeout_s: SimpleNamespace(decide=decide))
    monkeypatch.setattr(isolated_runtime, "execute", execute)
    output = tmp_path / "episode"
    summary = run.run_episode(output, backend="native", video=False)
    assert summary["success"] is False
    assert summary["error"].startswith("PermissionError:")
    assert "Access is denied" in summary["error"]
    assert summary["decisions"] == completed
    assert executed == [action] * completed
    assert json.loads((output / "summary.json").read_text(encoding="utf8")) == summary
    trace = (output / "error.txt").read_text(encoding="utf8")
    assert "Traceback (most recent call last):" in trace and "PermissionError" in trace
    actions = json.loads((output / "actions.json").read_text(encoding="utf8"))
    assert actions == [{"sequence": i, "action": action, "result": result} for i in range(completed)]
    pending = output / "steps" / f"{completed:03}"
    assert (pending / "request.json").is_file()
    assert not (pending / "action.json").exists()
    assert not (pending / "result.json").exists()
    assert not (output / "steps" / f"{completed + 1:03}").exists()
    experience = json.loads((output / "experience.json").read_text(encoding="utf8"))
    assert experience["status"] == "failed_episode"
    assert experience["trajectory"] == actions
    assert experience["outcome"]["error"] == summary["error"]


def test_unknown_task_does_not_create_an_episode(tmp_path):
    output = tmp_path / "episode"
    with pytest.raises(ValueError, match="Unknown task"):
        run.run_episode(output, task="not_a_task", video=False)
    assert not output.exists()


@pytest.mark.parametrize("task", ["wrench_turn", "tweezers_transfer"])
def test_cli_routes_requested_task(monkeypatch, tmp_path, task):
    captured = {}

    def episode(*args, **kwargs):
        captured.update(kwargs)
        return {"success": True, "error": None}

    monkeypatch.setattr(run, "run_episode", episode)
    monkeypatch.setattr(sys, "argv", ["dexhand-agent", "demo", "--task", task,
                                    "--output", str(tmp_path / "episode")])
    with pytest.raises(SystemExit) as result:
        run.main()
    assert result.value.code == 0
    assert captured["task"] == task


@pytest.mark.parametrize("names", [[], ["first.json"], ["second.json", "first.json"]])
def test_cli_loads_explicit_experiences_in_declared_order(monkeypatch, tmp_path, names):
    captured, loaded = {}, []
    paths = [tmp_path / name for name in names]

    def load(path):
        loaded.append(path)
        return {"id": path.stem}

    def episode(*args, **kwargs):
        captured.update(kwargs)
        return {"success": True, "error": None}

    monkeypatch.setattr(run, "load_experience", load)
    monkeypatch.setattr(run, "run_episode", episode)
    monkeypatch.setattr(sys, "argv", ["dexhand-agent", "demo", "--output", str(tmp_path / "episode"),
                                    *[value for path in paths for value in ("--experience", str(path))]])
    with pytest.raises(SystemExit) as result:
        run.main()
    assert result.value.code == 0
    assert loaded == paths
    assert captured["experience"] == ([{"id": path.stem} for path in paths] if paths else None)


def test_wrench_episode_retains_task_goal_and_experience(monkeypatch, tmp_path, isolated_runtime):
    from dexhand_agent import wrench

    class Runtime(isolated_runtime):
        initial_nut_angle_rad = 0.0

        def observe(self):
            observation = super().observe()
            observation.pop("button")
            observation["tool"]["name"] = "wrench"
            observation["nut"] = {"position_m": [0.53, 0.16, 0.245]}
            return observation

    class Policy:
        def decide(self, observation):
            assert observation["tool"]["name"] == "wrench"
            return {"operation": "finish", "targets": []}

    monkeypatch.setattr(wrench, "WrenchRuntime", Runtime)
    monkeypatch.setattr(wrench, "WrenchReference", Policy)
    output = tmp_path / "episode"
    summary = run.run_episode(output, task="wrench_turn", video=False)
    read = lambda name: json.loads((output / name).read_text(encoding="utf8"))
    assert summary["task_id"] == "wrench_turn"
    assert set(summary["gates"]) == {"lift", "turn", "withdraw"}
    assert read("metadata.json")["task_id"] == "wrench_turn"
    request = read("steps/000/request.json")
    assert request["task_id"] == "wrench_turn"
    assert request["max_decisions"] == 32
    assert "45" in request["goal"]
    assert read("experience.json")["applicability"]["task"] == "wrench_turn"


def test_tweezers_episode_retains_both_fixture_references(monkeypatch, tmp_path, isolated_runtime):
    from dexhand_agent import tweezers

    class Runtime(isolated_runtime):
        initial_object_height_m = 0.235

        def observe(self):
            observation = super().observe()
            observation.pop("button")
            observation["tool"]["name"] = "tweezers"
            observation["object"] = {"position_m": [0.64, 0.10, 0.235]}
            observation["goal"] = {"position_m": [0.55, 0.20, 0.225]}
            return observation

    class Policy:
        def decide(self, observation):
            return {"operation": "finish", "targets": []}

    monkeypatch.setattr(tweezers, "TweezersRuntime", Runtime)
    monkeypatch.setattr(tweezers, "TweezersReference", Policy)
    output = tmp_path / "episode"
    summary = run.run_episode(output, task="tweezers_transfer", video=False)
    assert summary["task_id"] == "tweezers_transfer"
    assert set(summary["gates"]) == {"pickup", "pinch_lift", "transport", "release"}
    request = json.loads((output / "steps/000/request.json").read_text(encoding="utf8"))
    assert request["max_decisions"] == 48
    experience = json.loads((output / "experience.json").read_text(encoding="utf8"))
    assert experience["applicability"]["task"] == "tweezers_transfer"
    assert set(experience["initial_observation"]) == {"tool", "object", "goal", "palm"}


def test_explicit_decision_budget_is_preserved(monkeypatch, tmp_path, isolated_runtime):
    class Policy:
        def decide(self, observation):
            return {"operation": "finish", "targets": []}

    monkeypatch.setattr(run, "ReferencePolicy", Policy)
    output = tmp_path / "episode"
    run.run_episode(output, max_decisions=7, video=False)
    request = json.loads((output / "steps/000/request.json").read_text(encoding="utf8"))
    assert request["max_decisions"] == 7


def test_cli_forwards_policy_for_wrench_demo(monkeypatch, tmp_path):
    captured = {}

    def episode(*args, **kwargs):
        captured.update(kwargs)
        return {"success": True, "error": None}

    policy = tmp_path / "trained-policy"
    monkeypatch.setattr(run, "run_episode", episode)
    monkeypatch.setattr(sys, "argv", ["dexhand-agent", "demo", "--task", "wrench_turn",
                                    "--policy", str(policy), "--output", str(tmp_path / "episode")])
    with pytest.raises(SystemExit) as result:
        run.main()
    assert result.value.code == 0
    assert captured["policy"] == policy
    assert captured["backend"] == "reference"


@pytest.mark.parametrize("command,task", [("demo", "hammer_button"),
                                         ("demo", "tweezers_transfer"),
                                         ("agent", "wrench_turn")])
def test_cli_rejects_incompatible_policy_before_episode(monkeypatch, tmp_path, capsys, command, task):
    def forbidden(*args, **kwargs):
        pytest.fail("Incompatible policy must not start an episode")

    output = tmp_path / "episode"
    monkeypatch.setattr(run, "run_episode", forbidden)
    monkeypatch.setattr(sys, "argv", ["dexhand-agent", command, "--task", task,
                                    "--policy", "trained-policy", "--output", str(output)])
    with pytest.raises(SystemExit) as result:
        run.main()
    assert result.value.code == 2
    assert "--policy requires demo --task wrench_turn" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("task,backend", [("hammer_button", "reference"),
                                         ("tweezers_transfer", "reference"),
                                         ("wrench_turn", "native"), ("wrench_turn", "cli")])
def test_episode_rejects_incompatible_policy_before_runtime(monkeypatch, tmp_path, task, backend):
    def forbidden(*args):
        pytest.fail("Incompatible policy must not create task components")

    monkeypatch.setattr(run, "_task_components", forbidden)
    output = tmp_path / "episode"
    with pytest.raises(ValueError, match="policy requires task wrench_turn and backend reference"):
        run.run_episode(output, task=task, backend=backend, policy=tmp_path / "policy", video=False)
    assert not output.exists()


def test_policy_load_failure_precedes_output_and_runtime(monkeypatch, tmp_path):
    from dexhand_agent import wrench, wrench_policy

    def fail_load(path):
        raise ValueError("Incompatible policy metadata: base_source_sha256")

    def forbidden(seed):
        pytest.fail("Invalid policy must not allocate a runtime")

    monkeypatch.setattr(wrench_policy, "load_policy", fail_load)
    monkeypatch.setattr(wrench, "WrenchRuntime", forbidden)
    output = tmp_path / "episode"
    with pytest.raises(ValueError, match="Incompatible policy metadata"):
        run.run_episode(output, task="wrench_turn", policy=tmp_path / "policy", video=False)
    assert not output.exists()


@pytest.fixture
def learned_runtime(monkeypatch, tmp_path, isolated_runtime):
    from dexhand_agent import wrench, wrench_policy

    events = []
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    (policy_dir / "policy.json").write_bytes(b'{"snapshot": "fake loader fixture"}\n')
    (policy_dir / "weights.npz").write_bytes(b"fake loader weights fixture")
    parameters = np.array([0.001, -0.002, 0.003, -0.001, 0.002])

    class Runtime(isolated_runtime):
        initial_nut_angle_rad = 0.0

        def __init__(self, seed):
            assert events == ["load"]
            events.append("runtime")
            super().__init__(seed)

        def observe(self):
            observation = super().observe()
            observation.pop("button")
            observation["tool"] = {"name": "wrench", "position_m": [0.535, -0.132, 0.24]}
            observation["nut"] = {"position_m": [0.53, 0.16, 0.245]}
            return observation

        def close(self):
            events.append("close")

    class Policy:
        def predict(self, context):
            events.append(("predict", context.tolist()))
            return parameters

    def load(path):
        assert path == policy_dir
        events.append("load")
        return Policy()

    monkeypatch.setattr(wrench_policy, "load_policy", load)
    monkeypatch.setattr(wrench, "WrenchRuntime", Runtime)
    monkeypatch.setattr(wrench_policy.ParameterWrenchReference, "decide",
                        lambda self, observation: {"operation": "finish", "targets": []})
    return policy_dir, parameters, events


def test_learned_episode_retains_initial_context_parameters_and_artifact_hashes(tmp_path, learned_runtime):
    from dexhand_agent.wrench_policy import PARAMETER_NAMES
    from dexhand_agent.wrench_rollout import QUALITY_CRITERIA

    policy_dir, parameters, events = learned_runtime
    output = tmp_path / "episode"
    summary = run.run_episode(output, task="wrench_turn", policy=policy_dir, video=False)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf8"))
    controller = metadata["controller"]
    assert metadata["backend"] == "reference"
    assert metadata["quality_criteria"] == QUALITY_CRITERIA
    assert controller["kind"] == "neural_context_parameters"
    assert controller["quality_criteria"] == QUALITY_CRITERIA
    assert controller["context"] == pytest.approx([0.5, -0.2])
    assert controller["parameters_m"] == dict(zip(PARAMETER_NAMES, parameters.tolist()))
    assert controller["policy_dir"] == str(policy_dir.resolve())
    for filename, field in [("policy.json", "policy_sha256"), ("weights.npz", "weights_sha256")]:
        assert controller[field] == hashlib.sha256((policy_dir / filename).read_bytes()).hexdigest()
    assert events[0:2] == ["load", "runtime"]
    assert len([event for event in events if isinstance(event, tuple)]) == 1
    assert events[-1] == "close"
    assert summary["task_id"] == "wrench_turn"
    assert summary["controller"] == controller
    assert json.loads((output / "summary.json").read_text(encoding="utf8"))["controller"] == controller
    assert summary["success"] is False
    assert summary["error"] is None
    assert set(summary["gates"]) == {"lift", "turn", "withdraw"}


def test_invalid_learned_parameters_close_runtime_without_valid_metadata(tmp_path, learned_runtime):
    policy_dir, parameters, events = learned_runtime
    parameters[0] = float("nan")
    output = tmp_path / "episode"
    with pytest.raises(ValueError, match="parameters must contain 5 finite values"):
        run.run_episode(output, task="wrench_turn", policy=policy_dir, video=False)
    assert events[-1] == "close"
    assert not (output / "metadata.json").exists()


@pytest.fixture
def wrench_sample():
    return {"time_s": 0.01, "wrench_pos_m": [0.535, -0.132, 0.27],
            "wrench_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "hand_tool_contacts": ["thumb", "index"], "tool_support_contacts": [],
            "tool_nut_contact": False, "head_nut_contact": False, "robot_nut_contact": False,
            "nut_angle_rad": 0.0, "head_clearance_m": 0.03,
            "max_contact_force_n": 25.0, "max_penetration_m": 0.002}


@pytest.mark.parametrize("field,value,reason", [
    ("max_contact_force_n", 25.087, "max_contact_force_n exceeds 25 N"),
    ("max_penetration_m", 0.0021, "max_penetration_m exceeds 0.002 m"),
    ("robot_nut_contact", True, "robot_nut_contact"),
])
def test_learned_quality_abort_retains_first_sample_and_stops_execution(
        monkeypatch, tmp_path, learned_runtime, wrench_sample, field, value, reason):
    from dexhand_agent import wrench

    policy_dir, _, events = learned_runtime
    wrench_sample[field] = value

    def execute(self, action):
        events.append("execute")
        self.data.time = wrench_sample["time_s"]
        self.on_sample(wrench_sample)
        events.append("continued_after_violation")
        return {"status": "finished", "time_s": 0.01}

    monkeypatch.setattr(wrench.WrenchRuntime, "execute", execute)
    output = tmp_path / "episode"
    summary = run.run_episode(output, task="wrench_turn", policy=policy_dir,
                              quality_guard=False, video=False)
    assert summary["error"] == "RuntimeError: Learned policy quality abort: " + reason
    assert json.loads((output / "samples.jsonl").read_text(encoding="utf8")) == wrench_sample
    assert summary["metrics"]["samples"] == 1
    assert summary["metrics"]["max_contact_force_n"] == wrench_sample["max_contact_force_n"]
    assert summary["metrics"]["max_penetration_m"] == wrench_sample["max_penetration_m"]
    assert events.count("execute") == 1
    assert "continued_after_violation" not in events
    assert events[-1] == "close"
    assert not (output / "steps/001").exists()
    assert reason in (output / "error.txt").read_text(encoding="utf8")
    actions = json.loads((output / "actions.json").read_text(encoding="utf8"))
    action = json.loads((output / "steps/000/action.json").read_text(encoding="utf8"))
    result = json.loads((output / "steps/000/result.json").read_text(encoding="utf8"))
    assert result == {"status": "interrupted", "reason": summary["error"], "time_s": wrench_sample["time_s"]}
    assert actions == [{"sequence": 0, "action": action, "result": result}]
    assert summary["decisions"] == 1
    assert summary["success"] is False
    experience = json.loads((output / "experience.json").read_text(encoding="utf8"))
    assert experience["status"] == "failed_episode"
    assert experience["trajectory"] == actions
    assert experience["outcome"]["decisions"] == 1


def test_learned_quality_boundary_allows_accepted_finish(monkeypatch, tmp_path, learned_runtime, wrench_sample):
    from dexhand_agent import wrench

    policy_dir, _, _ = learned_runtime

    def execute(self, action):
        self.on_sample(wrench_sample)
        return {"status": "finished", "time_s": 0.01}

    monkeypatch.setattr(wrench.WrenchRuntime, "execute", execute)
    summary = run.run_episode(tmp_path / "episode", task="wrench_turn", policy=policy_dir, video=False)
    assert summary["error"] is None
    assert summary["decisions"] == 1


def test_learned_rejected_control_is_retained_and_ends_episode(monkeypatch, tmp_path, learned_runtime):
    from dexhand_agent import wrench

    policy_dir, _, events = learned_runtime

    def execute(self, action):
        events.append("execute")
        raise InvalidAction("fixture rejected control")

    monkeypatch.setattr(wrench.WrenchRuntime, "execute", execute)
    output = tmp_path / "episode"
    summary = run.run_episode(output, task="wrench_turn", policy=policy_dir, video=False)
    assert summary["error"] == "RuntimeError: Learned policy control rejected: fixture rejected control"
    assert summary["decisions"] == 1
    actions = json.loads((output / "actions.json").read_text(encoding="utf8"))
    result = json.loads((output / "steps/000/result.json").read_text(encoding="utf8"))
    assert actions[0]["result"] == result
    assert result["status"] == "rejected"
    assert events.count("execute") == 1
    assert not (output / "steps/001").exists()


def test_learned_decision_limit_without_accepted_finish_is_failure(monkeypatch, tmp_path, learned_runtime):
    from dexhand_agent import wrench

    policy_dir, _, _ = learned_runtime
    monkeypatch.setattr(wrench.WrenchRuntime, "execute",
                        lambda self, action: {"status": "completed", "time_s": 0.01})
    summary = run.run_episode(tmp_path / "episode", task="wrench_turn", policy=policy_dir,
                              max_decisions=2, video=False)
    assert summary["error"] == "RuntimeError: Learned policy decision limit reached without an accepted finish"
    assert summary["decisions"] == 2


def test_default_wrench_retains_prior_quality_and_decision_limit_behavior(
        monkeypatch, tmp_path, isolated_runtime, wrench_sample):
    from dexhand_agent import wrench

    wrench_sample["max_contact_force_n"] = 25.087

    class Runtime(isolated_runtime):
        initial_nut_angle_rad = 0.0

        def observe(self):
            observation = super().observe()
            observation.pop("button")
            observation["tool"]["name"] = "wrench"
            observation["nut"] = {"position_m": [0.53, 0.16, 0.245]}
            return observation

        def execute(self, action):
            self.on_sample(wrench_sample)
            return {"status": "completed", "time_s": 0.01}

    monkeypatch.setattr(wrench, "WrenchRuntime", Runtime)
    monkeypatch.setattr(wrench.WrenchReference, "decide",
                        lambda self, observation: {"operation": "finish", "targets": []})
    summary = run.run_episode(tmp_path / "episode", task="wrench_turn", max_decisions=1, video=False)
    assert summary["error"] is None
    assert summary["decisions"] == 1
    assert summary["metrics"]["max_contact_force_n"] == 25.087


@pytest.fixture
def quality_guard_runtime(monkeypatch, isolated_runtime, wrench_sample):
    from dexhand_agent import wrench

    events = []

    class Runtime(isolated_runtime):
        initial_nut_angle_rad = 0.0

        def observe(self):
            observation = super().observe()
            observation.pop("button")
            observation["tool"]["name"] = "wrench"
            observation["nut"] = {"position_m": [0.53, 0.16, 0.245]}
            return observation

        def execute(self, action):
            events.append("execute")
            self.data.time = wrench_sample["time_s"]
            self.on_sample(wrench_sample)
            events.append("continued")
            return {"status": "finished", "time_s": self.data.time}

        def close(self):
            events.append("close")

    class Policy:
        def decide(self, *args):
            return {"operation": "finish", "targets": []}

    monkeypatch.setattr(wrench, "WrenchRuntime", Runtime)
    monkeypatch.setattr(wrench, "WrenchReference", Policy)
    monkeypatch.setattr(run, "FileBackend", lambda *args, **kwargs: Policy())
    return events


@pytest.mark.parametrize("backend", ["reference", "native"])
@pytest.mark.parametrize("field,value,reason", [
    ("max_contact_force_n", 25.087, "max_contact_force_n exceeds 25 N"),
    ("max_penetration_m", 0.0021, "max_penetration_m exceeds 0.002 m"),
    ("robot_nut_contact", True, "robot_nut_contact"),
])
def test_quality_guard_retains_violation_and_interrupted_action(
        tmp_path, quality_guard_runtime, wrench_sample, backend, field, value, reason):
    from dexhand_agent.wrench_rollout import QUALITY_CRITERIA

    wrench_sample[field] = value
    output = tmp_path / "episode"
    summary = run.run_episode(output, backend=backend, task="wrench_turn",
                              quality_guard=True, video=False)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf8"))
    assert metadata["quality_criteria"] == QUALITY_CRITERIA
    assert "controller" not in metadata
    assert summary["error"] == "RuntimeError: Wrench quality abort: " + reason
    assert json.loads((output / "samples.jsonl").read_text(encoding="utf8")) == wrench_sample
    assert summary["metrics"]["samples"] == 1
    assert summary["metrics"]["max_contact_force_n"] == wrench_sample["max_contact_force_n"]
    assert summary["metrics"]["max_penetration_m"] == wrench_sample["max_penetration_m"]
    result = json.loads((output / "steps/000/result.json").read_text(encoding="utf8"))
    action = json.loads((output / "steps/000/action.json").read_text(encoding="utf8"))
    actions = json.loads((output / "actions.json").read_text(encoding="utf8"))
    assert result == {"status": "interrupted", "reason": summary["error"],
                      "time_s": wrench_sample["time_s"]}
    assert actions == [{"sequence": 0, "action": action, "result": result}]
    assert summary["decisions"] == 1
    assert summary["success"] is False
    assert quality_guard_runtime == ["execute", "close"]
    assert not (output / "steps/001").exists()
    experience = json.loads((output / "experience.json").read_text(encoding="utf8"))
    assert experience["status"] == "failed_episode"
    assert experience["trajectory"] == actions


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_quality_guard_allows_exact_boundary(
        tmp_path, quality_guard_runtime, wrench_sample, backend):
    from dexhand_agent.wrench_rollout import QUALITY_CRITERIA

    output = tmp_path / "episode"
    summary = run.run_episode(output, backend=backend, task="wrench_turn",
                              quality_guard=True, video=False)
    assert summary["error"] is None
    assert summary["decisions"] == 1
    assert quality_guard_runtime == ["execute", "continued", "close"]
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf8"))
    assert metadata["quality_criteria"] == QUALITY_CRITERIA
    assert "controller" not in metadata


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_quality_guard_default_does_not_abort_old_wrench_execution(
        tmp_path, quality_guard_runtime, wrench_sample, backend):
    wrench_sample["max_contact_force_n"] = 25.087
    output = tmp_path / "episode"
    summary = run.run_episode(output, backend=backend, task="wrench_turn", video=False)
    assert summary["error"] is None
    assert quality_guard_runtime == ["execute", "continued", "close"]
    assert summary["metrics"]["max_contact_force_n"] == 25.087
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf8"))
    assert "quality_criteria" not in metadata


@pytest.mark.parametrize("task", ["hammer_button", "tweezers_transfer"])
def test_quality_guard_rejects_non_wrench_before_output_and_runtime(monkeypatch, tmp_path, task):
    def forbidden(*args):
        pytest.fail("Invalid quality guard must not create task components")

    monkeypatch.setattr(run, "_task_components", forbidden)
    output = tmp_path / "episode"
    with pytest.raises(ValueError, match="quality_guard requires task wrench_turn"):
        run.run_episode(output, task=task, quality_guard=True, video=False)
    assert not output.exists()


@pytest.mark.parametrize("command,backend", [("demo", "reference"), ("agent", "native")])
def test_cli_forwards_wrench_quality_guard(monkeypatch, tmp_path, command, backend):
    captured = {}

    def episode(*args, **kwargs):
        captured.update(kwargs)
        return {"success": True, "error": None}

    monkeypatch.setattr(run, "run_episode", episode)
    monkeypatch.setattr(sys, "argv", ["dexhand-agent", command, "--task", "wrench_turn",
                                    "--quality-guard", "--output", str(tmp_path / "episode")])
    with pytest.raises(SystemExit) as result:
        run.main()
    assert result.value.code == 0
    assert captured["quality_guard"] is True
    assert captured["backend"] == backend


@pytest.mark.parametrize("task", ["hammer_button", "tweezers_transfer"])
def test_cli_rejects_non_wrench_quality_guard_before_episode(monkeypatch, tmp_path, capsys, task):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid quality guard must not start an episode")

    monkeypatch.setattr(run, "run_episode", forbidden)
    output = tmp_path / "episode"
    monkeypatch.setattr(sys, "argv", ["dexhand-agent", "agent", "--task", task,
                                    "--quality-guard", "--output", str(output)])
    with pytest.raises(SystemExit) as result:
        run.main()
    assert result.value.code == 2
    assert "--quality-guard requires --task wrench_turn" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("backend", ["reference", "native", "cli"])
@pytest.mark.parametrize("enabled", [False, True])
def test_rich_feedback_is_fresh_shared_and_separate_from_basic_contacts(
        monkeypatch, tmp_path, isolated_runtime, backend, enabled):
    from dexhand_agent import wrench

    events, observations = [], []

    class Runtime(isolated_runtime):
        initial_nut_angle_rad = 0.0

        def __init__(self, seed):
            super().__init__(seed)
            self.data = SimpleNamespace(time=0.0)

        def observe(self):
            events.append("observe")
            observation = super().observe()
            observation.pop("button")
            observation["tool"]["name"] = "wrench"
            observation["nut"] = {"position_m": [0.53, 0.16, 0.245]}
            observation["contacts"] = {"basic_time_s": self.data.time}
            return observation

        def execute(self, action):
            events.append("execute")
            self.data.time += 0.1
            return {"status": "finished" if action["operation"] == "finish" else "completed",
                    "time_s": self.data.time}

    class Actor:
        def decide(self, value, image_paths=None):
            events.append("decide")
            observation = value["observation"] if "observation" in value else value
            observations.append(observation)
            return {"operation": "hold" if len(observations) == 1 else "finish", "targets": []}

    def feedback(runtime):
        assert events[-1] == "observe"
        events.append("feedback")
        return {"observed_time_s": runtime.data.time, "named_contacts": ["measured pair"]}

    monkeypatch.setitem(sys.modules, "dexhand_agent.wrench_sensing",
                        SimpleNamespace(read_wrench_feedback=feedback))
    monkeypatch.setattr(wrench, "WrenchRuntime", Runtime)
    monkeypatch.setattr(wrench, "WrenchReference", Actor)
    monkeypatch.setattr(run, "FileBackend", lambda *args, **kwargs: Actor())
    monkeypatch.setattr(run, "CodexBackend", lambda *args, **kwargs: Actor())
    output = tmp_path / "episode"
    guidance = [{"id": "explicitly supplied guidance"}]
    options = {"rich_feedback": True} if enabled else {}
    summary = run.run_episode(output, backend=backend, task="wrench_turn", video=False,
                              experience=guidance, **options)
    assert summary["error"] is None
    assert summary["decisions"] == 2
    assert events == (["observe", "feedback", "decide", "execute"] * 2 if enabled else
                      ["observe", "decide", "execute"] * 2)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf8"))
    assert metadata.get("feedback_contract") == ("wrench_geometry_and_named_contacts_v1" if enabled else None)
    assert "controller" not in metadata
    assert "quality_criteria" not in metadata
    assert metadata["experience"] == guidance
    for seq, observation in enumerate(observations):
        request = json.loads((output / f"steps/{seq:03}/request.json").read_text(encoding="utf8"))
        assert request["observation"] == observation
        assert request["experience"] == guidance
        assert observation["contacts"] == {"basic_time_s": seq * 0.1}
        if enabled:
            assert observation["wrench_feedback"] == {
                "observed_time_s": seq * 0.1, "named_contacts": ["measured pair"]}
        else:
            assert "wrench_feedback" not in observation
    experience = json.loads((output / "experience.json").read_text(encoding="utf8"))
    assert experience["initial_observation"]["tool"] == observations[0]["tool"]
    assert experience["initial_observation"]["nut"] == observations[0]["nut"]
    assert "wrench_feedback" not in experience["initial_observation"]


@pytest.mark.parametrize("task", ["hammer_button", "tweezers_transfer"])
def test_rich_feedback_rejects_non_wrench_before_output_and_runtime(monkeypatch, tmp_path, task):
    def forbidden(*args):
        pytest.fail("Invalid rich feedback must not create task components")

    monkeypatch.setattr(run, "_task_components", forbidden)
    output = tmp_path / "episode"
    with pytest.raises(ValueError, match="rich_feedback requires task wrench_turn"):
        run.run_episode(output, task=task, rich_feedback=True, video=False)
    assert not output.exists()


@pytest.mark.parametrize("command,backend", [("demo", "reference"), ("agent", "native"), ("agent", "cli")])
def test_cli_forwards_wrench_rich_feedback(monkeypatch, tmp_path, command, backend):
    captured = {}

    def episode(*args, **kwargs):
        captured.update(kwargs)
        return {"success": True, "error": None}

    monkeypatch.setattr(run, "run_episode", episode)
    monkeypatch.setattr(sys, "argv", ["dexhand-agent", command, "--task", "wrench_turn",
                                    "--rich-feedback", "--backend", "native" if backend == "reference" else backend,
                                    "--output", str(tmp_path / "episode")])
    with pytest.raises(SystemExit) as result:
        run.main()
    assert result.value.code == 0
    assert captured["rich_feedback"] is True
    assert captured["backend"] == backend


@pytest.mark.parametrize("task", ["hammer_button", "tweezers_transfer"])
def test_cli_rejects_non_wrench_rich_feedback_before_episode(monkeypatch, tmp_path, capsys, task):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid rich feedback must not start an episode")

    monkeypatch.setattr(run, "run_episode", forbidden)
    output = tmp_path / "episode"
    monkeypatch.setattr(sys, "argv", ["dexhand-agent", "agent", "--task", task,
                                    "--rich-feedback", "--output", str(output)])
    with pytest.raises(SystemExit) as result:
        run.main()
    assert result.value.code == 2
    assert "--rich-feedback requires --task wrench_turn" in capsys.readouterr().err
    assert not output.exists()


def test_rich_feedback_also_follows_learned_initial_context_observation(monkeypatch, tmp_path, learned_runtime):
    from dexhand_agent import wrench_policy

    policy_dir, _, _ = learned_runtime
    calls = []
    original_context = wrench_policy.context_from_observation

    def feedback(runtime):
        calls.append(runtime)
        return {"geometry": "measured"}

    def context(observation):
        assert observation["wrench_feedback"] == {"geometry": "measured"}
        return original_context(observation)

    monkeypatch.setitem(sys.modules, "dexhand_agent.wrench_sensing",
                        SimpleNamespace(read_wrench_feedback=feedback))
    monkeypatch.setattr(wrench_policy, "context_from_observation", context)
    output = tmp_path / "episode"
    summary = run.run_episode(output, task="wrench_turn", policy=policy_dir,
                              rich_feedback=True, video=False)
    assert summary["error"] is None
    assert len(calls) == 2
    assert calls[0] is calls[1]
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf8"))
    assert metadata["controller"]["kind"] == "neural_context_parameters"
    assert metadata["controller"]["context"] == pytest.approx([0.5, -0.2])
    assert metadata["feedback_contract"] == "wrench_geometry_and_named_contacts_v1"


@pytest.fixture
def placement_runtime(monkeypatch, quality_guard_runtime):
    from dexhand_agent import wrench

    instances, actions = [], []
    base_runtime = wrench.WrenchRuntime

    class Runtime(base_runtime):
        variant_metadata = {"variant_id": "placement-30deg-v1", "xml_sha256": "compiled fixture"}

        def __init__(self, seed, variant="placement-30deg-v1"):
            super().__init__(seed)
            self.variant_metadata = {"variant_id": variant, "xml_sha256": "compiled fixture"}
            instances.append((self, seed, variant))

        def execute(self, action):
            actions.append(action)
            return super().execute(action)

    def forbidden(*args):
        pytest.fail("A placement episode must not instantiate the base wrench runtime")

    def feedback(runtime):
        assert runtime is instances[-1][0]
        return {"measured_variant": runtime.variant_metadata["variant_id"]}

    monkeypatch.setattr(wrench, "WrenchRuntime", forbidden)
    monkeypatch.setitem(sys.modules, "dexhand_agent.wrench_placement",
                        SimpleNamespace(PlacementWrenchRuntime=Runtime))
    monkeypatch.setitem(sys.modules, "dexhand_agent.wrench_geometry",
                        SimpleNamespace(LongWrenchRuntime=Runtime))
    monkeypatch.setitem(sys.modules, "dexhand_agent.wrench_sensing",
                        SimpleNamespace(read_wrench_feedback=feedback))
    return Runtime, instances, actions, quality_guard_runtime


@pytest.mark.parametrize("backend,variant", [
    (backend, variant) for backend in ("reference", "native", "cli")
    for variant in ("placement-30deg-v1", "placement-minus20deg-v2", "placement-plus15deg-v3")
] + [(backend, "wrench-long-heavy-v1") for backend in ("native", "cli")])
def test_wrench_variant_routes_only_runtime_and_enforces_measured_checks(
        monkeypatch, tmp_path, placement_runtime, backend, variant):
    from dexhand_agent import wrench
    from dexhand_agent.wrench_rollout import QUALITY_CRITERIA

    runtime_type, instances, actions, events = placement_runtime
    action = {"operation": "finish", "targets": [], "memory": backend}

    class Actor:
        def decide(self, *args):
            return action

    monkeypatch.setattr(wrench, "WrenchReference", Actor)
    monkeypatch.setattr(run, "FileBackend", lambda *args, **kwargs: Actor())
    monkeypatch.setattr(run, "CodexBackend", lambda *args, **kwargs: Actor())
    output = tmp_path / "episode"
    summary = run.run_episode(output, backend=backend, task="wrench_turn", seed=5, video=False,
                              wrench_variant=variant, quality_guard=False, rich_feedback=False)
    assert summary["error"] is None
    assert set(summary["gates"]) == {"lift", "turn", "withdraw"}
    assert len(instances) == 1 and instances[0][1:] == (5, variant)
    assert actions == [action]
    assert events == ["execute", "continued", "close"]
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf8"))
    assert metadata["scene_variant"] == instances[0][0].variant_metadata
    assert metadata["quality_criteria"] == QUALITY_CRITERIA
    assert metadata["feedback_contract"] == "wrench_geometry_and_named_contacts_v1"
    assert metadata["backend"] == backend
    assert "controller" not in metadata
    request = json.loads((output / "steps/000/request.json").read_text(encoding="utf8"))
    assert request["goal"] == run.WRENCH_GOAL
    assert request["observation"]["wrench_feedback"] == {"measured_variant": variant}


@pytest.mark.parametrize("field,value,reason", [
    ("max_contact_force_n", 25.087, "max_contact_force_n exceeds 25 N"),
    ("max_penetration_m", 0.0021, "max_penetration_m exceeds 0.002 m"),
    ("robot_nut_contact", True, "robot_nut_contact"),
])
@pytest.mark.parametrize("variant", ["placement-30deg-v1", "placement-minus20deg-v2", "placement-plus15deg-v3",
                                     "wrench-long-heavy-v1"])
def test_wrench_variant_guard_cannot_be_disabled(
        tmp_path, placement_runtime, wrench_sample, field, value, reason, variant):
    wrench_sample[field] = value
    output = tmp_path / "episode"
    summary = run.run_episode(output, backend="native", task="wrench_turn", video=False,
                              wrench_variant=variant, quality_guard=False)
    assert summary["error"] == "RuntimeError: Wrench quality abort: " + reason
    assert json.loads((output / "samples.jsonl").read_text(encoding="utf8")) == wrench_sample
    actions = json.loads((output / "actions.json").read_text(encoding="utf8"))
    assert actions[0]["result"]["status"] == "interrupted"
    assert placement_runtime[3] == ["execute", "close"]


@pytest.mark.parametrize("options", [{}, {"wrench_variant": None}])
def test_default_wrench_does_not_load_variant_or_enable_its_checks(
        monkeypatch, tmp_path, quality_guard_runtime, wrench_sample, options):
    wrench_sample["max_contact_force_n"] = 25.087
    monkeypatch.setitem(sys.modules, "dexhand_agent.wrench_placement", None)
    monkeypatch.setitem(sys.modules, "dexhand_agent.wrench_geometry", None)
    monkeypatch.setitem(sys.modules, "dexhand_agent.wrench_sensing", None)
    output = tmp_path / "episode"
    summary = run.run_episode(output, task="wrench_turn", video=False, **options)
    assert summary["error"] is None
    assert quality_guard_runtime == ["execute", "continued", "close"]
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf8"))
    assert not {"scene_variant", "quality_criteria", "feedback_contract", "controller"} & metadata.keys()
    request = json.loads((output / "steps/000/request.json").read_text(encoding="utf8"))
    assert "wrench_feedback" not in request["observation"]


@pytest.mark.parametrize("options,message", [
    ({"task": "hammer_button", "wrench_variant": "placement-30deg-v1"},
     "wrench_variant requires task wrench_turn"),
    ({"task": "tweezers_transfer", "wrench_variant": "placement-30deg-v1"},
     "wrench_variant requires task wrench_turn"),
    ({"task": "wrench_turn", "wrench_variant": "unknown"}, "Unknown wrench_variant"),
    ({"task": "wrench_turn", "wrench_variant": "placement-30deg-v1", "policy": "missing-policy"},
     "wrench_variant cannot be combined with policy"),
    ({"task": "hammer_button", "wrench_variant": "placement-minus20deg-v2"},
     "wrench_variant requires task wrench_turn"),
    ({"task": "tweezers_transfer", "wrench_variant": "placement-minus20deg-v2"},
     "wrench_variant requires task wrench_turn"),
    ({"task": "wrench_turn", "wrench_variant": "placement-minus20deg-v2", "policy": "missing-policy"},
     "wrench_variant cannot be combined with policy"),
    ({"task": "hammer_button", "wrench_variant": "placement-plus15deg-v3"},
     "wrench_variant requires task wrench_turn"),
    ({"task": "tweezers_transfer", "wrench_variant": "placement-plus15deg-v3"},
     "wrench_variant requires task wrench_turn"),
    ({"task": "wrench_turn", "wrench_variant": "placement-plus15deg-v3", "policy": "missing-policy"},
     "wrench_variant cannot be combined with policy"),
    ({"task": "hammer_button", "wrench_variant": "wrench-long-heavy-v1"},
     "wrench_variant requires task wrench_turn"),
    ({"task": "tweezers_transfer", "wrench_variant": "wrench-long-heavy-v1"},
     "wrench_variant requires task wrench_turn"),
    ({"task": "wrench_turn", "wrench_variant": "wrench-long-heavy-v1", "policy": "missing-policy"},
     "wrench_variant cannot be combined with policy"),
    ({"task": "wrench_turn", "wrench_variant": "wrench-long-heavy-v1", "backend": "reference"},
     "wrench-long-heavy-v1 requires backend native or cli"),
    ({"task": "wrench_turn", "wrench_variant": "wrench-long-heavy-v1", "backend": "unknown"},
     "wrench-long-heavy-v1 requires backend native or cli"),
])
def test_invalid_wrench_variant_rejected_before_imports_output_and_runtime(
        monkeypatch, tmp_path, options, message):
    def forbidden(*args):
        pytest.fail("Invalid variant must not create task components")

    monkeypatch.setattr(run, "_task_components", forbidden)
    for name in ("wrench_placement", "wrench_geometry", "wrench_policy", "wrench_sensing", "wrench_rollout"):
        monkeypatch.setitem(sys.modules, "dexhand_agent." + name, None)
    output = tmp_path / "episode"
    with pytest.raises(ValueError, match=message):
        run.run_episode(output, video=False, **options)
    assert not output.exists()


@pytest.mark.parametrize("command,backend,variant", [
    (command, backend, variant)
    for command, backend in (("demo", "reference"), ("agent", "native"), ("agent", "cli"))
    for variant in ("placement-30deg-v1", "placement-minus20deg-v2", "placement-plus15deg-v3")
] + [("agent", backend, "wrench-long-heavy-v1") for backend in ("native", "cli")])
def test_cli_forwards_wrench_variant(monkeypatch, tmp_path, command, backend, variant):
    captured = {}

    def episode(*args, **kwargs):
        captured.update(kwargs)
        return {"success": True, "error": None}

    monkeypatch.setattr(run, "run_episode", episode)
    monkeypatch.setattr(sys, "argv", ["dexhand-agent", command, "--task", "wrench_turn",
                                    "--wrench-variant", variant, "--backend",
                                    "native" if backend == "reference" else backend,
                                    "--output", str(tmp_path / "episode")])
    with pytest.raises(SystemExit) as result:
        run.main()
    assert result.value.code == 0
    assert captured["wrench_variant"] == variant
    assert captured["backend"] == backend


@pytest.mark.parametrize("extra,message", [
    (["--task", "hammer_button"], "--wrench-variant requires --task wrench_turn"),
    (["--task", "tweezers_transfer"], "--wrench-variant requires --task wrench_turn"),
    (["--task", "wrench_turn", "--policy", "missing-policy"],
     "--wrench-variant cannot be combined with --policy"),
])
@pytest.mark.parametrize("variant", ["placement-30deg-v1", "placement-minus20deg-v2", "placement-plus15deg-v3",
                                     "wrench-long-heavy-v1"])
def test_cli_rejects_incompatible_wrench_variant_before_episode(monkeypatch, tmp_path, capsys, extra, message, variant):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid variant must not start an episode")

    monkeypatch.setattr(run, "run_episode", forbidden)
    output = tmp_path / "episode"
    monkeypatch.setattr(sys, "argv", ["dexhand-agent", "demo", "--wrench-variant", variant,
                                    "--output", str(output), *extra])
    with pytest.raises(SystemExit) as result:
        run.main()
    assert result.value.code == 2
    assert message in capsys.readouterr().err
    assert not output.exists()


def test_cli_rejects_long_wrench_demo_before_experience_loading_and_episode(monkeypatch, tmp_path, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("The unvalidated reference must not run the long wrench")

    monkeypatch.setattr(run, "run_episode", forbidden)
    monkeypatch.setattr(run, "load_experience", forbidden)
    output = tmp_path / "episode"
    monkeypatch.setattr(sys, "argv", ["dexhand-agent", "demo", "--task", "wrench_turn",
                                    "--wrench-variant", "wrench-long-heavy-v1", "--experience", "missing.json",
                                    "--output", str(output)])
    with pytest.raises(SystemExit) as result:
        run.main()
    assert result.value.code == 2
    assert "wrench-long-heavy-v1 requires agent --backend native or cli" in capsys.readouterr().err
    assert not output.exists()
