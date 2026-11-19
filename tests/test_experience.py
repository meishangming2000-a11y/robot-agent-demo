import hashlib
import json
from pathlib import Path

import pytest

from dexhand_agent.experience import export_experience, load_experience


@pytest.fixture
def episode(tmp_path):
    files = {
        "metadata.json": {"episode_id": "episode-123", "backend": "reference", "seed": 0,
                          "observation_mode": "simulator_state", "private_path": str(tmp_path)},
        "summary.json": {"episode_id": "episode-123", "protocol_version": 2, "success": True,
                         "gates": {name: {"passed": True} for name in ("lift", "press", "withdraw")},
                         "violations": [], "metrics": {"max_lift_m": 0.03}, "error": None,
                         "decisions": 2},
        "actions.json": [
            {"sequence": 0, "action": {"operation": "hold", "duration_s": 2.2},
             "result": {"status": "completed", "time_s": 3.2}},
            {"sequence": 1, "action": {"operation": "finish"},
             "result": {"status": "finished", "time_s": 3.2}},
        ],
        "steps/000/request.json": {"episode_id": "episode-123", "goal": "Held hammer presses button",
                                   "observation": {"mode": "simulator_state",
                                       "tool": {"name": "hammer", "position_m": [0.53, -0.13, 0.24]},
                                       "button": {"position_m": [0.53, 0.16, 0.23]},
                                       "robot": {"palm_position_m": [0.45, 0, 0.55],
                                                 "palm_quat_wxyz": [0, 0, 1, 0]}}},
    }
    for name, value in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf8")
    return tmp_path


def rewrite(episode, name, change):
    path = episode / name
    value = json.loads(path.read_text(encoding="utf8"))
    change(value)
    path.write_text(json.dumps(value), encoding="utf8")


def test_export_preserves_trace_and_hashes_without_copying_private_paths(episode):
    record = export_experience(episode)
    assert record["schema_version"] == 1
    assert record["status"] == "demonstrated_episode"
    assert record["origin_backend"] == "reference"
    assert record["trajectory"] == json.loads((episode / "actions.json").read_text(encoding="utf8"))
    for name, expected in record["source"]["files_sha256"].items():
        assert expected == hashlib.sha256((episode / name).read_bytes()).hexdigest()
    assert record["initial_observation"]["tool"]["position_m"] == [0.53, -0.13, 0.24]
    assert record["applicability"]["new_tools_verified"] is False
    assert record["applicability"]["other_layouts_verified"] is False
    assert "developer" in record["notes"]
    assert str(episode) not in json.dumps(record)
    path = episode / "experience.json"
    path.write_text(json.dumps(record), encoding="utf8")
    assert load_experience(path) == record


def test_failed_online_trial_stays_a_failed_record(episode):
    rewrite(episode, "metadata.json", lambda v: v.update(backend="native"))
    rewrite(episode, "summary.json", lambda v: v.update(success=False, gates={
        name: {"passed": False} for name in ("lift", "press", "withdraw")}))
    record = export_experience(episode)
    assert record["status"] == "failed_episode"
    assert record["outcome"]["success"] is False
    assert record["origin_backend"] == "native"
    assert record["trajectory"]


def test_backend_error_does_not_become_demonstrated_experience(episode):
    rewrite(episode, "summary.json", lambda v: v.update(error="BackendError: timeout"))
    assert export_experience(episode)["status"] == "failed_episode"


@pytest.mark.parametrize("name", ["summary.json", "steps/000/request.json"])
def test_mixed_episode_inputs_are_rejected(episode, name):
    rewrite(episode, name, lambda v: v.update(episode_id="different-episode"))
    with pytest.raises(ValueError, match="episode_id"):
        export_experience(episode)


def test_non_list_action_trace_is_rejected(episode):
    (episode / "actions.json").write_text("{}", encoding="utf8")
    with pytest.raises(ValueError, match="trajectory|actions"):
        export_experience(episode)


@pytest.mark.parametrize("payload", [
    "not json", "[]", "{}", '{"schema_version":2}',
    '{"schema_version":1,"outcome":{"max_lift_m":NaN}}',
    '{"schema_version":1,"outcome":{"max_lift_m":1e999}}',
])
def test_load_rejects_malformed_or_nonfinite_records(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf8")
    with pytest.raises(ValueError):
        load_experience(path)


def test_load_rejects_claimed_success_when_gates_or_error_disagree(episode):
    record = export_experience(episode)
    record["outcome"]["gates"]["press"]["passed"] = False
    path = episode / "tampered.json"
    path.write_text(json.dumps(record), encoding="utf8")
    with pytest.raises(ValueError, match="outcome|status"):
        load_experience(path)


def test_load_rejects_absolute_source_file_names(episode):
    record = export_experience(episode)
    hashes = record["source"]["files_sha256"]
    hashes[str(episode / "metadata.json")] = hashes.pop("metadata.json")
    path = episode / "absolute-source.json"
    path.write_text(json.dumps(record), encoding="utf8")
    with pytest.raises(ValueError, match="relative file names"):
        load_experience(path)


@pytest.fixture
def wrench_episode(episode):
    rewrite(episode, "metadata.json", lambda v: v.update(task_id="wrench_turn"))
    rewrite(episode, "summary.json", lambda v: v.update(
        task_id="wrench_turn", protocol_version=1,
        gates={name: {"passed": True} for name in ("lift", "turn", "withdraw")},
        metrics={"max_lift_m": 0.03, "max_nut_turn_rad": 0.8}))

    def wrench_request(value):
        value["goal"] = "Lift the held wrench, turn the passive nut, then withdraw"
        observation = value["observation"]
        del observation["button"]
        observation["tool"] = {
            "name": "wrench", "position_m": [0.53, -0.13, 0.24],
            "quat_wxyz": [1, 0, 0, 0], "head_center_m": [0.63, -0.13, 0.24],
            "initial_height_m": 0.24,
        }
        observation["nut"] = {
            "position_m": [0.53, 0.16, 0.245], "angle_rad": 0.0,
            "initial_angle_rad": 0.0, "top_m": 0.265, "half_height_m": 0.020,
        }

    rewrite(episode, "steps/000/request.json", wrench_request)
    return episode


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_wrench_export_and_load_preserve_nut_reference_outcome_and_provenance(wrench_episode, backend):
    supplied_experience = [{"id": "declared-reference", "origin_backend": "reference"}]
    rewrite(wrench_episode, "metadata.json", lambda v: v.update(
        backend=backend, experience=supplied_experience, experience_sha256="recorded-input-hash"))
    metadata_before = (wrench_episode / "metadata.json").read_bytes()
    record = export_experience(wrench_episode)
    request = json.loads((wrench_episode / "steps/000/request.json").read_text(encoding="utf8"))

    assert record["applicability"]["task"] == "wrench_turn"
    assert record["applicability"]["tool"] == "wrench"
    assert record["applicability"]["protocol_version"] == 1
    assert record["status"] == "demonstrated_episode"
    assert record["origin_backend"] == backend
    assert set(record["initial_observation"]) == {"tool", "nut", "palm"}
    assert record["initial_observation"]["tool"] == request["observation"]["tool"]
    assert record["initial_observation"]["nut"] == request["observation"]["nut"]
    assert set(record["outcome"]["gates"]) == {"lift", "turn", "withdraw"}
    assert record["trajectory"] == json.loads((wrench_episode / "actions.json").read_text(encoding="utf8"))
    assert record["applicability"]["new_tools_verified"] is False
    assert record["applicability"]["other_layouts_verified"] is False
    for name, digest in record["source"]["files_sha256"].items():
        assert digest == hashlib.sha256((wrench_episode / name).read_bytes()).hexdigest()
    assert (wrench_episode / "metadata.json").read_bytes() == metadata_before
    assert json.loads(metadata_before)["experience"] == supplied_experience
    assert ("Reference trajectory calibrated by the developer." if backend == "reference"
            else "Online actor episode recorded under its declared inputs.") in record["notes"]
    path = wrench_episode / "experience.json"
    path.write_text(json.dumps(record), encoding="utf8")
    assert load_experience(path) == record


def test_wrench_unsuccessful_physical_episode_stays_failed(wrench_episode):
    rewrite(wrench_episode, "summary.json", lambda v: v.update(
        success=False, gates={name: {"passed": False} for name in ("lift", "turn", "withdraw")}))
    record = export_experience(wrench_episode)
    assert record["status"] == "failed_episode"
    path = wrench_episode / "failed-experience.json"
    path.write_text(json.dumps(record), encoding="utf8")
    assert load_experience(path) == record


@pytest.mark.parametrize("gates", [
    {},
    {name: {"passed": True} for name in ("lift", "press", "withdraw")},
    {"lift": {"passed": True}, "turn": {"passed": False}, "withdraw": {"passed": True}},
])
def test_wrench_export_rejects_claimed_success_with_empty_wrong_or_failed_gates(wrench_episode, gates):
    rewrite(wrench_episode, "summary.json", lambda v: v.update(gates=gates))
    with pytest.raises(ValueError, match="gates|outcome"):
        export_experience(wrench_episode)


@pytest.mark.parametrize("gates", [
    {},
    {name: {"passed": True} for name in ("lift", "press", "withdraw")},
    {"lift": {"passed": True}, "turn": {"passed": False}, "withdraw": {"passed": True}},
])
def test_wrench_load_rejects_claimed_success_with_empty_wrong_or_failed_gates(wrench_episode, gates):
    record = export_experience(wrench_episode)
    record["outcome"]["gates"] = gates
    path = wrench_episode / "fake-gates.json"
    path.write_text(json.dumps(record), encoding="utf8")
    with pytest.raises(ValueError, match="gates|outcome"):
        load_experience(path)


def test_wrench_export_requires_a_nut_observation(wrench_episode):
    rewrite(wrench_episode, "steps/000/request.json", lambda v: v["observation"].pop("nut"))
    with pytest.raises(ValueError, match="nut"):
        export_experience(wrench_episode)


def test_wrench_load_requires_a_nut_reference(wrench_episode):
    record = export_experience(wrench_episode)
    del record["initial_observation"]["nut"]
    path = wrench_episode / "missing-nut.json"
    path.write_text(json.dumps(record), encoding="utf8")
    with pytest.raises(ValueError, match="nut"):
        load_experience(path)


def test_explicit_hammer_task_and_legacy_default_keep_their_fixture_and_gate_contract(episode):
    legacy = export_experience(episode)
    assert legacy["applicability"]["task"] == "held_hammer_press_spring_button"
    rewrite(episode, "metadata.json", lambda v: v.update(task_id="hammer_button"))
    explicit = export_experience(episode)
    assert explicit["applicability"]["task"] == "hammer_button"
    assert explicit["initial_observation"] == legacy["initial_observation"]
    assert explicit["outcome"] == legacy["outcome"]
    path = episode / "hammer-experience.json"
    path.write_text(json.dumps(explicit), encoding="utf8")
    assert load_experience(path) == explicit


def test_existing_reference_experience_loads_without_changing_legacy_content():
    path = Path(__file__).resolve().parents[1] / "experiments/reference-experience.json"
    assert load_experience(path) == json.loads(path.read_text(encoding="utf8"))


def test_unknown_metadata_task_is_rejected_even_when_legacy_gates_pass(episode):
    rewrite(episode, "metadata.json", lambda v: v.update(task_id="unknown_task"))
    with pytest.raises(ValueError, match="task"):
        export_experience(episode)


def test_unknown_record_task_cannot_turn_empty_gates_into_a_demonstration(episode):
    record = export_experience(episode)
    record["applicability"]["task"] = "unknown_task"
    record["outcome"]["gates"] = {}
    path = episode / "unknown-task.json"
    path.write_text(json.dumps(record), encoding="utf8")
    with pytest.raises(ValueError, match="task"):
        load_experience(path)


@pytest.fixture
def tweezers_episode(episode):
    rewrite(episode, "metadata.json", lambda v: v.update(task_id="tweezers_transfer"))
    rewrite(episode, "summary.json", lambda v: v.update(
        task_id="tweezers_transfer", protocol_version=1,
        gates={name: {"passed": True, "passed_at_s": passed_at,
                      "required_continuous_s": duration, "best_continuous_s": duration}
               for name, passed_at, duration in (("pickup", 2.0, 2.0),
                                                ("pinch_lift", 3.01, 1.0),
                                                ("transport", 3.52, 0.5),
                                                ("release", 4.53, 1.0))},
        metrics={"max_tool_lift_m": 0.03, "max_object_lift_m": 0.03,
                 "min_goal_distance_m": 0.01, "max_tip_object_clearance_m": 0.03}))

    def tweezers_request(value):
        value["task_id"] = "tweezers_transfer"
        value["goal"] = "Lift the free tweezers, pinch and carry the object, then release it at the goal"
        observation = value["observation"]
        del observation["button"]
        observation["tool"] = {
            "name": "tweezers", "position_m": [0.53, -0.13, 0.242],
            "quat_wxyz": [1, 0, 0, 0], "initial_height_m": 0.242,
            "left_tip_m": [0.62, -0.13, 0.242], "right_tip_m": [0.62, -0.13, 0.26],
        }
        observation["object"] = {"position_m": [0.34, 0.02, 0.230], "initial_height_m": 0.230}
        observation["goal"] = {"position_m": [0.34, 0.16, 0.225]}

    rewrite(episode, "steps/000/request.json", tweezers_request)
    return episode


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_tweezers_export_and_load_preserve_both_references_and_declared_inputs(tweezers_episode, backend):
    supplied_experience = [{"id": "wrench-reference", "origin_backend": "reference"}]
    rewrite(tweezers_episode, "metadata.json", lambda v: v.update(
        backend=backend, experience=supplied_experience, experience_sha256="declared-input-hash"))
    metadata_before = (tweezers_episode / "metadata.json").read_bytes()
    record = export_experience(tweezers_episode)
    request = json.loads((tweezers_episode / "steps/000/request.json").read_text(encoding="utf8"))

    assert record["status"] == "demonstrated_episode"
    assert record["origin_backend"] == backend
    assert record["applicability"]["task"] == "tweezers_transfer"
    assert record["applicability"]["tool"] == "tweezers"
    assert record["applicability"]["goal"] == request["goal"]
    assert record["applicability"]["protocol_version"] == 1
    assert record["applicability"]["new_tools_verified"] is False
    assert record["applicability"]["other_layouts_verified"] is False
    assert set(record["initial_observation"]) == {"tool", "object", "goal", "palm"}
    for name in ("tool", "object", "goal"):
        assert record["initial_observation"][name] == request["observation"][name]
    assert record["initial_observation"]["palm"] == {
        "position_m": request["observation"]["robot"]["palm_position_m"],
        "quat_wxyz": request["observation"]["robot"]["palm_quat_wxyz"],
    }
    summary = json.loads((tweezers_episode / "summary.json").read_text(encoding="utf8"))
    assert record["outcome"] == {name: summary[name] for name in (
        "success", "gates", "violations", "error", "metrics", "decisions")}
    assert list(record["outcome"]["gates"]) == ["pickup", "pinch_lift", "transport", "release"]
    assert record["trajectory"] == json.loads((tweezers_episode / "actions.json").read_text(encoding="utf8"))
    for name, digest in record["source"]["files_sha256"].items():
        assert digest == hashlib.sha256((tweezers_episode / name).read_bytes()).hexdigest()
    assert (tweezers_episode / "metadata.json").read_bytes() == metadata_before
    assert json.loads(metadata_before)["experience"] == supplied_experience
    assert ("Reference trajectory calibrated by the developer." if backend == "reference"
            else "Online actor episode recorded under its declared inputs.") in record["notes"]
    assert str(tweezers_episode) not in json.dumps(record)
    path = tweezers_episode / "experience.json"
    path.write_text(json.dumps(record), encoding="utf8")
    assert load_experience(path) == record


@pytest.mark.parametrize("stage", ["export", "load"])
@pytest.mark.parametrize("missing", ["object", "goal"])
def test_tweezers_requires_object_and_goal_references(tweezers_episode, stage, missing):
    if stage == "export":
        rewrite(tweezers_episode, "steps/000/request.json", lambda v: v["observation"].pop(missing))
        with pytest.raises(ValueError, match="reference"):
            export_experience(tweezers_episode)
    else:
        record = export_experience(tweezers_episode)
        del record["initial_observation"][missing]
        path = tweezers_episode / "missing-reference.json"
        path.write_text(json.dumps(record), encoding="utf8")
        with pytest.raises(ValueError, match="reference"):
            load_experience(path)


@pytest.mark.parametrize("stage", ["export", "load"])
@pytest.mark.parametrize("gates", [
    {},
    {name: {"passed": True} for name in ("lift", "turn", "withdraw")},
    {name: {"passed": True} for name in ("pickup", "pinch_lift", "transport")},
    {name: {"passed": True} for name in ("pickup", "pinch_lift", "transport", "release", "extra")},
])
def test_tweezers_rejects_empty_wrong_missing_or_extra_gate_sets(tweezers_episode, stage, gates):
    if stage == "export":
        rewrite(tweezers_episode, "summary.json", lambda v: v.update(gates=gates))
        with pytest.raises(ValueError, match="gates"):
            export_experience(tweezers_episode)
    else:
        record = export_experience(tweezers_episode)
        record["outcome"]["gates"] = gates
        path = tweezers_episode / "wrong-gates.json"
        path.write_text(json.dumps(record), encoding="utf8")
        with pytest.raises(ValueError, match="gates"):
            load_experience(path)


@pytest.mark.parametrize("stage", ["export", "load"])
@pytest.mark.parametrize("failed_gate", ["pickup", "pinch_lift", "transport", "release"])
def test_tweezers_rejects_claimed_success_with_any_failed_gate(tweezers_episode, stage, failed_gate):
    if stage == "export":
        rewrite(tweezers_episode, "summary.json", lambda v: v["gates"][failed_gate].update(passed=False))
        with pytest.raises(ValueError, match="physical gates"):
            export_experience(tweezers_episode)
    else:
        record = export_experience(tweezers_episode)
        record["outcome"]["gates"][failed_gate]["passed"] = False
        path = tweezers_episode / "fake-success.json"
        path.write_text(json.dumps(record), encoding="utf8")
        with pytest.raises(ValueError, match="physical gates"):
            load_experience(path)


@pytest.mark.parametrize("stage", ["export", "load"])
def test_tweezers_rejects_claimed_success_with_robot_contact_violation(tweezers_episode, stage):
    if stage == "export":
        rewrite(tweezers_episode, "summary.json", lambda v: v.update(violations=["robot_object_contact"]))
        with pytest.raises(ValueError, match="violations"):
            export_experience(tweezers_episode)
    else:
        record = export_experience(tweezers_episode)
        record["outcome"]["violations"] = ["robot_object_contact"]
        path = tweezers_episode / "violating-success.json"
        path.write_text(json.dumps(record), encoding="utf8")
        with pytest.raises(ValueError, match="violations"):
            load_experience(path)


@pytest.mark.parametrize("stage", ["export", "load"])
def test_unknown_task_with_tweezers_gates_is_rejected(tweezers_episode, stage):
    if stage == "export":
        rewrite(tweezers_episode, "metadata.json", lambda v: v.update(task_id="unknown_tweezers"))
        with pytest.raises(ValueError, match="task"):
            export_experience(tweezers_episode)
    else:
        record = export_experience(tweezers_episode)
        record["applicability"]["task"] = "unknown_tweezers"
        path = tweezers_episode / "unknown-tweezers.json"
        path.write_text(json.dumps(record), encoding="utf8")
        with pytest.raises(ValueError, match="task"):
            load_experience(path)


def test_tweezers_failed_episode_keeps_measurements_and_loads_as_failed(tweezers_episode):
    rewrite(tweezers_episode, "summary.json", lambda v: v.update(
        success=False, violations=["robot_object_contact"], gates={
            name: {"passed": name == "pickup"} for name in ("pickup", "pinch_lift", "transport", "release")}))
    record = export_experience(tweezers_episode)
    assert record["status"] == "failed_episode"
    assert record["outcome"]["success"] is False
    assert record["outcome"]["violations"] == ["robot_object_contact"]
    assert record["outcome"]["metrics"]["max_tool_lift_m"] == 0.03
    path = tweezers_episode / "failed-experience.json"
    path.write_text(json.dumps(record), encoding="utf8")
    assert load_experience(path) == record


def test_legacy_reference_reexport_is_identical_to_saved_experience():
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / "experiments/reference-experience.json").read_text(encoding="utf8"))
    assert export_experience(root / "evidence/reference-003-v2") == expected
