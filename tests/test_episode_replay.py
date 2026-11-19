import gzip
import json

import pytest

from tools.evaluate_episode import main, replay_episode


TASKS = [None, "hammer_button", "wrench_turn", "tweezers_transfer"]


def episode_data(task_id):
    sample = {
        "time_s": 0.0,
        "hand_tool_contacts": ["thumb", "index"],
        "tool_support_contacts": [],
        "head_clearance_m": 0.03125,
        "max_contact_force_n": 0.5,
        "max_penetration_m": 0.0001,
    }
    metrics = {"samples": 2, "last_time_s": 0.01}
    if task_id in (None, "hammer_button"):
        sample.update(hammer_pos_m=[0.0, 0.0, 0.375], button_q_m=0.0,
                      tool_button_contact=False, head_button_contact=False,
                      robot_button_contact=False)
        metrics.update(initial_height_m=0.25, max_lift_m=0.125,
                       max_button_depth_m=0.0, max_head_clearance_m=0.03125,
                       first_robot_button_contact_s=None)
        durations = {"lift": 2.0, "press": 0.5, "withdraw": 1.0}
    elif task_id == "wrench_turn":
        sample.update(wrench_pos_m=[0.0, 0.0, 0.375], nut_angle_rad=0.5,
                      tool_nut_contact=False, head_nut_contact=False,
                      robot_nut_contact=False)
        metrics.update(initial_height_m=0.25, initial_nut_angle_rad=0.25,
                       max_lift_m=0.125, max_nut_turn_rad=0.25,
                       max_head_clearance_m=0.03125, max_contact_force_n=0.5,
                       max_penetration_m=0.0001, first_robot_nut_contact_s=None)
        durations = {"lift": 2.0, "turn": 0.5, "withdraw": 1.0}
    else:
        sample.update(tool_pos_m=[0.0, 0.0, 0.375], object_pos_m=[0.0, 0.0, 0.25],
                      tool_object_contact=False, object_tip_contacts=[],
                      object_other_tool_contact=False, object_fixture_contacts=[],
                      object_goal_support=False, robot_object_contact=False,
                      object_to_goal_xy_m=0.125, object_bottom_clearance_m=0.0625,
                      object_speed_m_s=0.0, tip_object_clearance_m=0.03125)
        metrics.update(initial_tool_height_m=0.25, initial_object_height_m=0.125,
                       max_tool_lift_m=0.125, max_object_lift_m=0.125,
                       min_goal_distance_m=0.125, max_object_bottom_clearance_m=0.0625,
                       max_tip_object_clearance_m=0.03125, max_contact_force_n=0.5,
                       max_penetration_m=0.0001, first_robot_object_contact_s=None)
        durations = {"pickup": 2.0, "pinch_lift": 1.0, "transport": 0.5, "release": 1.0}
    first_gate = next(iter(durations))
    summary = {
        "protocol_version": 2 if task_id in (None, "hammer_button") else 1,
        "success": False,
        "gates": {
            name: {"passed": False, "passed_at_s": None,
                   "required_continuous_s": duration,
                   "best_continuous_s": 0.01 if name == first_gate else 0.0}
            for name, duration in durations.items()
        },
        "violations": [],
        "metrics": metrics,
    }
    if task_id is not None:
        summary["task_id"] = task_id
    return summary, [sample, {**sample, "time_s": 0.01}]


def save_episode(path, summary, samples, compressed=False):
    path.mkdir(exist_ok=True)
    (path / "summary.json").write_text(
        json.dumps({**summary, "backend": "离线测量"}, ensure_ascii=False), encoding="utf8")
    name = "samples.jsonl.gz" if compressed else "samples.jsonl"
    stream = gzip.open(path / name, "wt", encoding="utf8") if compressed else (
        (path / name).open("w", encoding="utf8"))
    with stream:
        for sample in samples:
            stream.write(json.dumps(sample) + "\n")


@pytest.mark.parametrize("task_id", TASKS)
@pytest.mark.parametrize("compressed", [False, True])
def test_replay_dispatches_and_exactly_recomputes_saved_measurements(tmp_path, task_id, compressed):
    summary, samples = episode_data(task_id)
    save_episode(tmp_path, summary, samples, compressed)

    assert replay_episode(tmp_path) == {"matches_saved_result": True, **summary}


@pytest.mark.parametrize("task_id", TASKS)
@pytest.mark.parametrize("field", ["success", "protocol_version", "gates", "violations", "metrics"])
def test_tampered_result_is_detected(tmp_path, task_id, field):
    summary, samples = episode_data(task_id)
    if field == "success":
        summary[field] = True
    elif field == "protocol_version":
        summary[field] += 1
    elif field == "gates":
        summary[field][next(iter(summary[field]))]["best_continuous_s"] += 1e-12
    elif field == "violations":
        summary[field] = ["invented_contact"]
    else:
        summary[field]["samples"] += 1
    save_episode(tmp_path, summary, samples)

    assert replay_episode(tmp_path)["matches_saved_result"] is False


@pytest.mark.parametrize("task_id,metric", [
    (None, "initial_height_m"),
    ("hammer_button", "initial_height_m"),
    ("wrench_turn", "initial_height_m"),
    ("wrench_turn", "initial_nut_angle_rad"),
    ("tweezers_transfer", "initial_tool_height_m"),
    ("tweezers_transfer", "initial_object_height_m"),
])
def test_missing_initial_measurements_are_rejected_clearly(tmp_path, task_id, metric):
    summary, samples = episode_data(task_id)
    del summary["metrics"][metric]
    save_episode(tmp_path, summary, samples)

    with pytest.raises(ValueError, match=rf"metrics\.{metric}"):
        replay_episode(tmp_path)


def test_unknown_task_is_rejected_before_replaying(tmp_path):
    save_episode(tmp_path, {"task_id": "unknown", "metrics": {}}, [])

    with pytest.raises(ValueError, match="[Uu]nknown task.*unknown"):
        replay_episode(tmp_path)


@pytest.mark.parametrize("tampered,exit_code", [(False, 0), (True, 1)])
def test_cli_reports_match_and_exit_status(tmp_path, monkeypatch, capsys, tampered, exit_code):
    summary, samples = episode_data("tweezers_transfer")
    summary["success"] = tampered
    save_episode(tmp_path, summary, samples, compressed=True)
    monkeypatch.setattr("sys.argv", ["evaluate_episode.py", str(tmp_path)])

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == exit_code
    assert json.loads(capsys.readouterr().out)["matches_saved_result"] is not tampered


def test_cli_reports_invalid_summary_without_a_traceback(tmp_path, monkeypatch, capsys):
    save_episode(tmp_path, {"task_id": "unknown", "metrics": {}}, [])
    monkeypatch.setattr("sys.argv", ["evaluate_episode.py", str(tmp_path)])

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "unknown" in message
    assert "Traceback" not in message
