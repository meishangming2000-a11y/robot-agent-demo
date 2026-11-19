"""Static provenance and numeric extraction checks; no physical runtime."""

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from dexhand_agent.experience import export_experience, load_experience
from tools.build_wrench_memory import build_memory, main, validate_skill_memory


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf8")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source(tmp_path):
    episode = tmp_path / "episode"
    identity = "synthetic-native-003"
    target = [0.53, 0.16, 0.29]
    summary = {"episode_id": identity, "backend": "native", "seed": 5, "task_id": "wrench_turn",
               "protocol_version": 1, "success": True, "error": None, "violations": [],
               "gates": {name: {"passed": True} for name in ("lift", "turn", "withdraw")},
               "metrics": {"samples": 1000}, "decisions": 59}
    write(episode / "metadata.json", {"episode_id": identity, "backend": "native", "seed": 5,
          "task_id": "wrench_turn", "observation_mode": "simulator_state"})
    write(episode / "summary.json", summary)
    actions = []
    for seq in range(59):
        error = {6: .009, 7: .005, 8: .002, 9: .001}.get(seq, .003)
        angle = 57.23 if seq == 28 else (69.23 if seq >= 58 else max(0, seq - 14) * 4.0)
        obs = {"mode": "simulator_state", "time_s": seq + 1., "tool": {"name": "wrench",
               "position_m": [.4, .1, .28], "head_center_m": [target[0], target[1], target[2] - error]},
               "robot": {"palm_position_m": [.4, .1, .34], "palm_quat_wxyz": [1, 0, 0, 0]},
               "nut": {"position_m": [.53, .16, .25], "angle_rad": math.radians(angle), "initial_angle_rad": 0.},
               "contacts": {"hand_tool_contacts": ["thumb", "index"], "tool_support_contacts": [],
               "tool_nut_contact": seq >= 13 and seq < 50, "head_nut_contact": seq >= 13 and seq < 50,
               "head_clearance_m": .034 if seq >= 58 else 0.},
               "wrench_feedback": {"head_nut": {"nut_hinge_axis_world": [0, 0, 1]}},
               "task_progress": {"gates": {name: {"passed": seq >= limit}
                   for name, limit in (("lift", 5), ("turn", 27), ("withdraw", 58))}}}
        request = {"episode_id": identity, "request_id": f"{identity}-{seq}", "sequence": seq,
                   "goal": "Public wrench task", "observation": obs}
        action = {"operation": "finish" if seq == 58 else "move", "targets": [],
                  "palm_target_m": [.4, .1, .34], "palm_quat_wxyz": [1, 0, 0, 0],
                  "duration_s": .5, "memory": "synthetic provenance fixture"}
        result = {"status": "finished" if seq == 58 else "completed", "time_s": seq + 1.5}
        actions.append({"sequence": seq, "action": action, "result": result})
        semantic = ("safe_height_align" if seq == 5 else
                    "same_target_tracking_correction" if seq in (6, 7, 8) else
                    "bounded_socket_descent" if 9 <= seq <= 13 else
                    "bounded_positive_turn" if 14 <= seq <= 27 else "hold")
        goal = target if 5 <= seq <= 8 else [.53, .16, .27]
        intent = {"request_id": request["request_id"], "observation_sha256": hashlib.sha256(
                  json.dumps(obs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                  "semantic_operation": semantic, "target_head_m": goal, "decision_origin": "online_model"}
        write(episode / f"steps/{seq:03}/request.json", request)
        write(episode / f"actor/intents/{seq:03}.json", intent)
    write(episode / "actions.json", actions)
    write(episode / "experience.json", export_experience(episode))
    event = {"rule": "working_end_tracking", "trigger_sequence": 6, "correction_sequence": 6,
             "after_sequence": 9, "intended_target_m": target,
             "before_position_error_m": .009, "after_position_error_m": .001,
             "fixed_target_action_sequences": [5, 6, 7, 8]}
    physical = {"matches_saved_result": True, "sampling_500hz_passed": True,
                "recomputed": {key: summary[key] for key in ("task_id", "protocol_version", "success", "gates", "violations", "metrics")}}
    common = {"action_binding": {"passed": True}, "physical": physical, "quality": {"passed": True},
              "accepted_finish": True, "no_runtime_error": True}
    write(episode / "audit-physical-v3.json", {**common, "recovery_accepted": False})
    write(episode / "audit-physical-v3-supplemental.json", {**common, "recovery_accepted": True,
          "recovery_events": [event], "audit_disclosure": "post-execution interface correction"})
    write(episode / "audit-correction-note.json", {"episode_id": identity,
          "change": {"numeric_rules_changed": False, "eligibility_thresholds_changed": False,
                     "action_or_intent_records_changed": False}})
    call_id = "withdraw-case"
    write(episode / f"actor/skill-calls/{call_id}-selection.json",
          {"sequence": 28, "diagnosis": "Selected engineered prior above 55 degree conservative margin"})
    write(episode / f"actor/skill-calls/{call_id}/initial.json",
          {"start_sequence": 28, "policy_state": {"version": "wrench_withdraw_v1"}})
    write(episode / f"actor/skill-calls/{call_id}/handoff.json",
          {"sequence": 58, "replies": 30, "policy_state": {"status": "clearance_check"}})
    counts = {"low_level_commands": 59, "online_action_choices": 29, "skill_selections": 1,
              "online_model_choices": 30, "local_motor_commands": 30, "unclassified_commands": 0}
    inputs = {p.relative_to(episode).as_posix(): digest(p) for p in episode.rglob("*.json")
              if not p.name.startswith("audit-") and p.name != "experience.json"}
    write(episode / "audit-local-v3.json", {"accepted": True, "trace": {"passed": True},
          "sources": {"passed": True}, "budgets": {"low_level_commands_at_most_160": True,
          "online_model_choices_at_most_32": True}, "counts": counts, "input_files_sha256": inputs,
          "calls": [{"skill_call_id": call_id, "invoked": True, "replayed_decisions": 31,
                     "local_motor_commands": 30, "status": "handoff", "errors": []}]})
    acceptance = tmp_path / "acceptance.json"
    write(acceptance, {"stage": 2, "accepted": True, "episode_id": identity, "seed": 5,
          "physical": summary, "counts": counts, "audit_disclosure": "Retrospective alias correction; original strict audit retained.",
          "recovery": {"unique_events": 1}, "source_sha256": {name: digest(episode / name) for name in (
          "summary.json", "experience.json", "audit-physical-v3.json",
          "audit-physical-v3-supplemental.json", "audit-local-v3.json", "audit-correction-note.json")}})
    return episode, acceptance


def rehash_acceptance(episode, acceptance, name):
    data = read(acceptance)
    data["source_sha256"][name] = digest(episode / name)
    write(acceptance, data)


def test_preserves_original_record_and_extracts_relative_single_episode_lessons(source, tmp_path):
    episode, acceptance = source
    original = load_experience(episode / "experience.json")
    memory = build_memory(episode, acceptance)
    skill = memory.pop("skill_memory")
    assert memory == original
    assert [lesson["id"] for lesson in skill["lessons"]] == [
        "fixed_head_target", "incremental_seating_turn", "engineered_withdraw_selection"]
    fixed = skill["lessons"][0]
    assert fixed["working_end_target"]["frame"] == "nut_axis"
    assert fixed["working_end_target"]["axial_offset_m"] == pytest.approx(.04)
    assert fixed["observed"]["before_error_m"] == pytest.approx(.009)
    assert fixed["observed"]["after_error_m"] == pytest.approx(.001)
    assert skill["acceptance_provenance"]["source_files_sha256"] == original["source"]["files_sha256"]
    assert skill["acceptance_provenance"]["original_strict_recovery_accepted"] is False
    assert "Retrospective" in skill["acceptance_provenance"]["audit_disclosure"]
    complete = {**memory, "skill_memory": skill}
    output = tmp_path / "memory.json"
    write(output, complete)
    assert load_experience(output) == complete
    assert len(complete["trajectory"]) == 59


@pytest.mark.parametrize("name", ["metadata.json", "actions.json", "steps/006/request.json"])
def test_source_hash_tamper_is_rejected(source, name):
    episode, acceptance = source
    (episode / name).write_bytes((episode / name).read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash|SHA"):
        build_memory(episode, acceptance)


def test_changed_export_cannot_relabel_the_original_source_episode(source):
    episode, acceptance = source
    record = read(episode / "experience.json")
    record["notes"] += " edited"
    write(episode / "experience.json", record)
    rehash_acceptance(episode, acceptance, "experience.json")
    with pytest.raises(ValueError, match="export"):
        build_memory(episode, acceptance)


@pytest.mark.parametrize("name,field", [
    ("audit-local-v3.json", "accepted"),
    ("audit-physical-v3-supplemental.json", "recovery_accepted"),
])
def test_unaccepted_source_audits_fail_even_when_their_hashes_are_updated(source, name, field):
    episode, acceptance = source
    data = read(episode / name)
    data[field] = False
    write(episode / name, data)
    rehash_acceptance(episode, acceptance, name)
    with pytest.raises(ValueError, match="accepted"):
        build_memory(episode, acceptance)


def test_stage_acceptance_and_source_identity_are_required(source):
    episode, acceptance = source
    data = read(acceptance)
    data["episode_id"] = "different"
    write(acceptance, data)
    with pytest.raises(ValueError, match="episode"):
        build_memory(episode, acceptance)


@pytest.mark.parametrize("offset", [float("nan"), -0.1, 0.1001])
def test_invalid_relative_memory_goal_is_rejected(source, offset):
    memory = build_memory(*source)
    memory["skill_memory"]["lessons"][0]["working_end_target"]["axial_offset_m"] = offset
    with pytest.raises(ValueError):
        validate_skill_memory(memory)


def test_cli_refuses_existing_output_and_does_not_overwrite(source, tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("preserve", encoding="utf8")
    with pytest.raises(FileExistsError):
        main([str(source[0]), str(source[1]), str(output)])
    assert output.read_text(encoding="utf8") == "preserve"

