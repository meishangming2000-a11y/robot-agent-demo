import gzip
import hashlib
import json
import math
from pathlib import Path

import pytest

from dexhand_agent.wrench_evaluate import WrenchEvaluator
from tools.audit_wrench_autonomy import audit_episode, main


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf8")).hexdigest()


@pytest.fixture
def recorded(tmp_path):
    """Synthetic continuous measurements and native mailbox evidence, no simulator."""
    root = tmp_path / "episode"
    protocol = tmp_path / "protocol.json"
    source = Path(__file__).parents[1] / "experiments/wrench-autonomy-protocol.json"
    save(protocol, read(source))
    samples = []
    evaluator = WrenchEvaluator(0.245, 0.0)
    for tick in range(2001):
        time_s = tick / 500
        sample = {
            "time_s": time_s, "wrench_pos_m": [0.418, 0.16, 0.275],
            "wrench_quat_wxyz": [1, 0, 0, 0],
            "hand_tool_contacts": ["thumb", "index"], "tool_support_contacts": [],
            "head_nut_contact": 2.3 <= time_s <= 2.802,
            "tool_nut_contact": 2.3 <= time_s <= 2.802,
            "robot_nut_contact": False, "nut_angle_rad": math.pi / 4 if time_s >= 2.3 else 0,
            "head_clearance_m": 0.04, "max_contact_force_n": 0.5,
            "max_penetration_m": 0.0001,
        }
        samples.append(sample)
        evaluator.update(sample)
    save(root / "summary.json", {**evaluator.summary(), "backend": "native", "error": None,
                                  "episode_id": "synthetic", "decisions": 4})
    save(root / "metadata.json", {"episode_id": "synthetic", "backend": "native"})
    (root / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples), encoding="utf8")
    history = []
    for seq, (time_s, x, operation) in enumerate([
        (2.0, 0.53, "align"), (2.1, 0.536, "correct_alignment"),
        (2.2, 0.531, "turn"), (4.0, 0.531, "finish"),
    ]):
        obs = {"time_s": time_s,
               "tool": {"head_center_m": [x, 0.16, 0.275]},
               "nut": {"position_m": [0.53, 0.16, 0.245], "top_m": 0.265,
                       "angle_rad": 0.0}, "contacts": samples[round(time_s * 500)]}
        request = {"request_id": f"synthetic-{seq}", "episode_id": "synthetic",
                   "sequence": seq, "observation": obs}
        action = {"operation": "finish" if operation == "finish" else "move",
                  "duration_s": 0.1, "memory": operation}
        result = {"status": "finished" if seq == 3 else "completed",
                  "time_s": [2.1, 2.2, 4.0, 4.0][seq]}
        history.append({"sequence": seq, "action": action, "result": result})
        save(root / f"steps/{seq:03}/request.json", request)
        save(root / f"steps/{seq:03}/action.json", action)
        save(root / f"steps/{seq:03}/result.json", result)
        save(root / f"actor/steps/{seq:03}/request.json", {**request, "images": ["fresh.png"]})
        save(root / f"actor/steps/{seq:03}/reply.json", {"request_id": request["request_id"], "action": action})
        save(root / f"actor/intents/{seq:03}.json", {
            "request_id": request["request_id"], "observation_sha256": digest(obs),
            "semantic_operation": operation, "target_head_m": [0.53, 0.16, 0.275],
            "diagnosis": "measured head relation", "correction_trigger":
                {"type": "my alignment spelling", "trigger_sequence": 1, "before": {"claimed": 999}}
                if seq == 1 else None,
        })
    save(root / "actions.json", history)
    return root, protocol


def test_alignment_recovery_uses_measurements_and_accepts_unfrozen_type_spelling(recorded):
    result = audit_episode(*recorded)
    assert result["physical"]["matches_saved_result"]
    assert result["action_binding"]["passed"]
    assert result["recovery_accepted"]
    event = result["recovery_events"][0]
    assert event["rule"] == "alignment_after_attempt"
    assert event["before_xy_error_m"] == pytest.approx(0.006)
    assert event["after_xy_error_m"] == pytest.approx(0.001)


def test_v3_protocol_preserves_existing_physical_recovery_audit(recorded):
    root, protocol = recorded
    source = Path(__file__).parents[1] / "experiments/wrench-autonomy-protocol-v3.json"
    save(protocol, read(source))
    result = audit_episode(root, protocol)
    assert result["physical"]["matches_saved_result"]
    assert result["action_binding"]["passed"]
    assert result["recovery_accepted"]
    assert result["recovery_events"][0]["rule"] == "alignment_after_attempt"


@pytest.mark.parametrize("relative,field,value", [
    ("steps/001/action.json", "memory", "root replacement"),
    ("actor/steps/001/reply.json", "request_id", "stale"),
    ("actor/intents/001.json", "observation_sha256", "0" * 64),
])
def test_action_or_observation_binding_tampering_fails(recorded, relative, field, value):
    root, protocol = recorded
    item = read(root / relative)
    item[field] = value
    save(root / relative, item)
    result = audit_episode(root, protocol)
    assert not result["action_binding"]["passed"]
    assert not result["recovery_accepted"]


def test_missing_intent_cannot_establish_native_recovery(recorded):
    root, protocol = recorded
    (root / "actor/intents/001.json").unlink()
    result = audit_episode(root, protocol)
    assert not result["action_binding"]["passed"]
    assert not result["recovery_accepted"]


def test_quality_violation_is_checked_even_if_evaluator_successfully_completes(recorded):
    root, protocol = recorded
    samples = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
    samples[-1]["max_contact_force_n"] = 25.001
    (root / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    stored = read(root / "summary.json")
    stored["metrics"]["max_contact_force_n"] = 25.001
    save(root / "summary.json", stored)
    result = audit_episode(root, protocol)
    assert result["physical"]["matches_saved_result"]
    assert result["physical"]["recomputed"]["success"]
    assert not result["quality"]["passed"]
    assert not result["recovery_accepted"]


def test_summary_fields_are_compared_exactly(recorded):
    root, protocol = recorded
    stored = read(root / "summary.json")
    stored["metrics"]["max_lift_m"] = 123
    save(root / "summary.json", stored)
    result = audit_episode(root, protocol)
    assert not result["physical"]["matches_saved_result"]
    assert not result["recovery_accepted"]


@pytest.mark.parametrize("change", ["penetration", "robot_nut", "runtime_error", "no_finish"])
def test_other_hard_failures_never_accept_recovery(recorded, change):
    root, protocol = recorded
    if change in {"penetration", "robot_nut"}:
        path = root / "samples.jsonl"
        samples = [json.loads(line) for line in path.read_text().splitlines()]
        samples[-1]["max_penetration_m" if change == "penetration" else "robot_nut_contact"] = (
            0.00201 if change == "penetration" else True)
        evaluator = WrenchEvaluator(0.245)
        for sample in samples:
            evaluator.update(sample)
        save(root / "summary.json", {**read(root / "summary.json"), **evaluator.summary()})
        path.write_text("".join(json.dumps(s) + "\n" for s in samples))
        rewrite_observation(root, 3, lambda obs: obs.update(contacts=samples[-1]))
    elif change == "runtime_error":
        save(root / "summary.json", {**read(root / "summary.json"), "error": "recorded failure"})
    else:
        history = read(root / "actions.json")
        history[-1]["result"]["status"] = "rejected"
        save(root / "actions.json", history)
        save(root / "steps/003/result.json", history[-1]["result"])
    assert not audit_episode(root, protocol)["recovery_accepted"]


@pytest.mark.parametrize("change", ["initial_approach", "rejected_only", "no_improvement"])
def test_planned_approach_rejection_and_ineffective_correction_do_not_count(recorded, change):
    root, protocol = recorded
    if change == "initial_approach":
        item = read(root / "actor/intents/000.json")
        item["semantic_operation"] = "approach"
        save(root / "actor/intents/000.json", item)
    elif change == "rejected_only":
        history = read(root / "actions.json")
        history[0]["result"]["status"] = "rejected"
        save(root / "actions.json", history)
        save(root / "steps/000/result.json", history[0]["result"])
    else:
        for relative in ("steps/002/request.json", "actor/steps/002/request.json",
                         "steps/003/request.json", "actor/steps/003/request.json"):
            item = read(root / relative)
            item["observation"]["tool"]["head_center_m"][0] = 0.535
            save(root / relative, item)
        for seq in (2, 3):
            intent = read(root / f"actor/intents/{seq:03}.json")
            intent["observation_sha256"] = digest(read(root / f"steps/{seq:03}/request.json")["observation"])
            save(root / f"actor/intents/{seq:03}.json", intent)
    assert not audit_episode(root, protocol)["recovery_accepted"]


def test_gzip_samples_and_output_refuse_overwrite(recorded, tmp_path, monkeypatch):
    root, protocol = recorded
    with gzip.open(root / "samples.jsonl.gz", "wb") as stream:
        stream.write((root / "samples.jsonl").read_bytes())
    (root / "samples.jsonl").unlink()
    output = tmp_path / "audit.json"
    monkeypatch.setattr("sys.argv", ["audit", str(root), str(protocol), "--output", str(output)])
    assert main() == 0
    assert read(output)["recovery_accepted"]
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        main()
    assert output.read_bytes() == before


def rewrite_observation(root, seq, change):
    for parent in ("steps", "actor/steps"):
        path = root / f"{parent}/{seq:03}/request.json"
        item = read(path)
        change(item["observation"])
        save(path, item)
    path = root / f"actor/intents/{seq:03}.json"
    intent = read(path)
    intent["observation_sha256"] = digest(item["observation"])
    save(path, intent)


def test_blocked_seating_requires_real_contact_and_later_contact_qualified_turn(recorded):
    root, protocol = recorded
    path = root / "actor/intents/000.json"
    intent = read(path)
    intent.update(semantic_operation="insert", target_head_m=[0.53, 0.16, 0.245])
    save(path, intent)
    samples = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
    samples[1050].update(head_nut_contact=True, tool_nut_contact=True)
    evaluator = WrenchEvaluator(0.245)
    for sample in samples:
        evaluator.update(sample)
    stored = read(root / "summary.json")
    save(root / "summary.json", {**stored, **evaluator.summary()})
    (root / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    rewrite_observation(root, 1, lambda obs: obs.update(contacts=samples[1050]))
    rewrite_observation(root, 2, lambda obs: obs["tool"].update(head_center_m=[0.536, 0.16, 0.271]))
    result = audit_episode(root, protocol)
    assert result["recovery_accepted"]
    assert any(event["rule"] == "blocked_seating" for event in result["recovery_events"])


def test_alignment_after_window_is_frozen(recorded):
    root, protocol = recorded
    value = read(protocol)
    value["recovery_acceptance"]["numeric_rules"]["alignment_after_attempt"]["after_window_maximum_decisions"] = 0
    save(protocol, value)
    assert not audit_episode(root, protocol)["recovery_accepted"]


def test_post_task_cosmetic_correction_is_not_physical_recovery(recorded):
    root, protocol = recorded
    samples = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
    times = [3.8, 3.9, 3.95, 4.0]
    for seq, time_s in enumerate(times):
        rewrite_observation(root, seq, lambda obs, t=time_s: obs.update(
            time_s=t, contacts=samples[round(t * 500)]))
    history = read(root / "actions.json")
    for seq, row in enumerate(history):
        row["result"]["time_s"] = times[min(seq + 1, 3)]
        save(root / f"steps/{seq:03}/result.json", row["result"])
    save(root / "actions.json", history)
    assert not audit_episode(root, protocol)["recovery_accepted"]


@pytest.fixture
def tracking(recorded):
    """A separate high, unsupported working-end tracking correction under v2."""
    root, protocol = recorded
    source = Path(__file__).parents[1] / "experiments/wrench-autonomy-protocol-v2.json"
    save(protocol, read(source))
    samples = []
    evaluator = WrenchEvaluator(0.245)
    for tick in range(2501):
        time_s = tick / 500
        sample = {
            "time_s": time_s, "wrench_pos_m": [0.418, 0.16, 0.31],
            "wrench_quat_wxyz": [1, 0, 0, 0],
            "hand_tool_contacts": ["thumb", "index"], "tool_support_contacts": [],
            "head_nut_contact": 3 <= time_s <= 3.502,
            "tool_nut_contact": 3 <= time_s <= 3.502,
            "robot_nut_contact": False, "nut_angle_rad": math.pi / 4 if time_s >= 3 else 0,
            "head_clearance_m": 0.04, "max_contact_force_n": 0.5,
            "max_penetration_m": 0.0001,
        }
        samples.append(sample)
        evaluator.update(sample)
    save(root / "summary.json", {**read(root / "summary.json"), **evaluator.summary()})
    (root / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    times = [2.0, 2.6, 2.7, 5.0]
    heads = [[0.53, 0.16, 0.31], [0.536, 0.16, 0.315],
             [0.5305, 0.16, 0.311], [0.53, 0.16, 0.31]]
    history = read(root / "actions.json")
    for seq in range(4):
        rewrite_observation(root, seq, lambda obs, i=seq: obs.update(
            time_s=times[i], contacts=samples[round(times[i] * 500)],
            tool={"head_center_m": heads[i]}))
        row = history[seq]
        row["action"]["duration_s"] = [0.6, 0.1, 2.3, 0.1][seq]
        row["result"]["time_s"] = times[min(seq + 1, 3)]
        save(root / f"steps/{seq:03}/action.json", row["action"])
        save(root / f"steps/{seq:03}/result.json", row["result"])
        save(root / f"actor/steps/{seq:03}/reply.json", {
            "request_id": f"synthetic-{seq}", "action": row["action"]})
        path = root / f"actor/intents/{seq:03}.json"
        intent = read(path)
        intent["target_head_m"] = [0.53, 0.16, 0.31] if seq < 2 else [0.53, 0.16, 0.26]
        if seq == 0:
            intent["semantic_operation"] = "level"
        save(path, intent)
    save(root / "actions.json", history)
    return root, protocol


def test_v2_tracking_accepts_same_target_3d_correction(tracking):
    result = audit_episode(*tracking)
    assert result["audit_version"] == 3
    assert result["recovery_accepted"]
    event = result["recovery_events"][0]
    assert event["rule"] == "working_end_tracking"
    assert event["before_position_error_m"] == pytest.approx(math.hypot(0.006, 0.005))
    assert event["after_position_error_m"] == pytest.approx(math.hypot(0.0005, 0.001))
    assert event["intended_target_m"] == [0.53, 0.16, 0.31]


def test_v1_does_not_inherit_tracking_rule(tracking):
    root, protocol = tracking
    source = Path(__file__).parents[1] / "experiments/wrench-autonomy-protocol.json"
    save(protocol, read(source))
    result = audit_episode(root, protocol)
    assert not result["recovery_events"]
    assert not result["recovery_accepted"]


@pytest.mark.parametrize("change", ["moving_goal", "approach", "lift", "short_motion", "late_window"])
def test_v2_tracking_excludes_changed_target_and_ineligible_attempts(tracking, change):
    root, protocol = tracking
    if change == "moving_goal":
        path = root / "actor/intents/001.json"
        intent = read(path)
        intent["target_head_m"][2] -= 0.001
        save(path, intent)
    elif change in {"approach", "lift"}:
        path = root / "actor/intents/000.json"
        intent = read(path)
        intent["semantic_operation"] = change
        save(path, intent)
    elif change == "short_motion":
        history = read(root / "actions.json")
        history[0]["action"]["duration_s"] = 0.49
        save(root / "actions.json", history)
        save(root / "steps/000/action.json", history[0]["action"])
        save(root / "actor/steps/000/reply.json", {
            "request_id": "synthetic-0", "action": history[0]["action"]})
    else:
        value = read(protocol)
        value["recovery_acceptance"]["numeric_rules"]["working_end_tracking"]["after_window_maximum_decisions"] = 0
        save(protocol, value)
    result = audit_episode(root, protocol)
    assert result["action_binding"]["passed"]
    assert not result["recovery_accepted"]


@pytest.mark.parametrize("change", ["before_lift", "low_target", "far_target", "one_hand", "supported"])
def test_v2_tracking_enforces_postlift_retention_and_safe_nearby_target(tracking, change):
    root, protocol = tracking
    samples_path = root / "samples.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
    if change == "before_lift":
        rewrite_observation(root, 0, lambda obs: obs.update(time_s=1.9, contacts=samples[950]))
    elif change in {"low_target", "far_target"}:
        target = [0.53, 0.16, 0.27] if change == "low_target" else [0.561, 0.16, 0.31]
        for seq in (0, 1):
            path = root / f"actor/intents/{seq:03}.json"
            save(path, {**read(path), "target_head_m": target})
        rewrite_observation(root, 1, lambda obs: obs["tool"].update(
            head_center_m=[target[0] + 0.006, target[1], target[2] + 0.005]))
        rewrite_observation(root, 2, lambda obs: obs["tool"].update(
            head_center_m=[target[0] + 0.0005, target[1], target[2] + 0.001]))
    else:
        samples[1300]["hand_tool_contacts" if change == "one_hand" else "tool_support_contacts"] = (
            ["thumb"] if change == "one_hand" else ["fixture"])
        evaluator = WrenchEvaluator(0.245)
        for sample in samples:
            evaluator.update(sample)
        save(root / "summary.json", {**read(root / "summary.json"), **evaluator.summary()})
        samples_path.write_text("".join(json.dumps(s) + "\n" for s in samples))
        rewrite_observation(root, 1, lambda obs: obs.update(contacts=samples[1300]))
    result = audit_episode(root, protocol)
    assert result["action_binding"]["passed"]
    assert result["physical"]["matches_saved_result"]
    assert not result["recovery_accepted"]


@pytest.mark.parametrize("target_delta,accepted", [(0.5e-6, True), (2e-6, False)])
def test_v2_tracking_uses_frozen_one_micrometre_target_tolerance(tracking, target_delta, accepted):
    root, protocol = tracking
    path = root / "actor/intents/001.json"
    intent = read(path)
    intent["target_head_m"][0] += target_delta
    save(path, intent)
    assert audit_episode(root, protocol)["recovery_accepted"] is accepted


def test_exact_safe_height_align_alias_accepts_numerically_eligible_tracking(tracking):
    root, protocol = tracking
    path = root / "actor/intents/000.json"
    save(path, {**read(path), "semantic_operation": "safe_height_align"})
    result = audit_episode(root, protocol)
    assert result["recovery_accepted"]
    assert result["recovery_events"][0]["rule"] == "working_end_tracking"
    assert result["semantic_operation_aliases"] == {"safe_height_align": "align"}
    assert result["semantic_operation_alias_scope"] == "working_end_tracking eligibility role only"
    assert "for native003" in result["audit_disclosure"]
    assert "numeric rules unchanged" in result["audit_disclosure"]
    assert "source freeze" in result["audit_disclosure"]


def test_safe_height_alias_does_not_accept_numeric_tracking_failure(tracking):
    root, protocol = tracking
    path = root / "actor/intents/000.json"
    save(path, {**read(path), "semantic_operation": "safe_height_align"})
    rewrite_observation(root, 2, lambda obs: obs["tool"].update(head_center_m=[0.535, 0.16, 0.315]))
    result = audit_episode(root, protocol)
    assert result["action_binding"]["passed"]
    assert result["physical"]["recomputed"]["success"]
    assert not result["recovery_accepted"]
    assert not result["recovery_events"]


def test_unknown_alignment_spelling_remains_ineligible(tracking):
    root, protocol = tracking
    path = root / "actor/intents/000.json"
    save(path, {**read(path), "semantic_operation": "safe_height_alignment"})
    result = audit_episode(root, protocol)
    assert result["action_binding"]["passed"]
    assert not result["recovery_accepted"]


def test_alias_is_not_applied_to_other_recovery_roles(recorded):
    root, protocol = recorded
    path = root / "actor/intents/000.json"
    save(path, {**read(path), "semantic_operation": "safe_height_align"})
    result = audit_episode(root, protocol)
    assert result["action_binding"]["passed"]
    assert result["physical"]["recomputed"]["success"]
    assert not result["recovery_accepted"]
