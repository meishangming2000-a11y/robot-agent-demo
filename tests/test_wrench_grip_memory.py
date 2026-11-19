"""Memory augmentation checks using saved samples only; no physical runtime."""
import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from dexhand_agent.experience import load_experience
from dexhand_agent.wrench_evaluate import WrenchEvaluator
from tools import build_wrench_grip_memory as builder


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf8")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rehash(source):
    _, d6, proof = source
    data = read(proof)
    data["source_sha256"] = {p.name: sha(p) for p in d6.iterdir()}
    write(proof, data)
    return sha(proof)


@pytest.fixture
def source(tmp_path, monkeypatch):
    original = Path(__file__).resolve().parents[1] / "experiences/wrench-native-v1.json"
    v1 = tmp_path / "experiences/v1.json"
    v1.parent.mkdir()
    v1.write_bytes(original.read_bytes())
    record = load_experience(v1)
    d6 = tmp_path / "runs/thumb-006"
    d6.mkdir(parents=True)
    source_actions = copy.deepcopy(record["trajectory"][:4])
    actions = copy.deepcopy(source_actions)
    actions[2]["action"]["targets"][9]["value"] += .02
    actions.append({"sequence": 4, "action": {"operation": "hold", "targets": [],
                    "palm_target_m": None, "palm_quat_wxyz": None, "duration_s": 2., "memory": "hold"},
                    "result": {"status": "completed", "time_s": 9.}})
    write(d6 / "actions-source.json", source_actions)
    write(d6 / "executed-actions-declaration.json", actions[:4])
    (d6 / "actions.jsonl").write_text("\n".join(json.dumps({**a, "event": event})
        for a in actions for event in ("before_execute", "after_execute")) + "\n", encoding="utf8")
    targets = actions[2]["action"]["targets"]
    ids = {t["actuator"]: i for i, t in enumerate(targets)}
    write(d6 / "actuator-ids.json", ids)
    evaluator = WrenchEvaluator(.24)
    with gzip.open(d6 / "samples.jsonl.gz", "wt", encoding="utf8") as f:
        for i in range(221):
            sample = {"time_s": 1 + i * .01, "wrench_pos_m": [.4, 0, .27],
                      "hand_tool_contacts": ["thumb", "index"], "tool_support_contacts": [],
                      "head_nut_contact": False, "tool_nut_contact": False, "robot_nut_contact": False,
                      "nut_angle_rad": 0., "head_clearance_m": .2, "max_contact_force_n": 4.,
                      "max_penetration_m": .0004}
            evaluator.update(sample)
            f.write(json.dumps({"sequence": 3, "physical_sample": sample,
                               "ctrl": [t["value"] for t in targets]}) + "\n")
    quality = {"all_normal_at_most25N": True, "all_global_penetration_at_most2mm": True,
               "zero_robot_nut_contacts": True}
    physical = evaluator.summary()
    summary = {**physical, "native_agent": False, "held_out": False, "robot_calls": 5, "error": None,
               "original_lift_completed": True, "postlift_hold_completed": True,
               "quality_pass": True, "pickup_evidence_pass": True, "exact_prefix_match": True,
               "protected_inputs_unchanged": True}
    write(d6 / "summary.json", summary)
    write(d6 / "evaluator-recomputed.json", {"exact_match": True, "evaluator": physical, "quality": quality})
    difference = {"exact_match": False, "fields_compared": 31, "maximum_numeric_abs_difference": .02,
                  "differences": [{"field": "/targets/9/value", "expected": targets[9]["value"] - .02,
                                   "actual": targets[9]["value"]}]}
    write(d6 / "one-parameter-difference.json", difference)
    write(d6 / "analysis.json", {
        "engineering_pickup_evidence": {"passed": True, "independently_recomputed_from_all_raw_samples": True,
             "postlift_hold_entirely_fixture_and_nut_free": True, "full_task_success": False},
        "protocol_actions": {"original_approach0_descent1_lift3_actionexact": {"0": True, "1": True, "3": True},
             "changed_grasp_equals_original_with_only_declared_target_replaced": True,
             "no_calibration_or_prelift_screen_or_extra_hold": True,
             "THJ4_command_during_lift_and_hold": targets[9]["value"],
             "THJ4_command_max_abs_deviation_during_lift_and_hold": 0.},
        "quality_independent_file_recheck": {**quality, "passed": True}})
    proof = tmp_path / "evidence/wrench-grip-v2/development-pickup.json"
    write(proof, {"kind": "engineering_partial_skill_evidence", "accepted_for_pickup_only": True,
          "full_task_validated": False, "native_agent": False, "held_out": False,
          "source": d6.relative_to(tmp_path).as_posix(), "physical": summary,
          "one_changed_parameter": {"actuator": "rh_A_THJ4", "original": targets[9]["value"] - .02,
                                    "candidate": targets[9]["value"], "range": [0., 1.22173]}})
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "PICKUP_PROOF_SHA256", rehash((v1, d6, proof)))
    return v1, d6, proof


def test_adds_one_partial_lesson_without_changing_native_record(source, tmp_path):
    original = load_experience(source[0])
    result = builder.build_grip_memory(*source)
    assert {k: v for k, v in result.items() if k != "skill_memory"} == {
        k: v for k, v in original.items() if k != "skill_memory"}
    before, after = original["skill_memory"], result["skill_memory"]
    assert after["lessons"][:3] == before["lessons"]
    assert after["acceptance_provenance"] == before["acceptance_provenance"]
    assert after["acquisition"] == before["acquisition"]
    lesson = after["lessons"][3]
    assert lesson["id"] == "calibrated_grasp_v2"
    targets = copy.deepcopy(original["trajectory"][2]["action"]["targets"])
    targets[9]["value"] += .02
    assert lesson["calibrated_targets"] == targets
    assert lesson["validation"]["kind"] == "engineering_pickup_only"
    assert lesson["validation"]["full_task_validated"] is False
    assert lesson["validation"]["native_agent"] is False
    assert after["parent_file_sha256"] == sha(source[0])
    output = tmp_path / "v2.json"
    write(output, result)
    assert load_experience(output) == result
    assert len(result["trajectory"]) == 59


@pytest.mark.parametrize("which", ["v1", "proof", "actions.jsonl"])
def test_immutable_source_hash_tamper_rejected(source, which):
    path = source[0] if which == "v1" else source[2] if which == "proof" else source[1] / which
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA"):
        builder.build_grip_memory(*source)


@pytest.mark.parametrize("field,value", [("full_task_validated", True), ("native_agent", True),
                                         ("accepted_for_pickup_only", False)])
def test_engineering_pickup_cannot_be_relabelled_full_or_native(source, monkeypatch, field, value):
    data = read(source[2]); data[field] = value; write(source[2], data)
    monkeypatch.setattr(builder, "PICKUP_PROOF_SHA256", sha(source[2]))
    with pytest.raises(ValueError, match="partial"):
        builder.build_grip_memory(*source)


def test_second_grasp_target_change_rejected_even_with_rehashed_sources(source, monkeypatch):
    path = source[1] / "actions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if row["sequence"] == 2: row["action"]["targets"][0]["value"] += .01
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf8")
    monkeypatch.setattr(builder, "PICKUP_PROOF_SHA256", rehash(source))
    with pytest.raises(ValueError, match="target|parameter|declaration"):
        builder.build_grip_memory(*source)


def test_saved_ctrl_drift_rejected_even_with_rehashed_sources(source, monkeypatch):
    path = source[1] / "samples.jsonl.gz"
    with gzip.open(path, "rt") as f: rows = [json.loads(line) for line in f]
    rows[-1]["ctrl"][0] += .01
    with gzip.open(path, "wt", encoding="utf8") as f:
        f.write("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setattr(builder, "PICKUP_PROOF_SHA256", rehash(source))
    with pytest.raises(ValueError, match="control"):
        builder.build_grip_memory(*source)


@pytest.mark.parametrize("field,value", [("error", "aborted"), ("quality_pass", False)])
def test_failed_run_rejected_with_updated_proof(source, monkeypatch, field, value):
    data = read(source[1] / "summary.json"); data[field] = value
    write(source[1] / "summary.json", data)
    proof = read(source[2]); proof["physical"] = data; write(source[2], proof)
    monkeypatch.setattr(builder, "PICKUP_PROOF_SHA256", rehash(source))
    with pytest.raises(ValueError, match="pickup|error"):
        builder.build_grip_memory(*source)


def test_cli_refuses_existing_output(source, tmp_path):
    output = tmp_path / "out.json"
    output.write_text("preserved", encoding="utf8")
    with pytest.raises(FileExistsError):
        builder.main([*(str(p) for p in source), str(output)])
    assert output.read_text() == "preserved"
