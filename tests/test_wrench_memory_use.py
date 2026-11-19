import copy
import hashlib
import json

import pytest

from tools import audit_wrench_memory_use as audit
from tools.audit_wrench_autonomy import canonical_sha


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf8")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    proof = tmp_path / "accepted.json"
    save(proof, {"stage": 2, "accepted": True, "episode_id": audit.SOURCE_ID,
                 "audit_disclosure": "corrected retrospective acceptance; original strict failure retained",
                 "source_sha256": {"summary.json": audit.SOURCE_HASHES["summary.json"]}})
    monkeypatch.setattr(audit, "ACCEPTANCE_PATH", proof)
    memory = {"schema_version": 1, "id": audit.SOURCE_ID, "origin_backend": "native",
              "status": "demonstrated_episode", "source": {"episode_id": audit.SOURCE_ID,
                                                           "files_sha256": dict(audit.SOURCE_HASHES)},
              "applicability": {"robot": "Panda+Shadow", "tool": "wrench", "task": "wrench_turn",
                                "observation_mode": "simulator_state", "source_seed": 5,
                                "new_tools_verified": False, "other_layouts_verified": False},
              "initial_observation": {"tool": {}, "nut": {}, "palm": {}}, "trajectory": [],
              "outcome": {"success": True, "gates": {k: {"passed": True} for k in ("lift", "turn", "withdraw")},
                          "violations": [], "error": None, "decisions": 0, "metrics": {}}, "notes": "source record",
              "skill_memory": {"schema_version": 1, "source_episode_id": audit.SOURCE_ID,
                  "kind": "single_accepted_native_episode_observational_memory",
                  "acceptance_provenance": {"episode_id": audit.SOURCE_ID, "stage": 2, "accepted": True,
                      "path": audit.ACCEPTANCE_RELATIVE, "sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
                      "source_files_sha256": dict(audit.SOURCE_HASHES),
                      "audit_disclosure": read(proof)["audit_disclosure"],
                      "original_strict_recovery_accepted": False, "corrected_retrospective_recovery_accepted": True},
                  "lessons": [{"id": "fixed_head_target", "condition": {"after_lift": True,
                      "retained_unsupported": True, "tool_nut_contact": False, "task": "wrench_turn"},
                      "working_end_target": {"frame": "nut_axis", "axial_offset_m": .04},
                      "evidence": [{"path": "runs/source/steps/005/action.json", "sha256": "a" * 64}]}]}}
    memory_path, episode = tmp_path / "memory.json", tmp_path / "episode"
    save(memory_path, memory)
    nominal = copy.deepcopy(memory)
    nominal.pop("skill_memory")
    nominal["id"] = nominal["source"]["episode_id"] = "nominal"
    experience = [nominal, memory]
    save(episode / "metadata.json", {"episode_id": "fresh", "backend": "native",
         "experience": experience, "experience_sha256": canonical_sha(experience)})
    save(episode / "summary.json", {"episode_id": "fresh", "backend": "native", "decisions": 2,
                                     "success": False, "error": None})
    history = []
    for seq in range(2):
        observation = {"time_s": 3.0 + seq, "nut": {"position_m": [.6, .2, .3]},
                       "wrench_feedback": {"head_nut": {"nut_hinge_axis_world": [0, 1, 0]}},
                       "contacts": {"hand_tool_contacts": ["thumb", "index"], "tool_support_contacts": [],
                                    "tool_nut_contact": False, "head_nut_contact": False},
                       "task_progress": {"gates": {"lift": {"passed": True, "passed_at_s": 2.0}}}}
        request = {"episode_id": "fresh", "sequence": seq, "request_id": f"fresh-{seq}",
                   "observation": observation, "experience": experience}
        action = {"operation": "move" if seq == 0 else "finish", "targets": [],
                  "palm_target_m": [.5, .2, .3] if seq == 0 else None,
                  "palm_quat_wxyz": [1, 0, 0, 0] if seq == 0 else None,
                  "duration_s": 1., "memory": "declared adoption"}
        result = {"status": "completed" if seq == 0 else "finished", "time_s": 4.0}
        history.append({"sequence": seq, "action": action, "result": result})
        save(episode / f"steps/{seq:03}/request.json", request)
        save(episode / f"steps/{seq:03}/action.json", action)
        save(episode / f"steps/{seq:03}/result.json", result)
        save(episode / f"actor/intents/{seq:03}.json", {"request_id": f"fresh-{seq}",
             "observation_sha256": canonical_sha(observation), "decision_origin": "online_model",
             "semantic_operation": "align" if seq == 0 else "finish",
             "target_head_m": [.6, .24, .3] if seq == 0 else None,
             "experience_refs": [{"experience_id": audit.SOURCE_ID, "lesson_id": "fixed_head_target"}] if seq == 0 else []})
    save(episode / "actions.json", history)
    return episode, memory_path


def update_requests(episode, change):
    for path in sorted((episode / "steps").glob("*/request.json")):
        value = read(path)
        change(value)
        save(path, value)


def test_declared_adoption_uses_observed_axis_and_does_not_require_physical_success(recorded):
    result = audit.audit_memory_use(*recorded)
    assert result["accepted"], result["errors"]
    assert result["adopted_sequences"] == [0]
    assert result["cited_lesson_ids"] == ["fixed_head_target"]
    assert result["request_count"] == 2
    assert result["memory_canonical_sha256"] == canonical_sha(read(recorded[1]))
    assert result["physical_success_claimed"] is False


@pytest.mark.parametrize("change", ["missing", "changed_payload", "metadata_digest", "one_request"])
def test_memory_declaration_must_match_exactly(recorded, change):
    episode, memory = recorded
    path = episode / "metadata.json"
    item = read(path)
    if change == "missing":
        item["experience"] = item["experience"][:1]
        item["experience_sha256"] = canonical_sha(item["experience"])
    elif change == "changed_payload":
        item["experience"][1]["notes"] = "changed after input"
        item["experience_sha256"] = canonical_sha(item["experience"])
    elif change == "metadata_digest":
        item["experience_sha256"] = "0" * 64
    else:
        request_path = episode / "steps/001/request.json"
        request = read(request_path)
        request["experience"] = request["experience"][:1]
        save(request_path, request)
    save(path, item)
    assert not audit.audit_memory_use(episode, memory)["accepted"]


@pytest.mark.parametrize("change", ["source_hash", "acceptance_hash", "offset", "source_id", "private_path"])
def test_memory_source_and_lesson_provenance_cannot_be_self_relabelled(recorded, change):
    episode, path = recorded
    item = read(path)
    if change == "source_hash":
        item["source"]["files_sha256"]["actions.json"] = "0" * 64
        item["skill_memory"]["acceptance_provenance"]["source_files_sha256"]["actions.json"] = "0" * 64
    elif change == "acceptance_hash":
        item["skill_memory"]["acceptance_provenance"]["sha256"] = "0" * 64
    elif change == "offset":
        item["skill_memory"]["lessons"][0]["working_end_target"]["axial_offset_m"] = .05
    elif change == "source_id":
        item["id"] = item["source"]["episode_id"] = "fresh"
    else:
        item["skill_memory"]["acceptance_provenance"]["path"] = "D:/private/acceptance.json"
    save(path, item)
    assert not audit.audit_memory_use(episode, path)["accepted"]


@pytest.mark.parametrize("change", ["unknown_lesson", "unknown_experience", "wrong_axis_goal",
                                   "local_only", "rejected", "before_lift", "supported", "stale_id"])
def test_references_and_observable_adoption_conditions_are_required(recorded, change):
    episode, memory = recorded
    path = episode / "actor/intents/000.json"
    intent = read(path)
    if change == "unknown_lesson":
        intent["experience_refs"][0]["lesson_id"] = "imaginary"
    elif change == "unknown_experience":
        intent["experience_refs"][0]["experience_id"] = "not_supplied"
    elif change == "wrong_axis_goal":
        intent["target_head_m"] = [.6, .2, .34]  # world Z does not equal the observed Y axis.
    elif change == "local_only":
        intent["decision_origin"] = "local_feedback_skill"
    elif change == "stale_id":
        intent["request_id"] = "old-request"
    elif change == "rejected":
        history = read(episode / "actions.json")
        history[0]["result"]["status"] = "rejected"
        save(episode / "actions.json", history)
        save(episode / "steps/000/result.json", history[0]["result"])
    else:
        request_path = episode / "steps/000/request.json"
        req = read(request_path)
        if change == "before_lift":
            req["observation"]["task_progress"]["gates"]["lift"]["passed"] = False
        else:
            req["observation"]["contacts"]["tool_support_contacts"] = ["rack"]
        save(request_path, req)
        intent["observation_sha256"] = canonical_sha(req["observation"])
    save(path, intent)
    assert not audit.audit_memory_use(episode, memory)["accepted"]


def test_baseline_absence_has_no_artificial_adoption_requirement(recorded):
    episode, memory = recorded
    metadata = read(episode / "metadata.json")
    metadata["experience"] = metadata["experience"][:1]
    metadata["experience_sha256"] = canonical_sha(metadata["experience"])
    save(episode / "metadata.json", metadata)
    update_requests(episode, lambda request: request.update(experience=metadata["experience"]))
    path = episode / "actor/intents/000.json"
    save(path, {**read(path), "experience_refs": []})
    result = audit.audit_memory_use(episode, memory, expected_present=False)
    assert result["accepted"], result["errors"]
    assert result["memory_present"] is False
    assert result["adopted_sequences"] == []
    save(path, {**read(path), "experience_refs": [{"experience_id": audit.SOURCE_ID, "lesson_id": "fixed_head_target"}]})
    assert not audit.audit_memory_use(episode, memory, expected_present=False)["accepted"]


def test_baseline_rejects_unexpected_native_memory(recorded):
    assert not audit.audit_memory_use(*recorded, expected_present=False)["accepted"]


def test_runtime_error_is_retained_as_nonacceptance(recorded):
    episode, memory = recorded
    path = episode / "summary.json"
    save(path, {**read(path), "error": "recorded backend interruption"})
    assert not audit.audit_memory_use(episode, memory)["accepted"]


def test_cli_never_overwrites_an_output(recorded, tmp_path, monkeypatch):
    episode, memory = recorded
    output = tmp_path / "audit.json"
    save(output, {"keep": 1})
    monkeypatch.setattr("sys.argv", ["audit", str(episode), str(memory), "--output", str(output)])
    with pytest.raises(FileExistsError):
        audit.main()
    assert read(output) == {"keep": 1}


NOMINAL_TARGETS = {
    "rh_A_FFJ3": .8, "rh_A_FFJ0": 1.5, "rh_A_MFJ3": .8, "rh_A_MFJ0": 1.5,
    "rh_A_RFJ3": .8, "rh_A_RFJ0": 1.5, "rh_A_LFJ3": .8, "rh_A_LFJ0": 1.5,
    "rh_A_THJ5": .9763313232421871, "rh_A_THJ4": .9253804236340059,
    "rh_A_THJ3": 0, "rh_A_THJ2": -3.4637906367758015e-05, "rh_A_THJ1": .45,
}
CANDIDATE_TARGETS = {**NOMINAL_TARGETS, "rh_A_THJ4": .9453804236340059}


def targets(mapping):
    return [{"actuator": name, "value": value} for name, value in mapping.items()]


def bind_memory(episode, path, memory):
    save(path, memory)
    metadata = read(episode / "metadata.json")
    metadata["experience"][1] = memory
    metadata["experience_sha256"] = canonical_sha(metadata["experience"])
    save(episode / "metadata.json", metadata)
    update_requests(episode, lambda request: request.update(experience=metadata["experience"]))


def bind_history(episode, history):
    save(episode / "actions.json", history)
    for row in history:
        save(episode / f"steps/{row['sequence']:03}/action.json", row["action"])
        save(episode / f"steps/{row['sequence']:03}/result.json", row["result"])
    summary = read(episode / "summary.json")
    save(episode / "summary.json", {**summary, "decisions": len(history)})


@pytest.fixture
def calibrated(recorded, tmp_path, monkeypatch):
    episode, memory_path = recorded
    source = tmp_path / "pickup-source"
    source.mkdir()
    events = [{"event": "after_execute", "sequence": i,
               "action": {"operation": "move", "targets": targets(CANDIDATE_TARGETS) if i == 2 else []},
               "result": {"status": "completed"}} for i in range(5)]
    (source / "actions.jsonl").write_text("\n".join(json.dumps(event) for event in events), encoding="utf8")
    source_hashes = {"actions.jsonl": hashlib.sha256((source / "actions.jsonl").read_bytes()).hexdigest()}
    proof = {
        "kind": "engineering_partial_skill_evidence", "accepted_for_pickup_only": True,
        "full_task_validated": False, "native_agent": False, "held_out": False,
        "source": "runs/wrench-placement-v1/development/thumb-006",
        "one_changed_parameter": {"actuator": "rh_A_THJ4", "original": NOMINAL_TARGETS["rh_A_THJ4"],
                                  "candidate": CANDIDATE_TARGETS["rh_A_THJ4"], "range": [0., 1.22173]},
        "physical": {"success": False, "gates": {"lift": {"passed": True}, "turn": {"passed": False},
                       "withdraw": {"passed": False}}, "pickup_evidence_pass": True, "quality_pass": True,
                       "violations": [], "error": None},
        "source_sha256": source_hashes, "no_prior_result_reclassified": True,
    }
    proof_path = tmp_path / "pickup-proof.json"
    save(proof_path, proof)
    proof_sha = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    monkeypatch.setattr(audit, "GRIP_PROOF_PATH", proof_path)
    monkeypatch.setattr(audit, "GRIP_PROOF_SHA256", proof_sha)
    monkeypatch.setattr(audit, "GRIP_SOURCE_PATH", source)
    memory = read(memory_path)
    memory["skill_memory"].update(revision=2, kind="single_native_episode_plus_partial_grasp_calibration")
    memory["skill_memory"]["lessons"].append({
        "id": "calibrated_grasp_v2", "calibrated_targets": targets(CANDIDATE_TARGETS),
        "validation": {"kind": "engineering_pickup_only", "full_task_validated": False,
                       "native_agent": False, "held_out": False, "source_files_sha256": source_hashes,
                       "lift_gate": proof["physical"]["gates"]["lift"], "quality_pass": True},
        "evidence": [{"path": "evidence/wrench-grip-v2/development-pickup.json", "sha256": proof_sha},
                     {"path": "runs/wrench-placement-v1/development/thumb-006/actions.jsonl",
                      "sha256": source_hashes["actions.jsonl"]}],
    })
    bind_memory(episode, memory_path, memory)
    history = read(episode / "actions.json")
    history[0]["action"]["targets"] = targets(CANDIDATE_TARGETS)
    history[1]["action"].update(operation="move", palm_target_m=[.5,.2,.3], palm_quat_wxyz=[1,0,0,0])
    history[1]["result"] = {"status": "completed", "time_s": 5.}
    bind_history(episode, history)
    for seq, role, lesson in ((0, "grasp", "calibrated_grasp_v2"), (1, "align", "fixed_head_target")):
        path = episode / f"actor/intents/{seq:03}.json"
        intent = read(path)
        intent.update(semantic_operation=role, target_head_m=None if seq == 0 else [.6,.24,.3],
                      experience_refs=[{"experience_id": audit.SOURCE_ID, "lesson_id": lesson}])
        save(path, intent)
    return episode, memory_path, proof_path, source


def test_required_calibrated_grasp_uses_exact_thirteen_targets_and_citation(calibrated):
    episode, memory, _, _ = calibrated
    result = audit.audit_memory_use(episode, memory, require_calibrated_grasp=True)
    assert result["accepted"], result["errors"]
    assert result["calibrated_grasp_applied"] is True
    assert result["calibrated_grasp_sequence"] == 0
    assert result["initial_grasp_targets"] == CANDIDATE_TARGETS
    assert result["later_grip_changes"] == []
    assert result["adopted_sequences"] == [1]


@pytest.mark.parametrize("change", ["nominal_thumb", "other_finger", "missing_citation", "local_origin", "duplicate_target"])
def test_required_grasp_rejects_nominal_or_unbound_configuration(calibrated, change):
    episode, memory, _, _ = calibrated
    history = read(episode / "actions.json")
    intent_path = episode / "actor/intents/000.json"
    intent = read(intent_path)
    if change == "nominal_thumb":
        history[0]["action"]["targets"] = targets(NOMINAL_TARGETS)
    elif change == "other_finger":
        history[0]["action"]["targets"][0]["value"] = .81
    elif change == "duplicate_target":
        history[0]["action"]["targets"][0] = history[0]["action"]["targets"][1]
    elif change == "missing_citation":
        intent["experience_refs"] = []
    else:
        intent["decision_origin"] = "local_feedback_skill"
    bind_history(episode, history)
    save(intent_path, intent)
    result = audit.audit_memory_use(episode, memory, require_calibrated_grasp=True)
    assert not result["accepted"]
    assert not result["calibrated_grasp_applied"]


@pytest.mark.parametrize("change", ["proof_full_task", "lesson_full_task", "lesson_native", "source_hash", "source_file", "proof_hash", "lesson_targets"])
def test_partial_grasp_source_cannot_be_promoted_or_changed(calibrated, monkeypatch, change):
    episode, memory_path, proof_path, source = calibrated
    memory = read(memory_path)
    lesson = memory["skill_memory"]["lessons"][-1]
    if change == "proof_full_task":
        proof = read(proof_path)
        proof["full_task_validated"] = True
        save(proof_path, proof)
        proof_sha = hashlib.sha256(proof_path.read_bytes()).hexdigest()
        monkeypatch.setattr(audit, "GRIP_PROOF_SHA256", proof_sha)
        lesson["evidence"][0]["sha256"] = proof_sha
    elif change == "lesson_full_task":
        lesson["validation"]["full_task_validated"] = True
    elif change == "lesson_native":
        lesson["validation"]["native_agent"] = True
    elif change == "source_hash":
        lesson["validation"]["source_files_sha256"]["actions.jsonl"] = "0" * 64
    elif change == "source_file":
        (source / "actions.jsonl").write_text("changed commands", encoding="utf8")
    elif change == "proof_hash":
        proof_path.write_text("{}", encoding="utf8")
    else:
        lesson["calibrated_targets"] = targets(NOMINAL_TARGETS)
    bind_memory(episode, memory_path, memory)
    result = audit.audit_memory_use(episode, memory_path, require_calibrated_grasp=True)
    assert not result["accepted"]


@pytest.mark.parametrize("status", ["completed", "rejected"])
def test_later_grip_reactions_are_reported_without_failing_adoption(calibrated, status):
    episode, memory, _, _ = calibrated
    request = read(episode / "steps/001/request.json")
    request.update(sequence=2, request_id="fresh-2")
    request["observation"]["time_s"] = 5.
    save(episode / "steps/002/request.json", request)
    save(episode / "actor/intents/002.json", {"request_id": "fresh-2", "decision_origin": "online_model",
         "observation_sha256": canonical_sha(request["observation"]), "semantic_operation": "grasp", "experience_refs": []})
    history = read(episode / "actions.json")
    history.append({"sequence": 2, "action": {**history[0]["action"],
         "targets": [{"actuator": "rh_A_THJ4", "value": NOMINAL_TARGETS["rh_A_THJ4"]}]},
         "result": {"status": status, "time_s": 6.}})
    bind_history(episode, history)
    result = audit.audit_memory_use(episode, memory, require_calibrated_grasp=True)
    assert result["accepted"], result["errors"]
    assert result["later_grip_changes"] == [{"sequence": 2, "result_status": status,
         "applied": status == "completed", "changes": {"rh_A_THJ4": {
             "previous": CANDIDATE_TARGETS["rh_A_THJ4"], "value": NOMINAL_TARGETS["rh_A_THJ4"]}}}]


def test_optional_calibrated_audit_does_not_change_default_v1_rules(recorded):
    assert audit.audit_memory_use(*recorded)["accepted"]
    assert not audit.audit_memory_use(*recorded, require_calibrated_grasp=True)["accepted"]


@pytest.mark.parametrize("change", ["interrupted", "nominal_targets"])
def test_actual_diagnostic_events_must_show_completed_calibrated_grasp(calibrated, monkeypatch, change):
    episode, memory_path, proof_path, source = calibrated
    action_path = source / "actions.jsonl"
    events = [json.loads(line) for line in action_path.read_text(encoding="utf8").splitlines()]
    if change == "interrupted":
        events[3]["result"]["status"] = "interrupted"
    else:
        events[2]["action"]["targets"] = targets(NOMINAL_TARGETS)
    action_path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf8")
    proof = read(proof_path)
    proof["source_sha256"]["actions.jsonl"] = hashlib.sha256(action_path.read_bytes()).hexdigest()
    save(proof_path, proof)
    proof_sha = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    monkeypatch.setattr(audit, "GRIP_PROOF_SHA256", proof_sha)
    memory = read(memory_path)
    lesson = memory["skill_memory"]["lessons"][-1]
    lesson["validation"]["source_files_sha256"] = proof["source_sha256"]
    lesson["evidence"][0]["sha256"] = proof_sha
    lesson["evidence"][1]["sha256"] = proof["source_sha256"]["actions.jsonl"]
    bind_memory(episode, memory_path, memory)
    result = audit.audit_memory_use(episode, memory_path, require_calibrated_grasp=True)
    assert not result["accepted"]
    assert any("Actual" in error for error in result["errors"])


def test_calibrated_cli_flag(calibrated, tmp_path, monkeypatch):
    episode, memory, _, _ = calibrated
    output = tmp_path / "calibrated-audit.json"
    monkeypatch.setattr("sys.argv", ["audit", str(episode), str(memory), "--require-calibrated-grasp", "--output", str(output)])
    assert audit.main() == 0
    assert read(output)["calibrated_grasp_applied"] is True
