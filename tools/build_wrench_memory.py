"""Build one source-backed observational memory; never replay robot targets."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dexhand_agent.experience import export_experience, load_experience


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    json.dumps(value, allow_nan=False)
    return value


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf8")).hexdigest()


def _file(directory, name):
    relative = Path(name)
    _require(not relative.is_absolute() and ".." not in relative.parts, "Source path must stay relative")
    return directory / relative


def _vector(value):
    _require(isinstance(value, list) and len(value) == 3, "Expected finite three-vector")
    _require(all(type(v) in (int, float) and math.isfinite(v) for v in value), "Expected finite three-vector")
    return value


def _relative(target, observation):
    target = _vector(target)
    center = _vector(observation["nut"]["position_m"])
    axis = _vector(observation["wrench_feedback"]["head_nut"]["nut_hinge_axis_world"])
    _require(abs(sum(v * v for v in axis) - 1) < 1e-9, "Nut axis must have unit length")
    delta = [a - b for a, b in zip(target, center)]
    offset = sum(a * b for a, b in zip(delta, axis))
    residual = math.sqrt(sum((d - offset * a) ** 2 for d, a in zip(delta, axis)))
    _require(residual <= 1e-6 and 0 < offset <= .1, "Target must be a bounded positive nut-axis offset")
    return {"frame": "nut_axis", "axial_offset_m": offset}


def validate_skill_memory(record):
    """Validate the optional compact memory without changing the schema-1 trace."""
    memory = record["skill_memory"]
    _require(memory["schema_version"] == 1 and memory["source_episode_id"] == record["id"], "Memory episode mismatch")
    provenance = memory["acceptance_provenance"]
    _require(provenance["accepted"] is True and provenance["stage"] == 2, "Memory requires accepted stage 2")
    _require(provenance["source_files_sha256"] == record["source"]["files_sha256"], "Memory source hashes differ")
    lessons = memory["lessons"]
    _require([v["id"] for v in lessons] == [
        "fixed_head_target", "incremental_seating_turn", "engineered_withdraw_selection"], "Unexpected lesson ids")
    for lesson in lessons:
        _require(isinstance(lesson["condition"], dict) and isinstance(lesson["observed"], dict), "Malformed lesson")
        _require(isinstance(lesson["action_outline"], str) and lesson["limitations"], "Missing lesson scope")
        _require(lesson["evidence"], "Lesson needs source evidence")
        for evidence in lesson["evidence"]:
            digest = evidence["sha256"]
            _require(isinstance(evidence["path"], str) and len(digest) == 64 and
                     all(c in "0123456789abcdef" for c in digest), "Malformed evidence hash")
    for target in [lessons[0]["working_end_target"], *[
            row["working_end_target"] for row in lessons[1]["observed"]["relative_targets"]]]:
        offset = target["axial_offset_m"]
        _require(target["frame"] == "nut_axis" and type(offset) in (int, float) and
                 math.isfinite(offset) and 0 < offset <= .1, "Invalid relative goal bound")
    json.dumps(record, allow_nan=False)
    return record


def build_memory(source_episode, acceptance_path):
    directory = Path(source_episode).resolve()
    acceptance_path = Path(acceptance_path).resolve()
    accepted = _read(acceptance_path)
    _require(accepted["accepted"] is True and accepted["stage"] == 2, "Source stage is not accepted")
    _require({"summary.json", "experience.json", "audit-physical-v3.json",
              "audit-physical-v3-supplemental.json", "audit-local-v3.json", "audit-correction-note.json"}
             <= set(accepted["source_sha256"]), "Missing accepted source audit hashes")
    for name, digest in accepted["source_sha256"].items():
        _require(_sha(_file(directory, name)) == digest, "Acceptance source SHA256 mismatch: " + name)
    original = load_experience(directory / "experience.json")
    for name, digest in original["source"]["files_sha256"].items():
        _require(_sha(_file(directory, name)) == digest, "Original four-file source hash mismatch: " + name)
    _require(original == export_experience(directory), "Original experience differs from four-file export")
    summary = _read(directory / "summary.json")
    _require(original["id"] == accepted["episode_id"] == summary["episode_id"], "Accepted episode mismatch")
    _require(original["origin_backend"] == "native" and original["applicability"]["task"] == "wrench_turn"
             and original["applicability"]["source_seed"] == accepted["seed"], "Source episode fields mismatch")
    _require(accepted["physical"] == summary and summary["success"] is True and summary["error"] is None,
             "Accepted physical result differs from source")
    strict = _read(directory / "audit-physical-v3.json")
    physical = _read(directory / "audit-physical-v3-supplemental.json")
    local = _read(directory / "audit-local-v3.json")
    note = _read(directory / "audit-correction-note.json")
    _require(physical["recovery_accepted"] is True and local["accepted"] is True, "Source audits are not accepted")
    _require(physical["physical"]["matches_saved_result"] and physical["physical"]["sampling_500hz_passed"]
             and physical["quality"]["passed"] and physical["action_binding"]["passed"]
             and physical["accepted_finish"] and physical["no_runtime_error"], "Physical audit checks failed")
    _require(physical["physical"]["recomputed"] == {
        k: summary[k] for k in ("task_id", "protocol_version", "success", "gates", "violations", "metrics")},
        "Physical recomputation differs from saved source")
    _require(local["trace"]["passed"] and local["sources"]["passed"] and all(local["budgets"].values()),
             "Local trace/source audit failed")
    _require(local["counts"] == accepted["counts"] and local["counts"]["low_level_commands"] == len(original["trajectory"])
             and local["counts"]["unclassified_commands"] == 0, "Accepted command counts differ")
    for name, digest in local["input_files_sha256"].items():
        _require(_sha(_file(directory, name)) == digest, "Local source hash mismatch: " + name)
    _require(strict["recovery_accepted"] is False and note["episode_id"] == original["id"] and
             all(note["change"][k] is False for k in (
                 "numeric_rules_changed", "eligibility_thresholds_changed", "action_or_intent_records_changed")),
             "Retrospective audit disclosure/source differs")

    label = directory.relative_to(ROOT).as_posix() if directory.is_relative_to(ROOT) else directory.name
    acceptance_label = acceptance_path.relative_to(ROOT).as_posix() if acceptance_path.is_relative_to(ROOT) else acceptance_path.name

    def evidence(name, sequences=None):
        _require(name in accepted["source_sha256"] or name in local["input_files_sha256"],
                 "Lesson evidence is not bound by an accepted source hash: " + name)
        item = {"path": label + "/" + name, "sha256": _sha(_file(directory, name))}
        if sequences is not None:
            item["sequences"] = sequences
        return item

    requests, intents = {}, {}
    def observation(sequence):
        if sequence not in requests:
            _require(all(name in local["input_files_sha256"] for name in (
                f"steps/{sequence:03}/request.json", f"actor/intents/{sequence:03}.json")),
                "Lesson request/intent is missing its local source audit hash")
            request = _read(directory / f"steps/{sequence:03}/request.json")
            intent = _read(directory / f"actor/intents/{sequence:03}.json")
            _require(request["episode_id"] == original["id"] and request["sequence"] == sequence
                     and request["request_id"] == intent["request_id"], "Request/intent episode binding differs")
            _require(_canonical_sha(request["observation"]) == intent["observation_sha256"],
                     "Intent observation hash mismatch")
            requests[sequence], intents[sequence] = request["observation"], intent
        return requests[sequence]

    events = [event for event in physical["recovery_events"] if event["rule"] == "working_end_tracking"]
    groups = {(e["trigger_sequence"], e["after_sequence"], tuple(e["intended_target_m"])) for e in events}
    _require(len(groups) == accepted["recovery"]["unique_events"] == 1, "Expected one physical recovery event")
    event = events[0]
    trigger, after = event["trigger_sequence"], event["after_sequence"]
    before_obs, after_obs = observation(trigger), observation(after)
    target = event["intended_target_m"]
    relative = _relative(target, before_obs)
    fixed_sequences = event["fixed_target_action_sequences"]
    for sequence in fixed_sequences:
        observation(sequence)
        _require(intents[sequence]["target_head_m"] == target, "Correction changed the intended fixed target")
    errors = [math.dist(target, obs["tool"]["head_center_m"]) for obs in (before_obs, after_obs)]
    _require(abs(errors[0] - event["before_position_error_m"]) < 1e-12 and
             abs(errors[1] - event["after_position_error_m"]) < 1e-12, "Observed correction errors differ from audit")
    contacts = before_obs["contacts"]
    retained = len(set(contacts["hand_tool_contacts"])) >= 2 and not contacts["tool_support_contacts"]
    _require(retained and not contacts["tool_nut_contact"] and not contacts["head_nut_contact"] and
             before_obs["task_progress"]["gates"]["lift"]["passed"], "Fixed-goal recovery condition differs")
    fixed = {
        "id": "fixed_head_target",
        "condition": {"task": "wrench_turn", "after_lift": True, "retained_unsupported": True, "tool_nut_contact": False},
        "working_end_target": relative,
        "action_outline": "Recompute the head target from the fresh nut center and hinge axis. Check actual head error after a completed move; keep this intended goal fixed while applying fresh observed palm-response correction using the explicit last accepted command.",
        "observed": {"trigger_sequence": trigger, "after_sequence": after, "action_sequences": fixed_sequences,
                     "observation_sequences": list(range(trigger, after + 1)), "before_error_m": errors[0],
                     "after_error_m": errors[1], "error_reduction_fraction": 1 - errors[1] / errors[0]},
        "evidence": [evidence("audit-physical-v3-supplemental.json"), evidence("actions.json", fixed_sequences),
                     *[evidence(f"steps/{s:03}/request.json", [s]) for s in range(trigger, after + 1)],
                     *[evidence(f"actor/intents/{s:03}.json", [s]) for s in fixed_sequences]],
        "limitations": ["One episode; this is observed tracking correction, not regrasp, training or universal calibration.",
                        "Recovery passed the corrected retrospective audit; the original strict audit did not accept its semantic label."],
    }
    _require(len(local["calls"]) == 1, "Expected one selected local withdrawal call")
    call = local["calls"][0]
    call_id = call["skill_call_id"]
    prefix = f"actor/skill-calls/{call_id}"
    initial, handoff = _read(directory / (prefix + "/initial.json")), _read(directory / (prefix + "/handoff.json"))
    selection = _read(directory / (prefix + "-selection.json"))
    start, end = initial["start_sequence"], handoff["sequence"]
    _require(call["invoked"] and not call["errors"] and handoff["policy_state"]["status"] == "clearance_check"
             and handoff["replies"] == call["local_motor_commands"] == end - start,
             "Local accepted handoff differs")
    relative_targets = []
    sequences = []
    for sequence in range(after, start):
        obs, next_obs = observation(sequence), observation(sequence + 1)
        operation = intents[sequence]["semantic_operation"]
        if operation not in ("bounded_socket_descent", "bounded_socket_seat", "bounded_positive_turn"):
            continue
        goal = intents[sequence]["target_head_m"]
        axis = obs["wrench_feedback"]["head_nut"]["nut_hinge_axis_world"]
        relative_targets.append({
            "sequence": sequence, "operation": operation, "working_end_target": _relative(goal, obs),
            "requested_axial_step_m": sum((a - b) * c for a, b, c in zip(goal, obs["tool"]["head_center_m"], axis)),
            "duration_s": original["trajectory"][sequence]["action"]["duration_s"],
            "nut_angle_before_deg": math.degrees(obs["nut"]["angle_rad"] - obs["nut"]["initial_angle_rad"]),
            "nut_angle_after_deg": math.degrees(next_obs["nut"]["angle_rad"] - next_obs["nut"]["initial_angle_rad"]),
        })
        sequences.append(sequence)
    _require(relative_targets, "No source incremental seating/turn goals")
    incremental = {
        "id": "incremental_seating_turn",
        "condition": {"task": "wrench_turn", "after_lift": True, "retained_unsupported": True},
        "action_outline": "Use fresh nut-axis-relative targets for small seating moves and incremental turns. Inspect actual contact, angle and continuous gate qualification; the recorded palm commands and relative heights are episode references, not a replay schedule.",
        "observed": {"relative_targets": relative_targets, "turn_gate_passed_at_s": summary["gates"]["turn"].get("passed_at_s")},
        "evidence": [evidence("actions.json", sequences),
                     *[evidence(f"actor/intents/{s:03}.json", [s]) for s in sequences],
                     *[evidence(f"steps/{s:03}/request.json", [s]) for s in range(after, start + 1)]],
        "limitations": ["Requested axial increments are not realized displacements; measured nut increments are not commanded tool rotations.",
                        "Check the independent dwell gate; crossing an instantaneous angle alone does not qualify it."],
    }
    entry, exit_obs = observation(start), observation(end)
    margin = re.search(r"above\s+([0-9.]+)\s+degree", selection["diagnosis"])
    _require(margin is not None, "Source selection lacks its declared engineered entry margin")
    withdrawal = {
        "id": "engineered_withdraw_selection",
        "condition": {"task": "wrench_turn", "after_lift": True, "after_turn": True,
                      "retained_unsupported": True, "engineered_entry_min_nut_deg": float(margin.group(1))},
        "action_outline": "Explicitly select the supplied observation-only WrenchWithdrawPolicy from a sufficient-turn retained state; inspect the independent withdrawal gate when it hands back clearance_check. This skill existed before this episode.",
        "observed": {"skill_call_id": call_id, "entry_nut_deg": math.degrees(entry["nut"]["angle_rad"] - entry["nut"]["initial_angle_rad"]),
                     "action_sequence_range": [start, end - 1], "local_motor_commands": call["local_motor_commands"],
                     "replayed_decisions": call["replayed_decisions"], "handoff_sequence": end,
                     "handoff_status": handoff["policy_state"]["status"],
                     "handoff_head_clearance_m": exit_obs["contacts"]["head_clearance_m"],
                     "independent_withdraw_gate_passed": exit_obs["task_progress"]["gates"]["withdraw"]["passed"]},
        "evidence": [evidence(prefix + "-selection.json", [start]), evidence(prefix + "/initial.json", [start]),
                     evidence(prefix + "/handoff.json", [end]), evidence(f"steps/{start:03}/request.json", [start]),
                     evidence(f"steps/{end:03}/request.json", [end]), evidence("audit-local-v3.json")],
        "limitations": ["The entry margin is an engineered prior selected online, not a newly discovered necessary threshold.",
                        "Local motor commands are not online model choices; clearance_check delegates final acceptance.",
                        "No transfer, efficiency benefit, new neural training or general success-rate claim."],
    }
    record = copy.deepcopy(original)
    record["skill_memory"] = {
        "schema_version": 1, "source_episode_id": original["id"],
        "kind": "single_accepted_native_episode_observational_memory",
        "acquisition": {
            **copy.deepcopy(local["counts"]),
            "source_brief": "origin_backend=native includes online choices plus an explicitly selected engineered local program.",
            "count_scope": "Recorded action/skill choices; not measured service inference calls, API calls or token costs.",
            "exact_model_identity_verified": False, "api_or_token_cost_measured": False,
        },
        "acceptance_provenance": {
            "episode_id": original["id"], "stage": accepted["stage"], "accepted": accepted["accepted"],
            "path": acceptance_label, "sha256": _sha(acceptance_path),
            "source_files_sha256": copy.deepcopy(original["source"]["files_sha256"]),
            "audit_disclosure": accepted["audit_disclosure"], "original_strict_recovery_accepted": strict["recovery_accepted"],
            "corrected_retrospective_recovery_accepted": physical["recovery_accepted"],
            "unique_physical_recovery_events": len(groups), "audits": [
                evidence(name) for name in ("audit-physical-v3.json", "audit-physical-v3-supplemental.json",
                                            "audit-local-v3.json", "audit-correction-note.json")]},
        "lessons": [fixed, incremental, withdrawal],
    }
    return validate_skill_memory(record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_episode", type=Path)
    parser.add_argument("acceptance", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite immutable memory: " + str(args.output))
    record = build_memory(args.source_episode, args.acceptance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "id": record["id"], "sha256": _sha(args.output),
                      "lessons": [lesson["id"] for lesson in record["skill_memory"]["lessons"]]}))


if __name__ == "__main__":
    main()

