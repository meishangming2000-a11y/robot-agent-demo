"""Audit declared native003 lesson adoption from episode files, without physics."""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dexhand_agent.experience import _validate
from tools.audit_wrench_autonomy import canonical_sha, read_json

SOURCE_ID = "02739d437afb43c2a30cb1347fb6f345"
SOURCE_HASHES = {
    "metadata.json": "d8a8f81c264ce98c7536705e5691c2f03d1775cc60830ba5a490c53adfffcb3c",
    "summary.json": "cc179b44c4feeabef520c02d86efbf4382d97b921331776ca0c9d3b202aa5409",
    "actions.json": "07232a3885257fab2bc6c71b7de5bd71a53bbc49a995e4a8ffda53a5ff9d5632",
    "steps/000/request.json": "78d80146216c5506c0d61f6a49ec4ddc84a6afa70ee162a6bb5924ede3095707",
}
ACCEPTANCE_RELATIVE = "evidence/wrench-recovery-v3/acceptance.json"
ACCEPTANCE_PATH = ROOT / ACCEPTANCE_RELATIVE
ALIGN_ROLES = {"align", "level", "safe_height_align"}
GRIP_PROOF_RELATIVE = "evidence/wrench-grip-v2/development-pickup.json"
GRIP_PROOF_PATH = ROOT / GRIP_PROOF_RELATIVE
GRIP_PROOF_SHA256 = "bce371a9b482964b475a7383105496dffad57ba266d245427c606c107eaade82"
GRIP_SOURCE_RELATIVE = "runs/wrench-placement-v1/development/thumb-006"
GRIP_SOURCE_PATH = ROOT / GRIP_SOURCE_RELATIVE
NOMINAL_GRASP_TARGETS = {
    "rh_A_FFJ3": .8, "rh_A_FFJ0": 1.5, "rh_A_MFJ3": .8, "rh_A_MFJ0": 1.5,
    "rh_A_RFJ3": .8, "rh_A_RFJ0": 1.5, "rh_A_LFJ3": .8, "rh_A_LFJ0": 1.5,
    "rh_A_THJ5": .9763313232421871, "rh_A_THJ4": .9253804236340059,
    "rh_A_THJ3": 0, "rh_A_THJ2": -3.4637906367758015e-05, "rh_A_THJ1": .45,
}
CALIBRATED_GRASP_TARGETS = {**NOMINAL_GRASP_TARGETS, "rh_A_THJ4": .9453804236340059}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _target_mapping(targets):
    require(isinstance(targets, list), "Grasp targets must be a list")
    result = {}
    for target in targets:
        require(isinstance(target, dict) and set(target) == {"actuator", "value"}
                and isinstance(target["actuator"], str) and target["actuator"] not in result
                and type(target["value"]) in (int, float) and math.isfinite(target["value"]),
                "Invalid or duplicate grasp target")
        result[target["actuator"]] = target["value"]
    return result


def _validate_calibrated_grasp(skill, lessons):
    require(skill.get("revision") == 2 and skill["kind"] == "single_native_episode_plus_partial_grasp_calibration",
            "Missing calibrated-grasp memory revision")
    lesson = lessons["calibrated_grasp_v2"]
    require(_target_mapping(lesson["calibrated_targets"]) == CALIBRATED_GRASP_TARGETS,
            "Calibrated grasp must change only nominal THJ4 by +.02 rad")
    require(hashlib.sha256(GRIP_PROOF_PATH.read_bytes()).hexdigest() == GRIP_PROOF_SHA256,
            "Calibrated-grasp proof hash mismatch")
    proof = read_json(GRIP_PROOF_PATH)
    require(proof["kind"] == "engineering_partial_skill_evidence" and proof["accepted_for_pickup_only"] is True
            and proof["full_task_validated"] is False and proof["native_agent"] is False
            and proof["held_out"] is False and proof["no_prior_result_reclassified"] is True
            and proof["source"] == GRIP_SOURCE_RELATIVE, "Grasp proof must remain engineering pickup-only evidence")
    require(proof["one_changed_parameter"] == {"actuator": "rh_A_THJ4",
            "original": NOMINAL_GRASP_TARGETS["rh_A_THJ4"], "candidate": CALIBRATED_GRASP_TARGETS["rh_A_THJ4"],
            "range": [0., 1.22173]}, "Grasp proof parameter differs from the frozen candidate")
    physical = proof["physical"]
    require(physical["success"] is False and physical["gates"]["lift"]["passed"] is True
            and all(physical["gates"][name]["passed"] is False for name in ("turn", "withdraw"))
            and physical["pickup_evidence_pass"] is True and physical["quality_pass"] is True
            and not physical["violations"] and physical["error"] is None,
            "Grasp diagnostic is not valid partial pickup evidence")
    validation = lesson["validation"]
    require(validation["kind"] == "engineering_pickup_only" and validation["full_task_validated"] is False
            and validation["native_agent"] is False and validation["held_out"] is False
            and validation["quality_pass"] is True and validation["lift_gate"] == physical["gates"]["lift"]
            and validation["source_files_sha256"] == proof["source_sha256"],
            "Calibrated-grasp validation/source claims differ from the partial proof")
    for name, expected in proof["source_sha256"].items():
        source = (GRIP_SOURCE_PATH / name).resolve()
        require(source.is_relative_to(GRIP_SOURCE_PATH.resolve())
                and hashlib.sha256(source.read_bytes()).hexdigest() == expected,
                f"Calibrated-grasp source hash mismatch: {name}")
    evidence = {item["path"]: item["sha256"] for item in lesson["evidence"]}
    require(len(evidence) == len(lesson["evidence"]) and evidence.get(GRIP_PROOF_RELATIVE) == GRIP_PROOF_SHA256,
            "Missing fixed calibrated-grasp proof evidence")
    for name, expected in evidence.items():
        if name != GRIP_PROOF_RELATIVE:
            prefix = GRIP_SOURCE_RELATIVE + "/"
            require(name.startswith(prefix) and proof["source_sha256"].get(name[len(prefix):]) == expected,
                    "Calibrated-grasp lesson evidence does not match the diagnostic source")
    # Actual execution results come from the event log. The preserved declaration
    # also carries old primary results and is not a diagnostic execution log.
    events = [json.loads(line) for line in (GRIP_SOURCE_PATH / "actions.jsonl").read_text(encoding="utf8").splitlines()]
    completed = [event for event in events if event.get("event") == "after_execute"]
    require([event["sequence"] for event in completed] == list(range(5))
            and all(event["result"]["status"] == "completed" for event in completed),
            "Actual calibrated-grasp diagnostic actions did not all complete")
    require(completed[2]["action"]["operation"] == "move"
            and _target_mapping(completed[2]["action"]["targets"]) == CALIBRATED_GRASP_TARGETS,
            "Actual diagnostic grasp does not match the calibrated thirteen targets")


def audit_memory_use(episode, memory, *, expected_present=True, require_calibrated_grasp=False):
    """Check supplied memory/citations; physical and causal acceptance is separate."""
    episode, memory = Path(episode).resolve(), Path(memory).resolve()
    errors, adopted, cited, count, digest, present = [], [], set(), 0, None, False
    metadata_consistent, episode_error = False, None
    grasp_sequence, initial_grasp, current_grasp, later_grip_changes = None, None, None, []
    try:
        require(not require_calibrated_grasp or expected_present, "Calibrated grasp requires supplied native memory")
        record = _validate(read_json(memory))
        digest = canonical_sha(record)
        skill = record["skill_memory"]
        provenance = skill["acceptance_provenance"]
        proof = read_json(ACCEPTANCE_PATH)  # Fixed project trust anchor, never a path from the memory.
        require(record["id"] == record["source"]["episode_id"] == skill["source_episode_id"] == SOURCE_ID
                and record["origin_backend"] == "native" and record["applicability"]["task"] == "wrench_turn",
                "Memory is not the accepted native003 wrench record")
        require(record["source"]["files_sha256"] == provenance["source_files_sha256"] == SOURCE_HASHES,
                "Accepted source four-file hashes differ")
        require(skill["schema_version"] == 1 and provenance["episode_id"] == SOURCE_ID
                and provenance["stage"] == proof["stage"] == 2 and provenance["accepted"] is True
                and proof["accepted"] is True and proof["episode_id"] == SOURCE_ID
                and proof["source_sha256"]["summary.json"] == SOURCE_HASHES["summary.json"], "Invalid accepted-source provenance")
        require(provenance["path"] == ACCEPTANCE_RELATIVE
                and provenance["sha256"] == hashlib.sha256(ACCEPTANCE_PATH.read_bytes()).hexdigest()
                and provenance["audit_disclosure"] == proof["audit_disclosure"]
                and provenance["original_strict_recovery_accepted"] is False
                and provenance["corrected_retrospective_recovery_accepted"] is True, "Acceptance proof/hash/disclosure mismatch")
        lessons = {lesson["id"]: lesson for lesson in skill["lessons"]}
        require(len(lessons) == len(skill["lessons"]), "Duplicate lesson id")
        lesson = lessons["fixed_head_target"]
        target_rule, condition = lesson["working_end_target"], lesson["condition"]
        offset = target_rule["axial_offset_m"]
        require(target_rule["frame"] == "nut_axis" and type(offset) in (int, float)
                and abs(offset - .04) <= 1e-12, "Fixed head target is not the source-derived nut-axis offset")
        require(condition["after_lift"] is True and condition["retained_unsupported"] is True
                and condition["tool_nut_contact"] is False and condition["task"] == "wrench_turn", "Lesson condition mismatch")
        require(bool(lesson["evidence"]), "Missing lesson evidence")
        for evidence in lesson["evidence"]:
            path = PurePosixPath(evidence["path"].replace("\\", "/"))
            require(not path.is_absolute() and not PureWindowsPath(evidence["path"]).drive and ".." not in path.parts
                    and bool(re.fullmatch(r"[0-9a-f]{64}", evidence["sha256"])), "Invalid project-relative lesson evidence")
        if require_calibrated_grasp:
            _validate_calibrated_grasp(skill, lessons)
        metadata, summary, history = (read_json(episode / name) for name in ("metadata.json", "summary.json", "actions.json"))
        current_id = metadata["episode_id"]
        require(current_id != SOURCE_ID and summary["episode_id"] == current_id
                and metadata["backend"] == summary["backend"] == "native", "Current/source episode identity mismatch")
        require(summary["decisions"] == len(history), "History/summary count mismatch")
        episode_error = summary.get("error")
        require(episode_error is None, "Episode contains a runtime/backend error")
        experiences = metadata["experience"]
        require(isinstance(experiences, list) and metadata["experience_sha256"] == canonical_sha(experiences), "Metadata experience digest mismatch")
        supplied = {value["id"]: value for value in experiences}
        require(len(supplied) == len(experiences), "Duplicate supplied experience id")
        present = SOURCE_ID in supplied
        require(present == expected_present, "Native memory presence does not match the audit mode")
        if expected_present:
            require(canonical_sha(supplied[SOURCE_ID]) == digest, "Declared memory payload differs from supplied memory")
        available = {key: {item["id"] for item in value.get("skill_memory", {}).get("lessons", [])}
                     for key, value in supplied.items()}
        paths, seen = sorted((episode / "steps").glob("*/request.json")), set()
        require(bool(paths), "No saved requests")
        for index, path in enumerate(paths):
            require(path.resolve().is_relative_to(episode), "Request path escapes episode")
            request = read_json(path)
            seq, obs = request["sequence"], request["observation"]
            require(type(seq) is int and seq == index and path.parent.name == f"{seq:03}"
                    and request["episode_id"] == current_id and request["request_id"] == f"{current_id}-{seq}"
                    and request["request_id"] not in seen, "Stale/noncontiguous request identity")
            seen.add(request["request_id"])
            count += 1
            require(canonical_sha(request["experience"]) == canonical_sha(experiences), "Request experience differs from metadata")
            intent_path = episode / f"actor/intents/{seq:03}.json"
            if seq >= len(history) and not intent_path.exists():
                continue
            require(intent_path.resolve().is_relative_to(episode), "Intent path escapes episode")
            intent = read_json(intent_path)
            require(intent["request_id"] == request["request_id"]
                    and intent["observation_sha256"] == canonical_sha(obs), "Intent/current observation binding mismatch")
            refs = intent.get("experience_refs", [])
            require(isinstance(refs, list), "experience_refs must be a list")
            for ref in refs:
                require(isinstance(ref, dict) and set(ref) == {"experience_id", "lesson_id"}
                        and ref["experience_id"] in available and ref["lesson_id"] in available[ref["experience_id"]], "Invalid experience/lesson reference")
                cited.add(ref["lesson_id"])
            if require_calibrated_grasp and seq < len(history):
                row = history[seq]
                first_grasp = (grasp_sequence is None and intent.get("decision_origin") == "online_model"
                               and intent.get("semantic_operation") == "grasp"
                               and row["action"].get("operation") == "move"
                               and row["result"].get("status") == "completed")
                if first_grasp or grasp_sequence is not None:
                    require(row["sequence"] == seq and row["action"] == read_json(path.parent / "action.json")
                            and row["result"] == read_json(path.parent / "result.json"), "Grasp action/result history mismatch")
                    mapping = _target_mapping(row["action"].get("targets", []))
                    if first_grasp:
                        require({"experience_id": SOURCE_ID, "lesson_id": "calibrated_grasp_v2"} in refs,
                                "First completed online grasp does not cite calibrated_grasp_v2")
                        require(mapping == CALIBRATED_GRASP_TARGETS, "First grasp differs from calibrated thirteen targets")
                        grasp_sequence, initial_grasp, current_grasp = seq, dict(mapping), dict(mapping)
                    else:
                        changes = {name: {"previous": current_grasp[name], "value": value}
                                   for name, value in mapping.items() if name in current_grasp and current_grasp[name] != value}
                        if changes:
                            applied = (row["action"].get("operation") == "move" and row["result"].get("status") == "completed")
                            later_grip_changes.append({"sequence": seq, "result_status": row["result"].get("status"),
                                                       "applied": applied, "changes": changes})
                            if applied:
                                current_grasp.update({name: item["value"] for name, item in changes.items()})
            used = {"experience_id": SOURCE_ID, "lesson_id": "fixed_head_target"} in refs
            if not (used and intent.get("decision_origin") == "online_model" and intent.get("semantic_operation") in ALIGN_ROLES and seq < len(history)):
                continue
            row = history[seq]
            require(row["sequence"] == seq and row["action"] == read_json(path.parent / "action.json")
                    and row["result"] == read_json(path.parent / "result.json"), "Adoption action/result history mismatch")
            if row["action"].get("operation") != "move" or row["result"].get("status") != "completed":
                continue
            axis = obs["wrench_feedback"]["head_nut"]["nut_hinge_axis_world"]
            require(len(axis) == 3 and abs(math.sqrt(sum(v*v for v in axis)) - 1) <= 1e-6, "Invalid observed nut axis")
            wanted = [p + v * offset for p, v in zip(obs["nut"]["position_m"], axis)]
            require(math.dist(intent["target_head_m"], wanted) <= 1e-6, "Working-end target does not follow current nut axis")
            lift, contacts = obs["task_progress"]["gates"]["lift"], obs["contacts"]
            if (lift["passed"] is True and lift["passed_at_s"] is not None and lift["passed_at_s"] <= obs["time_s"]
                    and len(set(contacts["hand_tool_contacts"])) >= 2 and not contacts["tool_support_contacts"]
                    and not contacts["tool_nut_contact"] and not contacts["head_nut_contact"]
                    and row["result"]["time_s"] > obs["time_s"]):
                adopted.append(seq)
        metadata_consistent = True
        require(not expected_present or bool(adopted), "No completed online post-lift unsupported alignment adopts fixed_head_target")
        require(not require_calibrated_grasp or grasp_sequence is not None, "No completed online calibrated grasp")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {"audit_version": 2 if require_calibrated_grasp else 1, "episode": str(episode), "accepted": not errors, "errors": errors,
            "expected_present": expected_present, "memory_present": present, "memory_canonical_sha256": digest,
            "source_episode_id": SOURCE_ID, "request_count": count, "metadata_and_requests_consistent": metadata_consistent,
            "cited_lesson_ids": sorted(cited), "adopted_sequences": adopted, "episode_error": episode_error,
            "online_alignment_roles": sorted(ALIGN_ROLES), "physical_success_claimed": False,
            "require_calibrated_grasp": require_calibrated_grasp, "calibrated_grasp_applied": grasp_sequence is not None,
            "calibrated_grasp_sequence": grasp_sequence, "initial_grasp_targets": initial_grasp,
            "later_grip_changes": later_grip_changes,
            "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "claim_scope": "Declared references and observable target adoption are consistent with the supplied record. This does not prove internal model use, causal memory benefit, physical success or transfer; combine independent physical/local audits."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("memory", type=Path)
    parser.add_argument("--expect-absent", action="store_true")
    parser.add_argument("--require-calibrated-grasp", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit_memory_use(args.episode, args.memory, expected_present=not args.expect_absent,
                              require_calibrated_grasp=args.require_calibrated_grasp)
    with args.output.open("x", encoding="utf8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
