"""Append the frozen D6 engineering pickup evidence to unchanged native memory v1."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dexhand_agent.experience import load_experience
from dexhand_agent.wrench_evaluate import WrenchEvaluator
from tools.build_wrench_memory import _file, _read, _require, _sha, validate_skill_memory

V1_SHA256 = "2e9a8b7374e6bb721601dd711436d6dce390e050f0dac8c658ef37ed37cd428f"
PICKUP_PROOF_SHA256 = "bce371a9b482964b475a7383105496dffad57ba266d245427c606c107eaade82"


def build_grip_memory(v1_path, d6_directory, pickup_proof):
    """Verify immutable inputs and add data only; never construct or drive a robot."""
    v1_path, directory, proof_path = (Path(p).resolve() for p in (v1_path, d6_directory, pickup_proof))
    _require(_sha(v1_path) == V1_SHA256, "Parent v1 SHA256 mismatch")
    _require(_sha(proof_path) == PICKUP_PROOF_SHA256, "Pickup proof SHA256 mismatch")
    original = validate_skill_memory(load_experience(v1_path))
    proof = _read(proof_path)
    _require(proof["kind"] == "engineering_partial_skill_evidence" and
             proof["accepted_for_pickup_only"] is True and all(proof[k] is False for k in
             ("full_task_validated", "native_agent", "held_out")), "Evidence must remain partial engineering pickup")
    relative_source = directory.relative_to(ROOT).as_posix()
    _require(proof["source"] == relative_source, "D6 source path mismatch")
    hashes = proof["source_sha256"]
    _require({"summary.json", "analysis.json", "actions-source.json", "actions.jsonl",
              "executed-actions-declaration.json", "one-parameter-difference.json",
              "evaluator-recomputed.json", "samples.jsonl.gz", "actuator-ids.json"} <= set(hashes),
             "Missing D6 source evidence")
    for name, digest in hashes.items():
        _require(_sha(_file(directory, name)) == digest, "D6 source SHA256 mismatch: " + name)
    summary = _read(directory / "summary.json")
    _require(summary == proof["physical"], "Pickup proof and summary differ")
    _require(summary["success"] is False and summary["native_agent"] is False and
             summary["held_out"] is False and summary["error"] is None and not summary["violations"],
             "Pickup cannot contain an error or claim native/full success")
    _require(all(summary[k] is True for k in ("original_lift_completed", "postlift_hold_completed",
             "quality_pass", "pickup_evidence_pass", "exact_prefix_match", "protected_inputs_unchanged")),
             "Incomplete or failed pickup evidence")
    lift = summary["gates"]["lift"]
    _require(lift["passed"] is True and lift["best_continuous_s"] >= lift["required_continuous_s"] >= 2 and
             all(summary["gates"][k]["passed"] is False for k in ("turn", "withdraw")),
             "Only the independent lift gate may pass")

    rows = [json.loads(line) for line in (directory / "actions.jsonl").read_text(encoding="utf8").splitlines()]
    before = [r for r in rows if r["event"] == "before_execute"]
    executed = [r for r in rows if r["event"] == "after_execute"]
    _require([r["sequence"] for r in before] == list(range(5)) == [r["sequence"] for r in executed] and
             all(r["result"]["status"] == "completed" for r in executed), "D6 actions did not all complete")
    _require(all(a["action"] == b["action"] for a, b in zip(before, executed)), "Executed action binding differs")
    source = _read(directory / "actions-source.json")
    declaration = _read(directory / "executed-actions-declaration.json")
    _require(all(executed[i]["action"] == declaration[i]["action"] for i in range(4)), "Action declaration mismatch")
    _require(all(executed[i]["action"] == source[i]["action"] for i in (0, 1, 3)), "More than grasp parameter changed")
    nominal = source[2]["action"]["targets"]
    targets = executed[2]["action"]["targets"]
    _require(len(nominal) == 13 and nominal == original["trajectory"][2]["action"]["targets"],
             "Original 13-target template mismatch")
    changed = proof["one_changed_parameter"]
    _require(changed["actuator"] == "rh_A_THJ4" and changed["original"] == nominal[9]["value"] and
             changed["candidate"] == .9453804236340059 and
             abs(changed["candidate"] - changed["original"] - .02) < 1e-15, "Unexpected calibrated parameter")
    expected = copy.deepcopy(source[2]["action"])
    expected["targets"][9]["value"] = changed["candidate"]
    _require(expected == executed[2]["action"], "Only the one declared target parameter may change")
    difference = _read(directory / "one-parameter-difference.json")
    _require(difference["exact_match"] is False and difference["differences"] == [{
        "field": "/targets/9/value", "expected": changed["original"], "actual": changed["candidate"]}] and
        abs(difference["maximum_numeric_abs_difference"] - .02) < 1e-15, "One-parameter difference evidence mismatch")
    analysis = _read(directory / "analysis.json")
    partial, protocol = analysis["engineering_pickup_evidence"], analysis["protocol_actions"]
    _require(all(partial[k] is True for k in ("passed", "independently_recomputed_from_all_raw_samples",
             "postlift_hold_entirely_fixture_and_nut_free")) and partial["full_task_success"] is False,
             "Independent partial pickup evidence failed")
    _require(all(protocol["original_approach0_descent1_lift3_actionexact"].values()) and
             protocol["changed_grasp_equals_original_with_only_declared_target_replaced"] is True and
             protocol["no_calibration_or_prelift_screen_or_extra_hold"] is True and
             protocol["THJ4_command_during_lift_and_hold"] == changed["candidate"] and
             protocol["THJ4_command_max_abs_deviation_during_lift_and_hold"] == 0,
             "Unexpected procedure or nonconstant control evidence")

    evaluator = WrenchEvaluator(summary["metrics"]["initial_height_m"], summary["metrics"]["initial_nut_angle_rad"])
    ids = _read(directory / "actuator-ids.json")
    control_samples = 0
    with gzip.open(directory / "samples.jsonl.gz", "rt", encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            sample = row["physical_sample"]
            evaluator.update(sample)
            _require(0 <= sample["max_contact_force_n"] <= 25 and
                     0 <= sample["max_penetration_m"] <= .002 and not sample["robot_nut_contact"],
                     "Raw sample quality failed")
            if row["sequence"] in (3, 4):
                control_samples += 1
                _require(all(row["ctrl"][ids[t["actuator"]]] == t["value"] for t in targets),
                         "Grasp control was not constant during lift/hold")
    recomputed = _read(directory / "evaluator-recomputed.json")
    physical = evaluator.summary()
    _require(control_samples > 0 and recomputed["exact_match"] is True and
             physical == recomputed["evaluator"] == {k: summary[k] for k in physical},
             "Independent saved-sample lift/quality replay mismatch")
    quality = recomputed["quality"]
    _require(all(quality.values()) and analysis["quality_independent_file_recheck"]["passed"] is True,
             "Independent pickup quality failed")

    def evidence(name, sequences=None):
        item = {"path": relative_source + "/" + name, "sha256": hashes[name]}
        if sequences is not None:
            item["sequences"] = sequences
        return item

    record = copy.deepcopy(original)
    memory = record["skill_memory"]
    memory.update({"revision": 2, "parent_file_sha256": _sha(v1_path),
        "kind": "single_native_episode_plus_partial_grasp_calibration",
        "augmentation": {"source": relative_source, "kind": "engineering_pickup_only",
            "native_agent": False, "full_task_validated": False, "neural_training": False,
            "disclosure": "Append D6's one-target engineering pickup result only. Original native003 record, three lessons and acceptance provenance remain unchanged; no D3-D5 pregrasp calibration is included."}})
    memory["lessons"].append({"id": "calibrated_grasp_v2",
        "condition": {"task": original["applicability"]["task"], "tool": original["applicability"]["tool"],
            "robot": original["applicability"]["robot"],
            "geometry_requirement": "Same public Panda/Shadow robot and wrench geometry; other geometries are unverified."},
        "calibrated_targets": copy.deepcopy(targets),
        "action_outline": "Use these 13 targets as v2's first-closing starting configuration, taking precedence over the old nominal trajectory targets. This engineering starting configuration passed development pickup only; inspect new observations and independently verify full-task and new-layout performance. Do not add D3-D5 pregrasp calibration.",
        "observed": {"changed_parameter": changed, "action_sequence": 2,
            "lift_and_hold_sequences": [3, 4], "constant_control_samples": control_samples,
            "max_lift_m": physical["metrics"]["max_lift_m"],
            "max_contact_force_n": physical["metrics"]["max_contact_force_n"],
            "max_penetration_m": physical["metrics"]["max_penetration_m"]},
        "validation": {"kind": "engineering_pickup_only", "full_task_validated": False,
            "native_agent": False, "held_out": False, "source_files_sha256": copy.deepcopy(hashes),
            "lift_gate": copy.deepcopy(lift), "quality_pass": True},
        "evidence": [{"path": proof_path.relative_to(ROOT).as_posix(), "sha256": _sha(proof_path)},
            evidence("actions.jsonl", [2, 3, 4]), evidence("summary.json"),
            evidence("one-parameter-difference.json", [2]), evidence("evaluator-recomputed.json"),
            evidence("analysis.json"), evidence("samples.jsonl.gz", [3, 4])],
        "limitations": ["One development seed5 placement; not a native or held-out result.",
            "Only pickup passed; turn/withdraw, full task and new layouts remain unvalidated for this augmentation.",
            "No learned neural weights, universal grasp mechanism or transfer success is claimed."]})
    json.dumps(record, allow_nan=False)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v1", "d6", "pickup_proof", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    record = build_grip_memory(args.v1, args.d6, args.pickup_proof)
    with args.output.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    _require(load_experience(args.output) == record, "Generated memory failed standard loading")


if __name__ == "__main__":
    main()
