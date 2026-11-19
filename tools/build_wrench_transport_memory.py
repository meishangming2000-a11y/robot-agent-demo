"""Build a partial transport lesson from saved evidence; never run simulation."""
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dexhand_agent.experience import export_experience
from dexhand_agent.wrench_evaluate import WrenchEvaluator
from tools.build_wrench_memory import _read, _require, _sha

EPISODE = "runs/wrench-placement-v2/reserved-minus20"
RUNS = "runs/wrench-placement-v3-development"
MANIFEST = "evidence/wrench-transport-v3/artifact-manifest.json"
AUDIT = RUNS + "/run-002-raised/candidate-audit.json"
MANIFEST_SHA256 = "d91f1943260d1838496d4e23ae08218b092e4c0e1eed77eab9ec8d3f4ff26467"
AUDIT_SHA256 = "a0a8faf563610f6ae2a26a704a893cf99b070e021067e918dbf2f4b7821610c2"


def build_transport_memory(root=ROOT):
    root = Path(root).resolve()
    _require(_sha(root / MANIFEST) == MANIFEST_SHA256, "Manifest SHA256 mismatch")
    manifest = _read(root / MANIFEST)
    for item in manifest:
        _require(_sha(root / item["path"]) == item["sha256"], "Source SHA256 mismatch: " + item["path"])
    _require(_sha(root / AUDIT) == AUDIT_SHA256, "Candidate audit SHA256 mismatch")
    audit, declared = _read(root / AUDIT), _read(root / RUNS / "run002-protocol.json")
    _require(audit["candidate_execution_audited"] and audit["prefix_exact_match"] and audit["hold_all_samples_aligned_safe"] and audit["samples"] == 6600 and audit["evaluator"]["exact_match"], "Incomplete candidate audit")
    _require(declared["engineering_run_index"] == declared["maximum_engineering_runs_total"] == 2 and _read(root / RUNS / "development-protocol.json")["maximum_engineering_runs"] == 2, "Expected exactly two declared runs")
    record = export_experience(root / EPISODE)
    _require(record["id"] == "47e79d2bd33144f59d4edc0797bc75f9" and record["status"] == "failed_episode" and not record["outcome"]["success"] and len(record["trajectory"]) == 8, "Expected original failed eight-action native episode")
    original = [json.loads(line) for line in (root / EPISODE / "samples.jsonl").read_text(encoding="utf8").splitlines()]
    _require(len(original) == 5800, "Original sample count mismatch")
    target, hold_errors, hold_tilts, all_actions = declared["feedback"]["target_head_m"], [], [], []
    for run, expected_count, action_count in (("run-001-exact", 5800, 7), ("run-002-raised", 6600, 9)):
        directory = root / RUNS / run
        metadata, summary = _read(directory / "metadata.json"), _read(directory / "summary.json")
        _require(summary["error"] is None and summary["first_bad"] is None and not summary["native_agent"] and not summary["held_out"] and summary["protected_inputs_unchanged"], "Engineering runs must have completed unchanged")
        protected = {item["path"]: item["sha256"] for item in metadata["protected_inputs_before"]}
        for name in (*record["source"]["files_sha256"], "samples.jsonl"):
            _require(_sha(root / EPISODE / name) == protected[EPISODE + "/" + name], "Failed source SHA256 mismatch")
        with zipfile.ZipFile(directory / "source-protocol-snapshot.zip") as archive:
            for item in metadata["source_snapshot_files"]:
                _require(hashlib.sha256(archive.read(item["path"])).hexdigest() == item["sha256"], "Snapshot source SHA256 mismatch")
        events = [json.loads(line) for line in (directory / "actions.jsonl").read_text(encoding="utf8").splitlines()]
        actions = [r for r in events if r["event"] == "after_execute"]
        _require([r["sequence"] for r in actions] == list(range(action_count)) and all(r["result"]["status"] == "completed" for r in actions), "All declared actions must have completed")
        _require([r["action"] for r in events if r["event"] == "before_execute"] == [r["action"] for r in actions], "Before/executed action binding mismatch")
        _require(all(actions[i]["action"] == record["trajectory"][i]["action"] for i in range(7 if run == "run-001-exact" else 5)), "Original action prefix mismatch")
        if run == "run-002-raised":
            raised = copy.deepcopy(record["trajectory"][5]["action"])
            raised["palm_target_m"][2] += .06
            _require(actions[5]["action"] == raised and summary["correction_calls"] == 2 and summary["hold_executed"], "Only seq5 Z +0.06 and two corrections/hold permitted")
            for i in (6, 7):
                args = actions[i]["feedback_input"]
                _require(args["target_head_m"] == target and args["target_tool_quat_wxyz"] == [1, 0, 0, 0] and args["duration_s"] == .6 and all(args["previous_" + k] == actions[i-1]["action"][k] for k in ("palm_target_m", "palm_quat_wxyz")), "Feedback target/previous command mismatch")
            _require(actions[8]["action"]["operation"] == "hold" and actions[8]["action"]["duration_s"] == 1, "Expected one-second hold")
        evaluator = WrenchEvaluator(summary["metrics"]["initial_height_m"], summary["metrics"]["initial_nut_angle_rad"])
        count, previous = 0, 1.0
        with gzip.open(directory / "samples.jsonl.gz", "rt", encoding="utf8") as stream:
            for line in stream:
                row = json.loads(line); sample = row["physical_sample"]; sequence = row["sequence"]
                _require(row["sample_index"] == count and abs(sample["time_s"] - previous - .002) < 1e-12, "500Hz sample ordering mismatch")
                _require(0 <= sample["max_contact_force_n"] <= 25 and 0 <= sample["max_penetration_m"] <= .002 and not sample["robot_nut_contact"], "Raw sample quality failed")
                if run == "run-001-exact" or count < 4100:
                    _require(sample == original[count], "Raw exact prefix mismatch")
                if run == "run-002-raised" and sequence >= 5:
                    _require(not sample["tool_nut_contact"] and not sample["tool_support_contacts"] and len(sample["hand_tool_contacts"]) >= 2, "Transport retention/contact failed")
                if run == "run-002-raised" and sequence == 8:
                    q = sample["wrench_quat_wxyz"]
                    error, tilt = math.dist(row["head_center_world_m"], target), math.degrees(math.acos(max(-1, min(1, 1-2*(q[1]**2+q[2]**2)))))
                    _require(error <= .002 and tilt <= 2, "Raw hold alignment failed")
                    hold_errors.append(error); hold_tilts.append(tilt)
                if count == actions[sequence]["last_sample_index"]:
                    obs = actions[sequence]["after"]["observation"]
                    _require(sample == obs["contacts"] and row["head_center_world_m"] == obs["tool"]["head_center_m"], "Action observation/raw sample binding mismatch")
                evaluator.update(sample); previous = sample["time_s"]; count += 1
        _require(count == expected_count and evaluator.summary() == {k: summary[k] for k in evaluator.summary()}, "Raw evaluator replay mismatch")
        all_actions.append(actions)
    _require(len(hold_errors) == 500, "Expected 500 qualified hold samples")
    actions = all_actions[1]
    errors = [math.dist(actions[i]["after"]["observation"]["tool"]["head_center_m"], target) for i in (5, 6, 7)]
    _require(errors[2] <= .002 < errors[0], "Correction did not establish fixed-target alignment")
    record["skill_memory"] = {"schema_version": 1, "source_episode_id": record["id"],
        "kind": "failed_native_episode_plus_partial_engineering_transport_repair",
        "full_task_validated": False, "native_repair_validated": False,
        "native_agent": False, "held_out": False, "neural_training": False,
        "acquisition": "Failed native episode plus two completed engineering diagnostics; no new neural network or weights.",
        "lessons": [{"id": "high_clearance_transport",
            "condition": {"after_lift": True, "retained_unsupported": True, "minimum_hand_parts": 2, "no_nut_or_fixture_contact": True, "geometry": "Same public wrench and Panda/Shadow robot; changed layouts unvalidated"},
            "working_end_target": {"frame": "nut_axis", "axial_offset_m": 0.10},
            "near_nut_alignment_target": {"frame": "nut_axis", "axial_offset_m": 0.04},
            "action_outline": "From fresh measured nut center/axis, recompute a high head waypoint at nut_center + 0.10m * nut_axis; transport and level the socket, then use fresh feedback_tool_pose_action observations and the most recently completed full palm command to correct the same waypoint. The engineering candidate needed two corrections. Recompute targets and inspect actual motion/contact; never blindly replay source coordinates. A safe endpoint does not constrain the whole IK/joint-interpolated path. The original +0.04m fixed_head_target is only a later near-nut alignment point after high transport, not the long-transport target or a safe path guarantee. Stop for nut/fixture contact, grip loss or quality breach; insertion/turn/full task require separate verification.",
            "observed": {"correction_calls": summary["correction_calls"], "head_error_before_correction_m": errors[0], "head_error_after_first_correction_m": errors[1], "head_error_after_correction_m": errors[2], "hold_samples": len(hold_errors), "hold_max_head_error_m": max(hold_errors), "hold_max_axis_tilt_deg": max(hold_tilts)},
            "validation": {"kind": "engineering_high_transport_and_fixed_target_alignment_only", "full_task_validated": False, "native_repair_validated": False, "native_agent": False, "held_out": False, "neural_training": False, "engineering_runs": 2},
            "evidence": [{"path": MANIFEST, "sha256": MANIFEST_SHA256}, *manifest]}]}
    return record


def main():
    record = build_transport_memory()
    output = ROOT / "experiences/wrench-transport-v3.json"
    with output.open("x", encoding="utf8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps({"path": str(output), "sha256": _sha(output), "status": record["status"]}))


if __name__ == "__main__":
    main()
