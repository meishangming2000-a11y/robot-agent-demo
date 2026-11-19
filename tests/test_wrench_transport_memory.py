"""The transport lesson must preserve its failed core and reject bad proof."""
import gzip
import hashlib
import json
import zipfile

import pytest

from dexhand_agent.experience import export_experience, load_experience
from dexhand_agent.wrench_evaluate import WrenchEvaluator
from tools import build_wrench_transport_memory as builder


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source(tmp_path, monkeypatch):
    """Synthetic saved records only: no real runs, assets, video or simulator."""
    def write(name, value):
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf8")
    def sample(i):
        return {"time_s": 1 + (i+1)*.002, "wrench_pos_m": [.5, .1, .3],
            "wrench_quat_wxyz": [1, 0, 0, 0], "hand_tool_contacts": ["index", "thumb"],
            "tool_support_contacts": [], "head_nut_contact": False, "tool_nut_contact": False,
            "robot_nut_contact": False, "nut_angle_rad": 0, "head_clearance_m": .1,
            "max_contact_force_n": 1, "max_penetration_m": 0}
    episode = "47e79d2bd33144f59d4edc0797bc75f9"
    tool = {"name": "wrench", "position_m": [.5, .1, .3], "quat_wxyz": [1, 0, 0, 0], "initial_height_m": .24}
    robot = {"palm_position_m": [.4, .2, .4], "palm_quat_wxyz": [1, 0, 0, 0]}
    write(builder.EPISODE + "/metadata.json", {"episode_id": episode, "backend": "native", "task_id": "wrench_turn", "seed": 5, "observation_mode": "simulator_state"})
    write(builder.EPISODE + "/steps/000/request.json", {"episode_id": episode, "goal": "test fixture", "observation": {"robot": robot, "tool": tool, "nut": {"position_m": [.505, .175, .245]}}})
    original = [{"sequence": i, "action": {"operation": "move" if i != 7 else "finish", "targets": [], "palm_target_m": [.4, .2, .3], "palm_quat_wxyz": [1, 0, 0, 0], "duration_s": .6, "memory": "fixture"}, "result": {"status": "completed"}} for i in range(8)]
    write(builder.EPISODE + "/actions.json", original)
    raw = tmp_path / builder.EPISODE / "samples.jsonl"
    raw.write_text("".join(json.dumps(sample(i))+"\n" for i in range(5800)), encoding="utf8")
    evaluator = WrenchEvaluator(.24, 0)
    for i in range(5800): evaluator.update(sample(i))
    write(builder.EPISODE + "/summary.json", {**evaluator.summary(), "episode_id": episode, "error": None, "decisions": 8})
    write(builder.RUNS + "/development-protocol.json", {"maximum_engineering_runs": 2})
    target = [.505, .175, .345]
    write(builder.RUNS + "/run002-protocol.json", {"engineering_run_index": 2, "maximum_engineering_runs_total": 2, "feedback": {"target_head_m": target}})
    protected = [{"path": builder.EPISODE+"/"+name, "sha256": sha(tmp_path / builder.EPISODE / name)} for name in ("metadata.json", "summary.json", "actions.json", "steps/000/request.json", "samples.jsonl")]
    for run, counts in (("run-001-exact", [900, 600, 750, 750, 1100, 1400, 300]), ("run-002-raised", [900, 600, 750, 750, 1100, 1400, 300, 300, 500])):
        name = builder.RUNS+"/"+run; actions = []; rows = []; evaluator = WrenchEvaluator(.24, 0)
        for sequence, count in enumerate(counts):
            command = json.loads(json.dumps(original[min(sequence, 6)]["action"]))
            if sequence == 5 and len(counts) == 9: command["palm_target_m"][2] += .06
            if sequence == 8: command.update(operation="hold", palm_target_m=None, palm_quat_wxyz=None, duration_s=1)
            head = [target[0]+({5: .021371365702789736, 6: .00569923707334752}.get(sequence, .00151797101932318)), *target[1:]]
            for _ in range(count):
                i = len(rows); s = sample(i); evaluator.update(s)
                rows.append({"sample_index": i, "sequence": sequence, "physical_sample": s, "head_center_world_m": head})
            action = {"event": "after_execute", "sequence": sequence, "action": command, "result": {"status": "completed"}, "last_sample_index": len(rows)-1, "after": {"observation": {"contacts": rows[-1]["physical_sample"], "tool": {"head_center_m": head}}}}
            if len(counts) == 9 and sequence in (6, 7):
                action["feedback_input"] = {"target_head_m": target, "target_tool_quat_wxyz": [1, 0, 0, 0], "duration_s": .6, **{"previous_"+k: actions[-1]["action"][k] for k in ("palm_target_m", "palm_quat_wxyz")}}
            actions.append(action)
        path = tmp_path / name; path.mkdir(parents=True)
        (path / "actions.jsonl").write_text("".join(json.dumps({**a, "event": event})+"\n" for a in actions for event in ("before_execute", "after_execute")), encoding="utf8")
        with gzip.open(path / "samples.jsonl.gz", "wt", encoding="utf8") as stream:
            stream.write("".join(json.dumps(r)+"\n" for r in rows))
        write(name+"/summary.json", {**evaluator.summary(), "error": None, "first_bad": None, "native_agent": False, "held_out": False, "protected_inputs_unchanged": True, "correction_calls": 2, "hold_executed": True})
        snapshot = b"fixture frozen source"
        with zipfile.ZipFile(path / "source-protocol-snapshot.zip", "w") as archive: archive.writestr("fixture.py", snapshot)
        write(name+"/metadata.json", {"protected_inputs_before": protected, "source_snapshot_files": [{"path": "fixture.py", "sha256": hashlib.sha256(snapshot).hexdigest()}]})
    write(builder.AUDIT, {"candidate_execution_audited": True, "prefix_exact_match": True, "hold_all_samples_aligned_safe": True, "samples": 6600, "evaluator": {"exact_match": True}})
    files = [p for p in tmp_path.rglob("*") if p.is_file() and not p.is_relative_to(tmp_path / builder.EPISODE)]
    write(builder.MANIFEST, [{"path": p.relative_to(tmp_path).as_posix(), "sha256": sha(p)} for p in files])
    monkeypatch.setattr(builder, "MANIFEST_SHA256", sha(tmp_path / builder.MANIFEST))
    monkeypatch.setattr(builder, "AUDIT_SHA256", sha(tmp_path / builder.AUDIT))
    return tmp_path


def rehash(root, monkeypatch):
    path = root / builder.MANIFEST
    manifest = json.loads(path.read_text(encoding="utf8"))
    for item in manifest:
        item["sha256"] = sha(root / item["path"])
    path.write_text(json.dumps(manifest), encoding="utf8")
    monkeypatch.setattr(builder, "MANIFEST_SHA256", sha(path))


def test_only_adds_partial_lesson_to_the_exact_failed_core(source, tmp_path):
    original = export_experience(source / builder.EPISODE)
    result = builder.build_transport_memory(source)
    assert {k: v for k, v in result.items() if k != "skill_memory"} == original
    assert result["status"] == "failed_episode" and result["outcome"]["success"] is False
    assert len(result["trajectory"]) == 8 and len(result["source"]["files_sha256"]) == 4
    memory = result["skill_memory"]
    assert memory["kind"] == "failed_native_episode_plus_partial_engineering_transport_repair"
    assert all(memory[k] is False for k in ("full_task_validated", "native_repair_validated", "held_out", "neural_training"))
    lesson, = memory["lessons"]
    assert lesson["id"] == "high_clearance_transport"
    assert lesson["working_end_target"] == {"frame": "nut_axis", "axial_offset_m": .10}
    assert lesson["near_nut_alignment_target"] == {"frame": "nut_axis", "axial_offset_m": .04}
    assert lesson["observed"]["correction_calls"] == 2
    assert lesson["observed"]["head_error_before_correction_m"] == pytest.approx(.021371365702789736)
    assert lesson["observed"]["head_error_after_correction_m"] == pytest.approx(.00151797101932318)
    assert all(lesson["validation"][k] is False for k in
               ("full_task_validated", "native_repair_validated", "held_out", "neural_training"))
    output = tmp_path / "memory.json"
    output.write_text(json.dumps(result), encoding="utf8")
    assert load_experience(output) == result


@pytest.mark.parametrize("name", ["manifest", "audit", "raw", "failed_core"])
def test_source_hash_tamper_is_rejected(source, name):
    paths = {"manifest": builder.MANIFEST, "audit": builder.AUDIT,
             "raw": builder.RUNS + "/run-002-raised/samples.jsonl.gz",
             "failed_core": builder.EPISODE + "/actions.json"}
    path = source / paths[name]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA"):
        builder.build_transport_memory(source)


def test_unqualified_raw_hold_is_rejected_even_with_rehashed_evidence(source, monkeypatch):
    path = source / builder.RUNS / "run-002-raised/samples.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf8") as stream:
        rows = [json.loads(line) for line in stream]
    # Keep the passing audit text; an interior hold sample exceeds 2 mm.
    rows[-250]["head_center_world_m"][0] += .01
    with gzip.open(path, "wt", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    rehash(source, monkeypatch)
    with pytest.raises(ValueError, match="hold"):
        builder.build_transport_memory(source)


def test_unfinished_candidate_rejected_despite_passing_audit(source, monkeypatch):
    path = source / builder.RUNS / "run-002-raised/summary.json"
    summary = json.loads(path.read_text(encoding="utf8"))
    summary["error"] = "interrupted"
    path.write_text(json.dumps(summary), encoding="utf8")
    rehash(source, monkeypatch)
    with pytest.raises(ValueError, match="completed"):
        builder.build_transport_memory(source)
