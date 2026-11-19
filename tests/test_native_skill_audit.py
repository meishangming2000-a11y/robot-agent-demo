import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.audit_native_skill import audit_episode, main
from tools.audit_wrench_autonomy import canonical_sha


ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf8")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


class TinyPolicy:
    """Deterministic observation-only replay dependency; never creates a model."""

    def __init__(self, initial_observation, previous_palm_action, max_steps=80):
        self.initial_sha = canonical_sha(initial_observation)
        self.previous = copy.deepcopy(previous_palm_action)
        self.max_steps = max_steps
        self.steps, self.phase, self.status, self.last_goal = 0, "extract", "ready", None

    def describe(self):
        return {"version": "synthetic-replay", "initial_observation_sha256": self.initial_sha,
                "previous_palm_action": self.previous, "max_steps": self.max_steps,
                "steps": self.steps, "phase": self.phase, "status": self.status,
                "last_goal": self.last_goal}

    def decide(self, observation):
        if self.steps >= 2:
            self.status = "clearance_check"
            return None
        self.steps += 1
        self.phase = "extract" if self.steps == 1 else "hold"
        self.status = "awaiting_result"
        self.last_goal = {"target_head_m": [0.1, 0, observation["time_s"] * .001]}
        return {"operation": "move" if self.steps == 1 else "hold", "targets": [],
                "palm_target_m": [0, 0, .2] if self.steps == 1 else None,
                "palm_quat_wxyz": [1, 0, 0, 0] if self.steps == 1 else None,
                "duration_s": .3, "memory": f"fresh time {observation['time_s']}"}


@pytest.fixture
def recorded(tmp_path):
    episode, freeze = tmp_path / "episode", tmp_path / "freeze.json"
    paths = ["dexhand_agent/wrench_recovery.py", "dexhand_agent/wrench_tools.py", "tools/drive_native_skill.py"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
    save(freeze, {"files_sha256": hashes})
    source_sha = hashes[paths[0]]
    online = {"operation": "move", "targets": [], "palm_target_m": [0, 0, .1],
              "palm_quat_wxyz": [1, 0, 0, 0], "duration_s": 1., "memory": "online"}
    finish = {"operation": "finish", "targets": [], "palm_target_m": None,
              "palm_quat_wxyz": None, "duration_s": .02, "memory": "honest finish"}
    requests = []
    for seq in range(4):
        requests.append({"request_id": f"test-{seq}", "episode_id": "test", "sequence": seq,
                         "observation": {"time_s": float(seq),
                                         "robot": {"palm_position_m": [0, 0, .09],
                                                   "palm_quat_wxyz": [1, 0, 0, 0]},
                                         "last_action": {"status": "ready"}}})
    call = episode / "actor/skill-calls/withdraw"
    selection = {"skill_call_id": "withdraw", "decision_origin": "online_model",
                 "request_id": "test-1", "sequence": 1,
                 "observation_sha256": canonical_sha(requests[1]["observation"]),
                 "policy": "WrenchWithdrawPolicy", "diagnosis": "choose a declared skill"}
    # Complete result/last_action bindings before capturing observations in logs.
    history = []
    results = [{"status": "completed", "time_s": 1.}, {"status": "completed", "time_s": 2.},
               {"status": "completed", "time_s": 3.}, {"status": "finished", "time_s": 3.}]
    for seq in range(1, 4):
        requests[seq]["observation"]["last_action"] = results[seq - 1]
    policy = TinyPolicy(requests[1]["observation"], online)
    selection["observation_sha256"] = canonical_sha(requests[1]["observation"])
    selection_path = episode / "actor/skill-calls/withdraw-selection.json"
    save(selection_path, selection)
    initial = {"skill_call_id": "withdraw", "invoked": True, "policy_state": policy.describe(),
               "policy_source_path": str(ROOT / paths[0]), "policy_source_sha256": source_sha,
               "start_request_id": "test-1", "start_sequence": 1,
               "start_observation_sha256": canonical_sha(requests[1]["observation"]),
               "actor_selection_path": str(selection_path),
               "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
               "maximum_replies": 80, "poll_interval_s": .05, "timeout_s": 120.}
    save(call / "initial.json", initial)
    local = {}
    for seq in (1, 2, 3):
        obs = requests[seq]["observation"]
        before = policy.describe()
        action = policy.decide(copy.deepcopy(obs))
        after = policy.describe()
        save(call / f"decisions/{seq:03}.json", {
            "request_id": f"test-{seq}", "sequence": seq, "observation_sha256": canonical_sha(obs),
            "action": action, "goal": policy.last_goal, "policy_state_before": before,
            "policy_state_after": after, "policy_step": policy.steps, "policy_source_sha256": source_sha})
        if action is not None:
            local[seq] = action
            save(episode / f"actor/intents/{seq:03}.json", {
                "request_id": f"test-{seq}", "observation_sha256": canonical_sha(obs),
                "semantic_operation": policy.phase, "target_head_m": policy.last_goal["target_head_m"],
                "diagnosis": f"Explicit local feedback policy synthetic-replay: {policy.status}",
                "correction_trigger": None, "decision_origin": "local_feedback_skill",
                "skill_call_id": "withdraw", "policy_source_sha256": source_sha,
                "policy_step": policy.steps, "goal": policy.last_goal})
    save(call / "handoff.json", {"status": "handoff", "skill_call_id": "withdraw", "replies": 2,
         "request_id": "test-3", "sequence": 3,
         "observation_sha256": canonical_sha(requests[3]["observation"]),
         "policy_state": policy.describe(), "policy_source_sha256": source_sha})
    for seq in range(4):
        action = online if seq == 0 else finish if seq == 3 else local[seq]
        history.append({"sequence": seq, "action": action, "result": results[seq]})
        for prefix in ("steps", "actor/steps"):
            save(episode / f"{prefix}/{seq:03}/request.json", requests[seq])
        save(episode / f"steps/{seq:03}/action.json", action)
        save(episode / f"steps/{seq:03}/result.json", results[seq])
        save(episode / f"actor/steps/{seq:03}/reply.json", {"request_id": f"test-{seq}", "action": action})
        if seq in (0, 3):
            save(episode / f"actor/intents/{seq:03}.json", {
                "request_id": f"test-{seq}", "observation_sha256": canonical_sha(requests[seq]["observation"]),
                "semantic_operation": action["operation"], "target_head_m": None,
                "diagnosis": "online choice", "correction_trigger": None, "decision_origin": "online_model"})
    save(episode / "actions.json", history)
    save(episode / "metadata.json", {"episode_id": "test", "backend": "native"})
    save(episode / "summary.json", {"episode_id": "test", "backend": "native", "decisions": 4,
                                     "success": False, "error": None})
    return episode, freeze


def test_exact_replay_validates_trace_without_claiming_physical_success(recorded):
    result = audit_episode(*recorded, policy_type=TinyPolicy)
    assert result["accepted"]
    assert result["trace"]["passed"]
    assert result["counts"] == {"low_level_commands": 4, "online_action_choices": 2,
                                 "skill_selections": 1, "online_model_choices": 3,
                                 "local_motor_commands": 2, "unclassified_commands": 0}
    assert result["calls"][0]["replayed_decisions"] == 3
    assert result["calls"][0]["local_motor_commands"] == 2
    assert result["budget_limits"] == {"max_online_choices": 32, "max_commands": 160}
    assert result["budgets"] == {"low_level_commands_at_most_160": True,
                                  "online_model_choices_at_most_32": True}
    assert "physical" in result["claim_scope"]


@pytest.mark.parametrize("relative,field,value", [
    ("actor/skill-calls/withdraw/initial.json", "policy_source_sha256", "0" * 64),
    ("actor/skill-calls/withdraw/decisions/001.json", "policy_step", 9),
    ("actor/skill-calls/withdraw/decisions/001.json", "observation_sha256", "0" * 64),
    ("actor/skill-calls/withdraw/handoff.json", "request_id", "stale"),
    ("actor/intents/001.json", "skill_call_id", "other"),
    ("actor/steps/001/reply.json", "request_id", "old"),
    ("actor/skill-calls/withdraw-selection.json", "sequence", 0),
    ("actor/intents/003.json", "decision_origin", "unknown"),
])
def test_binding_or_provenance_mutation_fails(recorded, relative, field, value):
    episode, freeze = recorded
    item = read(episode / relative)
    item[field] = value
    save(episode / relative, item)
    assert not audit_episode(episode, freeze, policy_type=TinyPolicy)["accepted"]


@pytest.mark.parametrize("field", ["action", "policy_state_before", "policy_state_after", "goal"])
def test_exact_decision_output_and_states_cannot_be_rewritten(recorded, field):
    episode, freeze = recorded
    path = episode / "actor/skill-calls/withdraw/decisions/001.json"
    item = read(path)
    item[field] = {"tampered": True}
    save(path, item)
    assert not audit_episode(episode, freeze, policy_type=TinyPolicy)["accepted"]


def test_previous_initial_command_must_be_owned_completed_and_full(recorded):
    episode, freeze = recorded
    history = read(episode / "actions.json")
    history[0]["result"]["status"] = "rejected"
    save(episode / "actions.json", history)
    save(episode / "steps/000/result.json", history[0]["result"])
    for prefix in ("steps", "actor/steps"):
        path = episode / f"{prefix}/001/request.json"
        item = read(path)
        item["observation"]["last_action"] = history[0]["result"]
        save(path, item)
    obs_sha = canonical_sha(item["observation"])
    path = episode / "actor/intents/001.json"
    intent = read(path)
    intent["observation_sha256"] = obs_sha
    save(path, intent)
    selection_path = episode / "actor/skill-calls/withdraw-selection.json"
    selection = read(selection_path)
    selection["observation_sha256"] = obs_sha
    save(selection_path, selection)
    path = episode / "actor/skill-calls/withdraw/initial.json"
    initial = read(path)
    initial["start_observation_sha256"] = obs_sha
    initial["selection_sha256"] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    save(path, initial)
    result = audit_episode(episode, freeze, policy_type=TinyPolicy)
    assert not result["accepted"]
    assert any("previous" in error.lower() for error in result["trace"]["errors"])


def test_null_component_does_not_count_as_a_full_accepted_palm_action(recorded):
    episode, freeze = recorded
    for relative in ("steps/000/action.json", "actor/steps/000/reply.json"):
        item = read(episode / relative)
        (item["action"] if "action" in item else item)["palm_quat_wxyz"] = None
        save(episode / relative, item)
    history = read(episode / "actions.json")
    history[0]["action"]["palm_quat_wxyz"] = None
    save(episode / "actions.json", history)
    assert not audit_episode(episode, freeze, policy_type=TinyPolicy)["accepted"]


def test_source_freeze_mismatch_never_instantiates_policy(recorded):
    episode, freeze = recorded
    item = read(freeze)
    item["files_sha256"]["dexhand_agent/wrench_tools.py"] = "0" * 64
    save(freeze, item)

    def forbidden(*args, **kwargs):
        pytest.fail("Unverified frozen sources must not be replayed")

    result = audit_episode(episode, freeze, policy_type=forbidden)
    assert not result["accepted"]
    assert not result["sources"]["passed"]


def test_missing_selection_or_local_trace_fails_closed(recorded):
    episode, freeze = recorded
    (episode / "actor/skill-calls/withdraw-selection.json").unlink()
    (episode / "actor/skill-calls/withdraw/decisions/001.json").unlink()
    result = audit_episode(episode, freeze, policy_type=TinyPolicy)
    assert not result["accepted"]


def test_record_paths_must_not_escape_episode(recorded, tmp_path):
    episode, freeze = recorded
    initial_path = episode / "actor/skill-calls/withdraw/initial.json"
    item = read(initial_path)
    item["actor_selection_path"] = str(tmp_path / "outside-selection.json")
    save(initial_path, item)
    assert not audit_episode(episode, freeze, policy_type=TinyPolicy)["accepted"]


def test_cli_output_refuses_overwrite(recorded, tmp_path, monkeypatch):
    episode, freeze = recorded
    output = tmp_path / "audit.json"
    save(output, {"keep": True})
    monkeypatch.setattr("sys.argv", ["audit", str(episode), "--source-freeze", str(freeze),
                                     "--output", str(output)])
    with pytest.raises(FileExistsError):
        main()
    assert read(output) == {"keep": True}


def terminal_call(episode, name, selected=False):
    initial = read(episode / "actor/skill-calls/withdraw/initial.json")
    initial.update(skill_call_id=name, invoked=False, start_request_id=None, start_sequence=None,
                   start_observation_sha256=None, actor_selection_path=None, selection_sha256=None)
    save(episode / f"actor/skill-calls/{name}/initial.json", initial)
    save(episode / f"actor/skill-calls/{name}/handoff.json", {
        "status": "episode_complete", "skill_call_id": name, "replies": 0,
        "request_id": None, "sequence": None, "observation_sha256": None,
        "policy_state": initial["policy_state"], "policy_source_sha256": initial["policy_source_sha256"]})
    if selected:
        selection = read(episode / "actor/skill-calls/withdraw-selection.json")
        selection["skill_call_id"] = name
        save(episode / f"actor/skill-calls/{name}-selection.json", selection)


def test_not_invoked_call_never_invents_selection_or_local_commands(recorded):
    episode, freeze = recorded
    terminal_call(episode, "terminal")
    result = audit_episode(episode, freeze, policy_type=TinyPolicy)
    assert result["accepted"]
    assert result["counts"]["online_model_choices"] == 3
    assert next(c for c in result["calls"] if c["skill_call_id"] == "terminal")["status"] == "not_invoked"


@pytest.mark.parametrize("extra,allowed", [(29, True), (30, False)])
def test_online_budget_counts_zero_reply_selections(recorded, extra, allowed):
    episode, freeze = recorded
    for n in range(extra):
        terminal_call(episode, f"terminal{n}", selected=True)
    result = audit_episode(episode, freeze, policy_type=TinyPolicy)
    assert result["trace"]["passed"]
    assert result["counts"]["online_model_choices"] == extra + 3
    assert result["budgets"]["online_model_choices_at_most_32"] is allowed
    assert result["accepted"] is allowed


@pytest.mark.parametrize("extra,allowed", [(30, True), (45, True), (46, False)])
def test_explicit_online_budget_does_not_raise_the_default(recorded, extra, allowed):
    episode, freeze = recorded
    for n in range(extra):
        terminal_call(episode, f"terminal{n}", selected=True)
    default = audit_episode(episode, freeze, policy_type=TinyPolicy)
    explicit = audit_episode(episode, freeze, policy_type=TinyPolicy, max_online_choices=48)
    assert default["trace"]["passed"] and explicit["trace"]["passed"]
    assert default["counts"] == explicit["counts"]
    assert default["counts"]["online_model_choices"] == extra + 3
    assert not default["accepted"]
    assert default["budget_limits"]["max_online_choices"] == 32
    assert explicit["budget_limits"] == {"max_online_choices": 48, "max_commands": 160}
    assert explicit["budgets"]["online_model_choices_at_most_48"] is allowed
    assert explicit["accepted"] is allowed


def extend_online_history(episode, count):
    """Retain the local handoff, then add complete online records before finish."""
    original = read(episode / "actions.json")
    history = original[:3]
    template = read(episode / "steps/003/request.json")
    for seq in range(3, count):
        request = copy.deepcopy(template)
        request.update(sequence=seq, request_id=f"test-{seq}")
        request["observation"].update(time_s=float(seq), last_action=history[-1]["result"])
        action = copy.deepcopy(original[-1 if seq == count - 1 else 0]["action"])
        result = {"status": "finished" if seq == count - 1 else "completed", "time_s": float(seq + 1)}
        history.append({"sequence": seq, "action": action, "result": result})
        for prefix in ("steps", "actor/steps"):
            save(episode / f"{prefix}/{seq:03}/request.json", request)
        save(episode / f"steps/{seq:03}/action.json", action)
        save(episode / f"steps/{seq:03}/result.json", result)
        save(episode / f"actor/steps/{seq:03}/reply.json", {"request_id": f"test-{seq}", "action": action})
        save(episode / f"actor/intents/{seq:03}.json", {
            "request_id": f"test-{seq}", "observation_sha256": canonical_sha(request["observation"]),
            "semantic_operation": action["operation"], "target_head_m": None,
            "diagnosis": "online choice", "correction_trigger": None, "decision_origin": "online_model"})
    save(episode / "actions.json", history)
    summary = read(episode / "summary.json")
    save(episode / "summary.json", {**summary, "decisions": count})


def test_explicit_command_budget_accepts_a_complete_trace_above_160(recorded):
    episode, freeze = recorded
    extend_online_history(episode, 161)
    default = audit_episode(episode, freeze, policy_type=TinyPolicy, max_online_choices=200)
    explicit = audit_episode(episode, freeze, policy_type=TinyPolicy, max_online_choices=200, max_commands=161)
    assert default["counts"]["low_level_commands"] == explicit["counts"]["low_level_commands"] == 161
    assert not default["budgets"]["low_level_commands_at_most_160"]
    assert not default["accepted"]
    assert explicit["trace"]["passed"], explicit["trace"]["errors"]
    assert explicit["budgets"]["low_level_commands_at_most_161"]
    assert explicit["budget_limits"] == {"max_online_choices": 200, "max_commands": 161}
    assert explicit["accepted"]


@pytest.mark.parametrize("name", ["max_online_choices", "max_commands"])
@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.5, "48", None])
def test_api_budget_limits_require_positive_integers(recorded, name, invalid):
    with pytest.raises(ValueError, match=name):
        audit_episode(*recorded, policy_type=TinyPolicy, **{name: invalid})


def test_cli_accepts_and_reports_explicit_budgets(recorded, tmp_path, monkeypatch):
    episode, freeze = recorded
    output = tmp_path / "explicit-budget.json"
    monkeypatch.setattr("dexhand_agent.wrench_recovery.WrenchWithdrawPolicy", TinyPolicy)
    monkeypatch.setattr("sys.argv", ["audit", str(episode), "--source-freeze", str(freeze),
                                    "--max-online-choices", "48", "--max-commands", "4", "--output", str(output)])
    assert main() == 0
    assert read(output)["budget_limits"] == {"max_online_choices": 48, "max_commands": 4}


@pytest.mark.parametrize("option", ["--max-online-choices", "--max-commands"])
@pytest.mark.parametrize("invalid", ["0", "-1"])
def test_cli_budget_limits_reject_nonpositive_integers(recorded, tmp_path, monkeypatch, capsys, option, invalid):
    episode, freeze = recorded
    output = tmp_path / "invalid-budget.json"
    monkeypatch.setattr("sys.argv", ["audit", str(episode), "--source-freeze", str(freeze),
                                    option, invalid, "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "positive" in capsys.readouterr().err
    assert not output.exists()


def test_low_level_budget_includes_every_history_slot(recorded):
    episode, freeze = recorded
    history = read(episode / "actions.json")
    while len(history) < 161:
        history.append({**copy.deepcopy(history[-1]), "sequence": len(history)})
    save(episode / "actions.json", history)
    summary = read(episode / "summary.json")
    summary["decisions"] = 161
    save(episode / "summary.json", summary)
    result = audit_episode(episode, freeze, policy_type=TinyPolicy)
    assert result["counts"]["low_level_commands"] == 161
    assert result["budgets"]["low_level_commands_at_most_160"] is False
    assert not result["accepted"]


def test_unexecuted_online_intent_still_counts_as_an_online_choice(recorded):
    episode, freeze = recorded
    intent = read(episode / "actor/intents/003.json")
    save(episode / "actor/intents/004.json", intent)
    result = audit_episode(episode, freeze, policy_type=TinyPolicy)
    assert result["counts"]["online_action_choices"] == 3
    assert result["counts"]["online_model_choices"] == 4
    assert not result["accepted"]  # Its duplicated request identity is invalid too.


def test_invoked_zero_reply_handoff_is_replayed_and_selection_is_counted(recorded):
    class NoMotionPolicy(TinyPolicy):
        def decide(self, observation):
            self.status = "needs_turn_margin"
            return None

    episode, freeze = recorded
    original = read(episode / "actions.json")
    req = read(episode / "steps/001/request.json")
    policy = NoMotionPolicy(req["observation"], original[0]["action"])
    before = policy.describe()
    assert policy.decide(req["observation"]) is None
    call = episode / "actor/skill-calls/withdraw"
    source_sha = read(call / "initial.json")["policy_source_sha256"]
    for seq in (2, 3):
        (call / f"decisions/{seq:03}.json").unlink()
        (episode / f"actor/intents/{seq:03}.json").unlink()
    save(call / "decisions/001.json", {"request_id": "test-1", "sequence": 1,
         "observation_sha256": canonical_sha(req["observation"]), "action": None, "goal": None,
         "policy_state_before": before, "policy_state_after": policy.describe(), "policy_step": 0,
         "policy_source_sha256": source_sha})
    save(call / "handoff.json", {"status": "handoff", "skill_call_id": "withdraw", "replies": 0,
         "request_id": "test-1", "sequence": 1, "observation_sha256": canonical_sha(req["observation"]),
         "policy_state": policy.describe(), "policy_source_sha256": source_sha})
    finish = original[3]["action"]
    result = {"status": "finished", "time_s": 1.}
    save(episode / "actions.json", [original[0], {"sequence": 1, "action": finish, "result": result}])
    save(episode / "steps/001/action.json", finish)
    save(episode / "steps/001/result.json", result)
    save(episode / "actor/steps/001/reply.json", {"request_id": "test-1", "action": finish})
    save(episode / "actor/intents/001.json", {"request_id": "test-1",
         "observation_sha256": canonical_sha(req["observation"]), "semantic_operation": "finish",
         "target_head_m": None, "diagnosis": "actor resumes", "correction_trigger": None,
         "decision_origin": "online_model"})
    save(episode / "summary.json", {"episode_id": "test", "backend": "native", "decisions": 2, "success": False})
    audited = audit_episode(episode, freeze, policy_type=NoMotionPolicy)
    assert audited["accepted"], audited["trace"]["errors"]
    assert audited["counts"]["local_motor_commands"] == 0
    assert audited["counts"]["online_model_choices"] == 3
    assert audited["calls"][0]["replayed_decisions"] == 1


def test_real_policy_exact_replay_uses_only_self_contained_saved_inputs(recorded):
    from dexhand_agent.wrench_recovery import WrenchWithdrawPolicy

    episode, freeze = recorded
    fixture = read(ROOT / "tests/fixtures/wrench_withdraw_v1.json")
    previous = fixture["previous_palm_action"]
    observations = [copy.deepcopy(fixture["initial_observation"])]
    observations[0]["time_s"] -= previous["duration_s"]
    observations[0]["last_action"] = {"status": "ready"}
    observations += [fixture["initial_observation"]]
    observations += [row["observation_after"] for row in fixture["sequence"]]
    requests = [{"request_id": f"test-{n}", "episode_id": "test", "sequence": n, "observation": obs}
                for n, obs in enumerate(observations)]
    policy = WrenchWithdrawPolicy(observations[1], previous)
    call = episode / "actor/skill-calls/withdraw"
    initial = read(call / "initial.json")
    selection_path = episode / "actor/skill-calls/withdraw-selection.json"
    selection = read(selection_path)
    selection["observation_sha256"] = canonical_sha(observations[1])
    save(selection_path, selection)
    initial.update(policy_state=policy.describe(), start_observation_sha256=canonical_sha(observations[1]),
                   selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest())
    save(call / "initial.json", initial)
    history = [{"sequence": 0, "action": previous, "result": observations[1]["last_action"]}]
    for seq in range(1, len(requests)):
        before = policy.describe()
        action = policy.decide(copy.deepcopy(observations[seq]))
        if seq <= len(fixture["sequence"]):
            expected_action = fixture["sequence"][seq - 1]["expected_action"]
            for field in ("operation", "targets", "duration_s"):
                assert action[field] == expected_action[field]
            for field in ("palm_target_m", "palm_quat_wxyz"):
                assert action[field] == pytest.approx(expected_action[field], abs=1e-12)
        else:
            assert action is None
        save(call / f"decisions/{seq:03}.json", {
            "request_id": f"test-{seq}", "sequence": seq, "observation_sha256": canonical_sha(observations[seq]),
            "action": action, "goal": policy.last_goal, "policy_state_before": before,
            "policy_state_after": policy.describe(), "policy_step": policy.steps,
            "policy_source_sha256": initial["policy_source_sha256"]})
        if action is not None:
            history.append({"sequence": seq, "action": action, "result": observations[seq + 1]["last_action"]})
            save(episode / f"actor/intents/{seq:03}.json", {
                "request_id": f"test-{seq}", "observation_sha256": canonical_sha(observations[seq]),
                "semantic_operation": policy.phase, "target_head_m": policy.last_goal["target_head_m"],
                "diagnosis": f"Explicit local feedback policy {initial['policy_state']['version']}: {policy.status}",
                "correction_trigger": None, "decision_origin": "local_feedback_skill", "skill_call_id": "withdraw",
                "policy_source_sha256": initial["policy_source_sha256"], "policy_step": policy.steps, "goal": policy.last_goal})
    final = len(requests) - 1
    history.append({"sequence": final, "action": {"operation": "finish", "targets": [], "palm_target_m": None,
        "palm_quat_wxyz": None, "duration_s": .02, "memory": "online finish"},
        "result": {"status": "finished", "time_s": observations[-1]["time_s"]}})
    for seq, row in enumerate(history):
        for prefix in ("steps", "actor/steps"):
            save(episode / f"{prefix}/{seq:03}/request.json", requests[seq])
        save(episode / f"steps/{seq:03}/action.json", row["action"])
        save(episode / f"steps/{seq:03}/result.json", row["result"])
        save(episode / f"actor/steps/{seq:03}/reply.json", {"request_id": f"test-{seq}", "action": row["action"]})
        if seq in (0, final):
            save(episode / f"actor/intents/{seq:03}.json", {
                "request_id": f"test-{seq}", "observation_sha256": canonical_sha(observations[seq]),
                "semantic_operation": "move" if seq == 0 else "finish", "target_head_m": None,
                "diagnosis": "online choice", "correction_trigger": None, "decision_origin": "online_model"})
    save(call / "handoff.json", {"status": "handoff", "skill_call_id": "withdraw", "replies": 66,
         "request_id": f"test-{final}", "sequence": final, "observation_sha256": canonical_sha(observations[-1]),
         "policy_state": policy.describe(), "policy_source_sha256": initial["policy_source_sha256"]})
    save(episode / "actions.json", history)
    save(episode / "summary.json", {"episode_id": "test", "backend": "native", "decisions": len(history), "success": False})
    result = audit_episode(episode, freeze)
    assert result["accepted"], result["trace"]["errors"]
    assert result["counts"]["local_motor_commands"] == 66
    assert result["calls"][0]["replayed_decisions"] == 67
