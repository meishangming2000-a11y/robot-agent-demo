"""The native actor's local skill bridge only replies to fresh file requests."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools import drive_native_skill as bridge


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf8")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf8")).hexdigest()


ACTION = {
    "operation": "move", "targets": [], "duration_s": 0.2,
    "palm_target_m": [0.1, 0.2, 0.3], "palm_quat_wxyz": [1.0, 0, 0, 0],
    "memory": "Explicit local feedback: 扳手",
}


class FakePolicy:
    def __init__(self, actions):
        self.actions = actions
        self.observations = []
        self.phase = "initial"
        self.status = "ready"
        self.steps = 0
        self.last_goal = None
        self.initial_observation_sha256 = "0" * 64

    def describe(self):
        return {
            "version": "fake_local_v1", "initial_observation_sha256": self.initial_observation_sha256,
            "previous_palm_action": {"original": True},
            "last_accepted_palm_action": {"accepted": self.steps},
            "max_steps": 80, "phase": self.phase, "status": self.status,
            "steps": self.steps, "last_goal": copy.deepcopy(self.last_goal),
        }

    def decide(self, observation):
        self.observations.append(copy.deepcopy(observation))
        self.steps += 1
        self.phase = "withdraw" if self.steps == 1 else "clearance_hold"
        self.last_goal = {"target_head_m": [0.11, 0.22, 0.33],
                          "target_tool_quat_wxyz": [1, 0, 0, 0]}
        result = self.actions[self.steps - 1]
        self.status = "handoff_check" if result is None else "waiting_acceptance"
        return copy.deepcopy(result)


def waiting(episode, sequence=3, *, request_id=None, relative=False, observation=None):
    actor = episode / "actor"
    request_id = request_id or f"episode-{sequence}"
    request_path = actor / f"steps/{sequence:03}/request.json"
    reply_path = request_path.with_name("reply.json")
    request = {
        "request_id": request_id, "episode_id": "episode", "sequence": sequence,
        "task_id": "wrench_turn", "goal": "Observed withdrawal",
        "observation": observation or {"time_s": sequence * 0.2, "label": "扳手",
                                       "wrench_feedback": {}, "last_action": {"status": "completed"}},
        "recent_actions": [], "action_schema": {}, "experience": [], "images": [],
    }
    save(request_path, request)
    pointer = {
        "status": "waiting", "request_id": request_id, "sequence": sequence,
        "request_path": str(request_path.relative_to(actor) if relative else request_path),
        "reply_path": str(reply_path.relative_to(actor) if relative else reply_path),
    }
    save(actor / "current.json", pointer)
    return pointer, request


def fake_clock(monkeypatch, on_sleep=None):
    now = [0.0]
    monkeypatch.setattr(bridge.time, "monotonic", lambda: now[0])

    def sleep(delay):
        now[0] += delay
        if on_sleep is not None:
            on_sleep()

    monkeypatch.setattr(bridge.time, "sleep", sleep)
    return now


def select(episode, policy, call_id):
    pointer = read(episode / "actor/current.json")
    request = read(episode / f"actor/steps/{pointer['sequence']:03}/request.json")
    policy.initial_observation_sha256 = digest(request["observation"])
    selection = {
        "skill_call_id": call_id, "decision_origin": "online_model",
        "request_id": request["request_id"], "sequence": request["sequence"],
        "observation_sha256": digest(request["observation"]),
        "policy": "WrenchWithdrawPolicy", "diagnosis": "Fresh observed withdrawal needs local feedback",
    }
    path = episode / f"actor/skill-calls/{call_id}-selection.json"
    save(path, selection)
    return path


def test_distinct_requests_bind_exact_actions_intents_and_call_snapshots(tmp_path, monkeypatch):
    policy = FakePolicy([ACTION, {**ACTION, "duration_s": 0.4}])
    first_pointer, first = waiting(tmp_path, relative=True)
    selection_path = select(tmp_path, policy, "withdraw-1")
    initial = policy.describe()
    publications = []
    publication = "rename" if bridge.os.name == "nt" else "link"
    publish = getattr(bridge.os, publication)

    def inspect_publication(source, destination):
        destination = Path(destination)
        if destination.name == "reply.json":
            sequence = int(destination.parent.name)
            intent = read(tmp_path / f"actor/intents/{sequence:03}.json")
            assert intent["request_id"] == read(source)["request_id"]
            publications.append(sequence)
        return publish(source, destination)

    monkeypatch.setattr(bridge.os, publication, inspect_publication)
    second_request = []

    def advance():
        assert read(tmp_path / "actor/steps/003/reply.json") == {
            "request_id": first["request_id"], "action": ACTION,
        }
        _, second = waiting(tmp_path, 4, observation={
            "last_action": {"status": "completed"}, "wrench_feedback": {},
            "label": "扳手", "time_s": 0.8,
        })
        second_request.append(second)

    fake_clock(monkeypatch, advance)
    result = bridge.drive_policy(tmp_path, policy, call_id="withdraw-1", maximum_replies=2)

    assert result["status"] == "reply_limit"
    assert result["replies"] == 2
    assert publications == [3, 4]
    assert policy.observations == [first["observation"], second_request[0]["observation"]]
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    call_dir = tmp_path / "actor/skill-calls/withdraw-1"
    snapshot = read(call_dir / "initial.json")
    assert snapshot["policy_state"] == initial
    assert snapshot["policy_source_sha256"] == source_sha
    assert Path(snapshot["policy_source_path"]).resolve() == Path(__file__).resolve()
    assert snapshot["start_request_id"] == first["request_id"]
    assert snapshot["start_sequence"] == first["sequence"]
    assert snapshot["start_observation_sha256"] == digest(first["observation"])
    assert Path(snapshot["actor_selection_path"]) == selection_path
    assert snapshot["selection_sha256"] == hashlib.sha256(selection_path.read_bytes()).hexdigest()
    for sequence, request in ((3, first), (4, second_request[0])):
        intent = read(tmp_path / f"actor/intents/{sequence:03}.json")
        assert intent["request_id"] == request["request_id"]
        assert intent["observation_sha256"] == digest(request["observation"])
        assert intent["semantic_operation"] == ("withdraw" if sequence == 3 else "clearance_hold")
        assert intent["target_head_m"] == policy.last_goal["target_head_m"]
        assert intent["goal"] == policy.last_goal
        assert intent["decision_origin"] == "local_feedback_skill"
        assert intent["skill_call_id"] == "withdraw-1"
        assert intent["policy_source_sha256"] == source_sha
        assert intent["policy_step"] == sequence - 2
        assert intent["correction_trigger"] is None
        assert "local" in intent["diagnosis"].lower()
        decision = read(call_dir / f"decisions/{sequence:03}.json")
        assert decision["action"] == policy.actions[sequence - 3]
        assert decision["policy_state_after"]["steps"] == sequence - 2
        assert decision["observation_sha256"] == intent["observation_sha256"]
    assert read(call_dir / "handoff.json") == result
    assert read(tmp_path / "actor/steps/004/reply.json") == {
        "request_id": "episode-4", "action": policy.actions[1],
    }
    assert not list(tmp_path.rglob("*.tmp"))
    assert first_pointer["status"] == "waiting"


def test_policy_none_leaves_same_request_available_to_actor(tmp_path):
    pointer, _ = waiting(tmp_path)
    policy = FakePolicy([None])
    select(tmp_path, policy, "handoff")

    result = bridge.drive_policy(tmp_path, policy, call_id="handoff")

    assert result["status"] == "handoff"
    assert result["replies"] == 0
    assert result["request_id"] == pointer["request_id"]
    assert result["policy_state"]["status"] == "handoff_check"
    assert read(tmp_path / "actor/current.json") == pointer
    assert not (tmp_path / "actor/steps/003/reply.json").exists()
    assert not (tmp_path / "actor/intents/003.json").exists()
    assert read(tmp_path / "actor/skill-calls/handoff/decisions/003.json")["action"] is None


@pytest.mark.parametrize("existing", ["reply", "intent"])
def test_existing_reply_or_intent_is_preserved_without_calling_policy(tmp_path, existing):
    waiting(tmp_path)
    path = (tmp_path / "actor/steps/003/reply.json" if existing == "reply" else
            tmp_path / "actor/intents/003.json")
    original = '{"original": "must survive"}\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(original, encoding="utf8")
    policy = FakePolicy([ACTION])
    select(tmp_path, policy, "preserve")

    with pytest.raises(bridge.NativeSkillError, match="[Ee]xisting|[Aa]lready"):
        bridge.drive_policy(tmp_path, policy, call_id="preserve")

    assert path.read_text(encoding="utf8") == original
    assert policy.observations == []


def test_same_waiting_request_is_never_decided_twice(tmp_path, monkeypatch):
    waiting(tmp_path)
    policy = FakePolicy([ACTION])
    select(tmp_path, policy, "once")
    fake_clock(monkeypatch)

    result = bridge.drive_policy(tmp_path, policy, call_id="once", timeout_s=0.12)

    assert result["status"] == "timeout"
    assert result["replies"] == 1
    assert len(policy.observations) == 1
    assert len(list((tmp_path / "actor/intents").glob("*.json"))) == 1


def test_reused_request_id_in_another_sequence_is_rejected(tmp_path, monkeypatch):
    waiting(tmp_path)
    fake_clock(monkeypatch, lambda: waiting(tmp_path, 4, request_id="episode-3"))
    policy = FakePolicy([ACTION])
    select(tmp_path, policy, "reuse")

    with pytest.raises(bridge.NativeSkillError, match="[Rr]eused|[Ss]tale"):
        bridge.drive_policy(tmp_path, policy, call_id="reuse", timeout_s=0.2)

    assert len(policy.observations) == 1
    assert not (tmp_path / "actor/steps/004/reply.json").exists()


def test_changed_current_request_after_decide_is_not_replied_to(tmp_path):
    waiting(tmp_path)
    policy = FakePolicy([ACTION])
    select(tmp_path, policy, "changed")
    decide = policy.decide

    def change_current(observation):
        action = decide(observation)
        waiting(tmp_path, 4)
        return action

    policy.decide = change_current
    with pytest.raises(bridge.NativeSkillError, match="[Cc]urrent|[Ss]tale"):
        bridge.drive_policy(tmp_path, policy, call_id="changed")

    assert not (tmp_path / "actor/steps/003/reply.json").exists()
    assert not (tmp_path / "actor/steps/004/reply.json").exists()


def test_missing_mailbox_times_out_without_any_reply(tmp_path, monkeypatch):
    policy = FakePolicy([ACTION])
    clock = fake_clock(monkeypatch)

    result = bridge.drive_policy(tmp_path, policy, call_id="timeout", timeout_s=0.12)

    assert result["status"] == "timeout"
    assert result["replies"] == 0
    assert 0.12 <= clock[0] < 0.17
    assert policy.observations == []
    assert not list(tmp_path.rglob("reply.json"))
    assert read(tmp_path / "actor/skill-calls/timeout/initial.json")["invoked"] is False


@pytest.mark.parametrize("terminal", ["summary", "done", "completed", "error", "timeout"])
def test_completed_or_stopped_episode_exits_cleanly(tmp_path, terminal):
    if terminal == "summary":
        save(tmp_path / "summary.json", {"success": False, "error": "retained result"})
    else:
        save(tmp_path / "actor/current.json", {"status": terminal})
    policy = FakePolicy([ACTION])

    result = bridge.drive_policy(tmp_path, policy, call_id="stopped")

    expected = {"error": "mailbox_error", "timeout": "mailbox_timeout"}.get(terminal, "episode_complete")
    assert result["status"] == expected
    assert result["replies"] == 0
    assert policy.observations == []
    assert read(tmp_path / "actor/skill-calls/stopped/handoff.json") == result
    assert read(tmp_path / "actor/skill-calls/stopped/initial.json")["start_request_id"] is None


@pytest.mark.parametrize("call_id", ["../escape", "a/b", "a\\b", "", ".", ".."])
def test_unsafe_call_id_never_writes(tmp_path, call_id):
    with pytest.raises(bridge.NativeSkillError, match="call_id"):
        bridge.drive_policy(tmp_path, FakePolicy([ACTION]), call_id=call_id)
    assert not list(tmp_path.iterdir())


def test_existing_call_is_never_rewritten(tmp_path):
    _, request = waiting(tmp_path)
    call_path = tmp_path / "actor/skill-calls/reused"
    save(call_path / "initial.json", {"original": True})
    original = (call_path / "initial.json").read_bytes()

    with pytest.raises(bridge.NativeSkillError, match="[Ee]xist|[Aa]lready"):
        bridge.drive_policy(tmp_path, FakePolicy([ACTION]), call_id="reused")

    assert (call_path / "initial.json").read_bytes() == original
    assert not (tmp_path / "actor/steps/003/reply.json").exists()
    assert read(tmp_path / "actor/steps/003/request.json") == request


@pytest.mark.parametrize("field", ["request_path", "reply_path"])
def test_pointer_paths_cannot_escape_actor(tmp_path, field):
    pointer, _ = waiting(tmp_path)
    pointer[field] = str(tmp_path / "outside.json")
    save(tmp_path / "actor/current.json", pointer)
    policy = FakePolicy([ACTION])
    select(tmp_path, policy, "escape")

    with pytest.raises(bridge.NativeSkillError, match="path"):
        bridge.drive_policy(tmp_path, policy, call_id="escape")

    assert not (tmp_path / "outside.json").exists()
    assert policy.observations == []


def test_policy_cannot_hide_finish_in_the_bridge(tmp_path):
    waiting(tmp_path)
    policy = FakePolicy([{"operation": "finish"}])
    select(tmp_path, policy, "finish")
    with pytest.raises(bridge.NativeSkillError, match="finish"):
        bridge.drive_policy(tmp_path, policy, call_id="finish")
    assert not (tmp_path / "actor/steps/003/reply.json").exists()


@pytest.mark.parametrize("invalid", ["missing", "origin", "observation", "initial_observation", "request_id"])
def test_fresh_call_requires_preexisting_matching_actor_selection(tmp_path, invalid):
    waiting(tmp_path)
    policy = FakePolicy([ACTION])
    path = select(tmp_path, policy, "selection")
    if invalid == "missing":
        path.unlink()
    elif invalid == "initial_observation":
        policy.initial_observation_sha256 = "f" * 64
    else:
        value = read(path)
        key = {"origin": "decision_origin", "observation": "observation_sha256"}.get(invalid, invalid)
        value[key] = "different"
        save(path, value)

    with pytest.raises(bridge.NativeSkillError, match="[Ss]election|[Oo]bservation"):
        bridge.drive_policy(tmp_path, policy, call_id="selection")

    assert policy.observations == []
    assert not (tmp_path / "actor/steps/003/reply.json").exists()
