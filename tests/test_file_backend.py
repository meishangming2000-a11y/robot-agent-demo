import copy
import errno
import json
import os
from pathlib import Path
import threading

import pytest

from dexhand_agent.file_backend import FileBackend, FileBackendError, _write_json


@pytest.fixture
def request_data():
    return {
        "request_id": "episode-step-2",
        "sequence": 2,
        "goal": "Lift the hammer",
        "observation": {"time_s": 0.5},
        "action_schema": {"type": "object"},
        "experience": [],
    }


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def actor_on_wait(monkeypatch, work_dir, reply):
    """Stand in for the external actor only when the backend yields to wait."""
    pointers = []

    def respond(delay):
        assert delay == 0.05
        pointer = read_json(work_dir / "current.json")
        assert pointer["status"] == "waiting"
        assert read_json(pointer["request_path"])["request_id"] == pointer["request_id"]
        pointers.append(pointer)
        text = reply(pointer) if callable(reply) else reply
        Path(pointer["reply_path"]).write_text(text, encoding="utf-8")

    monkeypatch.setattr("dexhand_agent.file_backend.time.sleep", respond)
    return pointers


def test_publishes_complete_atomic_request_and_correlated_action(tmp_path, monkeypatch, request_data):
    monkeypatch.chdir(tmp_path)
    work_dir = tmp_path / "mailbox"
    action = {"operation": "hold", "targets": [], "duration_s": 0.2}
    request_before = copy.deepcopy(request_data)
    publications = []
    replace = os.replace

    def inspect_publication(source, destination):
        assert Path(source) != Path(destination)
        payload = read_json(source)
        if Path(destination).name == "current.json":
            assert Path(payload["request_path"]).is_file()
        publications.append((Path(destination), payload))
        return replace(source, destination)

    monkeypatch.setattr("dexhand_agent.file_backend.os.replace", inspect_publication)
    pointers = actor_on_wait(
        monkeypatch, work_dir,
        lambda pointer: json.dumps({"request_id": pointer["request_id"], "action": action}),
    )

    result = FileBackend(work_dir).decide(request_data, [Path("front image.png"), Path("side.png")])

    assert result == action
    assert request_data == request_before
    step_dir = work_dir / "steps" / "002"
    published_request = read_json(step_dir / "request.json")
    assert published_request == {
        **request_data,
        "images": [str(tmp_path / "front image.png"), str(tmp_path / "side.png")],
    }
    assert pointers == [{
        "status": "waiting", "request_id": request_data["request_id"], "sequence": 2,
        "request_path": str(step_dir / "request.json"),
        "reply_path": str(step_dir / "reply.json"),
    }]
    assert [value["status"] for path, value in publications if path.name == "current.json"] == [
        "waiting", "responded",
    ]
    assert read_json(work_dir / "current.json")["status"] == "responded"
    assert not list(work_dir.rglob("*.tmp"))


def test_action_contents_are_left_for_runtime_validation(tmp_path, monkeypatch, request_data):
    action = {"unknown_action_field": "runtime must reject this"}
    actor_on_wait(monkeypatch, tmp_path, json.dumps({
        "request_id": request_data["request_id"], "action": action,
    }))
    assert FileBackend(tmp_path).decide(request_data, []) == action


@pytest.mark.parametrize("reply", [
    "not JSON",
    '{"request_id":',
    "[]",
    "null",
    '{"request_id":"episode-step-2"}',
    '{"request_id":"wrong-step","action":{}}',
    '{"request_id":"episode-step-2","action":{},"extra":true}',
    '{"request_id":"episode-step-2","action":[]}',
    '{"request_id":"episode-step-2","action":{"duration_s":NaN}}',
    '{"request_id":"episode-step-2","action":{"duration_s":Infinity}}',
    '{"request_id":"episode-step-2","action":{"duration_s":-Infinity}}',
    '{"request_id":"episode-step-2","action":{"targets":[{"value":1e999}]}}',
    '{"request_id":"wrong-step","request_id":"episode-step-2","action":{}}',
    '{"request_id":"episode-step-2","action":{"operation":"move","operation":"hold"}}',
    '{"request_id":"episode-step-2","action":{}} {}',
])
def test_bad_reply_is_a_single_error_with_original_reply_retained(tmp_path, monkeypatch, request_data, reply):
    pointers = actor_on_wait(monkeypatch, tmp_path, reply)

    with pytest.raises(FileBackendError):
        FileBackend(tmp_path).decide(request_data, [])

    assert len(pointers) == 1
    current = read_json(tmp_path / "current.json")
    assert current["status"] == "error"
    assert current["error"]
    assert Path(current["reply_path"]).read_text(encoding="utf-8") == reply
    assert not (Path(current["reply_path"]).parent / "action.json").exists()


def test_timeout_has_no_fabricated_action(tmp_path, monkeypatch, request_data):
    now = [0.0]
    waits = []
    monkeypatch.setattr("dexhand_agent.file_backend.time.monotonic", lambda: now[0])

    def wait(delay):
        waits.append(delay)
        now[0] += delay

    monkeypatch.setattr("dexhand_agent.file_backend.time.sleep", wait)
    with pytest.raises(FileBackendError, match="[Tt]imed out"):
        FileBackend(tmp_path, timeout_s=0.1).decide(request_data, [])

    current = read_json(tmp_path / "current.json")
    assert current["status"] == "timeout"
    assert waits == [0.05, 0.05]
    assert not Path(current["reply_path"]).exists()


def test_existing_sequence_is_not_reused_or_overwritten(tmp_path, monkeypatch, request_data):
    pointers = actor_on_wait(monkeypatch, tmp_path, json.dumps({
        "request_id": request_data["request_id"], "action": {"operation": "hold"},
    }))
    backend = FileBackend(tmp_path)
    backend.decide(request_data, [])
    original_request = (tmp_path / "steps" / "002" / "request.json").read_bytes()
    original_reply = (tmp_path / "steps" / "002" / "reply.json").read_bytes()

    with pytest.raises(FileBackendError):
        backend.decide({**request_data, "goal": "Replacement must not overwrite"}, [])

    assert len(pointers) == 1
    assert (tmp_path / "steps" / "002" / "request.json").read_bytes() == original_request
    assert (tmp_path / "steps" / "002" / "reply.json").read_bytes() == original_reply
    assert read_json(tmp_path / "current.json")["status"] == "error"


def test_following_sequence_uses_a_new_directory_and_reply(tmp_path, monkeypatch, request_data):
    actor_on_wait(monkeypatch, tmp_path, lambda pointer: json.dumps({
        "request_id": pointer["request_id"], "action": {"memory": pointer["request_id"]},
    }))
    backend = FileBackend(tmp_path)
    first = backend.decide(request_data, [])
    second_request = {**request_data, "sequence": 3, "request_id": "episode-step-3"}
    second = backend.decide(second_request, [])

    assert first == {"memory": "episode-step-2"}
    assert second == {"memory": "episode-step-3"}
    assert (tmp_path / "steps" / "002" / "reply.json").exists()
    assert (tmp_path / "steps" / "003" / "reply.json").exists()
    assert read_json(tmp_path / "current.json")["request_id"] == "episode-step-3"


def test_non_finite_request_is_not_published_as_waiting(tmp_path, monkeypatch, request_data):
    request_data["observation"]["time_s"] = float("nan")
    monkeypatch.setattr("dexhand_agent.file_backend.time.sleep", lambda _: pytest.fail("must not wait"))

    with pytest.raises(FileBackendError):
        FileBackend(tmp_path).decide(request_data, [])

    assert read_json(tmp_path / "current.json")["status"] == "error"
    assert not (tmp_path / "steps" / "002" / "request.json").exists()


@pytest.mark.parametrize("winerror", [5, 32])
def test_transient_status_replace_conflict_returns_action_once(
        tmp_path, monkeypatch, request_data, winerror):
    action = {"operation": "hold"}
    replies = []
    retries = []
    conflicts = []
    replace = os.replace

    def replace_with_conflict(source, destination):
        if Path(destination).name == "current.json" and read_json(source)["status"] == "responded":
            if len(conflicts) < 2:
                error = PermissionError(errno.EACCES, "reader holds current.json")
                error.winerror = winerror
                conflicts.append(error)
                raise error
        return replace(source, destination)

    def wait(delay):
        if delay < 0.05:
            retries.append(delay)
            return
        assert delay == 0.05
        pointer = read_json(tmp_path / "current.json")
        replies.append(pointer["request_id"])
        Path(pointer["reply_path"]).write_text(json.dumps({
            "request_id": pointer["request_id"], "action": action,
        }), encoding="utf-8")

    monkeypatch.setattr("dexhand_agent.file_backend.os.replace", replace_with_conflict)
    monkeypatch.setattr("dexhand_agent.file_backend.time.sleep", wait)

    assert FileBackend(tmp_path).decide(request_data, []) == action
    assert replies == [request_data["request_id"]]
    assert len(conflicts) == len(retries) == 2
    assert read_json(tmp_path / "current.json")["status"] == "responded"
    assert not list(tmp_path.rglob("*.tmp"))


def test_persistent_windows_replace_conflict_is_bounded(tmp_path, monkeypatch):
    path = tmp_path / "current.json"
    path.write_text('{"old":true}', encoding="utf-8")
    attempts = []
    waits = []
    error = PermissionError(errno.EACCES, "reader never releases current.json")
    error.winerror = 5

    def deny_replace(source, destination):
        attempts.append(destination)
        raise error

    monkeypatch.setattr("dexhand_agent.file_backend.os.replace", deny_replace)
    monkeypatch.setattr("dexhand_agent.file_backend.time.sleep", waits.append)

    with pytest.raises(PermissionError) as raised:
        _write_json(path, {"new": True})

    assert raised.value is error
    assert 1 < len(attempts) <= 100
    assert len(waits) == len(attempts) - 1
    assert 0 < sum(waits) <= 1
    assert read_json(path) == {"old": True}


@pytest.mark.parametrize("error,winerror", [
    (OSError(errno.EIO, "storage failed"), None),
    (PermissionError(errno.EACCES, "non-Windows access denied"), None),
    (PermissionError(errno.EACCES, "another Windows error"), 123),
])
def test_unrelated_replace_errors_are_not_retried(tmp_path, monkeypatch, error, winerror):
    if winerror is not None:
        error.winerror = winerror
    attempts = []

    def fail_replace(source, destination):
        attempts.append(destination)
        raise error

    monkeypatch.setattr("dexhand_agent.file_backend.os.replace", fail_replace)
    monkeypatch.setattr("dexhand_agent.file_backend.time.sleep",
                        lambda _: pytest.fail("unrelated errors must not retry"))

    with pytest.raises(OSError) as raised:
        _write_json(tmp_path / "current.json", {"new": True})

    assert raised.value is error
    assert len(attempts) == 1


@pytest.mark.parametrize("reply", [
    '{"request_id":"episode-step-2","action":{"operation":"hold"}}',
    "not JSON",
])
def test_final_status_write_failure_is_typed_and_preserves_primary_error(
        tmp_path, monkeypatch, request_data, reply):
    actor_on_wait(monkeypatch, tmp_path, reply)
    replace = os.replace
    error = OSError(errno.EIO, "status publication failed")

    def fail_final_status(source, destination):
        if Path(destination).name == "current.json" and read_json(source)["status"] != "waiting":
            raise error
        return replace(source, destination)

    monkeypatch.setattr("dexhand_agent.file_backend.os.replace", fail_final_status)

    with pytest.raises(FileBackendError, match="status publication failed") as raised:
        FileBackend(tmp_path).decide(request_data, [])

    assert raised.value.__cause__ is error
    if reply == "not JSON":
        assert isinstance(error.__context__, FileBackendError)
        assert isinstance(error.__context__.__cause__, ValueError)


@pytest.mark.skipif(os.name != "nt", reason="Windows reader sharing semantics")
def test_windows_reader_release_allows_atomic_status_replace(tmp_path, monkeypatch):
    path = tmp_path / "current.json"
    path.write_text('{"old":true}', encoding="utf-8")
    reader_open = threading.Event()
    release_reader = threading.Event()
    conflicts = []
    replace = os.replace

    def read_until_conflict():
        with path.open("r", encoding="utf-8"):
            reader_open.set()
            release_reader.wait(2)

    def replace_and_release(source, destination):
        try:
            return replace(source, destination)
        except PermissionError as exc:
            conflicts.append(exc.winerror)
            release_reader.set()
            raise

    reader = threading.Thread(target=read_until_conflict)
    reader.start()
    try:
        assert reader_open.wait(2)
        monkeypatch.setattr("dexhand_agent.file_backend.os.replace", replace_and_release)
        _write_json(path, {"new": True})
    finally:
        release_reader.set()
        reader.join(2)

    assert not reader.is_alive()
    assert conflicts and set(conflicts) <= {5, 32}
    assert read_json(path) == {"new": True}
