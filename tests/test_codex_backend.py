import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def request_data():
    return {
        "request_id": "episode-1-step-2",
        "episode_id": "episode-1",
        "sequence": 2,
        "goal": "Lift the hammer",
        "observation": {"time_s": 0.5},
        "action_schema": {"type": "object", "properties": {"operation": {"type": "string"}}},
        "experience": [{"verified": True, "text": "Close fingers before lifting"}],
    }


@pytest.fixture
def action():
    return {
        "operation": "hold",
        "targets": {},
        "palm_target_m": None,
        "duration_s": 0.5,
        "memory": "Check contact next",
    }


def cli_stub(monkeypatch, *, reply=None, returncode=0, timeout=False):
    calls = []
    monkeypatch.setattr("shutil.which", lambda name: "C:/Program Files/Codex/codex.exe")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write('{"type":"thread.started"}\n')
        kwargs["stderr"].write("test diagnostic\n")
        if reply is not None:
            Path(command[command.index("--output-last-message") + 1]).write_text(reply, encoding="utf-8")
        if timeout:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_cli_action_and_evidence(tmp_path, monkeypatch, request_data, action):
    from dexhand_agent.codex_backend import CodexBackend

    calls = cli_stub(monkeypatch, reply=json.dumps(action))
    images = [tmp_path / "front image.png", tmp_path / "side.png"]
    backend = CodexBackend(tmp_path / "decisions", timeout_s=12)

    assert backend.decide(request_data, images) == action

    command, kwargs = calls[0]
    assert isinstance(command, list)
    assert command[:2] == ["C:/Program Files/Codex/codex.exe", "exec"]
    assert command[-1] == "-"
    for flag in ["--ephemeral", "--skip-git-repo-check", "--json"]:
        assert flag in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--color") + 1] == "never"
    assert "--model" not in command and "-m" not in command
    assert "--ignore-user-config" not in command
    assert not any("dangerously" in value for value in command)
    assert [command[i + 1] for i, value in enumerate(command) if value == "--image"] == [str(p) for p in images]
    assert not kwargs.get("shell", False)
    assert kwargs["timeout"] == 12
    for key in ["goal", "observation", "action_schema", "experience"]:
        assert json.dumps(request_data[key], ensure_ascii=False) in kwargs["input"]

    step_dir = Path(kwargs["cwd"])
    assert step_dir.parent == backend.work_dir
    assert json.loads((step_dir / "request.json").read_text(encoding="utf-8")) == request_data
    assert json.loads((step_dir / "jsonschema.json").read_text(encoding="utf-8")) == request_data["action_schema"]
    assert Path(command[command.index("--output-schema") + 1]) == step_dir / "jsonschema.json"
    assert json.loads((step_dir / "reply.json").read_text(encoding="utf-8")) == action
    assert (step_dir / "stdout.jsonl").read_text(encoding="utf-8") == '{"type":"thread.started"}\n'
    assert (step_dir / "stderr.txt").read_text(encoding="utf-8") == "test diagnostic\n"
    meta = json.loads((step_dir / "backend_meta.json").read_text(encoding="utf-8"))
    assert meta["exit_code"] == 0
    assert meta["timed_out"] is False
    assert meta["elapsed_s"] >= 0
    assert meta["status"] == "ok"


@pytest.mark.parametrize("returncode, timeout", [(7, False), (0, True)])
def test_cli_failure_preserves_evidence(tmp_path, monkeypatch, request_data, action, returncode, timeout):
    from dexhand_agent.codex_backend import BackendError, CodexBackend

    calls = cli_stub(monkeypatch, reply=json.dumps(action), returncode=returncode, timeout=timeout)
    with pytest.raises(BackendError):
        CodexBackend(tmp_path).decide(request_data, [])

    step_dir = Path(calls[0][1]["cwd"])
    meta = json.loads((step_dir / "backend_meta.json").read_text(encoding="utf-8"))
    assert meta["timed_out"] is timeout
    assert meta["exit_code"] == (None if timeout else returncode)
    assert meta["status"] != "ok"
    assert (step_dir / "stdout.jsonl").stat().st_size > 0
    assert (step_dir / "stderr.txt").stat().st_size > 0
    assert json.loads((step_dir / "reply.json").read_text(encoding="utf-8")) == action


@pytest.mark.parametrize("reply", [None, "", "not JSON", "[]", '{"duration_s": NaN}', '{"duration_s": Infinity}', '{"duration_s": -Infinity}', '{"targets": {"finger": 1e999}}'])
def test_missing_or_invalid_reply_is_failure(tmp_path, monkeypatch, request_data, reply):
    from dexhand_agent.codex_backend import BackendError, CodexBackend

    calls = cli_stub(monkeypatch, reply=reply)
    with pytest.raises(BackendError):
        CodexBackend(tmp_path).decide(request_data, [])

    step_dir = Path(calls[0][1]["cwd"])
    meta = json.loads((step_dir / "backend_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] != "ok"
    assert meta["exit_code"] == 0
    if reply is not None:
        assert (step_dir / "reply.json").read_text(encoding="utf-8") == reply


def test_request_id_cannot_escape_or_overwrite_evidence(tmp_path, monkeypatch, request_data, action):
    from dexhand_agent.codex_backend import CodexBackend

    calls = cli_stub(monkeypatch, reply=json.dumps(action))
    request_data["request_id"] = "../../outside\\CON: bad?"
    backend = CodexBackend(tmp_path)
    backend.decide(request_data, [])
    backend.decide(request_data, [])

    step_dirs = [Path(kwargs["cwd"]) for _, kwargs in calls]
    assert len(set(step_dirs)) == 2
    for directory in step_dirs:
        assert directory.parent == tmp_path
        assert all(character not in directory.name for character in '/\\:?')
        assert (directory / "request.json").is_file()


def test_missing_cli_is_recorded(tmp_path, monkeypatch, request_data):
    from dexhand_agent.codex_backend import BackendError, CodexBackend

    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(BackendError):
        CodexBackend(tmp_path).decide(request_data, [])
    meta_path, = tmp_path.glob("*/backend_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["status"] != "ok"
    assert meta["exit_code"] is None
