import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import quickstart


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(quickstart, "ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "dexhand-agent"\n', encoding="utf8")
    (tmp_path / "requirements-tested.txt").write_text("numpy==2.2.4\n", encoding="utf8")
    (tmp_path / "assets").mkdir()
    payload = b"fixture mesh bytes"
    asset = tmp_path / "assets/vendor/robot/mesh.obj"
    manifest = {"repository": "https://example.invalid/models", "revision": "a" * 40,
                "license_by_model": {"robot": "Apache-2.0"}, "files": [{
                    "path": "robot/mesh.obj", "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload), "url": "https://example.invalid/robot/mesh.obj"}]}
    (tmp_path / "assets/manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    events = []
    failures = set()

    def process(command, *, cwd, check):
        assert Path(cwd) == tmp_path
        assert check is True
        if "venv" in command:
            event = "venv"
            python = tmp_path / ".venv" / ("Scripts/python.exe" if quickstart.os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_bytes(b"external interpreter stand-in")
        elif "ensurepip" in command:
            event = "ensurepip"
        elif "install" in command:
            event = "install"
            assert "-e" in command and ".[video]" in command
            assert str(tmp_path / "requirements-tested.txt") in command
        elif "check" in command:
            event = "check"
        elif command[-1].endswith("fetch_assets.py"):
            event = "fetch"
        elif "dexhand_agent" in command:
            event = "demo"
        else:
            raise AssertionError(command)
        events.append(event)
        if event in failures:
            raise subprocess.CalledProcessError(7, command)
        if event == "fetch":
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.write_bytes(payload)
        if event == "demo":
            task = command[command.index("--task") + 1]
            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True)
            gates = {"hammer_button": ("lift", "press", "withdraw"),
                     "wrench_turn": ("lift", "turn", "withdraw"),
                     "tweezers_transfer": ("pickup", "pinch_lift", "transport", "release")}[task]
            summary = {"task_id": task, "success": True, "error": None,
                       "backend": "reference", "gates": {name: {"passed": True} for name in gates},
                       "metrics": {}, "violations": []}
            (output / "summary.json").write_text(json.dumps(summary), encoding="utf8")
        return subprocess.CompletedProcess(command, 0)

    def installed(command, **kwargs):
        assert command[-2:] == ["freeze", "--exclude-editable"]
        return "numpy==2.2.4\n"

    monkeypatch.setattr(quickstart.subprocess, "run", process)
    monkeypatch.setattr(quickstart.subprocess, "check_output", installed)
    return tmp_path, asset, payload, events, failures


def test_first_run_prepares_isolated_dependencies_and_verified_assets(checkout):
    root, asset, payload, events, _ = checkout
    assert quickstart.main(["--output", "runs/first"]) == 0
    assert events[0] == "venv"
    assert events.index("install") < events.index("fetch") < events.index("demo")
    assert asset.read_bytes() == payload
    assert (root / ".venv/.dexhand-setup.json").is_file()


def test_second_run_reuses_verified_setup_without_install_or_download(checkout):
    _, _, _, events, _ = checkout
    quickstart.main(["--output", "runs/first"])
    events.clear()
    assert quickstart.main(["--output", "runs/second"]) == 0
    assert events == ["demo"]


def test_asset_corruption_triggers_repair_before_demo(checkout):
    _, asset, payload, events, _ = checkout
    quickstart.main(["--output", "runs/first"])
    asset.write_bytes(b"corrupted cache")
    events.clear()
    assert quickstart.main(["--output", "runs/repaired"]) == 0
    assert events == ["fetch", "demo"]
    assert asset.read_bytes() == payload


def test_dependency_declaration_change_invalidates_setup_cache(checkout):
    root, _, _, events, _ = checkout
    quickstart.main(["--output", "runs/first"])
    (root / "requirements-tested.txt").write_text("numpy==2.2.4\nPillow==11.1.0\n", encoding="utf8")
    events.clear()
    assert quickstart.main(["--output", "runs/updated"]) == 0
    assert "install" in events
    assert events[-1] == "demo"


def test_existing_output_is_rejected_without_preparing_or_overwriting(checkout):
    root, _, _, events, _ = checkout
    output = root / "runs/existing"
    output.mkdir(parents=True)
    (output / "summary.json").write_text("preserve this result", encoding="utf8")
    with pytest.raises(SystemExit) as error:
        quickstart.main(["--output", "runs/existing"])
    assert error.value.code == 2
    assert not events
    assert (output / "summary.json").read_text(encoding="utf8") == "preserve this result"


@pytest.mark.parametrize("failed_stage", ["install", "fetch"])
def test_setup_failure_preserves_exit_status_and_never_runs_demo(checkout, failed_stage):
    _, _, _, events, failures = checkout
    failures.add(failed_stage)
    assert quickstart.main(["--output", "runs/not-started"]) == 7
    assert "demo" not in events


@pytest.fixture
def task_process(checkout, monkeypatch):
    original = quickstart.subprocess.run
    calls, failed_tasks = [], set()

    def process(command, **kwargs):
        if "dexhand_agent" in command:
            task = command[command.index("--task") + 1]
            output = Path(command[command.index("--output") + 1])
            calls.append((task, output))
            if task in failed_tasks:
                raise subprocess.CalledProcessError(9, command)
        return original(command, **kwargs)

    monkeypatch.setattr(quickstart.subprocess, "run", process)
    return calls, failed_tasks


def test_all_routes_tasks_serially_and_retains_suite_results(checkout, task_process):
    root, _, _, events, _ = checkout
    calls, _ = task_process
    assert quickstart.main(["--task", "all", "--output", "runs/suite"]) == 0
    tasks = ["hammer_button", "wrench_turn", "tweezers_transfer"]
    assert calls == [(task, root / "runs/suite" / task) for task in tasks]
    assert events.count("install") == events.count("fetch") == 1
    suite = json.loads((root / "runs/suite/suite.json").read_text(encoding="utf8"))
    assert [run["task_id"] for run in suite["task_runs"]] == tasks
    assert all(run["returncode"] == 0 and run["summary"]["success"] for run in suite["task_runs"])
    assert (root / "runs/suite/report.html").is_file()
    assert all((output / "report.html").is_file() for _, output in calls)


@pytest.mark.parametrize("policy", [False, True])
def test_all_continues_after_failed_task_and_returns_failure(checkout, task_process, policy):
    root, _, _, _, _ = checkout
    calls, failed_tasks = task_process
    failed_tasks.add("wrench_turn")
    arguments = ["--task", "all", "--output", "runs/partial"]
    if policy:
        arguments.extend(["--policy", "models/trained"])
    assert quickstart.main(arguments) == 9
    assert [task for task, _ in calls] == ["hammer_button", "wrench_turn", "tweezers_transfer"]
    runs = json.loads((root / "runs/partial/suite.json").read_text(encoding="utf8"))["task_runs"]
    assert runs[1]["returncode"] == 9 and runs[1]["summary"] is None
    assert runs[1]["error"]
    report = (root / "runs/partial/report.html").read_text(encoding="utf8")
    assert "No physical record available" in report


@pytest.mark.parametrize("failed_stage", ["install", "fetch"])
def test_all_setup_failure_stops_before_any_task(checkout, task_process, failed_stage):
    root, _, _, events, failures = checkout
    calls, _ = task_process
    failures.add(failed_stage)
    assert quickstart.main(["--task", "all", "--output", "runs/stopped"]) == 7
    assert not calls and "demo" not in events
    assert not (root / "runs/stopped/suite.json").exists()


def test_all_cached_setup_does_not_reinstall_or_download(checkout, task_process):
    _, _, _, events, _ = checkout
    quickstart.main(["--task", "all", "--output", "runs/first"])
    events.clear()
    assert quickstart.main(["--task", "all", "--output", "runs/cached"]) == 0
    assert events == ["demo", "demo", "demo"]


@pytest.mark.parametrize("task", ["hammer_button", "wrench_turn", "tweezers_transfer"])
def test_single_task_uses_requested_task_without_suite_subdirectory(checkout, task_process, task):
    root, _, _, _, _ = checkout
    calls, _ = task_process
    assert quickstart.main(["--task", task, "--output", "runs/single"]) == 0
    assert calls == [(task, root / "runs/single")]


@pytest.mark.parametrize("summary", [None, {"success": False, "error": None},
                                    {"success": True, "error": "render failed"}])
@pytest.mark.parametrize("task", ["wrench_turn", "all"])
@pytest.mark.parametrize("policy", [False, True])
def test_zero_exit_without_physical_success_is_failure(checkout, task_process, monkeypatch, summary, task, policy):
    root, _, _, _, _ = checkout
    original = quickstart.subprocess.run

    def process(command, **kwargs):
        result = original(command, **kwargs)
        if "--task" in command and command[command.index("--task") + 1] == "wrench_turn":
            path = Path(command[command.index("--output") + 1]) / "summary.json"
            if summary is None:
                path.unlink()
            else:
                path.write_text(json.dumps(summary), encoding="utf8")
        return result

    monkeypatch.setattr(quickstart.subprocess, "run", process)
    arguments = ["--task", task, "--output", "runs/incomplete"]
    if policy:
        arguments.extend(["--policy", "models/trained"])
    assert quickstart.main(arguments) != 0
    assert (root / "runs/incomplete/report.html").is_file()
    if task == "all":
        runs = json.loads((root / "runs/incomplete/suite.json").read_text(encoding="utf8"))["task_runs"]
        assert runs[1]["returncode"] == 0 and runs[1]["summary"] == summary
        assert runs[2]["task_id"] == "tweezers_transfer"


def test_policy_is_resolved_from_checkout_and_forwarded_to_wrench(checkout, monkeypatch):
    root, _, _, _, _ = checkout
    commands = []
    original = quickstart.subprocess.run

    def process(command, **kwargs):
        if "dexhand_agent" in command:
            commands.append(command)
        return original(command, **kwargs)

    monkeypatch.setattr(quickstart.subprocess, "run", process)
    assert quickstart.main(["--task", "wrench_turn", "--policy", "models/trained",
                            "--output", "runs/learned"]) == 0
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--policy") + 1] == str((root / "models/trained").resolve())
    assert command[command.index("--task") + 1] == "wrench_turn"


def test_all_policy_is_forwarded_only_to_wrench(checkout, monkeypatch):
    root, _, _, _, _ = checkout
    commands = []
    original = quickstart.subprocess.run

    def process(command, **kwargs):
        if "dexhand_agent" in command:
            commands.append(command)
        return original(command, **kwargs)

    monkeypatch.setattr(quickstart.subprocess, "run", process)
    assert quickstart.main(["--task", "all", "--policy", "pretrained/wrench-xy-v1",
                            "--output", "runs/learned-suite"]) == 0
    assert [command[command.index("--task") + 1] for command in commands] == list(quickstart.TASKS)
    for command in commands:
        task = command[command.index("--task") + 1]
        if task == "wrench_turn":
            assert command[command.index("--policy") + 1] == str((root / "pretrained/wrench-xy-v1").resolve())
        else:
            assert "--policy" not in command
    suite = json.loads((root / "runs/learned-suite/suite.json").read_text(encoding="utf8"))
    assert suite["success"] is True
    assert len(suite["task_runs"]) == 3


@pytest.mark.parametrize("task", ["hammer_button", "tweezers_transfer"])
def test_policy_requires_wrench_before_setup_or_output(checkout, capsys, task):
    root, _, _, events, _ = checkout
    with pytest.raises(SystemExit) as error:
        quickstart.main(["--task", task, "--policy", "models/trained", "--output", "runs/rejected"])
    assert error.value.code == 2
    assert "--policy requires --task wrench_turn or --task all" in capsys.readouterr().err
    assert not events
    assert not (root / "runs/rejected").exists()


def test_policy_does_not_change_default_task(checkout, capsys):
    _, _, _, events, _ = checkout
    with pytest.raises(SystemExit) as error:
        quickstart.main(["--policy", "models/trained", "--output", "runs/rejected"])
    assert error.value.code == 2
    assert "--policy requires --task wrench_turn or --task all" in capsys.readouterr().err
    assert not events
