"""Prepare this checkout and run one or all local tool-use references."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from dexhand_agent.reporting import write_episode_report, write_suite_report


ROOT = Path(__file__).resolve().parent
TASKS = ("hammer_button", "wrench_turn", "tweezers_transfer")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _run(command):
    subprocess.run([str(item) for item in command], cwd=ROOT, check=True)


def _installed(python):
    try:
        packages = subprocess.check_output(
            [str(python), "-m", "pip", "freeze", "--exclude-editable"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
        return _digest(packages.encode("utf8"))
    except (OSError, subprocess.CalledProcessError):
        return None


def _assets_verified():
    manifest = json.loads((ROOT / "assets/manifest.json").read_text(encoding="utf8"))
    return bool(manifest["files"]) and all(
        (ROOT / "assets/vendor" / entry["path"]).is_file()
        and _digest((ROOT / "assets/vendor" / entry["path"]).read_bytes()) == entry["sha256"]
        for entry in manifest["files"])


def _prepare():
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = environment / ".dexhand-setup.json"
    declarations = _digest((ROOT / "pyproject.toml").read_bytes() + b"\0"
                           + (ROOT / "requirements-tested.txt").read_bytes())
    installed = _installed(python) if python.is_file() else None
    try:
        cached = json.loads(marker.read_text(encoding="utf8"))
    except (OSError, ValueError):
        cached = {}
    expected = {"declarations_sha256": declarations, "installed_sha256": installed}
    if installed is None or cached != expected:
        print("Preparing the isolated project environment.", flush=True)
        if not python.is_file():
            _run([sys.executable, "-m", "venv", environment])
        elif installed is None:
            _run([python, "-m", "ensurepip", "--upgrade"])
        _run([python, "-m", "pip", "install", "-c", ROOT / "requirements-tested.txt", "-e", ".[video]"])
        _run([python, "-m", "pip", "check"])
        installed = _installed(python)
        if installed is None:
            raise RuntimeError("Could not verify the isolated environment")
        expected["installed_sha256"] = installed
        marker.write_text(json.dumps(expected) + "\n", encoding="utf8")
    print("Checking pinned public robot assets.", flush=True)
    if not _assets_verified():
        _run([python, ROOT / "tools/fetch_assets.py"])
        if not _assets_verified():
            raise RuntimeError("Downloaded robot assets do not match the manifest")
    return python


def _run_task(python, task, output, policy=None):
    print(f"Running the local {task} reference.", flush=True)
    returncode, error = 0, None
    try:
        command = [python, "-m", "dexhand_agent", "demo", "--task", task, "--output", output]
        if policy is not None:
            command.extend(["--policy", policy])
        _run(command)
    except subprocess.CalledProcessError as exc:
        returncode, error = exc.returncode, f"Command exited with status {exc.returncode}."
    except OSError as exc:
        returncode, error = 1, f"Could not start task: {exc}"
    try:
        summary = json.loads((output / "summary.json").read_text(encoding="utf8"))
        if not isinstance(summary, dict):
            raise ValueError("summary.json must contain an object")
    except (OSError, ValueError) as exc:
        summary = None
        error = error or f"No physical record available: {exc}"
    report = output / "report.html"
    if not report.is_file():
        write_episode_report(output, summary)
    print(f"Report: {report}", flush=True)
    return {"task_id": task, "output": output.name, "returncode": returncode,
            "summary": summary, "error": error}


def _launch(output, task="hammer_button", policy=None):
    python = _prepare()
    if task != "all":
        run = _run_task(python, task, output, policy=policy)
        success = (run["summary"] is not None and run["summary"].get("success") is True
                   and run["summary"].get("error") is None)
        return run["returncode"] or (0 if success else 1)
    output.mkdir(parents=True, exist_ok=False)
    task_runs = [_run_task(python, name, output / name,
                           policy=policy if name == "wrench_turn" else None) for name in TASKS]
    success = all(run["returncode"] == 0 and run["summary"] is not None
                  and run["summary"].get("success") is True
                  and run["summary"].get("error") is None for run in task_runs)
    (output / "suite.json").write_text(
        json.dumps({"success": success, "task_runs": task_runs}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf8")
    print(f"Report: {write_suite_report(output, task_runs)}", flush=True)
    return next((run["returncode"] for run in task_runs if run["returncode"]), 0 if success else 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New output directory (default: a unique name under runs/)")
    parser.add_argument("--task", choices=[*TASKS, "all"], default="hammer_button")
    parser.add_argument("--policy", type=Path,
                        help="Wrench-specific learned parameter policy directory (--task wrench_turn or all)")
    args = parser.parse_args(argv)
    if args.policy is not None and args.task not in ("wrench_turn", "all"):
        parser.error("--policy requires --task wrench_turn or --task all")
    name = "quickstart-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = (ROOT / (args.output or Path("runs") / name)).resolve()
    if output.exists():
        parser.error("Use a new output directory; existing results are never overwritten")
    try:
        return _launch(output, args.task,
                       policy=(ROOT / args.policy).resolve() if args.policy is not None else None)
    except subprocess.CalledProcessError as exc:
        print(f"Stopped: command exited with status {exc.returncode}.", file=sys.stderr)
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
