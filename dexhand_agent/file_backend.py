"""File exchange for a persistent Codex actor; the runtime validates its actions."""

import json
import os
from pathlib import Path
import time

from .codex_backend import BackendError


class FileBackendError(BackendError):
    """The file decision failed; current.json records error or timeout status."""


def _write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    for attempt in range(50):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            # A Windows reader may briefly prevent atomic replacement.
            if getattr(exc, "winerror", None) not in (5, 32) or attempt == 49:
                raise
            time.sleep(0.01)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


class FileBackend:
    def __init__(self, work_dir: Path, timeout_s=600):
        self.work_dir = Path(work_dir).resolve()
        self.timeout_s = timeout_s

    def decide(self, request: dict, image_paths: list[Path]) -> dict:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        sequence = request["sequence"]
        step_dir = self.work_dir / "steps" / f"{sequence:03}"
        request_path = step_dir / "request.json"
        reply_path = step_dir / "reply.json"
        pointer = {
            "status": "waiting",
            "request_id": request["request_id"],
            "sequence": sequence,
            "request_path": str(request_path),
            "reply_path": str(reply_path),
        }
        try:
            # Never consume a stale reply from a previous attempt at this sequence.
            step_dir.mkdir(parents=True, exist_ok=False)
            _write_json(request_path, {
                **request,
                "images": [str(Path(path).resolve()) for path in image_paths],
            })
            _write_json(self.work_dir / "current.json", pointer)
            deadline = time.monotonic() + self.timeout_s
            while not reply_path.exists():
                if time.monotonic() >= deadline:
                    pointer["status"] = "timeout"
                    pointer["error"] = f"Timed out waiting {self.timeout_s}s for {reply_path}"
                    raise FileBackendError(pointer["error"])
                time.sleep(0.05)

            reply = json.loads(reply_path.read_text(encoding="utf-8-sig"),
                               object_pairs_hook=_unique_object)
            # Reject JSON constants and numeric overflow, including nested values.
            json.dumps(reply, allow_nan=False)
            if not isinstance(reply, dict) or set(reply) != {"request_id", "action"}:
                raise ValueError("Reply must contain only request_id and action")
            if reply["request_id"] != request["request_id"]:
                raise ValueError("Reply request_id does not match the current request")
            if not isinstance(reply["action"], dict):
                raise ValueError("Reply action must be a JSON object")
            pointer["status"] = "responded"
            return reply["action"]
        except (OSError, ValueError, TypeError, KeyError) as exc:
            pointer["status"] = "error"
            pointer["error"] = f"{type(exc).__name__}: {exc}"
            raise FileBackendError(f"File decision failed; {pointer['error']}") from exc
        finally:
            try:
                _write_json(self.work_dir / "current.json", pointer)
            except OSError as exc:
                raise FileBackendError(
                    f"File decision status publication failed; {type(exc).__name__}: {exc}"
                ) from exc
