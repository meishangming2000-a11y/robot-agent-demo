"""Explicitly drive a local feedback policy through the native actor's mailbox.

The invoking actor first retains its online selection at
actor/skill-calls/<call_id>-selection.json. This utility never selects a skill,
changes a policy command, marks a task successful, or sends a finish action.
"""

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import time
import uuid


class NativeSkillError(ValueError):
    """A stale mailbox, reused output, or invalid invocation was refused."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canonical(value).encode("utf8")).hexdigest()


def _copy(value):
    return json.loads(_canonical(value))


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise NativeSkillError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _read(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique)
    _canonical(value)
    return value


def _inside(actor, path):
    if not path.resolve().is_relative_to(actor):
        raise NativeSkillError(f"Written path escapes episode actor directory: {path}")
    return path


def _write_new(actor, path, value):
    """Publish complete JSON atomically without replacing an existing file."""
    _inside(actor, path)
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = actor / f".{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            os.rename(temporary, path)  # Windows rename fails if destination exists.
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _mailbox_path(actor, raw, expected):
    if not isinstance(raw, str):
        raise NativeSkillError("Mailbox path must be a string")
    _inside(actor, expected)
    candidate = Path(raw)
    candidates = ([candidate] if candidate.is_absolute() else
                  [actor / candidate, actor.parent / candidate, Path.cwd() / candidate])
    if not any(path.resolve() == expected.resolve() for path in candidates):
        raise NativeSkillError(f"Mailbox path does not match current sequence: {raw}")
    return expected


def _waiting(actor, pointer):
    sequence, request_id = pointer.get("sequence"), pointer.get("request_id")
    if type(sequence) is not int or sequence < 0 or not isinstance(request_id, str) or not request_id:
        raise NativeSkillError("Invalid current request identity")
    step = actor / f"steps/{sequence:03}"
    request_path = _mailbox_path(actor, pointer.get("request_path"), step / "request.json")
    reply_path = _mailbox_path(actor, pointer.get("reply_path"), step / "reply.json")
    request = _read(request_path)
    if (not isinstance(request, dict) or request.get("request_id") != request_id
            or type(request.get("sequence")) is not int or request["sequence"] != sequence
            or not isinstance(request.get("observation"), dict)):
        raise NativeSkillError("Current pointer and request identity differ")
    return request, request_path, reply_path


def drive_policy(episode_path, policy, *, call_id, maximum_replies=80,
                 poll_interval_s=0.05, timeout_s=120.0):
    """Reply once per fresh request, returning a retained JSON handoff summary.

    Relative mailbox paths may be relative to the actor, episode, or working
    directory, but must resolve to the current actor/steps/NNN files. Existing
    replies, intents and call directories are never replaced. A policy returning
    None leaves that same request unanswered for the online actor to resume.
    """
    if not isinstance(call_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", call_id):
        raise NativeSkillError("call_id must contain 1-64 letters, digits, underscores or hyphens")
    if type(maximum_replies) is not int or maximum_replies < 1:
        raise NativeSkillError("maximum_replies must be a positive integer")
    for name, value in (("poll_interval_s", poll_interval_s), ("timeout_s", timeout_s)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise NativeSkillError(f"{name} must be finite and positive")
    episode = Path(episode_path).resolve(strict=True)
    actor = episode / "actor"
    if actor.resolve() != actor:
        raise NativeSkillError("Actor path must remain inside the episode")
    call_dir = _inside(actor, actor / "skill-calls" / call_id)
    source_path = Path(inspect.getsourcefile(type(policy))).resolve()
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    initial_state = _copy(policy.describe())
    if not isinstance(initial_state, dict):
        raise NativeSkillError("Policy describe() must return a JSON object")
    policy_state = initial_state
    try:
        call_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise NativeSkillError(f"Existing skill call cannot be reused: {call_id}") from exc
    initial = {
        "skill_call_id": call_id, "invoked": False, "policy_state": initial_state,
        "policy_source_path": str(source_path), "policy_source_sha256": source_sha,
        "start_request_id": None, "start_sequence": None, "start_observation_sha256": None,
        "actor_selection_path": None, "selection_sha256": None,
        "maximum_replies": maximum_replies, "poll_interval_s": poll_interval_s,
        "timeout_s": timeout_s,
    }
    initial_written = False
    replies, seen, last_sequence = 0, {}, -1
    current_id, current_sequence, observation_sha = None, None, None
    deadline = time.monotonic() + timeout_s

    def finish(status, **details):
        nonlocal initial_written
        if not initial_written:
            _write_new(actor, call_dir / "initial.json", initial)
            initial_written = True
        result = {
            "status": status, "skill_call_id": call_id, "replies": replies,
            "request_id": current_id, "sequence": current_sequence,
            "observation_sha256": observation_sha, "policy_state": policy_state,
            "policy_source_sha256": source_sha, **details,
        }
        _write_new(actor, call_dir / "handoff.json", result)
        return result

    try:
        while time.monotonic() < deadline:
            if (episode / "summary.json").is_file():
                return finish("episode_complete")
            pointer_path = actor / "current.json"
            pointer = _read(pointer_path) if pointer_path.is_file() else None
            if pointer is not None:
                if not isinstance(pointer, dict):
                    raise NativeSkillError("Current mailbox pointer must be an object")
                status = pointer.get("status")
                if status in {"done", "completed", "finished"}:
                    return finish("episode_complete")
                if status in {"error", "timeout"}:
                    return finish(f"mailbox_{status}")
                if status not in {"waiting", "responded"}:
                    raise NativeSkillError(f"Unknown current mailbox status: {status}")
                if status == "waiting":
                    request, request_path, reply_path = _waiting(actor, pointer)
                    current_id, current_sequence = request["request_id"], request["sequence"]
                    observation_sha = _sha(request["observation"])
                    binding = {"pointer": pointer, "request_sha256": _sha(request)}
                    if current_id in seen:
                        previous = seen[current_id]
                        if (binding != previous["binding"] or current_sequence != last_sequence
                                or not reply_path.is_file() or _read(reply_path) != previous["reply"]):
                            raise NativeSkillError("Stale or reused current request/reply")
                    else:
                        if current_sequence <= last_sequence:
                            raise NativeSkillError("Stale or reused request sequence")
                        intent_path = _inside(actor, actor / f"intents/{current_sequence:03}.json")
                        if reply_path.exists() or intent_path.exists():
                            raise NativeSkillError("Existing reply or intent cannot be reused")
                        if not initial_written:
                            selection_path = _inside(actor, actor / f"skill-calls/{call_id}-selection.json")
                            if not selection_path.is_file():
                                raise NativeSkillError("Actor selection must exist before skill invocation")
                            selection = _read(selection_path)
                            expected = {
                                "skill_call_id": call_id, "decision_origin": "online_model",
                                "request_id": current_id, "sequence": current_sequence,
                                "observation_sha256": observation_sha, "policy": "WrenchWithdrawPolicy",
                            }
                            if (not isinstance(selection, dict) or set(selection) != {*expected, "diagnosis"}
                                    or any(selection.get(key) != value for key, value in expected.items())
                                    or not isinstance(selection["diagnosis"], str) or not selection["diagnosis"].strip()):
                                raise NativeSkillError("Actor selection does not match the fresh request")
                            if initial_state.get("initial_observation_sha256") != observation_sha:
                                raise NativeSkillError("Policy initial observation does not match selection")
                            initial.update({
                                "invoked": True, "start_request_id": current_id,
                                "start_sequence": current_sequence, "start_observation_sha256": observation_sha,
                                "actor_selection_path": str(selection_path),
                                "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                            })
                            _write_new(actor, call_dir / "initial.json", initial)
                            initial_written = True
                        before = _copy(policy.describe())
                        action = _copy(policy.decide(_copy(request["observation"])))
                        policy_state = _copy(policy.describe())
                        goal = _copy(policy.last_goal)
                        if action is not None and not isinstance(action, dict):
                            raise NativeSkillError("Policy action must be a JSON object or None")
                        if action is not None and action.get("operation") == "finish":
                            raise NativeSkillError("Local skill cannot send finish; hand control to the actor")
                        decision = {
                            "request_id": current_id, "sequence": current_sequence,
                            "observation_sha256": observation_sha, "action": action, "goal": goal,
                            "policy_state_before": before, "policy_state_after": policy_state,
                            "policy_step": policy.steps, "policy_source_sha256": source_sha,
                        }
                        _write_new(actor, call_dir / f"decisions/{current_sequence:03}.json", decision)
                        if (_read(pointer_path) != pointer or _sha(_read(request_path)) != binding["request_sha256"]):
                            raise NativeSkillError("Current request changed while policy was deciding")
                        if action is None:
                            return finish("handoff")
                        intent = {
                            "request_id": current_id, "observation_sha256": observation_sha,
                            "semantic_operation": policy.phase,
                            "target_head_m": goal.get("target_head_m") if isinstance(goal, dict) else None,
                            "diagnosis": f"Explicit local feedback policy {initial_state.get('version', '')}: {policy.status}",
                            "correction_trigger": None, "decision_origin": "local_feedback_skill",
                            "skill_call_id": call_id, "policy_source_sha256": source_sha,
                            "policy_step": policy.steps, "goal": goal,
                        }
                        _write_new(actor, intent_path, intent)
                        reply = {"request_id": current_id, "action": action}
                        if _read(pointer_path) != pointer:
                            raise NativeSkillError("Current request changed before reply publication")
                        _write_new(actor, reply_path, reply)
                        seen[current_id] = {"binding": binding, "reply": reply}
                        last_sequence = current_sequence
                        replies += 1
                        if replies >= maximum_replies:
                            return finish("reply_limit")
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(poll_interval_s, remaining))
        return finish("timeout")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        finish("error", error=f"{type(exc).__name__}: {exc}")
        if isinstance(exc, NativeSkillError):
            raise
        raise NativeSkillError(str(exc)) from exc
