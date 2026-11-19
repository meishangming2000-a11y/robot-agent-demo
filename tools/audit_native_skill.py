"""Replay declared native skill traces from saved observations, without physics."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_wrench_autonomy import canonical_sha, read_json

SOURCES = ("dexhand_agent/wrench_recovery.py", "dexhand_agent/wrench_tools.py", "tools/drive_native_skill.py")
SELECTION_KEYS = {"skill_call_id", "decision_origin", "request_id", "sequence",
                  "observation_sha256", "policy", "diagnosis"}
ACTION_KEYS = {"operation", "targets", "palm_target_m", "palm_quat_wxyz", "duration_s", "memory"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _action_types(action):
    _require(isinstance(action, dict) and set(action) == ACTION_KEYS, "Incomplete action schema")
    _require(action["operation"] in {"move", "hold", "finish"} and isinstance(action["memory"], str),
             "Invalid action operation or memory type")
    _require(type(action["duration_s"]) in (int, float), "Invalid action duration type")
    _require(isinstance(action["targets"], list), "Invalid actuator targets type")
    for entry in action["targets"]:
        _require(isinstance(entry, dict) and set(entry) == {"actuator", "value"}
                 and isinstance(entry["actuator"], str) and type(entry["value"]) in (int, float),
                 "Invalid actuator target type")
    for key, size in (("palm_target_m", 3), ("palm_quat_wxyz", 4)):
        value = action[key]
        _require(value is None or isinstance(value, list) and len(value) == size
                 and all(type(v) in (int, float) for v in value), f"Invalid {key} type")


def audit_episode(episode, source_freeze=None, *, policy_type=None, max_online_choices=32, max_commands=160):
    """Validate saved bindings/replay only; acceptance does not certify physics."""
    for name, value in (("max_online_choices", max_online_choices), ("max_commands", max_commands)):
        _require(type(value) is int and value > 0, f"{name} must be a positive integer")
    episode = Path(episode).resolve()
    inputs, errors, requests, intents, covered = {}, [], {}, {}, {}

    def load(path):
        path = Path(path).resolve()
        _require(path.is_relative_to(episode), f"Record path escapes episode: {path}")
        value = read_json(path)
        inputs[path.relative_to(episode).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return value

    history, metadata, summary = (load(episode / name) for name in
                                  ("actions.json", "metadata.json", "summary.json"))
    episode_id = metadata["episode_id"]
    if metadata.get("backend") != "native" or summary.get("backend") != "native":
        errors.append("Episode is not recorded as native")
    if summary.get("episode_id") != episode_id or summary.get("decisions") != len(history):
        errors.append("Summary identity or decision count mismatch")
    if source_freeze is None:
        binding = load(episode / "actor-binding.json")
        source_freeze = episode / binding["source_freeze"]
    source_freeze = Path(source_freeze).resolve()
    freeze = read_json(source_freeze)
    source_errors, source_hashes = [], {}
    for name in SOURCES:
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        source_hashes[name] = digest
        if freeze.get("files_sha256", {}).get(name) != digest:
            source_errors.append(f"Frozen source mismatch: {name}")
    if source_errors:
        errors.extend(source_errors)
    source_sha = source_hashes[SOURCES[0]]

    def request(seq):
        _require(type(seq) is int and 0 <= seq <= len(history) and seq < max_commands, "Invalid request sequence")
        if seq not in requests:
            value = load(episode / f"steps/{seq:03}/request.json")
            actor = load(episode / f"actor/steps/{seq:03}/request.json")
            _require(type(value.get("sequence")) is int and value["sequence"] == seq
                     and value.get("episode_id") == episode_id and isinstance(value.get("observation"), dict)
                     and isinstance(value.get("request_id"), str) and bool(value["request_id"]), "Request identity mismatch")
            _require(all(r["request_id"] != value["request_id"] for r in requests.values()),
                     "Request id reused across sequences")
            _require({k: v for k, v in value.items() if k != "images"}
                     == {k: v for k, v in actor.items() if k != "images"}, "Actor request mismatch")
            if seq > 0:
                _require(value["observation"].get("last_action") == history[seq - 1]["result"],
                         "Fresh observation last_action differs from own preceding result")
            requests[seq] = value
        return requests[seq]

    for seq, row in enumerate(history):
        try:
            _require(type(row["sequence"]) is int and row["sequence"] == seq, "Noncontiguous history")
            req = request(seq)
            action, result = (load(episode / f"steps/{seq:03}/{name}.json") for name in ("action", "result"))
            reply = load(episode / f"actor/steps/{seq:03}/reply.json")
            intent = load(episode / f"actor/intents/{seq:03}.json")
            intents[seq] = intent
            _action_types(action)
            _require(set(reply) == {"request_id", "action"} and reply["request_id"] == req["request_id"],
                     "Strict reply identity mismatch")
            _require(canonical_sha(action) == canonical_sha(row["action"]) == canonical_sha(reply["action"])
                     and result == row["result"], "Raw reply/executed action/result mismatch")
            _require(intent["request_id"] == req["request_id"]
                     and intent["observation_sha256"] == canonical_sha(req["observation"]), "Intent identity mismatch")
            _require(intent.get("decision_origin") in {"online_model", "local_feedback_skill"}, "Unknown decision origin")
            _require(isinstance(intent.get("semantic_operation"), str)
                     and isinstance(intent.get("diagnosis"), str)
                     and (intent.get("correction_trigger") is None or isinstance(intent["correction_trigger"], dict))
                     and "target_head_m" in intent and "correction_trigger" in intent, "Invalid intent field type")
            target = intent["target_head_m"]
            _require(target is None or isinstance(target, list) and len(target) == 3
                     and all(type(v) in (int, float) for v in target), "Invalid intent target type")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"Sequence {seq}: {exc}")

    # A proposed online reply still consumed a choice if it never executed.
    for path in sorted((episode / "actor/intents").glob("*.json")):
        try:
            seq = int(path.stem)
            _require(path.name == f"{seq:03}.json", "Invalid intent filename")
            if seq in intents:
                continue
            intent = load(path)
            intents[seq] = intent
            req = request(seq)
            _require(intent["request_id"] == req["request_id"]
                     and intent["observation_sha256"] == canonical_sha(req["observation"])
                     and intent.get("decision_origin") == "online_model", "Unexecuted intent identity/origin mismatch")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"Unexecuted intent {path.name}: {exc}")

    call_root = episode / "actor/skill-calls"
    selections, calls = {}, []
    for path in sorted(call_root.glob("*-selection.json")):
        call_id = path.name[:-len("-selection.json")]
        try:
            value = load(path)
            _require(set(value) == SELECTION_KEYS and value["skill_call_id"] == call_id
                     and value["decision_origin"] == "online_model" and value["policy"] == "WrenchWithdrawPolicy"
                     and isinstance(value["diagnosis"], str) and bool(value["diagnosis"].strip()), "Invalid selection schema")
            req = request(value["sequence"])
            _require(value["request_id"] == req["request_id"]
                     and value["observation_sha256"] == canonical_sha(req["observation"]), "Selection fresh request mismatch")
            selections[call_id] = value
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"Selection {call_id}: {exc}")
    directories = sorted((p for p in call_root.iterdir() if p.is_dir()), key=lambda p: p.name) if call_root.exists() else []
    for directory in directories:
        call_id, call_errors = directory.name, []
        record = {"skill_call_id": call_id, "invoked": None, "replayed_decisions": 0, "local_motor_commands": 0}
        try:
            _require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", call_id)), "Unsafe skill call id")
            initial, handoff = load(directory / "initial.json"), load(directory / "handoff.json")
            record["invoked"] = initial["invoked"]
            _require(initial["skill_call_id"] == handoff["skill_call_id"] == call_id, "Call id mismatch")
            _require(Path(initial["policy_source_path"]).resolve() == (ROOT / SOURCES[0]).resolve()
                     and initial["policy_source_sha256"] == handoff["policy_source_sha256"] == source_sha,
                     "Policy source path/hash mismatch")
            decisions = sorted((directory / "decisions").glob("*.json"))
            if initial["invoked"] is False:
                _require(not decisions and handoff["replies"] == 0
                         and all(initial[key] is None for key in ("start_request_id", "start_sequence",
                             "start_observation_sha256", "actor_selection_path", "selection_sha256"))
                         and all(handoff[key] is None for key in ("request_id", "sequence", "observation_sha256"))
                         and initial["policy_state"] == handoff["policy_state"], "Invalid not-invoked record")
                record["status"] = "not_invoked"
                calls.append(record)
                continue
            _require(initial["invoked"] is True and bool(decisions), "Invoked call has no decision trace")
            _require(not source_errors, "Unverified frozen sources cannot be replayed")
            selection_path = call_root / f"{call_id}-selection.json"
            _require(call_id in selections and Path(initial["actor_selection_path"]).resolve() == selection_path.resolve(),
                     "Missing selection or selection path mismatch")
            _require(initial["selection_sha256"] == hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                     "Original selection file hash mismatch")
            start = initial["start_sequence"]
            req = request(start)
            _require(initial["start_request_id"] == req["request_id"]
                     and initial["start_observation_sha256"] == canonical_sha(req["observation"])
                     and selections[call_id]["sequence"] == start, "Initial fresh request mismatch")
            previous = next((row["action"] for row in reversed(history[:start])
                             if row["result"].get("status") == "completed" and row["action"].get("operation") == "move"
                             and row["action"].get("palm_target_m") is not None
                             and row["action"].get("palm_quat_wxyz") is not None), None)
            _require(previous is not None and initial["policy_state"]["previous_palm_action"] == previous,
                     "Initial previous action is not the latest own completed full palm action")
            if policy_type is None:
                from dexhand_agent.wrench_recovery import WrenchWithdrawPolicy
                factory = WrenchWithdrawPolicy
            else:
                factory = policy_type
            policy = factory(copy.deepcopy(req["observation"]), copy.deepcopy(previous),
                             max_steps=initial["policy_state"]["max_steps"])
            _require(canonical_sha(policy.describe()) == canonical_sha(initial["policy_state"]), "Initial policy state mismatch")
            last_seq, replies, last_action = start - 1, 0, "not_called"
            for path in decisions:
                decision = load(path)
                seq = decision["sequence"]
                _require(type(seq) is int and seq == last_seq + 1 and path.name == f"{seq:03}.json"
                         and last_action is not None, "Skipped/duplicate decision or decision after handoff")
                req = request(seq)
                _require(decision["request_id"] == req["request_id"]
                         and decision["observation_sha256"] == canonical_sha(req["observation"])
                         and decision["policy_source_sha256"] == source_sha, "Decision fresh source binding mismatch")
                before = policy.describe()
                predicted = policy.decide(copy.deepcopy(req["observation"]))
                after = policy.describe()
                expected = {"action": predicted, "goal": policy.last_goal, "policy_state_before": before,
                            "policy_state_after": after, "policy_step": policy.steps}
                _require(all(canonical_sha(decision[k]) == canonical_sha(v) for k, v in expected.items()),
                         "Exact policy action/state/goal replay mismatch")
                record["replayed_decisions"] += 1
                if predicted is not None:
                    _action_types(predicted)
                    _require(predicted["operation"] != "finish" and seq < len(history), "Local policy cannot finish or invent execution")
                    _require(seq not in covered and canonical_sha(predicted) == canonical_sha(history[seq]["action"]),
                             "Local action differs from execution or overlaps another call")
                    goal = policy.last_goal
                    expected_intent = {"request_id": req["request_id"], "observation_sha256": canonical_sha(req["observation"]),
                        "semantic_operation": policy.phase, "target_head_m": goal.get("target_head_m") if isinstance(goal, dict) else None,
                        "diagnosis": f"Explicit local feedback policy {initial['policy_state'].get('version', '')}: {policy.status}",
                        "correction_trigger": None, "decision_origin": "local_feedback_skill", "skill_call_id": call_id,
                        "policy_source_sha256": source_sha, "policy_step": policy.steps, "goal": goal}
                    _require(seq in intents and canonical_sha(intents[seq]) == canonical_sha(expected_intent), "Local intent differs from replay/bridge contract")
                    covered[seq] = call_id
                    replies += 1
                last_seq, last_action = seq, predicted
            _require(handoff["replies"] == replies <= initial["maximum_replies"], "Handoff reply count mismatch")
            _require(canonical_sha(handoff["policy_state"]) == canonical_sha(policy.describe()), "Handoff policy state mismatch")
            end = request(handoff["sequence"])
            _require(handoff["request_id"] == end["request_id"]
                     and handoff["observation_sha256"] == canonical_sha(end["observation"])
                     and last_seq <= handoff["sequence"] <= last_seq + 1, "Handoff fresh observation mismatch")
            status = handoff["status"]
            _require(status in {"handoff", "reply_limit", "episode_complete", "timeout", "mailbox_error", "mailbox_timeout", "error"},
                     "Unknown handoff status")
            if status in {"handoff", "reply_limit"}:
                _require(handoff["sequence"] == last_seq and
                         (last_action is None if status == "handoff" else replies == initial["maximum_replies"]), "Invalid handoff termination")
            record.update(local_motor_commands=replies, status=status)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            call_errors.append(str(exc))
        record["errors"] = call_errors
        calls.append(record)
        errors.extend(f"Call {call_id}: {e}" for e in call_errors)
    for call_id in selections:
        if call_id not in {p.name for p in directories}:
            errors.append(f"Selection {call_id}: missing call record")
    local = {seq for seq, intent in intents.items()
             if seq < len(history) and intent.get("decision_origin") == "local_feedback_skill"}
    if local != set(covered):
        errors.append("Local execution coverage does not match replayed call decisions")
    online = sum(intent.get("decision_origin") == "online_model" for intent in intents.values())
    executed_online = sum(seq < len(history) and intent.get("decision_origin") == "online_model"
                          for seq, intent in intents.items())
    selection_count = len(list(call_root.glob("*-selection.json")))
    counts = {"low_level_commands": len(history), "online_action_choices": online,
              "skill_selections": selection_count, "online_model_choices": online + selection_count,
              "local_motor_commands": len(local), "unclassified_commands": len(history) - executed_online - len(local)}
    budgets = {f"low_level_commands_at_most_{max_commands}": len(history) <= max_commands,
               f"online_model_choices_at_most_{max_online_choices}": counts["online_model_choices"] <= max_online_choices}
    return {"audit_version": 1, "episode": str(episode), "accepted": not errors and all(budgets.values()),
            "trace": {"passed": not errors, "errors": errors}, "counts": counts, "budgets": budgets, "calls": calls,
            "budget_limits": {"max_online_choices": max_online_choices, "max_commands": max_commands},
            "sources": {"passed": not source_errors, "errors": source_errors, "files_sha256": source_hashes,
                        "freeze_path": str(source_freeze), "freeze_sha256": hashlib.sha256(source_freeze.read_bytes()).hexdigest()},
            "input_files_sha256": inputs, "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "claim_scope": "Saved files are consistent with the frozen observation-only policy and declared budgets. This does not certify physical task/recovery success, authenticate actor identity, or exclude undisclosed intervention; combine with the separate physical audit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--source-freeze", type=Path)
    parser.add_argument("--max-online-choices", type=int, default=32)
    parser.add_argument("--max-commands", type=int, default=160)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_online_choices < 1 or args.max_commands < 1:
        parser.error("Budget limits must be positive integers")
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        result = audit_episode(args.episode, args.source_freeze, max_online_choices=args.max_online_choices,
                               max_commands=args.max_commands)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"accepted": False, "audit_error": f"{type(exc).__name__}: {exc}"}
    with args.output.open("x", encoding="utf8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
