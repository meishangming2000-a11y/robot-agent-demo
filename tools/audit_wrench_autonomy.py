"""Audit saved native wrench decisions and recovery evidence without simulation."""

import argparse
import bisect
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dexhand_agent.wrench_evaluate import WrenchEvaluator

SEMANTIC_OPERATION_ALIASES = {"safe_height_align": "align"}


def canonical_sha(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate JSON field: {key}")
            value[key] = item
        return value

    value = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique)
    canonical_sha(value)  # Reject NaN, Infinity and overflow throughout the record.
    return value


def audit_episode(episode, protocol_path):
    episode, protocol_path = Path(episode), Path(protocol_path)
    protocol = read_json(protocol_path)
    if protocol["protocol_id"] not in {"wrench-native-recovery-development-v1",
                                       "wrench-native-recovery-development-v2",
                                       "wrench-native-recovery-development-v3"}:
        raise ValueError("Unsupported recovery protocol")
    stored = read_json(episode / "summary.json")
    metadata = read_json(episode / "metadata.json")
    history = read_json(episode / "actions.json")
    errors, records, requests, intents = [], [], {}, {}
    if not history:
        errors.append("No completed decision history")
    if stored.get("backend") != "native" or metadata.get("backend") != "native":
        errors.append("Episode is not recorded as native")
    if stored.get("decisions") != len(history):
        errors.append("Summary decision count differs from history")
    if len(history) > protocol["maximum_decisions_per_attempt"]:
        errors.append("Protocol decision limit exceeded")
    if metadata.get("episode_id") != stored.get("episode_id"):
        errors.append("Metadata and summary episode ids differ")
    request_ids = set()
    for expected, row in enumerate(history):
        try:
            seq = row["sequence"]
            if seq != expected:
                raise ValueError("History sequences must be contiguous from zero")
            step, actor = episode / f"steps/{seq:03}", episode / f"actor/steps/{seq:03}"
            request, action, result = (read_json(step / name) for name in
                                       ("request.json", "action.json", "result.json"))
            actor_request, reply = (read_json(actor / name) for name in ("request.json", "reply.json"))
            intent = read_json(episode / f"actor/intents/{seq:03}.json")
            observation_sha = canonical_sha(request["observation"])
            checks = {
                "request_identity": request["sequence"] == seq
                    and request["episode_id"] == stored["episode_id"]
                    and request["request_id"] not in request_ids,
                "actor_request": {k: v for k, v in actor_request.items() if k != "images"}
                    == {k: v for k, v in request.items() if k != "images"},
                "raw_reply": set(reply) == {"request_id", "action"}
                    and reply["request_id"] == request["request_id"],
                "executed_action": action == row["action"] == reply.get("action"),
                "result": result == row["result"],
                "intent_identity": intent["request_id"] == request["request_id"]
                    and intent["observation_sha256"] == observation_sha,
            }
            for name, passed in checks.items():
                if not passed:
                    errors.append(f"Sequence {seq}: {name} mismatch")
            request_ids.add(request["request_id"])
            requests[seq], intents[seq] = request, intent
            records.append({"sequence": seq, "request_id": request["request_id"],
                            "observation_sha256": observation_sha,
                            "raw_reply_action_sha256": canonical_sha(reply.get("action")),
                            "executed_action_sha256": canonical_sha(action), "checks": checks})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"Sequence {expected}: {type(exc).__name__}: {exc}")

    metrics = stored["metrics"]
    evaluator = WrenchEvaluator(metrics["initial_height_m"], metrics["initial_nut_angle_rad"])
    limits = protocol["quality_criteria"]
    sample_path = episode / "samples.jsonl"
    stream = sample_path.open(encoding="utf8") if sample_path.is_file() else gzip.open(
        episode / "samples.jsonl.gz", "rt", encoding="utf8")
    samples, quality_failures, cadence_failures = [], [], []
    with stream:
        for index, line in enumerate(stream):
            sample = json.loads(line)
            canonical_sha(sample)
            evaluator.update(sample)
            if samples and sample["time_s"] - samples[-1]["time_s"] > 1 / 500 + 1e-8:
                cadence_failures.append(index)
            reasons = []
            if sample["max_contact_force_n"] > limits["maximum_individual_contact_normal_n"]:
                reasons.append("contact_normal")
            if sample["max_penetration_m"] > limits["maximum_global_penetration_m"]:
                reasons.append("penetration")
            if sample["robot_nut_contact"]:
                reasons.append("robot_nut_contact")
            if reasons:
                quality_failures.append({"sample_index": index, "time_s": sample["time_s"],
                                         "reasons": reasons})
            samples.append(sample)
    recomputed = evaluator.summary()
    mismatches = [key for key, value in recomputed.items() if stored.get(key) != value]
    times = [s["time_s"] for s in samples]
    observation_errors = []
    for seq, request in requests.items():
        obs = request["observation"]
        time_s = obs["time_s"]
        index = bisect.bisect_left(times, time_s - 1e-8)
        # The initial request can precede the first physical integration sample.
        if seq == 0 and times and time_s < times[0]:
            continue
        if index == len(times) or abs(times[index] - time_s) > 1e-8:
            observation_errors.append(f"Sequence {seq}: no matching physical sample")
        elif any(obs["contacts"].get(k) != v for k, v in samples[index].items()):
            observation_errors.append(f"Sequence {seq}: contacts differ from physical sample")

    finish = bool(history and history[-1]["action"].get("operation") == "finish"
                  and history[-1]["result"].get("status") == "finished")
    no_error = stored.get("error") is None and not (episode / "error.txt").exists() and all(
        row["result"].get("status") != "interrupted" for row in history)
    physical_pass = bool(samples and not mismatches and recomputed["success"]
                         and not observation_errors and not cadence_failures)
    quality_pass = bool(samples and not quality_failures)
    binding_pass = not errors
    events, attempts = [], []
    if binding_pass and not observation_errors:
        rules = protocol["recovery_acceptance"]["numeric_rules"]
        lift_time = recomputed["gates"]["lift"]["passed_at_s"]
        turn_time = recomputed["gates"]["turn"]["passed_at_s"]

        def observation(seq):
            return requests[seq]["observation"]

        def actual_motion(seq):
            return (history[seq]["result"].get("status") == "completed"
                    and history[seq]["action"].get("operation") == "move"
                    and seq + 1 in requests
                    and observation(seq + 1)["time_s"] > observation(seq)["time_s"])

        def retained_unsupported(obs):
            contacts = obs["contacts"]
            return (len(set(contacts["hand_tool_contacts"])) >= 2
                    and not contacts["tool_support_contacts"]
                    and not contacts["tool_nut_contact"] and not contacts["head_nut_contact"])

        def eligible(seq, kind):
            if seq < 1 or not actual_motion(seq - 1):
                return False
            before, now = observation(seq - 1), observation(seq)
            prior = intents[seq - 1]
            if lift_time is None or now["time_s"] < lift_time:
                return False
            if not any(gate["passed_at_s"] is not None and gate["passed_at_s"] > now["time_s"]
                       for name, gate in recomputed["gates"].items() if name != "lift"):
                return False  # A cosmetic adjustment after all gates is not recovery.
            head, nut = now["tool"]["head_center_m"], now["nut"]
            if kind == "working_end_tracking":
                target = prior.get("target_head_m")
                role = SEMANTIC_OPERATION_ALIASES.get(prior["semantic_operation"], prior["semantic_operation"])
                return (target is not None and role in {"level", "align"}
                        and before["time_s"] >= lift_time
                        and history[seq - 1]["action"].get("duration_s", 0) >= 0.5
                        and now["time_s"] - before["time_s"] >= 0.5 - 1e-9
                        and retained_unsupported(now)
                        and math.dist(target[:2], nut["position_m"][:2]) <= 0.03 + 1e-9
                        and target[2] >= nut["top_m"] + 0.01 - 1e-9
                        and math.dist(head, target) >= rules[kind]["trigger_position_error_m_at_least"])
            if kind == "alignment_after_attempt":
                rule = rules[kind]
                error = math.dist(head[:2], nut["position_m"][:2])
                return (prior["semantic_operation"] in {"align", "insert", "correct_alignment"}
                        and 0 <= head[2] - nut["top_m"] <= 0.015 + 1e-9
                        and error >= rule["trigger_xy_error_m_at_least"])
            target = prior.get("target_head_m")
            return (prior["semantic_operation"] == "insert" and target is not None
                    and target[2] <= nut["position_m"][2] + 0.008 + 1e-9
                    and now["contacts"]["head_nut_contact"]
                    and head[2] - target[2] >= rules[kind]["trigger_head_above_intended_target_m_at_least"]
                    and now["contacts"]["nut_angle_rad"] - before["contacts"]["nut_angle_rad"]
                    < rules[kind]["trigger_nut_angle_advance_rad_below"])

        for seq, intent in intents.items():
            trigger = intent.get("correction_trigger")
            if not trigger:
                continue
            attempt = {"correction_sequence": seq, "raw_trigger": trigger, "passed_rules": []}
            attempts.append(attempt)
            trigger_seq = trigger.get("trigger_sequence") if isinstance(trigger, dict) else None
            if (not isinstance(trigger, dict) or not trigger.get("type")
                    or not isinstance(trigger_seq, int) or isinstance(trigger_seq, bool)
                    or not 1 <= trigger_seq <= seq or not actual_motion(seq)):
                attempt["reason"] = "Missing eligible trigger identity or executed correction motion"
                continue
            kinds = ["alignment_after_attempt", "blocked_seating"]
            if "working_end_tracking" in rules:
                kinds.append("working_end_tracking")
            for kind in kinds:
                possible = [i for i in range(1, seq + 1) if eligible(i, kind)]
                if not possible or trigger_seq != possible[0]:
                    continue
                before = observation(trigger_seq)
                head = before["tool"]["head_center_m"]
                for after_seq in range(seq + 1, len(history)):
                    valid_actions = sum(history[i]["result"].get("status") == "completed"
                                        for i in range(trigger_seq, after_seq))
                    window = (after_seq - trigger_seq if kind != "blocked_seating"
                              else valid_actions)
                    maximum = rules[kind].get("after_window_maximum_decisions", 6)
                    if window > maximum:
                        break
                    after = observation(after_seq)
                    if after["time_s"] <= before["time_s"]:
                        continue
                    event = {"rule": kind, "trigger_sequence": trigger_seq,
                             "correction_sequence": seq, "after_sequence": after_seq,
                             "trigger_time_s": before["time_s"], "after_time_s": after["time_s"],
                             "raw_trigger_type": trigger["type"]}
                    if kind == "working_end_tracking":
                        rule = rules[kind]
                        target = intents[trigger_seq - 1]["target_head_m"]
                        initial = math.dist(head, target)
                        final = math.dist(after["tool"]["head_center_m"], target)
                        # The after request observes the last action's result. Its
                        # own intent is the next action, outside this measured window.
                        target_sequences = range(trigger_seq - 1, after_seq)
                        same_target = all(intents[i].get("target_head_m") is not None
                                          and math.dist(intents[i]["target_head_m"], target)
                                          <= rule["same_intended_target_tolerance_m"]
                                          for i in target_sequences)
                        passed = (same_target and retained_unsupported(after)
                                  and final <= rule["after_position_error_m_at_most"]
                                  and (initial - final) / initial >= rule["relative_error_reduction_at_least"])
                        event.update(intended_target_m=target, before_position_error_m=initial,
                                     after_position_error_m=final,
                                     fixed_target_action_sequences=list(target_sequences))
                    elif kind == "alignment_after_attempt":
                        rule = rules[kind]
                        initial = math.dist(head[:2], before["nut"]["position_m"][:2])
                        final = math.dist(after["tool"]["head_center_m"][:2],
                                          after["nut"]["position_m"][:2])
                        passed = (final <= rule["after_xy_error_m_at_most"]
                                  and (initial - final) / initial >= rule["relative_error_reduction_at_least"])
                        event.update(before_xy_error_m=initial, after_xy_error_m=final)
                    else:
                        target_z = intents[trigger_seq - 1]["target_head_m"][2]
                        initial = abs(head[2] - target_z)
                        final = abs(after["tool"]["head_center_m"][2] - target_z)
                        passed = (initial - final >= 0.003 - 1e-9 and turn_time is not None
                                  and turn_time > before["time_s"] and turn_time >= after["time_s"])
                        event.update(intended_target_z_m=target_z, before_error_m=initial,
                                     after_error_m=final, contact_qualified_turn_at_s=turn_time)
                    if passed:
                        events.append(event)
                        attempt["passed_rules"].append(kind)
                        break
    return {
        "audit_version": 3, "episode": str(episode.resolve()),
        "semantic_operation_aliases": dict(SEMANTIC_OPERATION_ALIASES),
        "semantic_operation_alias_scope": "working_end_tracking eligibility role only",
        "audit_disclosure": "Historical post-execution interface correction for native003; its original strict audit is retained; numeric rules unchanged. Consult each episode's source freeze for the auditor used before execution.",
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluator_source_sha256": hashlib.sha256(
            Path(sys.modules[WrenchEvaluator.__module__].__file__).read_bytes()).hexdigest(),
        "action_binding": {"passed": binding_pass, "errors": errors, "sequences": records,
                           "claim_scope": "Saved replies match executed actions; files alone do not authenticate actor identity or exclude undisclosed strategic intervention."},
        "physical": {"matches_saved_result": not mismatches, "mismatched_fields": mismatches,
                     "recomputed": recomputed, "observation_errors": observation_errors,
                     "sampling_500hz_passed": bool(samples and not cadence_failures),
                     "first_cadence_failure_index": next(iter(cadence_failures), None)},
        "quality": {"passed": quality_pass, "violating_samples": len(quality_failures),
                    "first_violation": next(iter(quality_failures), None)},
        "accepted_finish": finish, "no_runtime_error": no_error,
        "recovery_attempts": attempts, "recovery_events": events,
        "unsupported_rules": {"grip_loss": "Continuous loss/restoration window audit is not implemented; no grip-loss acceptance is claimed."},
        "recovery_accepted": bool(binding_pass and physical_pass and quality_pass
                                  and finish and no_error and events),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit_episode(args.episode, args.protocol)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"audit_error": f"{type(exc).__name__}: {exc}", "recovery_accepted": False}
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf8") as stream:
            stream.write(text)
    else:
        print(text, end="")
    return 0 if result["recovery_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
