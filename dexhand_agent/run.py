"""Run one bounded episode and retain inputs, controls, measurements, and video."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import uuid

import mujoco
import numpy as np
from PIL import Image

from .codex_backend import BackendError, CodexBackend
from .evaluate import TaskEvaluator
from .experience import export_experience, load_experience
from .file_backend import FileBackend
from .reference import ReferencePolicy
from .reporting import write_episode_report
from .runtime import ACTION_SCHEMA, HandRuntime, InvalidAction
from .scene import ROOT


GOAL = ("Lift the free hammer at least 20 mm unsupported for 2 continuous seconds, "
        "use its head to depress the button at least 8 mm for 0.5 continuous seconds, "
        "then withdraw the head at least 30 mm above the resting cap while retaining "
        "the tool for 1 second and letting the cap return. Keep at least two hand "
        "parts in tool contact. Never touch the cap directly with the robot.")
WRENCH_GOAL = ("Lift the free wrench at least 20 mm with no fixture or nut support for 2 continuous seconds. "
               "Use its head to turn the passive braked nut at least positive 45 degrees from its initial "
               "angle for 0.5 continuous seconds. Withdraw the head at least 30 mm clear of the nut while "
               "retaining the unsupported tool and the nut angle for 1 second. Keep at least two distinct "
               "hand parts contacting the tool. Never touch the nut directly with the robot.")
TWEEZERS_GOAL = ("Pick up the free tweezers at least 20 mm unsupported for 2 seconds with at least two "
                 "distinct hand parts in contact. Use both working tips to lift the free object at least "
                 "20 mm for 1 second, without other tool parts, robot or fixtures supporting it. "
                 "Transport it within 20 mm XY of the goal with its bottom at least 10 mm above the goal "
                 "platform for 0.5 seconds. Release it stably onto that platform, speed below 0.02 m/s, "
                 "then keep both tips at least 20 mm clear and retain the unsupported tweezers for "
                 "1 second. Never touch the object directly with the robot.")


def _task_components(task):
    if task == "hammer_button":
        return HandRuntime, lambda runtime: TaskEvaluator(runtime.initial_height_m), ReferencePolicy, GOAL
    if task == "wrench_turn":
        from .wrench import WrenchRuntime, WrenchReference
        from .wrench_evaluate import WrenchEvaluator
        return (WrenchRuntime, lambda runtime: WrenchEvaluator(runtime.initial_height_m,
                runtime.initial_nut_angle_rad), WrenchReference, WRENCH_GOAL)
    if task == "tweezers_transfer":
        from .tweezers import TweezersRuntime, TweezersReference
        from .tweezers_evaluate import TweezersEvaluator
        return (TweezersRuntime, lambda runtime: TweezersEvaluator(runtime.initial_height_m,
                runtime.initial_object_height_m), TweezersReference, TWEEZERS_GOAL)
    raise ValueError(f"Unknown task: {task}")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf8")


def run_episode(output, backend="reference", seed=0, max_decisions=None, experience=None, video=True,
                task="hammer_button", policy=None, quality_guard=False, rich_feedback=False,
                wrench_variant=None):
    if wrench_variant is not None:
        if wrench_variant not in ("placement-30deg-v1", "placement-minus20deg-v2", "placement-plus15deg-v3",
                                  "wrench-long-heavy-v1"):
            raise ValueError(f"Unknown wrench_variant: {wrench_variant}")
        if task != "wrench_turn":
            raise ValueError("wrench_variant requires task wrench_turn")
        if policy is not None:
            raise ValueError("wrench_variant cannot be combined with policy")
        if wrench_variant == "wrench-long-heavy-v1" and backend not in ("native", "cli"):
            raise ValueError("wrench-long-heavy-v1 requires backend native or cli")
        quality_guard = rich_feedback = True
    if quality_guard and task != "wrench_turn":
        raise ValueError("quality_guard requires task wrench_turn")
    if rich_feedback and task != "wrench_turn":
        raise ValueError("rich_feedback requires task wrench_turn")
    if rich_feedback:
        from .wrench_sensing import read_wrench_feedback
    quality_guard = quality_guard or policy is not None
    if quality_guard:
        from .wrench_rollout import QUALITY_CRITERIA
    controller = None
    if policy is not None:
        if task != "wrench_turn" or backend != "reference":
            raise ValueError("policy requires task wrench_turn and backend reference")
        from .wrench_policy import (PARAMETER_NAMES, ParameterWrenchReference,
                                    context_from_observation, load_policy)
        policy_dir = Path(policy).resolve()
        learned_policy = load_policy(policy_dir)
        controller = {"kind": "neural_context_parameters", "policy_dir": str(policy_dir),
                      "quality_criteria": QUALITY_CRITERIA,
                      "policy_sha256": hashlib.sha256((policy_dir / "policy.json").read_bytes()).hexdigest(),
                      "weights_sha256": hashlib.sha256((policy_dir / "weights.npz").read_bytes()).hexdigest()}
    runtime_type, make_evaluator, reference_type, goal = _task_components(task)
    if wrench_variant is not None:
        if wrench_variant == "wrench-long-heavy-v1":
            from .wrench_geometry import LongWrenchRuntime
            runtime_type = LongWrenchRuntime
        else:
            from .wrench_placement import PlacementWrenchRuntime
            runtime_type = PlacementWrenchRuntime
    if max_decisions is None:
        max_decisions = 48 if task == "tweezers_transfer" else 32
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = runtime_type(seed) if wrench_variant is None else runtime_type(seed, variant=wrench_variant)
    evaluator = make_evaluator(runtime)
    if policy is None:
        reference = reference_type()
    else:
        try:
            observation = runtime.observe()
            if rich_feedback:
                observation["wrench_feedback"] = read_wrench_feedback(runtime)
            context = context_from_observation(observation)
            reference = ParameterWrenchReference(learned_policy.predict(context))
            controller.update({"context": context.tolist(),
                               "parameters_m": dict(zip(PARAMETER_NAMES, reference.parameters_m.tolist()))})
        except Exception:
            runtime.close()
            raise
    actor = (CodexBackend(output / "codex", timeout_s=240) if backend == "cli" else
             FileBackend(output / "actor", timeout_s=600) if backend == "native" else None)
    episode_id = uuid.uuid4().hex
    metadata = {"episode_id": episode_id, "backend": backend, "seed": seed, "task_id": task,
                "mujoco": mujoco.__version__,
                "observation_mode": "simulator_state", "source_sha256": {
                    p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (ROOT / "dexhand_agent").glob("*.py")},
                "experience": experience or [],
                "experience_sha256": hashlib.sha256(json.dumps(experience or [], sort_keys=True,
                    separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest(),
                "asset_manifest_sha256": hashlib.sha256((ROOT / "assets/manifest.json").read_bytes()).hexdigest(),
                "max_decisions": max_decisions}
    if wrench_variant is not None:
        metadata["scene_variant"] = runtime.variant_metadata
    if controller is not None:
        metadata["controller"] = controller
    if quality_guard:
        metadata["quality_criteria"] = QUALITY_CRITERIA
    if rich_feedback:
        metadata["feedback_contract"] = "wrench_geometry_and_named_contacts_v1"
    write_json(output / "metadata.json", metadata)
    writer = None
    cv2 = None
    if video:
        import cv2
        writer = cv2.VideoWriter(str(output / "episode.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 480))
        if not writer.isOpened():
            runtime.close()
            raise RuntimeError("Video writer failed; install the video extra or pass --no-video")
    next_frame_s = runtime.data.time
    sample_stream = (output / "samples.jsonl").open("w", encoding="utf8")
    def retain(sample):
        nonlocal next_frame_s
        evaluator.update(sample)
        sample_stream.write(json.dumps(sample, allow_nan=False, separators=(",", ":")) + "\n")
        if quality_guard:
            reasons = []
            if sample["max_contact_force_n"] > QUALITY_CRITERIA["max_contact_normal_n"]:
                reasons.append(f"max_contact_force_n exceeds {QUALITY_CRITERIA['max_contact_normal_n']:g} N")
            if sample["max_penetration_m"] > QUALITY_CRITERIA["max_global_penetration_m"]:
                reasons.append(f"max_penetration_m exceeds {QUALITY_CRITERIA['max_global_penetration_m']:g} m")
            if sample["robot_nut_contact"]:
                reasons.append("robot_nut_contact")
            if reasons:
                label = "Learned policy" if policy is not None else "Wrench"
                raise RuntimeError(label + " quality abort: " + "; ".join(reasons))
        if writer is not None and sample["time_s"] + 1e-9 >= next_frame_s:
            rgb = np.concatenate([runtime.render("overview", width=640, height=480),
                                  runtime.render("front", width=640, height=480)], axis=1)
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gates = evaluator.summary()["gates"]
            label = f"{backend} | sim {sample['time_s']:.1f}s | " + " ".join(
                f"{name}:{'PASS' if gate['passed'] else '...'}" for name, gate in gates.items())
            cv2.rectangle(frame, (0, 0), (1280, 25), (25, 25, 25), -1)
            cv2.putText(frame, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1, cv2.LINE_AA)
            writer.write(frame)
            next_frame_s += 0.1
    runtime.on_sample = retain
    history, memory = [], ""
    started = time.monotonic()
    error = None
    finished = False
    try:
        for sequence in range(max_decisions):
            step_dir = output / "steps" / f"{sequence:03}"
            step_dir.mkdir(parents=True)
            observation = runtime.observe()
            if rich_feedback:
                observation["wrench_feedback"] = read_wrench_feedback(runtime)
            observation["task_progress"] = evaluator.summary()
            observation["previous_memory"] = memory
            image_paths = []
            for camera in ("overview", "front"):
                path = step_dir / f"{camera}.png"
                Image.fromarray(runtime.render(camera, width=640, height=480)).save(path)
                image_paths.append(path)
            request = {"request_id": f"{episode_id}-{sequence}", "episode_id": episode_id,
                       "sequence": sequence, "task_id": task, "max_decisions": max_decisions,
                       "goal": goal, "observation": observation,
                       "recent_actions": history[-3:], "action_schema": ACTION_SCHEMA,
                       "experience": experience or []}
            write_json(step_dir / "request.json", request)
            action = actor.decide(request, image_paths) if actor else reference.decide(observation)
            write_json(step_dir / "action.json", action)
            try:
                result = runtime.execute(action)
            except InvalidAction as exc:
                result = {"status": "rejected", "reason": str(exc), "time_s": runtime.data.time}
                runtime.last_result = result
            except (RuntimeError, ValueError) as exc:
                result = {"status": "interrupted", "reason": f"{type(exc).__name__}: {exc}",
                          "time_s": runtime.data.time}
                runtime.last_result = result
                history.append({"sequence": sequence, "action": action, "result": result})
                write_json(step_dir / "result.json", result)
                raise
            memory = action.get("memory", "")
            history.append({"sequence": sequence, "action": action, "result": result})
            write_json(step_dir / "result.json", result)
            print(json.dumps({"step": sequence, "time_s": runtime.data.time,
                              "action_status": result["status"], "memory": memory,
                              "gates": {k: v["passed"] for k, v in evaluator.summary()["gates"].items()}}), flush=True)
            if policy is not None and result["status"] == "rejected":
                raise RuntimeError("Learned policy control rejected: " + result["reason"])
            if result["status"] == "finished":
                finished = True
                break
        if policy is not None and not finished:
            raise RuntimeError("Learned policy decision limit reached without an accepted finish")
    except (BackendError, RuntimeError, ValueError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf8")
    finally:
        runtime.close()
        sample_stream.close()
        if writer is not None:
            writer.release()
        summary = {**evaluator.summary(), "backend": backend, "seed": seed, "task_id": task,
                   "episode_id": episode_id,
                   "decisions": len(history), "elapsed_wall_s": time.monotonic() - started,
                   "error": error}
        if controller is not None:
            summary["controller"] = controller
        write_json(output / "actions.json", history)
        write_json(output / "summary.json", summary)
        if (output / "steps/000/request.json").is_file():
            write_json(output / "experience.json", export_experience(output))
        write_episode_report(output, summary)
        print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["demo", "agent"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", choices=["hammer_button", "wrench_turn", "tweezers_transfer"],
                        default="hammer_button")
    parser.add_argument("--max-decisions", type=int,
                        help="Decision limit (default: tweezers 48, other tasks 32)")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--experience", type=Path, action="append",
                        help="Explicit experience JSON record; repeat to supply records in order")
    parser.add_argument("--policy", type=Path,
                        help="Learned parameter policy directory (demo --task wrench_turn only)")
    parser.add_argument("--quality-guard", action="store_true",
                        help="Abort wrench episodes on the learned policy's simulation quality limits")
    parser.add_argument("--rich-feedback", action="store_true",
                        help="Include measured wrench geometry and named contact details")
    parser.add_argument("--wrench-variant", choices=["placement-30deg-v1", "placement-minus20deg-v2", "placement-plus15deg-v3",
                                                  "wrench-long-heavy-v1"],
                        help="Use a declared wrench variant with quality guard and rich feedback")
    parser.add_argument("--backend", choices=["native", "cli"], default="native",
                        help="Native waits for the Codex skill's file replies; cli starts codex exec")
    args = parser.parse_args()
    if args.wrench_variant is not None:
        if args.task != "wrench_turn":
            parser.error("--wrench-variant requires --task wrench_turn")
        if args.policy is not None:
            parser.error("--wrench-variant cannot be combined with --policy")
        if args.wrench_variant == "wrench-long-heavy-v1" and args.command != "agent":
            parser.error("wrench-long-heavy-v1 requires agent --backend native or cli")
    if args.policy is not None and (args.command != "demo" or args.task != "wrench_turn"):
        parser.error("--policy requires demo --task wrench_turn")
    if args.quality_guard and args.task != "wrench_turn":
        parser.error("--quality-guard requires --task wrench_turn")
    if args.rich_feedback and args.task != "wrench_turn":
        parser.error("--rich-feedback requires --task wrench_turn")
    summary = run_episode(args.output, backend="reference" if args.command == "demo" else args.backend,
                          seed=args.seed, max_decisions=args.max_decisions, video=not args.no_video,
                          task=args.task, policy=args.policy, quality_guard=args.quality_guard,
                          rich_feedback=args.rich_feedback, wrench_variant=args.wrench_variant,
                          experience=[load_experience(path) for path in args.experience] if args.experience else None)
    raise SystemExit(0 if summary["success"] and summary["error"] is None else 1)


if __name__ == "__main__":
    main()
