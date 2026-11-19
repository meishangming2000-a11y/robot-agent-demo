"""Independently check saved synchronized data and replay measured tick controls.

No controller, task policy or rendering is invoked. Acceptance describes saved
data consistency, not task success, training benefit or held-out performance.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


TIME_TOLERANCE = 1e-10
REPLAY_TOLERANCE = 1e-9
PINNED_MUJOCO = "3.5.0"


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _close(a, b, tolerance=TIME_TOLERANCE):
    return bool(np.all(np.abs(np.asarray(a) - np.asarray(b)) <= tolerance))


def _index(value, size):
    return type(value) is int and 0 <= value < size


def _check(directory, report):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"),
                          parse_constant=lambda value: _require(False, "non-finite JSON"))
    _require(manifest["format"] == "dexhand-synchronized-v1", "unknown recording format")
    _require(manifest["mujoco_version"] == mujoco.__version__ == PINNED_MUJOCO,
             "MuJoCo version must match the pinned compiled-model version")
    _require(manifest["arrays_file"] == "samples.npz", "unexpected arrays file")
    binary = manifest["provenance"]["model_binary"]
    _require(binary["file"] == "model.mjb", "unexpected model binary file")
    model_path = directory / "model.mjb"
    for path in (manifest_path, directory / "samples.npz"):
        report["input_sha256"][path.name] = _sha(path)
    model_bytes = model_path.read_bytes()
    report["input_sha256"]["model.mjb"] = hashlib.sha256(model_bytes).hexdigest()
    _require(len(model_bytes) == binary["bytes"]
             and report["input_sha256"]["model.mjb"] == binary["sha256"],
             "model binary bytes/SHA256 mismatch")
    # MuJoCo's Windows native file loader cannot open every Unicode path.
    # Its VFS accepts an ASCII key bound to the exact bytes verified above.
    model = mujoco.MjModel.from_binary_path("model.mjb", assets={"model.mjb": model_bytes})
    timestep = float(model.opt.timestep)
    _require(np.isfinite(timestep) and timestep > 0
             and _close(manifest["simulation_timestep_s"], timestep), "model timestep mismatch")
    report["provenance_source_sha256"] = manifest["provenance"].get("source_sha256", {})
    with np.load(directory / "samples.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state_size = mujoco.mj_stateSize(model, spec)
    frame_count = len(arrays["time_s"])
    tick_count = len(arrays["tick_time_s"])
    _require(frame_count > 0, "recording has no frames")
    camera = manifest["camera"]
    expected = {"time_s": (frame_count,), "qpos": (frame_count, model.nq),
                "qvel": (frame_count, model.nv), "ctrl": (frame_count, model.nu),
                "integration_state": (frame_count, state_size),
                "rgb": (frame_count, camera["height"], camera["width"], 3),
                "action_index": (frame_count,), "tick_time_s": (tick_count,),
                "tick_start_time_s": (tick_count,), "tick_ctrl": (tick_count, model.nu),
                "tick_action_index": (tick_count,)}
    _require(set(arrays) == set(expected) == set(manifest["arrays"]), "array names mismatch")
    for name, shape in expected.items():
        value = arrays[name]
        dtype = np.dtype("uint8" if name == "rgb" else
                         "int64" if name.endswith("action_index") else "float64")
        _require(value.dtype == dtype, f"{name} dtype mismatch")
        _require(value.shape == shape, f"{name} shape mismatch")
        _require(np.isfinite(value).all(), f"{name} must be finite")
        _require(manifest["arrays"][name]["shape"] == list(shape)
                 and manifest["arrays"][name]["dtype"] == str(dtype),
                 f"{name} manifest shape/dtype mismatch")
    _require(sum(a.nbytes for a in arrays.values()) == manifest["raw_array_bytes"],
             "raw array byte count mismatch")
    components, offset = [], 0
    for name, flag in mujoco.mjtState.__members__.items():
        value = int(flag)
        if name.startswith("mjSTATE_") and value and not value & (value - 1) and int(spec) & value:
            size = mujoco.mj_stateSize(model, flag)
            components.append({"name": name, "value": value, "offset": offset, "size": size})
            offset += size
    state_spec = manifest["state_spec"]
    _require(state_spec["value"] == int(spec) and state_spec["size"] == state_size
             and state_spec["components"] == components, "integration state specification mismatch")
    for component in components:
        field = {"mjSTATE_TIME": "time_s", "mjSTATE_QPOS": "qpos",
                 "mjSTATE_QVEL": "qvel", "mjSTATE_CTRL": "ctrl"}.get(component["name"])
        if field:
            start = component["offset"]
            stored = arrays["integration_state"][:, start:start + component["size"]]
            _require(_close(stored, arrays[field].reshape(frame_count, -1), 1e-12),
                     f"integration state slice mismatch: {field}")
    times, tick_times = arrays["time_s"], arrays["tick_time_s"]
    starts = arrays["tick_start_time_s"]
    _require(np.all(np.diff(times) > 0), "frame timestamps must increase")
    _require(np.all(np.diff(tick_times) > 0) and np.all(np.diff(starts) > 0),
             "tick timestamps must increase")
    _require(_close(times[0], manifest["start_time_s"]), "initial frame time mismatch")
    _require(_close(tick_times - starts, timestep), "tick widths do not match timestep")
    if tick_count:
        _require(_close(starts[0], times[0]) and _close(starts[1:], tick_times[:-1]),
                 "tick times are not continuous from the initial frame")
    frame_ticks = [-1]
    for frame in range(1, frame_count):
        tick = int(np.searchsorted(tick_times, times[frame] - TIME_TOLERANCE))
        _require(tick < tick_count and _close(tick_times[tick], times[frame]),
                 "frame time does not match a physical tick")
        _require(_close(arrays["ctrl"][frame], arrays["tick_ctrl"][tick]),
                 "frame control differs from incoming tick control")
        _require(arrays["action_index"][frame] == arrays["tick_action_index"][tick],
                 "frame action differs from incoming tick action")
        frame_ticks.append(tick)
    _require(arrays["action_index"][0] == -1, "initial frame must have no incoming action")
    status = manifest["status"]
    _require(status in ("completed", "failed"), "episode needs completed/failed status")
    final = manifest["final_snapshot"]
    if final["available"]:
        _require(final["sample_index"] == frame_count - 1, "final boundary index mismatch")
        _require(_close(times[-1], tick_times[-1] if tick_count else times[0]),
                 "available final boundary omits physical ticks")
    else:
        _require(status == "failed" and bool(final.get("error"))
                 and final["last_valid_sample_index"] == frame_count - 1
                 and _close(final["last_valid_time_s"], times[-1]),
                 "unavailable final boundary must be labelled failed with last valid state")
    _require(not manifest["truncated"] or status == "failed", "truncated episode must be failed")
    actions = manifest["actions"]
    expected_actions = np.full(tick_count, -1, dtype=np.int64)
    complete, incomplete = set(), set()
    previous_end = times[0]
    for index, action in enumerate(actions):
        _require(action["index"] == index and isinstance(action["action"], dict),
                 "action index/command mismatch")
        start, end = action["start_time_s"], action["end_time_s"]
        start_row, end_row = action["start_sample_index"], action["end_sample_index"]
        _require(_index(start_row, frame_count) and _close(times[start_row], start)
                 and start >= previous_end - TIME_TOLERANCE, "action start boundary mismatch")
        result = action.get("result")
        if end is None:
            _require(end_row is None and index == len(actions) - 1 and status == "failed",
                     "incomplete action must be the final action in a failed episode")
            incomplete.add(index)
            interval_end = tick_times[-1] if tick_count else start
        else:
            _require(_index(end_row, frame_count) and _close(times[end_row], end) and end >= start,
                     "action end boundary mismatch")
            _require(isinstance(result, dict)
                     and result.get("status") in ("completed", "finished", "rejected", "interrupted"),
                     "ended action needs explicit result status")
            _require(_close(result["time_s"], end), "action result time differs from boundary")
            result_status = result["status"]
            elapsed = end - start
            if result_status == "completed":
                duration = action["action"].get("duration_s")
                _require(isinstance(duration, (float, int)) and np.isfinite(duration) and duration > 0,
                         "completed action needs finite positive duration")
                _require(_close(elapsed, max(1, round(duration / timestep)) * timestep),
                         "completed action elapsed time differs from executed duration")
                complete.add(index)
            elif result_status in ("finished", "rejected"):
                _require(_close(elapsed, 0), "finish/rejection elapsed time must be zero")
                if result_status == "finished":
                    complete.add(index)
            else:
                _require(status == "failed", "interrupted action requires failed episode label")
                incomplete.add(index)
            interval_end = previous_end = end
        mask = (starts >= start - TIME_TOLERANCE) & (tick_times <= interval_end + TIME_TOLERANCE)
        _require(np.all(expected_actions[mask] == -1), "actions overlap physical tick intervals")
        expected_actions[mask] = index
    _require(np.array_equal(expected_actions, arrays["tick_action_index"]),
             "incoming tick action differs from outgoing action boundaries")
    _require(status == "failed" or np.all(expected_actions >= 0),
             "completed episode has undeclared tick actions")
    events = manifest["events"]
    for event in events:
        row = event["sample_index"]
        _require(_index(row, frame_count) and _close(times[row], event["time_s"]),
                 "event boundary time/index mismatch")
        if event["reason"] in ("action_start", "action_end"):
            index = event["action_index"]
            _require(_index(index, len(actions)), "event action index mismatch")
            key = "start_sample_index" if event["reason"] == "action_start" else "end_sample_index"
            _require(actions[index][key] == row, "event outgoing action boundary mismatch")
    for index, action in enumerate(actions):
        for reason in ("action_start", "action_end"):
            if reason == "action_end" and action["end_time_s"] is None:
                continue
            _require(sum(e["reason"] == reason and e["action_index"] == index for e in events) == 1,
                     "missing or duplicate action boundary event")
    _require(any(e["reason"] == "initial" and e["sample_index"] == 0 for e in events),
             "missing initial boundary label")
    _require(not final["available"] or any(e["reason"] == "final"
             and e["sample_index"] == frame_count - 1 for e in events), "missing final boundary label")
    eligible = []
    for frame in range(frame_count - 1):
        ids = arrays["tick_action_index"][frame_ticks[frame] + 1:frame_ticks[frame + 1] + 1]
        eligible.append(bool(len(ids) and ids[0] in complete and np.all(ids == ids[0])))
    report["counts"].update(frames=frame_count, ticks=tick_count, actions=len(actions),
                            complete_actions=len(complete), incomplete_actions=len(incomplete),
                            complete_frame_intervals=sum(eligible),
                            excluded_frame_intervals=len(eligible) - sum(eligible),
                            undeclared_ticks=int(np.sum(expected_actions < 0)))
    report["episode_labels"] = {"status": status, "truncated": manifest["truncated"],
                                "outcome": manifest["outcome"], "final_snapshot": final}
    # Bound work to five spaced frame intervals, including interior action states.
    intervals = np.unique(np.linspace(0, frame_count - 2, min(5, frame_count - 1), dtype=int))
    for frame in intervals:
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, arrays["integration_state"][frame], spec)
        mujoco.mj_forward(model, data)
        first, last = frame_ticks[frame] + 1, frame_ticks[frame + 1] + 1
        for tick in range(first, last):
            data.ctrl[:] = arrays["tick_ctrl"][tick]
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
        deviations = {}
        for name in ("qpos", "qvel", "ctrl"):
            deviations[name] = float(np.max(np.abs(getattr(data, name) - arrays[name][frame + 1]), initial=0))
            report["max_deviations"][name] = max(report["max_deviations"][name], deviations[name])
        report["replayed_intervals"].append({"from_frame": int(frame), "to_frame": int(frame + 1),
                                             "ticks": last - first, "max_deviations": deviations})
        report["counts"]["replayed_intervals"] += 1
        _require(_close(data.time, times[frame + 1])
                 and all(np.isfinite(v) and v <= REPLAY_TOLERANCE for v in deviations.values()),
                 f"physics replay mismatch at frame interval {frame}->{frame + 1}")


def validate_episode(episode):
    """Return a JSON-safe report; invalid data is accepted=False with a reason."""
    report = {"validator_version": 2, "episode": str(Path(episode).resolve()),
              "accepted": False, "errors": [], "input_sha256": {},
              "source_sha256": {"tools/check_learning_data.py": _sha(Path(__file__))},
              "tolerances": {"time_s": TIME_TOLERANCE, "replay_absolute": REPLAY_TOLERANCE},
              "counts": {"replayed_intervals": 0}, "replayed_intervals": [],
              "max_deviations": {"qpos": 0., "qvel": 0., "ctrl": 0.},
              "claim": "Saved data integrity/alignment and sampled physics continuation only."}
    try:
        _check(Path(episode), report)
    except (KeyError, ValueError, TypeError, OSError, IndexError) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["accepted"] = not report["errors"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    report = validate_episode(args.episode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"accepted": report["accepted"], "counts": report["counts"],
                      "errors": report["errors"]}))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
