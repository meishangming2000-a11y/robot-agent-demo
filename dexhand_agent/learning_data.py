"""Synchronous RGB, measured state, and executed-control recording for MuJoCo."""

import json
from pathlib import Path

import mujoco
import numpy as np


class SynchronizedRecorder:
    """Record an existing runtime without changing its state or controls.

    The runtime supplies ``model``, ``data``, and an ``on_sample`` callback invoked
    after every physics tick and its mj_forward. Use one thread, a fixed model,
    and one recorder per runtime. Declare commands BEFORE executing them::

        rec = SynchronizedRecorder(runtime, new_directory, provenance={...})
        try:
            rec.begin_action(original_action)
            result = runtime.execute(original_action)
            rec.end_action(result)
            rec.finalize(status="completed", outcome=independent_evaluation)
        except Exception as exc:
            rec.finalize(status="failed", outcome={"error": str(exc)})
            raise
        finally:
            rec.close()

    ``snapshot(reason)`` forces a frame; begin/end/finalize do this automatically.
    Equal-time boundaries share one row. Row action_index identifies the command
    producing the incoming physical tick (-1 before any tick or if undeclared).
    Actions/events identify the outgoing command and its exact boundary row.
    tick_ctrl is the control actually present after each tick, interpreted over
    [tick_start_time_s, tick_time_s]; this assumes controls are held during each
    tick, as in HandRuntime. Targets in original commands are NOT executed chunks.

    RGB is the visual student input. Full qpos/qvel/integration_state include
    object truth and are privileged teacher data, never an implied student input.
    Outcome and provenance are supplied by the caller, never inferred here.

    The default raw-data budget is 96 MiB, including reserved final-state space.
    Exhaustion captures the current tick/final frame and raises BufferError; the
    caller can finalize that truncated recording with failure status. Finalizing
    with status="failed" preserves previous valid rows if a final snapshot is
    unavailable, and records that snapshot's error without inventing a row.
    Other statuses require a valid final snapshot. Finalizing
    detaches the callback and closes only this recorder's renderer. close() alone
    discards unsaved data and never closes the runtime. Output must not exist.
    """

    def __init__(self, runtime, output_dir, *, sample_period_s=.1, width=160,
                 height=120, camera="overview", provenance=None,
                 max_bytes=96 * 1024**2):
        if not np.isfinite(sample_period_s) or sample_period_s <= 0:
            raise ValueError("sample_period_s must be positive and finite")
        if (not isinstance(width, int) or not isinstance(height, int)
                or width <= 0 or height <= 0):
            raise ValueError("image dimensions must be positive integers")
        if not isinstance(max_bytes, int) or not 0 < max_bytes <= 96 * 1024**2:
            raise ValueError("max_bytes must be positive and at most 96 MiB")
        self.runtime = runtime
        self.output_dir = Path(output_dir)
        self.period = float(sample_period_s)
        self.width, self.height, self.camera = width, height, camera
        self.max_bytes = max_bytes
        self._spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._state_size = mujoco.mj_stateSize(runtime.model, self._spec)
        self._rows, self._ticks = [], []
        self._actions, self._events = [], []
        self._active = None
        self._last_tick_action = -1
        self._last_tick_time = float(runtime.data.time)
        self._origin = self._last_tick_time
        self._next_sample_time = self._origin + self.period
        self._closed = False
        self._truncated = False
        self._previous_callback = runtime.on_sample
        self._callback = self._on_sample
        self._base = self._metadata(self._json_copy(provenance or {}))
        self._used_bytes = len(json.dumps(self._base).encode("utf-8")) + 8192
        self._row_bytes = (16 + runtime.data.qpos.nbytes + runtime.data.qvel.nbytes
                           + runtime.data.ctrl.nbytes + self._state_size * 8
                           + width * height * 3)
        self._tick_bytes = 24 + runtime.data.ctrl.nbytes
        self._reserve = self._row_bytes + self._tick_bytes + 1024
        if self._used_bytes + self._row_bytes + self._reserve > max_bytes:
            raise ValueError("budget too small for metadata, initial and final frames")
        # mkdir is exclusive even for an empty directory: no existing data touched.
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._renderer = None
        try:
            self._renderer = mujoco.Renderer(runtime.model, height=height, width=width)
            self.snapshot("initial")
            runtime.on_sample = self._callback
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _json_copy(value):
        return json.loads(json.dumps(value, allow_nan=False))

    def _metadata(self, provenance):
        model = self.runtime.model
        joints = []
        for i in range(model.njnt):
            kind = int(model.jnt_type[i])
            units = {
                int(mujoco.mjtJoint.mjJNT_FREE): (["m"] * 3 + ["unitless_wxyz"] * 4,
                                                ["m/s"] * 3 + ["rad/s"] * 3),
                int(mujoco.mjtJoint.mjJNT_BALL): (["unitless_wxyz"] * 4, ["rad/s"] * 3),
                int(mujoco.mjtJoint.mjJNT_SLIDE): (["m"], ["m/s"]),
                int(mujoco.mjtJoint.mjJNT_HINGE): (["rad"], ["rad/s"]),
            }[kind]
            joints.append({"id": i, "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
                           "type": kind, "qpos_adr": int(model.jnt_qposadr[i]),
                           "qvel_adr": int(model.jnt_dofadr[i]),
                           "qpos_units": units[0], "qvel_units": units[1]})
        actuators = [{"id": i, "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
                      "ctrl_units": "model-defined actuator input",
                      "ctrlrange": model.actuator_ctrlrange[i].tolist(),
                      "transmission_type": int(model.actuator_trntype[i]),
                      "transmission_ids": model.actuator_trnid[i].tolist(),
                      "gain_type": int(model.actuator_gaintype[i]),
                      "bias_type": int(model.actuator_biastype[i]),
                      "dynamics_type": int(model.actuator_dyntype[i])}
                     for i in range(model.nu)]
        components = []
        offset = 0
        for name, flag in mujoco.mjtState.__members__.items():
            value = int(flag)
            if not name.startswith("mjSTATE_") or value & (value - 1):
                continue
            if int(self._spec) & value:
                size = mujoco.mj_stateSize(model, flag)
                components.append({"name": name, "value": value, "offset": offset,
                                   "size": size})
                offset += size
        return {"format": "dexhand-synchronized-v1", "arrays_file": "samples.npz",
                "mujoco_version": mujoco.__version__, "provenance": provenance,
                "sample_period_s": self.period, "start_time_s": self._origin,
                "simulation_timestep_s": float(model.opt.timestep),
                "camera": {"name": self.camera, "width": self.width, "height": self.height,
                           "channels": "RGB", "pixel_range": [0, 255]},
                "state_spec": {"name": "mjSTATE_INTEGRATION", "value": int(self._spec),
                               "size": self._state_size, "components": components,
                               "restore": "mj_setState(same_model, data, row, spec); mj_forward(model, data)"},
                "joints": joints, "actuators": actuators,
                "semantics": {
                    "frames": "All row fields read synchronously after the same mj_forward; equal-time boundaries share a row.",
                    "action_index": "Command producing incoming tick; -1 initial or undeclared. Use action start_sample_index for outgoing commands.",
                    "tick_ctrl": "Actual ctrl over [tick_start_time_s,tick_time_s], assuming controls held within a tick; command targets are not exact executed chunks.",
                    "teacher": "Full simulator qpos, qvel and integration_state include object truth and are privileged teacher data.",
                    "student": "Only rgb is designated visual student input; no policy input or success is inferred.",
                    "restore_limit": "Requires identical model/assets/options; does not restore caller logic, callbacks or external random generators."}}

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("recorder is closed")

    def _charge(self, size, *, reserve=True):
        if self._used_bytes + size + (self._reserve if reserve else 0) > self.max_bytes:
            self._truncated = True
            raise BufferError("recorder budget exhausted")
        self._used_bytes += size

    def _event(self, reason, sample_index, action_index=None):
        event = {"reason": reason, "time_s": float(self.runtime.data.time),
                 "sample_index": sample_index,
                 "action_index": self._active if action_index is None else action_index}
        self._charge(len(json.dumps(event).encode("utf-8")), reserve=False)
        self._events.append(event)

    def _capture(self, *, emergency=False):
        data, model = self.runtime.data, self.runtime.model
        state = np.empty(self._state_size, dtype=np.float64)
        mujoco.mj_getState(model, data, state, self._spec)
        if not np.isfinite(state).all() or not np.isfinite(data.time):
            raise ValueError("non-finite simulator state")
        time_s = float(data.time)
        if self._rows and time_s < self._rows[-1][0]:
            raise ValueError("simulation time must be monotonic")
        if self._rows and time_s == self._rows[-1][0]:
            if not np.array_equal(state, self._rows[-1][4]):
                raise ValueError("state changed without increasing simulation time")
            return len(self._rows) - 1
        self._charge(self._row_bytes, reserve=not emergency)
        try:
            self._renderer.update_scene(data, camera=self.camera)
            rgb = self._renderer.render()
            if rgb.dtype != np.uint8 or rgb.shape != (self.height, self.width, 3):
                raise ValueError("renderer must return uint8 HxWx3 RGB")
        except BaseException:
            self._used_bytes -= self._row_bytes
            raise
        self._rows.append((time_s, data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(),
                           state, rgb.copy(), self._last_tick_action))
        return len(self._rows) - 1

    def snapshot(self, reason="manual"):
        """Force a synchronized row (or reuse an identical equal-time row)."""
        self._ensure_open()
        index = self._capture(emergency=self._truncated)
        self._event(str(reason), index)
        return index

    def begin_action(self, action):
        """Copy an original structured command and its boundary BEFORE execute."""
        self._ensure_open()
        if self._active is not None:
            raise ValueError("end the previous action before beginning another")
        action = self._json_copy(action)
        if not isinstance(action, dict):
            raise ValueError("action must be a JSON object")
        boundary = self._capture()
        index = len(self._actions)
        entry = {"index": index, "action": action, "start_time_s": float(self.runtime.data.time),
                 "start_sample_index": boundary, "end_time_s": None, "end_sample_index": None}
        self._charge(len(json.dumps(entry).encode("utf-8")))
        self._actions.append(entry)
        self._active = index
        self._event("action_start", boundary)
        return index

    def end_action(self, result=None):
        """Record the actual end boundary; result is caller-supplied runtime output."""
        self._ensure_open()
        if self._active is None:
            raise ValueError("no action is active")
        result = self._json_copy(result)
        index = self._capture(emergency=self._truncated)
        self._charge(len(json.dumps(result).encode("utf-8")), reserve=False)
        self._actions[self._active].update(end_time_s=float(self.runtime.data.time),
                                           end_sample_index=index, result=result)
        self._event("action_end", index)
        self._active = None
        return index

    def _on_sample(self, sample):
        try:
            time_s = float(self.runtime.data.time)
            if not np.isfinite(time_s) or time_s <= self._last_tick_time:
                raise ValueError("physical tick time must increase and be finite")
            if not np.isfinite(self.runtime.data.ctrl).all():
                raise ValueError("non-finite controls")
            self._last_tick_action = self._active if self._active is not None else -1
            tick = (time_s, self._last_tick_time, self.runtime.data.ctrl.copy(),
                    self._last_tick_action)
            try:
                self._charge(self._tick_bytes)
            except BufferError:
                self._truncated = True
                self._charge(self._tick_bytes, reserve=False)
            self._ticks.append(tick)
            self._last_tick_time = time_s
            if not self._truncated and time_s >= self._next_sample_time - 1e-12:
                try:
                    self._capture()
                except BufferError:
                    self._truncated = True
                while self._next_sample_time <= time_s + 1e-12:
                    self._next_sample_time += self.period
            if self._truncated:
                self._capture(emergency=True)
                raise BufferError("recorder budget exhausted; finalize this truncated recording")
        finally:
            if self._previous_callback is not None:
                self._previous_callback(sample)

    def _arrays(self):
        names = ("time_s", "qpos", "qvel", "ctrl", "integration_state", "rgb", "action_index")
        arrays = {name: np.asarray([row[i] for row in self._rows],
                                  dtype=np.int64 if name == "action_index" else
                                  np.uint8 if name == "rgb" else np.float64)
                  for i, name in enumerate(names)}
        for i, name in enumerate(("tick_time_s", "tick_start_time_s", "tick_ctrl", "tick_action_index")):
            arrays[name] = np.asarray([row[i] for row in self._ticks],
                                      dtype=np.int64 if name == "tick_action_index" else np.float64)
        arrays["tick_ctrl"] = arrays["tick_ctrl"].reshape(len(self._ticks), self.runtime.model.nu)
        return arrays

    def finalize(self, *, status, outcome):
        """Write compressed NPZ plus JSON manifest, preserving external outcome."""
        self._ensure_open()
        outcome = self._json_copy(outcome)
        try:
            final_snapshot = {"available": True, "sample_index": self.snapshot("final")}
        except Exception as exc:
            if status != "failed":
                raise
            attempted_time = float(self.runtime.data.time)
            final_snapshot = {"available": False, "error": f"{type(exc).__name__}: {exc}",
                              "attempted_time_s": attempted_time if np.isfinite(attempted_time) else None,
                              "last_valid_sample_index": len(self._rows) - 1,
                              "last_valid_time_s": self._rows[-1][0]}
        arrays = self._arrays()
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("non-finite output array")
        for name in ("time_s", "tick_time_s", "tick_start_time_s"):
            if np.any(np.diff(arrays[name]) <= 0):
                raise ValueError("timestamps must strictly increase")
        metadata = dict(self._base, status=str(status), outcome=outcome,
                        truncated=self._truncated, actions=self._actions, events=self._events,
                        final_snapshot=final_snapshot,
                        raw_array_bytes=sum(value.nbytes for value in arrays.values()))
        metadata["arrays"] = {}
        for name, value in arrays.items():
            role = ("visual_student_input" if name == "rgb" else
                    "privileged_teacher" if name in ("qpos", "qvel", "integration_state") else
                    "executed_control" if name in ("ctrl", "tick_ctrl") else "alignment")
            units = ("s" if name.endswith("time_s") else "uint8 RGB intensity" if name == "rgb" else
                     "joint-specific; see joints" if name in ("qpos", "qvel") else
                     "model-defined actuator input" if name in ("ctrl", "tick_ctrl") else
                     "component-specific MuJoCo SI state; see state_spec" if name == "integration_state" else
                     "zero-based action index; -1 undeclared")
            metadata["arrays"][name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                                         "role": role, "units": units}
        manifest_text = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        # NPY/ZIP framing and JSON are also bounded before creating output files.
        if (metadata["raw_array_bytes"] + len(manifest_text.encode("utf-8")) + 4096
                > self.max_bytes):
            raise BufferError("serialized recording exceeds budget")
        with (self.output_dir / "samples.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        manifest_path = self.output_dir / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            stream.write(manifest_text)
        self.close()
        return manifest_path

    def close(self):
        """Restore the saved callback if still attached; never close the runtime."""
        if self._closed:
            return
        if self.runtime.on_sample is self._callback:
            self.runtime.on_sample = self._previous_callback
        if self._renderer is not None:
            self._renderer.close()
        self._closed = True
