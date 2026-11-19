"""Named robot controls, local dynamics, and declared state observations."""

import math

import mujoco
import numpy as np

from .scene import make_model


ACTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "operation": {"type": "string", "enum": ["move", "hold", "finish"]},
        "targets": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                    "properties": {"actuator": {"type": "string"}, "value": {"type": "number"}},
                    "required": ["actuator", "value"]}},
        "palm_target_m": {"anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}, {"type": "null"}]},
        "palm_quat_wxyz": {"anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}, {"type": "null"}]},
        "duration_s": {"type": "number", "minimum": 0.02, "maximum": 3.0},
        "memory": {"type": "string"},
    }, "required": ["operation", "targets", "palm_target_m", "palm_quat_wxyz", "duration_s", "memory"],
}


class InvalidAction(ValueError):
    pass


def _quaternion(rotation):
    result = np.zeros(4)
    mujoco.mju_mat2Quat(result, np.asarray(rotation).reshape(9))
    return result


class HandRuntime:
    def __init__(self, seed=0, model_factory=None, tool_body="hammer"):
        self.model, self.data = (model_factory or make_model)(seed)
        self.tool_body = tool_body
        self.arm_qadr = np.array([self.model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.arm_dadr = np.array([self.model.joint(f"joint{i}").dofadr[0] for i in range(1, 8)])
        self.arm_aids = np.array([self.model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self.actuator_ids = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
                             for i in range(self.model.nu)}
        for _ in range(500):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.initial_height_m = float(self.data.body(tool_body).xpos[2])
        self.samples = []
        self.on_sample = None
        self._renderer = None
        self.last_result = {"status": "ready"}

    def _ik(self, position, quaternion, controls):
        model, actual = self.model, self.data
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = actual.qpos
        for name in ("rh_A_WRJ1", "rh_A_WRJ2"):
            aid = self.actuator_ids[name]
            jid = model.actuator_trnid[aid, 0]
            scratch.qpos[model.jnt_qposadr[jid]] = controls[aid]
        sid = model.site("palm_frame").id
        target_rotation = np.zeros(9)
        mujoco.mju_quat2Mat(target_rotation, quaternion)
        target_rotation = target_rotation.reshape(3, 3)
        jac_pos, jac_rot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        lower = np.array([model.joint(f"joint{i}").range[0] for i in range(1, 8)])
        upper = np.array([model.joint(f"joint{i}").range[1] for i in range(1, 8)])
        for _ in range(160):
            mujoco.mj_forward(model, scratch)
            pos_error = position - scratch.site_xpos[sid]
            current_rotation = scratch.site_xmat[sid].reshape(3, 3)
            difference = _quaternion(target_rotation @ current_rotation.T)
            if difference[0] < 0:
                difference *= -1
            rot_error = np.zeros(3)
            mujoco.mju_quat2Vel(rot_error, difference, 1.0)
            if np.linalg.norm(pos_error) < 0.0004 and np.linalg.norm(rot_error) < 0.005:
                return scratch.qpos[self.arm_qadr].copy()
            mujoco.mj_jacSite(model, scratch, jac_pos, jac_rot, sid)
            jac = np.vstack([jac_pos[:, self.arm_dadr], 0.25 * jac_rot[:, self.arm_dadr]])
            error = np.r_[pos_error, 0.25 * rot_error]
            change = jac.T @ np.linalg.solve(jac @ jac.T + 0.002**2 * np.eye(6), error)
            scratch.qpos[self.arm_qadr] = np.clip(scratch.qpos[self.arm_qadr] + np.clip(change, -0.12, 0.12), lower, upper)
        raise InvalidAction("Palm target is not reachable at the requested orientation")

    def _validate(self, action):
        if not isinstance(action, dict) or set(action) - set(ACTION_SCHEMA["properties"]):
            raise InvalidAction("Action must be an object with documented fields")
        operation = action.get("operation")
        if operation not in ("move", "hold", "finish"):
            raise InvalidAction("Unknown operation")
        duration = action.get("duration_s", 0.2)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0.02 <= duration <= 3:
            raise InvalidAction("duration_s must be finite and between 0.02 and 3")
        entries = action.get("targets", [])
        if not isinstance(entries, list):
            raise InvalidAction("targets must be a list of actuator/value objects")
        targets = {}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"actuator", "value"} or not isinstance(entry["actuator"], str):
                raise InvalidAction("Each target must contain actuator and value")
            if entry["actuator"] in targets:
                raise InvalidAction("Duplicate actuator target")
            targets[entry["actuator"]] = entry["value"]
        controls = self.data.ctrl.copy()
        for name, value in targets.items():
            if name not in self.actuator_ids:
                raise InvalidAction(f"Unknown actuator: {name}")
            aid = self.actuator_ids[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise InvalidAction(f"Invalid actuator value: {name}")
            low, high = self.model.actuator_ctrlrange[aid]
            if not low <= value <= high:
                raise InvalidAction(f"{name} must be within [{low}, {high}]")
            controls[aid] = value
        position, quaternion = action.get("palm_target_m"), action.get("palm_quat_wxyz")
        if operation != "move" and (targets or position is not None or quaternion is not None):
            raise InvalidAction("Only move accepts actuator or palm targets")
        if position is not None or quaternion is not None:
            if any(name in targets for name in (f"actuator{i}" for i in range(1, 8))):
                raise InvalidAction("Choose arm actuator targets or a palm pose per action")
            try:
                position = np.asarray(position if position is not None else self.data.site("palm_frame").xpos, dtype=float)
                quaternion = np.asarray(quaternion if quaternion is not None else _quaternion(self.data.site("palm_frame").xmat), dtype=float)
            except (ValueError, TypeError) as exc:
                raise InvalidAction("Palm position and wxyz quaternion must be finite vectors") from exc
            if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(quaternion).all():
                raise InvalidAction("Palm position and wxyz quaternion must be finite vectors")
            norm = np.linalg.norm(quaternion)
            if abs(norm - 1) > 0.02:
                raise InvalidAction("Palm quaternion must have unit length")
            quaternion = quaternion / norm
            if np.linalg.norm(position - self.data.site("palm_frame").xpos) > 0.18 * duration + 0.01:
                raise InvalidAction("Palm translation exceeds 0.18 m/s for this duration")
            controls[self.arm_aids] = self._ik(position, quaternion, controls)
        return operation, float(duration), controls

    def execute(self, action):
        operation, duration, controls = self._validate(action)
        if operation == "finish":
            self.last_result = {"status": "finished", "time_s": self.data.time}
            return self.last_result
        before = self.data.ctrl.copy()
        steps = max(1, round(duration / self.model.opt.timestep))
        for step in range(steps):
            fraction = (step + 1) / steps
            fraction = fraction * fraction * (3 - 2 * fraction)
            self.data.ctrl[:] = before + fraction * (controls - before)
            mujoco.mj_step(self.model, self.data)
            # mj_step integrates qpos after its forward pass. Refresh geometry,
            # contacts, and force estimates so every exported field is current.
            mujoco.mj_forward(self.model, self.data)
            if not all(np.isfinite(x).all() for x in (self.data.qpos, self.data.qvel)):
                raise RuntimeError("Non-finite physical state")
            sample = self.physical_sample()
            self.samples.append(sample)
            if self.on_sample is not None:
                self.on_sample(sample)
        self.last_result = {"status": "completed", "time_s": self.data.time}
        return self.last_result

    def physical_sample(self):
        model, data = self.model, self.data
        tool = model.body("hammer").id
        cap = model.geom("button_cap").id
        head = model.geom("hammer_head").id
        hand_contacts, support_contacts = set(), set()
        head_button, tool_button, robot_button = False, False, False
        maximum_force, penetration = 0.0, 0.0
        for contact_id, contact in enumerate(data.contact):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_id, force)
            if force[0] <= 0.05:
                continue
            maximum_force = max(maximum_force, float(force[0]))
            penetration = max(penetration, -float(contact.dist))
            geoms = [int(contact.geom1), int(contact.geom2)]
            bodies = [int(model.geom_bodyid[g]) for g in geoms]
            for index in (0, 1):
                other = 1 - index
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bodies[other]) or "world"
                if bodies[index] == tool:
                    if geoms[other] == cap:
                        tool_button = True
                    for prefix, label in (("rh_th", "thumb"), ("rh_ff", "index"), ("rh_mf", "middle"),
                                          ("rh_rf", "ring"), ("rh_lf", "little"), ("rh_palm", "palm")):
                        if name.startswith(prefix):
                            hand_contacts.add(label)
                    if bodies[other] == 0:
                        support_contacts.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geoms[other]))
                if geoms[index] == cap and (name.startswith("rh_") or name.startswith("link")):
                    robot_button = True
            if set(geoms) == {head, cap}:
                head_button = True
        head_center = data.geom("hammer_head").xpos
        head_radius_z = abs(data.geom("hammer_head").xmat.reshape(3, 3)[2]) @ model.geom_size[head]
        return {"time_s": float(data.time), "hammer_pos_m": data.body("hammer").xpos.tolist(),
                "hammer_quat_wxyz": data.body("hammer").xquat.tolist(),
                "hand_tool_contacts": sorted(hand_contacts), "tool_support_contacts": sorted(support_contacts),
                "head_button_contact": head_button, "tool_button_contact": tool_button,
                "robot_button_contact": robot_button,
                "button_q_m": float(data.qpos[model.joint("button_slide").qposadr[0]]),
                "head_clearance_m": float(head_center[2] - head_radius_z - 0.24),
                "max_contact_force_n": maximum_force, "max_penetration_m": penetration}

    def robot_observation(self):
        joints = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name.startswith("joint") or name.startswith("rh_"):
                joints[name] = float(self.data.qpos[self.model.jnt_qposadr[i]])
        actuators = {}
        for name, aid in self.actuator_ids.items():
            actuators[name] = {"measured": float(self.data.actuator_length[aid]),
                              "velocity": float(self.data.actuator_velocity[aid]),
                              "commanded": float(self.data.ctrl[aid]),
                              "range": self.model.actuator_ctrlrange[aid].tolist(),
                              "transmission": "tendon" if self.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_TENDON else "joint"}
        return {"joints_rad": joints, "actuators": actuators,
                "palm_position_m": self.data.site("palm_frame").xpos.tolist(),
                "palm_quat_wxyz": _quaternion(self.data.site("palm_frame").xmat).tolist(),
                "palm_axes_world": self.data.site("palm_frame").xmat.reshape(3, 3).tolist()}

    def observe(self):
        return {"mode": "simulator_state", "time_s": float(self.data.time),
                "physics_step": round(self.data.time / self.model.opt.timestep),
                "robot": self.robot_observation(),
                "tool": {"name": "hammer", "position_m": self.data.body("hammer").xpos.tolist(),
                         "quat_wxyz": self.data.body("hammer").xquat.tolist(),
                         "head_center_m": self.data.geom("hammer_head").xpos.tolist(),
                         "initial_height_m": self.initial_height_m},
                "button": {"position_m": self.data.body("button").xpos.tolist(), "rest_top_m": 0.24},
                "contacts": self.physical_sample(), "last_action": self.last_result}

    def render(self, camera="overview", width=960, height=720):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
