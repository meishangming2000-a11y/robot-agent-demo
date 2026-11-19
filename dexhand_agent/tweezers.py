"""Original passive spring forceps: a free backbone and one movable leaf.

The fixed left working tip belongs to the backbone; the right working tip
belongs to an unactuated spring hinge. Robot finger contact closes the leaf,
and the passive spring reopens it. This is a lumped elastic-tweezer model.
"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .runtime import HandRuntime
from .scene import make_model, scene_xml


TWEEZERS_SPEC = {
    "approximation": "rigid backbone/fixed lower tip (left label) plus spring upper tip (right label)",
    "tool_position_m": [0.53, -0.13, 0.242],
    "source_object_position_m": [0.34, 0.02, 0.230],
    "goal_position_m": [0.34, 0.16, 0.225],
    "object_half_size_m": 0.005,
    "object_mass_kg": 0.001,
    "backbone_mass_kg": 0.044,
    "leaf_mass_kg": 0.003,
    "leaf_stiffness_nm_rad": 0.030,
    "leaf_springref_rad": -0.10,
    "leaf_damping_nms": 0.0008,
    "leaf_range_rad": [0, 0.25],
    "left_tip_local_m": [-0.19, 0, -0.080],
    "right_tip_rest_local_m": [-0.19, 0, -0.035],
    "hinge_local_m": [-0.015, -0.035, -0.035],
    "hinge_axis": [0, -1, 0],
    "input_pad_local_m": [-0.040, 0, -0.052],
    "input_pad_half_size_m": [0.015, 0.020, 0.002],
    "left_tip_half_size_m": [0.002, 0.010, 0.002],
    "right_tip_half_size_m": [0.005, 0.010, 0.002],
    "tip_friction": [1.2, 0.005, 0.0001],
    "position_variation_m": 0.010,
    "support_rail_half_spacing_m": 0.0045,
    "support_rail_width_m": 0.001,
    "source_rail_y_bounds_relative_m": [-0.008, 0.024],
    "goal_rail_y_bounds_relative_m": [-0.024, 0.024],
    "support_rail_top_m": 0.225,
    "metal_contact_solref": [0.004, 1],
    "metal_contact_solimp": [0.99, 0.999, 0.0001],
    "reference_decision_budget": 48,
    "placement_mechanism": "two-tip transport; open at goal; fixed lower tip seats object onto split rails before withdrawal",
}


def _action(stage, duration_s, *, palm_target_m=None, palm_quat_wxyz=None, targets=None):
    return {"operation": "finish" if stage == "finish" else
            ("move" if palm_target_m is not None or targets else "hold"),
            "targets": [{"actuator": name, "value": value} for name, value in (targets or {}).items()],
            "palm_target_m": palm_target_m, "palm_quat_wxyz": palm_quat_wxyz,
            "duration_s": duration_s, "memory": f"Engineering tweezers reference: {stage}"}


def tweezers_scene_xml(seed=0):
    root = ET.fromstring(scene_xml(seed))
    root.set("model", "panda_shadow_passive_spring_tweezers")
    world = root.find("worldbody")
    for element in list(world):
        name = element.get("name", "")
        if name in {"hammer", "button", "button_base"} or name.startswith("rack_"):
            world.remove(element)
    # Fixed close views expose the actual tip/object contacts for visual review.
    for label, center in (("source", [0.34, 0.02, 0.25]), ("goal", [0.34, 0.16, 0.24])):
        camera_pos = np.array(center) + [0.14, 0.14, 0.11]
        ET.SubElement(world, "camera", name=f"tweezers_{label}_detail",
                      pos=" ".join(map(str, camera_pos)),
                      xyaxes="-0.707107 0.707107 0 -0.3435 -0.3435 0.8741", fovy="30")
    rng = np.random.default_rng(seed)
    offset = rng.uniform(-0.01, 0.01, 2) if seed else np.zeros(2)
    x, y = np.array(TWEEZERS_SPEC["tool_position_m"][:2]) + offset
    for dx in (-0.075, 0.055):
        ET.SubElement(world, "geom", name=f"tweezer_rack_{'left' if dx < 0 else 'right'}",
                      type="box", pos=f"{x+dx} {y} 0.181", size="0.009 0.015 0.051",
                      rgba="0.25 0.3 0.36 1")
    ET.SubElement(world, "geom", name="tweezer_rack_leaf", type="box",
                  pos=f"{x-0.13} {y-0.035} 0.1645", size="0.008 0.008 0.0245",
                  rgba="0.25 0.3 0.36 1")
    tool = ET.SubElement(world, "body", name="tweezers", pos=f"{x} {y} 0.242")
    ET.SubElement(tool, "freejoint", name="tweezers_free")
    ET.SubElement(tool, "geom", name="tweezer_backbone", type="capsule",
                  fromto="-0.085 0 0 0.080 0 0", size="0.010", mass="0.035",
                  friction="1.1 0.005 0.0005", condim="6", rgba="0.25 0.43 0.65 1")
    ET.SubElement(tool, "geom", name="tweezer_fixed_arm", type="capsule",
                  fromto="-0.085 -0.035 -0.050 -0.190 -0.035 -0.050", size="0.003", mass="0.005",
                  rgba="0.65 0.70 0.76 1")
    ET.SubElement(tool, "geom", name="tweezer_fixed_neck", type="capsule",
                  fromto="-0.19 -0.035 -0.050 -0.19 -0.010 -0.080", size="0.002", mass="0.001",
                  rgba="0.65 0.70 0.76 1")
    ET.SubElement(tool, "geom", name="tweezer_fixed_bridge", type="capsule",
                  fromto="-0.085 0 0 -0.085 -0.035 -0.050", size="0.002", mass="0.001",
                  rgba="0.65 0.70 0.76 1")
    ET.SubElement(tool, "geom", name="tweezer_hinge_bridge", type="capsule",
                  fromto="-0.015 0 0 -0.015 -0.035 -0.035", size="0.002", mass="0.001",
                  rgba="0.65 0.70 0.76 1")
    ET.SubElement(tool, "geom", name="tweezer_tip_left", type="box",
                  pos="-0.19 0 -0.080", size="0.002 0.010 0.002", mass="0.001",
                  friction="1.2 0.005 0.0001", rgba="0.25 0.8 0.8 1")
    leaf = ET.SubElement(tool, "body", name="tweezer_leaf", pos="-0.015 -0.035 -0.035")
    ET.SubElement(leaf, "joint", name="tweezer_leaf_hinge", type="hinge", axis="0 -1 0",
                  range="0 0.25", stiffness="0.030", springref="-0.10", damping="0.0008",
                  armature="0.000001", solreflimit="0.004 1", solimplimit="0.99 0.999 0.0001")
    ET.SubElement(leaf, "geom", name="tweezer_moving_arm", type="capsule",
                  fromto="0 0 0 -0.175 0 0", size="0.002", mass="0.0014",
                  rgba="0.75 0.79 0.85 1", friction="1.1 0.01 0.001")
    ET.SubElement(leaf, "geom", name="tweezer_moving_neck", type="capsule",
                  fromto="-0.175 0 0 -0.175 0.025 0", size="0.002", mass="0.0005",
                  rgba="0.75 0.79 0.85 1")
    ET.SubElement(leaf, "geom", name="tweezer_tip_right", type="box",
                  pos="-0.175 0.035 0", size="0.005 0.010 0.002", mass="0.0005",
                  friction="1.2 0.005 0.0001", rgba="0.9 0.65 0.2 1")
    for label, body, pos in (("left", tool, "-0.19 0 -0.080"),
                             ("right", leaf, "-0.175 0.035 0")):
        ET.SubElement(body, "site", name=f"tweezer_tip_{label}_center", pos=pos, size="0.002", group="4")
    ET.SubElement(leaf, "geom", name="tweezer_input_pad", type="box",
                  pos="-0.025 0.035 -0.017", size="0.015 0.020 0.002", mass="0.0005",
                  rgba="0.75 0.79 0.85 1", friction="1.1 0.01 0.001")
    ET.SubElement(leaf, "geom", name="tweezer_input_lever", type="capsule",
                  fromto="0 0 0 -0.025 0.035 -0.017", size="0.0015", mass="0.0001",
                  rgba="0.75 0.79 0.85 1")
    # Adjacent hinge bodies otherwise have MuJoCo parent-child collisions
    # suppressed. Their physically disjoint neck/tip geometry remains separate.
    objpos = np.array(TWEEZERS_SPEC["source_object_position_m"])
    objpos[:2] += offset
    for name, center in (("source_platform", objpos[:2]),
                          ("goal_platform", TWEEZERS_SPEC["goal_position_m"][:2])):
        for side, dx in (("", -0.0045), ("_right", 0.0045)):
            ET.SubElement(world, "geom", name=name+side, type="box", pos=f"{center[0]+dx} {center[1]+(0.008 if name == 'source_platform' else 0.0)} 0.223",
                          size=f"0.0005 {0.016 if name == 'source_platform' else 0.024} 0.002", friction="0.7 0.005 0.0001",
                          rgba="0.28 0.45 0.34 1" if name == "source_platform" else "0.3 0.45 0.65 1")
            ET.SubElement(world, "geom", name=name+side+"_post", type="box",
                          pos=f"{center[0]+dx} {center[1]+0.022} 0.1805", size="0.0005 0.002 0.0405",
                          rgba="0.28 0.45 0.34 1")
    obj = ET.SubElement(world, "body", name="small_object", pos=" ".join(map(str, objpos)))
    ET.SubElement(obj, "freejoint", name="object_free")
    ET.SubElement(obj, "geom", name="small_object_geom", type="box", size="0.005 0.005 0.005",
                  mass="0.001", friction="1.2 0.005 0.0001", solref="0.004 1",
                  solimp="0.99 0.999 0.0001", rgba="0.85 0.3 0.65 1")
    for label in ("left", "right"):
        ET.SubElement(root.find("contact"), "pair", geom1=f"tweezer_tip_{label}", geom2="small_object_geom",
                      solref="0.004 1", solimp="0.99 0.999 0.0001", friction="1.2 1.2 0.005 0.0001 0.0001")
    return ET.tostring(root, encoding="unicode")


def make_tweezers_model(seed=0):
    return make_model(seed, scene_factory=tweezers_scene_xml)


def _world_aabb(model, data, gid):
    local = model.geom_aabb[gid]
    rotation = data.geom_xmat[gid].reshape(3, 3)
    return data.geom_xpos[gid] + rotation @ local[:3], np.abs(rotation) @ local[3:]


def _gap(first, second):
    return float(np.linalg.norm(np.maximum(np.abs(first[0] - second[0]) - first[1] - second[1], 0)))


class TweezersRuntime(HandRuntime):
    def __init__(self, seed=0):
        super().__init__(seed, model_factory=make_tweezers_model, tool_body="tweezers")
        self.initial_object_height_m = float(self.data.body("small_object").xpos[2])

    def physical_sample(self):
        model, data = self.model, self.data
        tool_ids = {model.body(name).id for name in ("tweezers", "tweezer_leaf")}
        obj_id = model.body("small_object").id
        obj_geom = model.geom("small_object_geom").id
        tips = {model.geom(f"tweezer_tip_{label}").id: label for label in ("left", "right")}
        hand, support, object_tips, fixtures = set(), set(), set(), set()
        tool_object = other_tool = goal = robot_object = False
        maxforce = penetration = 0.0
        tip_normal = {"left": 0.0, "right": 0.0}
        object_force = object_penetration = 0.0
        for cid, contact in enumerate(data.contact):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, cid, force)
            maxforce = max(maxforce, float(force[0]))
            penetration = max(penetration, -float(contact.dist))
            geoms = [int(contact.geom1), int(contact.geom2)]
            bodies = [int(model.geom_bodyid[g]) for g in geoms]
            for index in (0, 1):
                other = 1 - index
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bodies[other]) or "world"
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geoms[other]) or "unnamed"
                if bodies[index] in tool_ids:
                    if bodies[other] == obj_id:
                        tool_object = True
                        object_force = max(object_force, float(force[0]))
                        object_penetration = max(object_penetration, -float(contact.dist))
                        if geoms[index] in tips:
                            tip_normal[tips[geoms[index]]] = max(tip_normal[tips[geoms[index]]], float(force[0]))
                        if geoms[index] in tips and force[0] > 0.005:
                            object_tips.add(tips[geoms[index]])
                        if geoms[index] not in tips:
                            other_tool = True
                    for prefix, label in (("rh_th", "thumb"), ("rh_ff", "index"), ("rh_mf", "middle"),
                                          ("rh_rf", "ring"), ("rh_lf", "little"), ("rh_palm", "palm")):
                        if name.startswith(prefix) and force[0] > 0.05:
                            hand.add(label)
                    if bodies[other] == 0:
                        support.add(geom_name)
                if bodies[index] == obj_id:
                    if name.startswith("rh_") or name.startswith("link"):
                        robot_object = True
                    if bodies[other] == 0:
                        if geom_name in {"goal_platform", "goal_platform_right"}:
                            goal = True
                        else:
                            fixtures.add(geom_name)
        obj_bound = _world_aabb(model, data, obj_geom)
        bounds = [_world_aabb(model, data, tip) for tip in tips]
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, obj_id, velocity, 0)
        return {"time_s": float(data.time), "tool_pos_m": data.body("tweezers").xpos.tolist(),
                "tool_quat_wxyz": data.body("tweezers").xquat.tolist(),
                "object_pos_m": data.body("small_object").xpos.tolist(),
                "hand_tool_contacts": sorted(hand), "tool_support_contacts": sorted(support),
                "tool_object_contact": tool_object, "object_tip_contacts": sorted(object_tips),
                "object_other_tool_contact": other_tool, "object_fixture_contacts": sorted(fixtures),
                "object_goal_support": goal, "robot_object_contact": robot_object,
                "object_to_goal_xy_m": float(np.linalg.norm(data.body("small_object").xpos[:2] -
                                                           TWEEZERS_SPEC["goal_position_m"][:2])),
                "object_bottom_clearance_m": float(obj_bound[0][2] - obj_bound[1][2] - TWEEZERS_SPEC["goal_position_m"][2]),
                "object_speed_m_s": float(np.linalg.norm(velocity[3:])),
                "tip_object_clearance_m": min(_gap(bound, obj_bound) for bound in bounds),
                "tip_gap_m": _gap(bounds[0], bounds[1]),
                "max_contact_normal_n_by_tip": tip_normal,
                "max_tool_object_contact_force_n": object_force,
                "max_tool_object_penetration_m": object_penetration,
                "leaf_angle_rad": float(data.qpos[model.joint("tweezer_leaf_hinge").qposadr[0]]),
                "leaf_constraint_torque_nm": float(data.qfrc_constraint[model.joint("tweezer_leaf_hinge").dofadr[0]]),
                "leaf_passive_torque_nm": float(data.qfrc_passive[model.joint("tweezer_leaf_hinge").dofadr[0]]),
                "max_contact_force_n": maxforce, "max_penetration_m": penetration}

    def observe(self):
        return {"mode": "simulator_state", "time_s": float(self.data.time),
                "physics_step": round(self.data.time / self.model.opt.timestep),
                "robot": self.robot_observation(),
                "tool": {"name": "tweezers", "position_m": self.data.body("tweezers").xpos.tolist(),
                         "quat_wxyz": self.data.body("tweezers").xquat.tolist(),
                         "left_tip_m": self.data.site("tweezer_tip_left_center").xpos.tolist(),
                         "right_tip_m": self.data.site("tweezer_tip_right_center").xpos.tolist(),
                         "initial_height_m": self.initial_height_m},
                "object": {"position_m": self.data.body("small_object").xpos.tolist(),
                           "initial_height_m": self.initial_object_height_m},
                "goal": {"position_m": TWEEZERS_SPEC["goal_position_m"]},
                "contacts": self.physical_sample(), "last_action": self.last_result}


class TweezersReference:
    """Observation-only engineering reference; no evaluator or object writes."""

    def __init__(self):
        self._phase = "approach"
        self._grasp = None
        self._step = 0
        self._pose_bias = np.array([0.0, -0.005, 0.018])
        self._last_tip_target = None
        self._press = 0.78

    def _level(self, obs, stage):
        return self._pose_tip(obs, obs["tool"]["left_tip_m"], stage)

    def _pose_tip(self, obs, tip_target, stage, duration=1.0, targets=None):
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, obs["tool"]["quat_wxyz"])
        change = rotation.reshape(3, 3).T
        palm = np.array(obs["robot"]["palm_position_m"])
        tip = np.array(obs["tool"]["left_tip_m"])
        if self._last_tip_target is not None:
            self._pose_bias = np.clip(self._pose_bias + 0.6 * (self._last_tip_target - tip), -0.04, 0.04)
        self._last_tip_target = np.array(tip_target)
        target = np.array(tip_target) + change @ (palm - tip) + self._pose_bias
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (change @ np.asarray(obs["robot"]["palm_axes_world"])).reshape(9))
        duration = min(3.0, max(duration, float(np.linalg.norm(target - palm)) / 0.16))
        return _action(stage, duration, palm_target_m=target.tolist(), palm_quat_wxyz=quat.tolist(), targets=targets)

    def decide(self, obs):
        phase = self._phase
        if phase == "approach":
            self._grasp = np.array(obs["tool"]["position_m"]) + [0, 0.025, 0.082]
            self._phase = "descend"
            return _action("tweezers_approach", 1.8, palm_target_m=(self._grasp + [0, 0, 0.09]).tolist(),
                           palm_quat_wxyz=[0, 0, 1, 0])
        if phase == "descend":
            self._phase = "close"
            return _action("tweezers_descend", 1.2, palm_target_m=self._grasp.tolist())
        if phase == "close":
            targets = {}
            for finger in ("FF", "MF", "RF", "LF"):
                targets[f"rh_A_{finger}J3"] = 0.8
                targets[f"rh_A_{finger}J0"] = 1.5
            targets.update({"rh_A_THJ5": 0.9763313232421871, "rh_A_THJ4": 0.9253804236340059,
                            "rh_A_THJ3": 0, "rh_A_THJ2": 0, "rh_A_THJ1": 0.45})
            self._phase = "lift"
            return _action("tweezers_close_backbone", 1.5, targets=targets)
        if phase == "lift":
            self._phase = "pickup_hold"
            return _action("tweezers_lift", 1.5, palm_target_m=(self._grasp + [0, 0, 0.14]).tolist())
        if phase == "pickup_hold":
            self._phase = "test_close"
            return _action("tweezers_pickup_hold", 2.2)
        if phase == "test_close":
            self._phase = "test_open"
            return _action("tweezers_test_close_leaf", 1.2, targets={"rh_A_FFJ3": 0.78, "rh_A_FFJ0": 0.0})
        if phase == "test_open":
            self._phase = "repeat_close"
            return _action("tweezers_test_reopen_leaf", 2.0, targets={"rh_A_FFJ3": 0.8, "rh_A_FFJ0": 1.5})
        if phase == "repeat_close":
            self._phase = "repeat_open"
            return _action("tweezers_repeat_close", 1.2, targets={"rh_A_FFJ3": 0.78, "rh_A_FFJ0": 0.0})
        if phase == "repeat_open":
            self._phase = "level"
            return _action("tweezers_repeat_open", 1.2, targets={"rh_A_FFJ3": 0.8, "rh_A_FFJ0": 1.5})
        if phase == "level":
            self._step += 1
            if self._step >= 3:
                self._phase = "align"
                self._step = 0
            return self._level(obs, f"tweezers_level_{self._step}")
        if phase in ("align", "descend_object", "squeeze", "object_lift", "transport", "place", "release", "withdraw", "clear_hold"):
            self._step += 1
            obj = np.array(obs["object"]["position_m"])
            target = obj + [0, 0, -0.0075]
            targets, duration = None, 1.0
            count, following = {"align": (3, "descend_object"), "descend_object": (3, "squeeze"),
                                "squeeze": (6, "object_lift"), "object_lift": (1, "transport"),
                                "transport": (2, "place"),
                                "place": (3, "release"), "release": (5, "withdraw"),
                                "withdraw": (4, "clear_hold"), "clear_hold": (1, "finish")}[phase]
            if phase == "align":
                target[1] -= 0.040
                duration = 1.6
            elif phase == "descend_object":
                target[1] -= 0.040 * (1 - self._step / count)
            elif phase == "squeeze":
                targets = {"rh_A_FFJ0": max(0.0, 1.5 - self._step * 0.25), "rh_A_FFJ3": 0.78}
                duration = 0.6
            elif phase == "object_lift":
                self._press = max(0.48, self._press - 0.10)
                targets = {"rh_A_FFJ0": 0.0, "rh_A_FFJ3": self._press}
                target[2] = obs["object"]["initial_height_m"] + 0.0425
                duration = 1.5
            else:
                self._press = max(0.48, self._press - 0.10)
                offset = obj - np.array(obs["tool"]["left_tip_m"])
                target = np.array(obs["goal"]["position_m"]) + [0, 0, 0.055] - offset
                targets = {"rh_A_FFJ0": 0.0, "rh_A_FFJ3": self._press}
                duration = 1.0 if self._step == 1 else 0.7
                if phase in ("place", "release", "withdraw", "clear_hold"):
                    target[2] = obs["goal"]["position_m"][2] + 0.005 - offset[2] + (0.025 if self._step == 1 else 0.010 if self._step == 2 else 0.0)
                    duration = 0.8
                if phase in ("release", "withdraw", "clear_hold"):
                    target[2] = obs["goal"]["position_m"][2] - 0.015
                    targets = {"rh_A_FFJ0": 1.2, "rh_A_FFJ3": 0.48}
                    if phase == "release":
                        target = np.array(obs["goal"]["position_m"]) + [0, 0, 0.007 if self._step <= 2 else -0.0035]
                        duration = 0.8
                if phase in ("withdraw", "clear_hold"):
                    target = np.array(obs["goal"]["position_m"]) + [0, 0, -0.015]
                    if phase == "clear_hold" or self._step >= 3:
                        target[1] -= 0.045
                    duration = 0.9
                if phase == "clear_hold":
                    target[2] += 0.050
                    duration = 2.0
            action = self._pose_tip(obs, target, f"tweezers_{phase}_{self._step}", duration, targets)
            if self._step >= count:
                self._phase, self._step = following, 0
            return action
        return _action("finish", 0.02)
