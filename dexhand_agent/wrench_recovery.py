"""Bounded observation-only wrench withdrawal after a confirmed tool turn.

This engineering skill was demonstrated after a recorded seed-5 prefix. It is
not a general grasp/turn policy and never certifies task success.
"""
import copy
import hashlib
import json
import math

import mujoco
import numpy as np

from .wrench_tools import _array, _rotation, feedback_tool_pose_action


def _matrix(quaternion):
    result = np.zeros(9)
    mujoco.mju_quat2Mat(result, np.asarray(quaternion, dtype=float))
    return result.reshape(3, 3)


def _quaternion(matrix):
    result = np.zeros(4)
    mujoco.mju_mat2Quat(result, np.asarray(matrix).reshape(9))
    return result.tolist()


def _bounded(vector, maximum):
    vector = np.asarray(vector)
    size = np.linalg.norm(vector)
    return vector if size <= maximum else vector * maximum / size


def _rotation_step(matrix, maximum):
    quaternion = np.asarray(_quaternion(matrix))
    if quaternion[0] < 0:
        quaternion = -quaternion
    angle = 2 * math.atan2(np.linalg.norm(quaternion[1:]), quaternion[0])
    if angle <= maximum:
        return matrix
    axis = quaternion[1:] / np.linalg.norm(quaternion[1:])
    return _matrix(np.r_[math.cos(maximum / 2), axis * math.sin(maximum / 2)])


def _align_axis(matrix, axis, maximum):
    current = matrix[:, 2]
    cross = np.cross(current, axis)
    size = np.linalg.norm(cross)
    if size < 1e-12:
        return matrix.copy()
    angle = min(math.atan2(size, np.clip(current @ axis, -1, 1)), maximum)
    return _matrix(np.r_[math.cos(angle / 2), cross / size * math.sin(angle / 2)]) @ matrix


class WrenchWithdrawPolicy:
    """Issue small robot actions using fresh public observations only.

    The caller supplies the last complete accepted palm move and must return the
    observation after each issued action. Rejection/interruption hands control
    back without adopting that command. Contact/grasp variation can invalidate
    this local recovery method; the caller owns the independent final gate.
    """

    VERSION = "wrench_withdraw_v1"

    def __init__(self, initial_observation, previous_palm_action, max_steps=80):
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not isinstance(previous_palm_action, dict) or previous_palm_action.get("operation") != "move":
            raise ValueError("previous_palm_action must be a complete accepted palm move")
        _array(previous_palm_action.get("palm_target_m"), (3,), "previous palm target")
        _rotation(previous_palm_action.get("palm_quat_wxyz"), "previous palm quaternion")
        self._initial_sha256 = hashlib.sha256(json.dumps(
            initial_observation, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf8")).hexdigest()
        self._initial_previous = copy.deepcopy(previous_palm_action)
        self._previous = copy.deepcopy(previous_palm_action)
        error = np.linalg.norm(np.asarray(previous_palm_action["palm_target_m"]) -
                               initial_observation["robot"]["palm_position_m"])
        self._position_cap_m = min(.012, max(.007, error + .003))
        self.max_steps = max_steps
        self.steps = 0
        self.phase = "unload"
        self.status = "ready"
        self.last_goal = None
        self.reason = None
        self._pending = None
        self._pending_time = None
        self._clear_count = 0
        self._stopped = False

    def describe(self):
        return copy.deepcopy({
            "version": self.VERSION, "initial_observation_sha256": self._initial_sha256,
            "previous_palm_action": self._initial_previous,
            "last_accepted_palm_action": self._previous,
            "max_steps": self.max_steps, "steps": self.steps, "phase": self.phase,
            "status": self.status, "reason": self.reason, "last_goal": self.last_goal,
            "position_cap_m": float(self._position_cap_m),
        })

    def _handoff(self, status, reason):
        self.status, self.reason, self._stopped = status, reason, True
        return None

    def _issue(self, action, observation):
        self._pending = copy.deepcopy(action)
        self._pending_time = float(observation["time_s"])
        self.steps += 1
        self.status = "awaiting_result"
        return action

    def decide(self, observation):
        if self._stopped:
            return None
        hold_completed = False
        if self._pending is not None:
            result = observation["last_action"]
            if result["status"] in ("rejected", "interrupted"):
                return self._handoff(result["status"], result.get("reason"))
            if result["status"] != "completed":
                return self._handoff("action_not_completed", "Previous skill action was not accepted")
            if (float(observation["time_s"]) <= self._pending_time or
                    float(result["time_s"]) <= self._pending_time):
                return self._handoff("stale_observation", "Need observation after the issued action")
            if self._pending["operation"] == "move":
                self._previous = self._pending
                self._clear_count = (self._clear_count + 1
                                     if not observation["contacts"]["tool_nut_contact"] else 0)
            else:
                hold_completed = True
            self._pending = None

        contacts = observation["contacts"]
        if (contacts["max_contact_force_n"] > 25 or contacts["max_penetration_m"] > .002
                or contacts["robot_nut_contact"]):
            return self._handoff("quality_violation", "Existing wrench quality criterion failed")
        if len(set(contacts["hand_tool_contacts"])) < 2:
            return self._handoff("grip_lost", "Fewer than two hand parts retain the tool")
        angle = math.degrees(observation["nut"]["angle_rad"] - observation["nut"]["initial_angle_rad"])
        if self.steps == 0 and angle < 55:
            return self._handoff("needs_turn_margin", "Initial retained angle is below 55 degrees")
        if angle < 45:
            return self._handoff("angle_lost", "Retained nut angle fell below 45 degrees")
        if hold_completed:
            return self._handoff("clearance_check", "Caller must inspect the independent withdrawal gate")
        if self.steps >= self.max_steps:
            return self._handoff("budget_exhausted", "Skill action budget exhausted")
        if (not contacts["tool_nut_contact"] and contacts["head_clearance_m"] >= .032
                and (self._clear_count >= 2 or self.steps == 0)):
            self.phase = "hold"
            return self._issue({
                "operation": "hold", "targets": [], "palm_target_m": None, "palm_quat_wxyz": None,
                "duration_s": 1.2, "memory": "Measured clear retained hold",
            }, observation)

        head = np.asarray(observation["tool"]["head_center_m"])
        center = np.asarray(observation["nut"]["position_m"])
        feedback = observation["wrench_feedback"]["head_nut"]
        axis = np.asarray(feedback["nut_hinge_axis_world"])
        tool = _matrix(observation["tool"]["quat_wxyz"])
        lateral = center - head
        lateral -= axis * (lateral @ axis)
        if self.phase == "unload" and (
                (np.linalg.norm(lateral) < .0005 and feedback["axis_angle_deg"] < 1.5)
                or self.steps >= 16):
            self.phase = "extract"
        target = head + _bounded(lateral, .00035)
        if self.phase == "extract":
            target += axis * (.0015 if contacts["tool_nut_contact"] else .005)
        desired = _align_axis(tool, axis, math.radians(.8 if self.phase == "unload" else .5))
        self.last_goal = {"target_head_m": target.tolist(), "target_tool_quat_wxyz": _quaternion(desired)}
        action = feedback_tool_pose_action(
            observation, target.tolist(), _quaternion(desired),
            previous_palm_target_m=self._previous["palm_target_m"],
            previous_palm_quat_wxyz=self._previous["palm_quat_wxyz"],
            duration_s=.3, memory=f"Measured centering and axis {self.phase} {self.steps}",
        )
        palm = np.asarray(observation["robot"]["palm_position_m"])
        palm_rotation = np.asarray(observation["robot"]["palm_axes_world"])
        raw_position = np.asarray(action["palm_target_m"])
        raw_rotation = _matrix(action["palm_quat_wxyz"])
        action["palm_target_m"] = (palm + _bounded(raw_position - palm, self._position_cap_m)).tolist()
        action["palm_quat_wxyz"] = _quaternion(
            _rotation_step(raw_rotation @ palm_rotation.T, math.radians(5)) @ palm_rotation)
        return self._issue(action, observation)

