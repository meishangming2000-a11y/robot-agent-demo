"""Pure tool-to-palm pose conversion from the current public observation."""

import mujoco
import numpy as np


def _array(value, shape, name):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite array with shape {shape}") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array


def _rotation(quaternion, name):
    quaternion = _array(quaternion, (4,), name)
    norm = np.linalg.norm(quaternion)
    if abs(norm - 1) > 1e-6:
        raise ValueError(f"{name} must have unit length")
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, quaternion / norm)
    return matrix.reshape(3, 3)


def tool_pose_action(observation, target_head_m, target_tool_quat_wxyz=None, *,
                     duration_s=1.0, compliance_offset_m=(0, 0, 0), memory=""):
    """Return a move preserving the currently observed tool-to-palm transform.

    Positions and the optional compliance offset are world-frame metres.
    Quaternions are unit wxyz. Omitting the tool quaternion keeps its observed
    orientation. This function only computes an action; it does not execute it.
    """
    tool = observation["tool"]
    robot = observation["robot"]
    head = _array(tool["head_center_m"], (3,), "tool.head_center_m")
    palm = _array(robot["palm_position_m"], (3,), "robot.palm_position_m")
    target = _array(target_head_m, (3,), "target_head_m")
    offset = _array(compliance_offset_m, (3,), "compliance_offset_m")
    current = _rotation(tool["quat_wxyz"], "tool.quat_wxyz")
    desired = current if target_tool_quat_wxyz is None else _rotation(
        target_tool_quat_wxyz, "target_tool_quat_wxyz")
    palm_rotation = _array(robot["palm_axes_world"], (3, 3), "robot.palm_axes_world")
    if (not np.allclose(palm_rotation.T @ palm_rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(palm_rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError("robot.palm_axes_world must be a rotation matrix")
    if (isinstance(duration_s, bool) or not isinstance(duration_s, (int, float))
            or not np.isfinite(duration_s) or not 0.02 <= duration_s <= 3):
        raise ValueError("duration_s must be finite and between 0.02 and 3")
    if not isinstance(memory, str):
        raise ValueError("memory must be a string")

    delta = desired @ current.T
    position = target + delta @ (palm - head) + offset
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, (delta @ palm_rotation).reshape(9))
    return {"operation": "move", "targets": [], "palm_target_m": position.tolist(),
            "palm_quat_wxyz": quaternion.tolist(), "duration_s": float(duration_s),
            "memory": memory}


def palm_command_error(observation, previous_palm_target_m, previous_palm_quat_wxyz):
    """Return local E in actual_palm = commanded_palm @ E as plain JSON.

    Supply the complete accepted command that produced this observation. This
    is a local engineering calibration, not a learned task policy; a changing
    grasp or load can invalidate it. Contact performance is not established.
    """
    robot = observation["robot"]
    actual_position = _array(robot["palm_position_m"], (3,), "robot.palm_position_m")
    actual_rotation = _array(robot["palm_axes_world"], (3, 3), "robot.palm_axes_world")
    if (not np.allclose(actual_rotation.T @ actual_rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(actual_rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError("robot.palm_axes_world must be a rotation matrix")
    previous_position = _array(previous_palm_target_m, (3,), "previous_palm_target_m")
    previous_rotation = _rotation(previous_palm_quat_wxyz, "previous_palm_quat_wxyz")
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, (previous_rotation.T @ actual_rotation).reshape(9))
    return {"translation_m": (previous_rotation.T @ (actual_position - previous_position)).tolist(),
            "quat_wxyz": quaternion.tolist()}


def calibrated_tool_pose_action(observation, target_head_m, target_tool_quat_wxyz=None, *,
                                command_error, duration_s=1.0, memory=""):
    """Apply nominal_palm @ inverse(E), using the caller's fixed calibration.

    The nominal pose uses the current observed grasp. E is never reestimated or
    mutated here. Its local constant-rigid-error assumption does not guarantee
    realization under changing contact or grasp, or establish task success.
    """
    action = tool_pose_action(observation, target_head_m, target_tool_quat_wxyz,
                              duration_s=duration_s, memory=memory)
    if not isinstance(command_error, dict) or set(command_error) != {"translation_m", "quat_wxyz"}:
        raise ValueError("command_error must contain translation_m and quat_wxyz only")
    error_position = _array(command_error["translation_m"], (3,), "command_error.translation_m")
    error_rotation = _rotation(command_error["quat_wxyz"], "command_error.quat_wxyz")
    nominal_rotation = _rotation(action["palm_quat_wxyz"], "nominal palm quaternion")
    command_rotation = nominal_rotation @ error_rotation.T
    action["palm_target_m"] = (np.asarray(action["palm_target_m"]) -
                               command_rotation @ error_position).tolist()
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, command_rotation.reshape(9))
    action["palm_quat_wxyz"] = quaternion.tolist()
    return action


def feedback_tool_pose_action(observation, target_head_m, target_tool_quat_wxyz=None, *,
                              previous_palm_target_m, previous_palm_quat_wxyz,
                              duration_s=1.0, memory=""):
    """Invert an estimated constant local command-to-realized palm pose bias.

    Supply the complete previous accepted palm command that produced this
    observation, in world metres and unit wxyz. Recompute from each fresh
    observation: changing load or grasp can change the estimated rigid bias.
    This pure calculation does not execute an action or guarantee realization.
    """
    error = palm_command_error(observation, previous_palm_target_m, previous_palm_quat_wxyz)
    return calibrated_tool_pose_action(observation, target_head_m, target_tool_quat_wxyz,
                                       command_error=error, duration_s=duration_s, memory=memory)


