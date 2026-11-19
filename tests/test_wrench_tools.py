"""Pure tool-frame geometry from public observations, without a physics model."""

import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from dexhand_agent.runtime import ACTION_SCHEMA
from dexhand_agent.wrench_tools import (
    calibrated_tool_pose_action, feedback_tool_pose_action, palm_command_error, tool_pose_action,
)


def rotation(quaternion):
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    return matrix.reshape(3, 3)


@pytest.fixture
def observation():
    half = np.sqrt(0.5)
    return {
        "tool": {"position_m": [-1.0, 2.0, 3.0],
                 "head_center_m": [0.8, -0.2, 0.4],
                 "quat_wxyz": [half, 0.0, 0.0, half]},
        "robot": {"palm_position_m": [0.7, -0.1, 0.6],
                  "palm_axes_world": [[1, 0, 0], [0, 0, -1], [0, 1, 0]]},
    }


def test_nonidentity_grasp_moves_head_site_and_preserves_relative_pose(observation):
    half = np.sqrt(0.5)
    action = tool_pose_action(
        observation, [0.4, 0.2, 0.3], [half, 0, half, 0],
        duration_s=1.2, compliance_offset_m=[0.001, -0.002, 0.003],
        memory="Explicit observed-pose correction",
    )
    # Current tool is Rz(90), current palm Rx(90), desired tool Ry(90).
    # Their noncommuting delta maps the observed palm-head offset to (0.2, 0.1, -0.1).
    np.testing.assert_allclose(action["palm_target_m"], [0.601, 0.298, 0.203], atol=1e-14)
    np.testing.assert_allclose(rotation(action["palm_quat_wxyz"]),
                               [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-14)
    assert set(action) == set(ACTION_SCHEMA["required"])
    assert action["operation"] == "move"
    assert action["targets"] == []
    assert action["duration_s"] == 1.2
    assert action["memory"] == "Explicit observed-pose correction"


def test_omitting_tool_orientation_only_translates_without_default_bias(observation):
    action = tool_pose_action(observation, [0.85, -0.17, 0.38])
    np.testing.assert_allclose(action["palm_target_m"], [0.75, -0.07, 0.58], atol=1e-14)
    np.testing.assert_allclose(rotation(action["palm_quat_wxyz"]),
                               observation["robot"]["palm_axes_world"], atol=1e-14)
    assert action["memory"] == ""


def test_call_is_stateless_and_does_not_mutate_inputs(observation):
    before = copy.deepcopy(observation)
    target = np.array([0.4, 0.2, 0.3])
    offset = np.array([0.001, -0.002, 0.003])
    first = tool_pose_action(observation, target, compliance_offset_m=offset)
    assert observation == before
    np.testing.assert_array_equal(target, [0.4, 0.2, 0.3])
    np.testing.assert_array_equal(offset, [0.001, -0.002, 0.003])
    changed = copy.deepcopy(observation)
    changed["robot"]["palm_position_m"][0] += 0.02
    second = tool_pose_action(changed, target, compliance_offset_m=offset)
    np.testing.assert_allclose(np.subtract(second["palm_target_m"], first["palm_target_m"]),
                               [0.02, 0, 0], atol=1e-14)
    assert tool_pose_action(observation, target, compliance_offset_m=offset) == first


def test_uses_actual_retained_request_observation_without_other_state():
    request_path = (Path(__file__).resolve().parents[1] / "evidence" / "kit-suite-v1" /
                    "wrench_turn" / "steps" / "010" / "request.json")
    request = json.loads(request_path.read_text(encoding="utf8"))
    obs = request["observation"]
    before = copy.deepcopy(obs)
    target = np.asarray(obs["tool"]["head_center_m"]) + [0.02, -0.01, 0.03]
    # Only the documented current tool/head/palm fields are passed to the helper.
    public_pose = {
        "tool": {name: obs["tool"][name] for name in ("quat_wxyz", "head_center_m")},
        "robot": {name: obs["robot"][name] for name in ("palm_position_m", "palm_axes_world")},
    }
    action = tool_pose_action(public_pose, target)
    np.testing.assert_allclose(action["palm_target_m"],
                               np.asarray(obs["robot"]["palm_position_m"]) + [0.02, -0.01, 0.03],
                               atol=1e-12)
    assert obs == before


@pytest.mark.parametrize("section,key,value", [
    ("tool", "head_center_m", [0, 0]),
    ("tool", "head_center_m", [0, np.nan, 0]),
    ("tool", "quat_wxyz", [1, 0, 0]),
    ("tool", "quat_wxyz", [np.inf, 0, 0, 0]),
    ("tool", "quat_wxyz", [0, 0, 0, 0]),
    ("tool", "quat_wxyz", [2, 0, 0, 0]),
    ("robot", "palm_position_m", [[0, 0, 0]]),
    ("robot", "palm_position_m", [0, 0, -np.inf]),
    ("robot", "palm_axes_world", np.eye(2).tolist()),
    ("robot", "palm_axes_world", [[1, 0, 0], [0, 1, 0], [0, 0, np.nan]]),
    ("robot", "palm_axes_world", [[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    ("robot", "palm_axes_world", [[2, 0, 0], [0, 1, 0], [0, 0, 1]]),
])
def test_rejects_invalid_observed_poses(observation, section, key, value):
    observation[section][key] = value
    with pytest.raises(ValueError):
        tool_pose_action(observation, [0.4, 0.2, 0.3])


@pytest.mark.parametrize("kwargs", [
    {"target_head_m": [0, 0]},
    {"target_head_m": [0, np.inf, 0]},
    {"target_tool_quat_wxyz": [1, 0, 0]},
    {"target_tool_quat_wxyz": [1, 0, np.nan, 0]},
    {"target_tool_quat_wxyz": [1.01, 0, 0, 0]},
    {"compliance_offset_m": [0, 0]},
    {"compliance_offset_m": [0, 0, np.nan]},
    {"duration_s": 0.019},
    {"duration_s": 3.001},
    {"duration_s": np.nan},
    {"duration_s": True},
    {"memory": None},
])
def test_rejects_invalid_requested_action(observation, kwargs):
    arguments = {"target_head_m": [0.4, 0.2, 0.3], **kwargs}
    with pytest.raises(ValueError):
        tool_pose_action(observation, **arguments)


@pytest.mark.parametrize("duration", [0.02, 3.0])
def test_accepts_action_duration_boundaries(observation, duration):
    action = tool_pose_action(observation, [0.4, 0.2, 0.3], duration_s=duration)
    assert action["duration_s"] == duration


def quaternion(matrix):
    result = np.zeros(4)
    mujoco.mju_mat2Quat(result, np.asarray(matrix, dtype=float).reshape(9))
    return result


def test_feedback_inverts_fixed_local_rigid_bias_with_noncommuting_rotations():
    # Analytic plant: actual_palm = commanded_palm @ fixed_local_bias.
    command_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    bias_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    tool_in_palm_rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    command_position = np.array([0.3, -0.2, 0.5])
    bias_position = np.array([0.03, 0.04, -0.02])
    head_in_palm = np.array([0.11, 0.03, 0.02])
    actual_rotation = command_rotation @ bias_rotation
    actual_position = command_position + command_rotation @ bias_position
    observation = {
        "robot": {"palm_position_m": actual_position.tolist(),
                  "palm_axes_world": actual_rotation.tolist()},
        "tool": {"head_center_m": (actual_position + actual_rotation @ head_in_palm).tolist(),
                 "quat_wxyz": quaternion(actual_rotation @ tool_in_palm_rotation).tolist()},
    }
    target_head = np.array([0.6, 0.1, 0.4])
    target_rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    action = feedback_tool_pose_action(
        observation, target_head, quaternion(target_rotation),
        previous_palm_target_m=command_position,
        previous_palm_quat_wxyz=quaternion(command_rotation),
        duration_s=0.8, memory="Explicit accepted-command feedback",
    )
    next_rotation = rotation(action["palm_quat_wxyz"])
    realized_rotation = next_rotation @ bias_rotation
    realized_position = np.asarray(action["palm_target_m"]) + next_rotation @ bias_position
    np.testing.assert_allclose(realized_position + realized_rotation @ head_in_palm,
                               target_head, atol=1e-14)
    np.testing.assert_allclose(realized_rotation @ tool_in_palm_rotation,
                               target_rotation, atol=1e-14)
    # Reusing the previous world translation error without rotating it is not
    # the inverse of this same body-fixed plant.
    nominal = tool_pose_action(observation, target_head, quaternion(target_rotation))
    mixed_position = np.asarray(nominal["palm_target_m"]) + command_position - actual_position
    mixed_head = mixed_position + next_rotation @ bias_position + realized_rotation @ head_in_palm
    assert np.linalg.norm(mixed_head - target_head) > 0.01
    assert set(action) == set(ACTION_SCHEMA["required"])
    assert action["operation"] == "move"
    assert action["targets"] == []
    assert action["duration_s"] == 0.8
    assert action["memory"] == "Explicit accepted-command feedback"


@pytest.mark.parametrize("target_quaternion", [None, [np.sqrt(0.5), 0, np.sqrt(0.5), 0]])
def test_feedback_zero_bias_matches_nominal(observation, target_quaternion):
    previous_position = observation["robot"]["palm_position_m"]
    previous_quaternion = quaternion(observation["robot"]["palm_axes_world"])
    expected = tool_pose_action(observation, [0.4, 0.2, 0.3], target_quaternion)
    actual = feedback_tool_pose_action(
        observation, [0.4, 0.2, 0.3], target_quaternion,
        previous_palm_target_m=previous_position,
        previous_palm_quat_wxyz=previous_quaternion,
    )
    np.testing.assert_allclose(actual["palm_target_m"], expected["palm_target_m"], atol=1e-14)
    np.testing.assert_allclose(rotation(actual["palm_quat_wxyz"]),
                               rotation(expected["palm_quat_wxyz"]), atol=1e-14)
    for key in ("operation", "targets", "duration_s", "memory"):
        assert actual[key] == expected[key]


def test_feedback_does_not_mutate_observation_or_previous_command(observation):
    before = copy.deepcopy(observation)
    previous_position = np.array([0.72, -0.09, 0.61])
    previous_quaternion = np.array([np.sqrt(0.5), np.sqrt(0.5), 0, 0])
    position_before, quaternion_before = previous_position.copy(), previous_quaternion.copy()
    feedback_tool_pose_action(observation, [0.4, 0.2, 0.3],
                             previous_palm_target_m=previous_position,
                             previous_palm_quat_wxyz=previous_quaternion)
    assert observation == before
    np.testing.assert_array_equal(previous_position, position_before)
    np.testing.assert_array_equal(previous_quaternion, quaternion_before)


@pytest.mark.parametrize("changes", [
    {"previous_palm_target_m": None},
    {"previous_palm_target_m": [0, 0]},
    {"previous_palm_target_m": [[0, 0, 0]]},
    {"previous_palm_target_m": [0, np.nan, 0]},
    {"previous_palm_target_m": [0, 0, np.inf]},
    {"previous_palm_quat_wxyz": None},
    {"previous_palm_quat_wxyz": [1, 0, 0]},
    {"previous_palm_quat_wxyz": [0, 0, 0, 0]},
    {"previous_palm_quat_wxyz": [2, 0, 0, 0]},
    {"previous_palm_quat_wxyz": [1, 0, 0, np.nan]},
    {"previous_palm_quat_wxyz": [1, 0, np.inf, 0]},
])
def test_feedback_rejects_invalid_previous_command(observation, changes):
    arguments = {"previous_palm_target_m": [0.7, -0.1, 0.6],
                 "previous_palm_quat_wxyz": [1, 0, 0, 0], **changes}
    with pytest.raises(ValueError):
        feedback_tool_pose_action(observation, [0.4, 0.2, 0.3], **arguments)


def test_palm_command_calibration_is_local_json_and_inverts_noncommuting_plant(observation):
    command_position = np.array([0.3, -0.2, 0.5])
    command_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    error_position = np.array([0.03, 0.04, -0.02])
    error_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    calibration_observation = {
        "robot": {"palm_position_m": (command_position + command_rotation @ error_position).tolist(),
                  "palm_axes_world": (command_rotation @ error_rotation).tolist()},
    }
    error = palm_command_error(calibration_observation, command_position, quaternion(command_rotation))
    np.testing.assert_allclose(error["translation_m"], error_position, atol=1e-14)
    np.testing.assert_allclose(rotation(error["quat_wxyz"]), error_rotation, atol=1e-14)
    assert set(error) == {"translation_m", "quat_wxyz"}
    error = json.loads(json.dumps(error, allow_nan=False))
    action = calibrated_tool_pose_action(
        observation, [0.4, 0.2, 0.3], [np.sqrt(0.5), 0, np.sqrt(0.5), 0],
        command_error=error, duration_s=0.7, memory="Frozen free-space calibration",
    )
    nominal = tool_pose_action(observation, [0.4, 0.2, 0.3], [np.sqrt(0.5), 0, np.sqrt(0.5), 0])
    # Apply the independent fixed-local-bias plant to the returned command.
    command_rotation = rotation(action["palm_quat_wxyz"])
    np.testing.assert_allclose(np.asarray(action["palm_target_m"]) + command_rotation @ error_position,
                               nominal["palm_target_m"], atol=1e-14)
    np.testing.assert_allclose(command_rotation @ error_rotation,
                               rotation(nominal["palm_quat_wxyz"]), atol=1e-14)
    assert set(action) == set(ACTION_SCHEMA["required"])
    assert action["operation"] == "move" and action["targets"] == []
    assert action["duration_s"] == 0.7
    assert action["memory"] == "Frozen free-space calibration"


def test_frozen_calibration_is_not_reestimated_from_current_palm_error(observation):
    previous_position = [0.72, -0.09, 0.61]
    previous_quaternion = quaternion(observation["robot"]["palm_axes_world"])
    frozen = palm_command_error(observation, previous_position, previous_quaternion)
    current = copy.deepcopy(observation)
    current["robot"]["palm_position_m"] = [0.74, -0.12, 0.64]
    current["robot"]["palm_axes_world"] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    instantaneous = palm_command_error(current, previous_position, previous_quaternion)
    assert instantaneous != frozen
    action = calibrated_tool_pose_action(current, [0.4, 0.2, 0.3], command_error=frozen)
    nominal = tool_pose_action(current, [0.4, 0.2, 0.3])
    realized_rotation = rotation(action["palm_quat_wxyz"])
    np.testing.assert_allclose(realized_rotation @ rotation(frozen["quat_wxyz"]),
                               rotation(nominal["palm_quat_wxyz"]), atol=1e-14)
    np.testing.assert_allclose(np.asarray(action["palm_target_m"]) +
                               realized_rotation @ frozen["translation_m"],
                               nominal["palm_target_m"], atol=1e-14)
    reestimated = calibrated_tool_pose_action(current, [0.4, 0.2, 0.3],
                                              command_error=instantaneous)
    assert np.linalg.norm(np.subtract(action["palm_target_m"], reestimated["palm_target_m"])) > 0.01


def test_calibration_and_application_do_not_mutate_inputs(observation):
    before = copy.deepcopy(observation)
    position = np.array([0.72, -0.09, 0.61])
    previous_quaternion = quaternion(observation["robot"]["palm_axes_world"])
    position_before, quaternion_before = position.copy(), previous_quaternion.copy()
    error = palm_command_error(observation, position, previous_quaternion)
    error_before = copy.deepcopy(error)
    calibrated_tool_pose_action(observation, [0.4, 0.2, 0.3], command_error=error)
    assert observation == before
    assert error == error_before
    np.testing.assert_array_equal(position, position_before)
    np.testing.assert_array_equal(previous_quaternion, quaternion_before)


@pytest.mark.parametrize("target_quaternion", [None, [np.sqrt(0.5), 0, np.sqrt(0.5), 0]])
def test_explicit_calibration_matches_existing_feedback(observation, target_quaternion):
    previous_position = [0.72, -0.09, 0.61]
    previous_quaternion = quaternion(observation["robot"]["palm_axes_world"])
    error = palm_command_error(observation, previous_position, previous_quaternion)
    calibrated = calibrated_tool_pose_action(
        observation, [0.4, 0.2, 0.3], target_quaternion, command_error=error)
    feedback = feedback_tool_pose_action(
        observation, [0.4, 0.2, 0.3], target_quaternion,
        previous_palm_target_m=previous_position, previous_palm_quat_wxyz=previous_quaternion)
    np.testing.assert_allclose(calibrated["palm_target_m"], feedback["palm_target_m"], atol=1e-14)
    np.testing.assert_allclose(rotation(calibrated["palm_quat_wxyz"]),
                               rotation(feedback["palm_quat_wxyz"]), atol=1e-14)


@pytest.mark.parametrize("error", [
    None, [], {}, {"translation_m": [0, 0, 0]},
    {"translation_m": [0, 0, 0], "quat_wxyz": [1, 0, 0, 0], "extra": 1},
    {"translation_m": None, "quat_wxyz": [1, 0, 0, 0]},
    {"translation_m": [0, 0], "quat_wxyz": [1, 0, 0, 0]},
    {"translation_m": [0, np.nan, 0], "quat_wxyz": [1, 0, 0, 0]},
    {"translation_m": [0, 0, np.inf], "quat_wxyz": [1, 0, 0, 0]},
    {"translation_m": [0, 0, 0], "quat_wxyz": None},
    {"translation_m": [0, 0, 0], "quat_wxyz": [1, 0, 0]},
    {"translation_m": [0, 0, 0], "quat_wxyz": [0, 0, 0, 0]},
    {"translation_m": [0, 0, 0], "quat_wxyz": [2, 0, 0, 0]},
    {"translation_m": [0, 0, 0], "quat_wxyz": [1, 0, np.nan, 0]},
])
def test_calibrated_action_rejects_invalid_calibration_json(observation, error):
    with pytest.raises(ValueError):
        calibrated_tool_pose_action(observation, [0.4, 0.2, 0.3], command_error=error)


@pytest.mark.parametrize("field,value", [
    ("palm_position_m", [0, 0]),
    ("palm_position_m", [0, np.nan, 0]),
    ("palm_axes_world", np.eye(2).tolist()),
    ("palm_axes_world", [[1, 0, 0], [0, 1, 0], [0, 0, np.inf]]),
    ("palm_axes_world", [[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
])
def test_calibration_rejects_invalid_measured_palm(observation, field, value):
    observation["robot"][field] = value
    with pytest.raises(ValueError):
        palm_command_error(observation, [0.7, -0.1, 0.6], [1, 0, 0, 0])



