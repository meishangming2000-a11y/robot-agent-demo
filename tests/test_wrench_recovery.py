"""Observation-only recovery contracts against preserved physical observations."""

import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from dexhand_agent.runtime import ACTION_SCHEMA
from dexhand_agent.wrench_recovery import WrenchWithdrawPolicy


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/wrench_withdraw_v1.json"


def rotation(quaternion):
    result = np.zeros(9)
    mujoco.mju_quat2Mat(result, np.asarray(quaternion))
    return result.reshape(3, 3)


def quaternion(matrix):
    result = np.zeros(4)
    mujoco.mju_mat2Quat(result, np.asarray(matrix).reshape(9))
    return result.tolist()


@pytest.fixture(scope="module")
def recorded():
    data = json.loads(FIXTURE.read_text(encoding="utf8"))
    return [{"observation": data["initial_observation"], "action": data["previous_palm_action"]}] + [
        {"observation": row["observation_after"], "action": row["expected_action"]}
        for row in data["sequence"]
    ]


def observation(record):
    return copy.deepcopy(record["observation"])


def make_policy(recorded, **kwargs):
    return WrenchWithdrawPolicy(observation(recorded[0]), recorded[0]["action"], **kwargs)


def test_reproduces_entire_preserved_successful_physical_recovery(recorded):
    policy = make_policy(recorded)
    assert policy.steps == 0
    assert policy.last_goal is None
    for index in range(1, len(recorded)):
        actual = policy.decide(observation(recorded[index - 1]))
        expected = recorded[index]["action"]
        assert set(actual) == set(ACTION_SCHEMA["required"])
        assert actual["operation"] == expected["operation"]
        assert actual["targets"] == expected["targets"]
        assert actual["duration_s"] == expected["duration_s"]
        if expected["palm_target_m"] is None:
            assert actual["palm_target_m"] is None
            assert actual["palm_quat_wxyz"] is None
        else:
            np.testing.assert_allclose(actual["palm_target_m"], expected["palm_target_m"], atol=2e-14, rtol=0)
            np.testing.assert_allclose(rotation(actual["palm_quat_wxyz"]),
                                       rotation(expected["palm_quat_wxyz"]), atol=2e-14, rtol=0)
    assert policy.decide(observation(recorded[-1])) is None
    assert policy.status == "clearance_check"
    assert policy.phase == "hold"
    assert policy.steps == 66
    assert "success" not in policy.describe()
    assert policy.decide(observation(recorded[-1])) is None


@pytest.mark.parametrize("status", ["rejected", "interrupted"])
def test_unaccepted_action_is_not_adopted_as_previous_command(recorded, status):
    policy = make_policy(recorded)
    before = policy.describe()["last_accepted_palm_action"]
    policy.decide(observation(recorded[0]))
    obs = observation(recorded[1])
    obs["last_action"] = {"status": status, "reason": "external execution failure", "time_s": obs["time_s"]}
    assert policy.decide(obs) is None
    assert policy.status == status
    assert policy.describe()["last_accepted_palm_action"] == before


def test_completed_result_requires_a_fresh_observation(recorded):
    policy = make_policy(recorded)
    obs = observation(recorded[0])
    policy.decide(obs)
    assert policy.decide(obs) is None
    assert policy.status == "stale_observation"
    assert policy.describe()["last_accepted_palm_action"] == recorded[0]["action"]


def test_actual_low_margin_contact_state_hands_back_without_action():
    data = json.loads(FIXTURE.read_text(encoding="utf8"))
    obs = data["low_margin_observation"]
    previous = data["low_margin_previous_palm_action"]
    policy = WrenchWithdrawPolicy(obs, previous)
    assert policy.decide(obs) is None
    assert policy.status == "needs_turn_margin"
    assert policy.steps == 0


@pytest.mark.parametrize("field,value,status", [
    ("max_contact_force_n", 25.01, "quality_violation"),
    ("max_penetration_m", 0.00201, "quality_violation"),
    ("robot_nut_contact", True, "quality_violation"),
    ("hand_tool_contacts", ["thumb"], "grip_lost"),
])
def test_quality_and_grip_handoff_does_not_claim_success(recorded, field, value, status):
    obs = observation(recorded[0])
    obs["contacts"][field] = value
    policy = WrenchWithdrawPolicy(obs, recorded[0]["action"])
    assert policy.decide(obs) is None
    assert policy.status == status
    assert "success" not in policy.describe()


def test_clearance_hold_returns_for_external_gate_check_even_without_progress_field(recorded):
    obs = observation(recorded[-1])
    obs.pop("task_progress", None)
    obs["contacts"]["head_clearance_m"] = 0.04
    previous = recorded[-2]["action"]
    policy = WrenchWithdrawPolicy(obs, previous)
    action = policy.decide(obs)
    assert action["operation"] == "hold" and action["duration_s"] == 1.2
    assert action["palm_target_m"] is None and action["targets"] == []
    after = copy.deepcopy(obs)
    after["time_s"] += 1.2
    after["last_action"] = {"status": "completed", "time_s": after["time_s"]}
    # The caller supplies no independent withdraw-gate PASS here.
    assert policy.decide(after) is None
    assert policy.status == "clearance_check"
    assert "success" not in policy.describe()


def test_budget_handoff_after_confirming_last_command(recorded):
    policy = make_policy(recorded, max_steps=1)
    sent = policy.decide(observation(recorded[0]))
    assert policy.decide(observation(recorded[1])) is None
    assert policy.status == "budget_exhausted"
    assert policy.describe()["last_accepted_palm_action"] == sent


def test_description_and_inputs_are_json_safe_and_not_mutated(recorded):
    obs = observation(recorded[0])
    previous = copy.deepcopy(recorded[0]["action"])
    before_obs, before_previous = copy.deepcopy(obs), copy.deepcopy(previous)
    policy = WrenchWithdrawPolicy(obs, previous)
    description = policy.describe()
    assert json.loads(json.dumps(description, allow_nan=False)) == description
    assert description["version"] == "wrench_withdraw_v1"
    assert len(description["initial_observation_sha256"]) == 64
    assert description["initial_observation_sha256"] == hashlib.sha256(json.dumps(
        obs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest()
    assert description["previous_palm_action"] == previous
    policy.decide(obs)
    assert set(policy.last_goal) == {"target_head_m", "target_tool_quat_wxyz"}
    assert obs == before_obs and previous == before_previous
    description["last_accepted_palm_action"]["palm_target_m"][0] = 999
    assert policy.describe()["last_accepted_palm_action"] == previous


def transform_observation(obs, yaw, shift):
    obs = copy.deepcopy(obs)
    robot, tool, nut = obs["robot"], obs["tool"], obs["nut"]
    for owner, key in ((robot, "palm_position_m"), (tool, "head_center_m"), (nut, "position_m")):
        owner[key] = (yaw @ owner[key] + shift).tolist()
    robot["palm_axes_world"] = (yaw @ robot["palm_axes_world"]).tolist()
    tool["quat_wxyz"] = quaternion(yaw @ rotation(tool["quat_wxyz"]))
    feedback = obs["wrench_feedback"]["head_nut"]
    feedback["nut_hinge_axis_world"] = (yaw @ feedback["nut_hinge_axis_world"]).tolist()
    return obs


def test_policy_is_equivariant_to_world_yaw_and_translation(recorded):
    angle = 0.7
    yaw = np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    shift = np.array([0.1, -0.2, 0.05])
    previous = copy.deepcopy(recorded[0]["action"])
    previous["palm_target_m"] = (yaw @ previous["palm_target_m"] + shift).tolist()
    previous["palm_quat_wxyz"] = quaternion(yaw @ rotation(previous["palm_quat_wxyz"]))
    original = make_policy(recorded)
    transformed = WrenchWithdrawPolicy(transform_observation(observation(recorded[0]), yaw, shift), previous)
    for index in range(0, 7):
        obs = observation(recorded[index])
        first = original.decide(obs)
        second = transformed.decide(transform_observation(obs, yaw, shift))
        np.testing.assert_allclose(second["palm_target_m"],
                                   yaw @ first["palm_target_m"] + shift, atol=1e-13, rtol=0)
        np.testing.assert_allclose(rotation(second["palm_quat_wxyz"]),
                                   yaw @ rotation(first["palm_quat_wxyz"]), atol=1e-13, rtol=0)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_rejects_invalid_step_budget(recorded, limit):
    with pytest.raises(ValueError):
        make_policy(recorded, max_steps=limit)

