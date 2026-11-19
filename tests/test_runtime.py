import numpy as np
import pytest
import mujoco

from dexhand_agent.runtime import HandRuntime, InvalidAction


@pytest.fixture(scope="module")
def runtime():
    return HandRuntime()


def test_rejected_action_does_not_advance_physics(runtime):
    before = runtime.data.qpos.copy()
    time_before = runtime.data.time
    with pytest.raises(InvalidAction):
        runtime.execute({"operation": "move", "targets": [{"actuator": "not_an_actuator", "value": 1.0}], "duration_s": 0.5})
    assert runtime.data.time == time_before
    np.testing.assert_array_equal(runtime.data.qpos, before)


def test_range_violation_does_not_advance_physics(runtime):
    before = runtime.data.qpos.copy()
    with pytest.raises(InvalidAction):
        runtime.execute({"operation": "move", "targets": [{"actuator": "rh_A_FFJ3", "value": 100.0}], "duration_s": 0.5})
    np.testing.assert_array_equal(runtime.data.qpos, before)


def test_named_actuation_retains_other_targets(runtime):
    before = runtime.data.ctrl.copy()
    target = runtime.model.actuator("rh_A_FFJ3").id
    runtime.execute({"operation": "move", "targets": [{"actuator": "rh_A_FFJ3", "value": 0.3}], "duration_s": 0.7})
    np.testing.assert_array_equal(np.delete(runtime.data.ctrl, target), np.delete(before, target))
    assert abs(runtime.data.actuator_length[target] - 0.3) < 0.06


def test_public_observation_distinguishes_joint_and_tendon(runtime):
    observation = runtime.observe()
    assert "rh_FFJ1" in observation["robot"]["joints_rad"]
    assert "rh_FFJ0" not in observation["robot"]["joints_rad"]
    assert observation["robot"]["actuators"]["rh_A_FFJ0"]["transmission"] == "tendon"
    assert "commanded" in observation["robot"]["actuators"]["rh_A_FFJ0"]


def test_unreachable_palm_target_does_not_advance_physics(runtime):
    time_before = runtime.data.time
    with pytest.raises(InvalidAction):
        runtime.execute({"operation": "move", "palm_target_m": [5, 5, 5], "duration_s": 1.0})
    assert runtime.data.time == time_before


def test_observation_geometry_matches_current_joint_state(runtime):
    runtime.execute({"operation": "move", "targets": [{"actuator": "actuator2", "value": -0.30}], "duration_s": 0.1})
    scratch = mujoco.MjData(runtime.model)
    scratch.qpos[:] = runtime.data.qpos
    mujoco.mj_forward(runtime.model, scratch)
    np.testing.assert_allclose(runtime.observe()["robot"]["palm_position_m"], scratch.site("palm_frame").xpos, atol=1e-10, rtol=0)


@pytest.mark.parametrize("field,value", [
    ("palm_target_m", [1, 2, "bad"]),
    ("palm_target_m", [1, 2, {}]),
    ("palm_quat_wxyz", [1, 0, 0, "bad"]),
])
def test_malformed_palm_target_is_a_rejection_without_physics(runtime, field, value):
    time_before = runtime.data.time
    with pytest.raises(InvalidAction):
        runtime.execute({"operation": "move", field: value, "duration_s": 1.0})
    assert runtime.data.time == time_before
