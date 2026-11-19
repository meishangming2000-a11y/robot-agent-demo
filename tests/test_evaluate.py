import math

import pytest

from dexhand_agent.evaluate import TaskEvaluator


DT = 0.002


def measurement(time_s, **changes):
    sample = {
        "time_s": time_s,
        "hammer_pos_m": [0.0, 0.0, 0.23],
        "hammer_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "hand_tool_contacts": ["thumb", "index"],
        "tool_support_contacts": [],
        "head_button_contact": False,
        "robot_button_contact": False,
        "button_q_m": 0.0,
        "head_clearance_m": 0.03,
    }
    sample.update(changes)
    if "tool_button_contact" not in changes:
        sample["tool_button_contact"] = sample["head_button_contact"]
    return sample


def hold(evaluator, start_tick, duration_s, **changes):
    """Supply every measured 2 ms tick, including both interval endpoints."""
    final_tick = start_tick + round(duration_s / DT)
    for tick in range(start_tick, final_tick + 1):
        evaluator.update(measurement(tick * DT, **changes))
    return final_tick + 1


def complete_flow(evaluator, start_tick=0):
    tick = hold(evaluator, start_tick, 2.0)
    tick = hold(
        evaluator, tick, 0.5, head_button_contact=True, button_q_m=-0.008,
        head_clearance_m=-0.008,
    )
    return hold(evaluator, tick, 1.0)


def test_ordered_physical_flow_records_pass_times_and_measured_durations():
    evaluator = TaskEvaluator(initial_height_m=0.2)
    final_tick = complete_flow(evaluator)

    result = evaluator.summary()
    assert result["success"] is True
    assert result["violations"] == []
    assert result["gates"]["lift"]["passed_at_s"] == pytest.approx(2.0)
    assert result["gates"]["press"]["passed_at_s"] == pytest.approx(2.502)
    assert result["gates"]["withdraw"]["passed_at_s"] == pytest.approx(3.504)
    assert result["gates"]["lift"]["best_continuous_s"] == pytest.approx(2.0)
    assert result["gates"]["press"]["best_continuous_s"] == pytest.approx(0.5)
    assert result["gates"]["withdraw"]["best_continuous_s"] == pytest.approx(1.0)
    assert all(gate["passed"] for gate in result["gates"].values())
    assert result["metrics"]["samples"] == final_tick
    assert result["metrics"]["max_lift_m"] == pytest.approx(0.03)
    assert result["metrics"]["max_button_depth_m"] == pytest.approx(0.008)


@pytest.mark.parametrize("head_contact", [True, False])
def test_button_support_cannot_establish_airborne_hold(head_contact):
    evaluator = TaskEvaluator(initial_height_m=0.2)
    tick = hold(evaluator, 0, 3.0, tool_button_contact=True,
                head_button_contact=head_contact, button_q_m=-0.008)
    tick = hold(evaluator, tick, 0.5, tool_button_contact=True,
                head_button_contact=True, button_q_m=-0.008)
    hold(evaluator, tick, 1.0)
    assert not evaluator.summary()["success"]
    assert not evaluator.summary()["gates"]["lift"]["passed"]


@pytest.mark.parametrize(
    "changes",
    [
        {"hand_tool_contacts": []},
        {"hand_tool_contacts": ["index", "index"]},
        {"tool_support_contacts": ["rack_head"]},
        {"tool_support_contacts": ["table"]},
        {"tool_support_contacts": ["floor"]},
        {"hammer_pos_m": [0.0, 0.0, 0.219]},
    ],
)
def test_height_or_button_motion_without_an_unsupported_grasp_cannot_pass(changes):
    evaluator = TaskEvaluator(initial_height_m=0.2)
    hold(
        evaluator, 0, 4.0, head_button_contact=True, button_q_m=-0.012,
        **changes,
    )

    result = evaluator.summary()
    assert result["success"] is False
    assert not any(gate["passed"] for gate in result["gates"].values())
    assert result["gates"]["lift"]["best_continuous_s"] == 0.0


@pytest.mark.parametrize(
    "changes",
    [
        {"hand_tool_contacts": []},
        {"hand_tool_contacts": ["palm"]},
        {"tool_support_contacts": ["rack_left"]},
        {"tool_support_contacts": ["table"]},
        {"head_button_contact": False},
        {"button_q_m": -0.0079},
    ],
)
def test_dropped_or_supported_tool_and_unrelated_button_motion_cannot_press(changes):
    evaluator = TaskEvaluator(initial_height_m=0.2)
    tick = hold(evaluator, 0, 2.0)
    pressing = {"head_button_contact": True, "button_q_m": -0.008}
    pressing.update(changes)
    tick = hold(evaluator, tick, 0.6, **pressing)
    hold(evaluator, tick, 1.1)

    result = evaluator.summary()
    assert result["gates"]["lift"]["passed"] is True
    assert result["gates"]["press"]["passed"] is False
    assert result["gates"]["withdraw"]["passed"] is False
    assert result["success"] is False


def test_one_robot_button_contact_permanently_prevents_success():
    evaluator = TaskEvaluator(initial_height_m=0.2)
    evaluator.update(measurement(0.0, robot_button_contact=True))
    evaluator.update(measurement(DT, robot_button_contact=True))
    complete_flow(evaluator, start_tick=2)

    result = evaluator.summary()
    assert all(gate["passed"] for gate in result["gates"].values())
    assert result["violations"] == ["robot_button_contact"]
    assert result["success"] is False


def test_robot_contact_after_completed_gates_also_invalidates_the_episode():
    evaluator = TaskEvaluator(initial_height_m=0.2)
    tick = complete_flow(evaluator)
    assert evaluator.summary()["success"] is True

    evaluator.update(measurement(tick * DT, robot_button_contact=True))
    assert evaluator.summary()["success"] is False


def test_a_robot_press_is_not_a_held_hammer_press():
    evaluator = TaskEvaluator(initial_height_m=0.2)
    tick = hold(evaluator, 0, 2.0)
    tick = hold(
        evaluator, tick, 0.6, head_button_contact=True,
        robot_button_contact=True, button_q_m=-0.012,
    )
    hold(evaluator, tick, 1.1)

    result = evaluator.summary()
    assert result["gates"]["press"]["passed"] is False
    assert result["violations"] == ["robot_button_contact"]
    assert result["success"] is False


def test_earlier_press_and_withdrawal_do_not_count_before_the_previous_gate():
    evaluator = TaskEvaluator(initial_height_m=0.2)
    tick = hold(evaluator, 0, 0.6, head_button_contact=True, button_q_m=-0.012)
    tick = hold(evaluator, tick, 1.1)
    assert evaluator.summary()["gates"]["press"]["best_continuous_s"] == 0.0
    tick = hold(evaluator, tick, 1.1)
    result = evaluator.summary()
    assert result["gates"]["lift"]["passed"] is True
    assert result["gates"]["press"]["passed"] is False
    assert result["gates"]["withdraw"]["best_continuous_s"] == 0.0

    tick = hold(evaluator, tick, 0.5, head_button_contact=True, button_q_m=-0.008)
    hold(evaluator, tick, 0.998)
    assert evaluator.summary()["success"] is False
    evaluator.update(measurement((tick + 500) * DT))
    assert evaluator.summary()["success"] is True


def test_interrupted_holds_do_not_add_up_to_a_continuous_lift():
    evaluator = TaskEvaluator(initial_height_m=0.2)
    tick = hold(evaluator, 0, 1.2)
    evaluator.update(measurement(tick * DT, hand_tool_contacts=["thumb"]))
    tick = hold(evaluator, tick + 1, 1.2)

    result = evaluator.summary()
    assert result["gates"]["lift"]["passed"] is False
    assert result["gates"]["lift"]["best_continuous_s"] == pytest.approx(1.2)
    assert result["success"] is False


def test_a_sparse_gap_restarts_the_lift_interval_instead_of_filling_missing_ticks():
    evaluator = TaskEvaluator(initial_height_m=0.2)
    hold(evaluator, 0, 1.2)
    hold(evaluator, 1500, 1.2)

    result = evaluator.summary()
    assert result["gates"]["lift"]["passed"] is False
    assert result["gates"]["lift"]["best_continuous_s"] == pytest.approx(1.2)
    evaluator.update(measurement(10.0))
    assert evaluator.summary()["gates"]["lift"]["passed"] is False


def test_press_interruption_and_withdrawal_gap_each_restart_the_active_interval():
    evaluator = TaskEvaluator(initial_height_m=0.2)
    tick = hold(evaluator, 0, 2.0)
    tick = hold(evaluator, tick, 0.3, head_button_contact=True, button_q_m=-0.008)
    evaluator.update(measurement(tick * DT, head_button_contact=False))
    tick = hold(evaluator, tick + 1, 0.3, head_button_contact=True, button_q_m=-0.008)
    assert evaluator.summary()["gates"]["press"]["passed"] is False
    assert evaluator.summary()["gates"]["press"]["best_continuous_s"] == pytest.approx(0.3)
    tick = hold(evaluator, tick, 0.2, head_button_contact=True, button_q_m=-0.008)
    tick = hold(evaluator, tick, 0.6)
    tick = hold(evaluator, tick + 20, 0.6)
    assert evaluator.summary()["gates"]["withdraw"]["passed"] is False
    assert evaluator.summary()["gates"]["withdraw"]["best_continuous_s"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    "changes",
    [
        {"hand_tool_contacts": ["index"]},
        {"tool_support_contacts": ["rack_right"]},
        {"tool_button_contact": True},
        {"head_clearance_m": 0.0299},
        {"button_q_m": -0.0011},
        {"button_q_m": 0.0011},
    ],
)
def test_withdrawal_requires_grasp_clearance_and_button_return_together(changes):
    evaluator = TaskEvaluator(initial_height_m=0.2)
    tick = hold(evaluator, 0, 2.0)
    tick = hold(evaluator, tick, 0.5, head_button_contact=True, button_q_m=-0.008)
    hold(evaluator, tick, 1.1, **changes)

    result = evaluator.summary()
    assert result["gates"]["press"]["passed"] is True
    assert result["gates"]["withdraw"]["passed"] is False
    assert result["success"] is False


def test_declared_boundary_values_pass_with_floating_point_tick_times():
    evaluator = TaskEvaluator(initial_height_m=0.2)
    tick = hold(evaluator, 0, 2.0, hammer_pos_m=[0, 0, 0.22])
    tick = hold(evaluator, tick, 0.5, head_button_contact=True, button_q_m=-0.008)
    hold(evaluator, tick, 1.0, head_clearance_m=0.03, button_q_m=0.001)
    assert evaluator.summary()["success"] is True


@pytest.mark.parametrize("time_s", [0.0, -0.002, math.inf, math.nan])
def test_non_increasing_or_non_finite_timestamps_are_rejected_without_counting(time_s):
    evaluator = TaskEvaluator(initial_height_m=0.2)
    evaluator.update(measurement(0.0))

    with pytest.raises(ValueError, match="time_s"):
        evaluator.update(measurement(time_s))
    result = evaluator.summary()
    assert result["metrics"]["samples"] == 1
    assert result["metrics"]["last_time_s"] == 0.0


def test_summary_before_any_measurements_has_no_success_or_measured_duration():
    result = TaskEvaluator(initial_height_m=0.2).summary()
    assert result["success"] is False
    assert result["metrics"]["samples"] == 0
    assert result["metrics"]["last_time_s"] is None
    assert all(gate["passed_at_s"] is None for gate in result["gates"].values())
    assert all(gate["best_continuous_s"] == 0.0 for gate in result["gates"].values())
