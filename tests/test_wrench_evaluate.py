import math

import pytest

from dexhand_agent.wrench_evaluate import WrenchEvaluator


DT = 0.01
ANGLE = math.pi / 4


def measurement(time_s, **changes):
    sample = {
        "time_s": time_s,
        "wrench_pos_m": [0.0, 0.0, 0.23],
        "wrench_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "hand_tool_contacts": ["thumb", "index"],
        "tool_support_contacts": [],
        "tool_nut_contact": False,
        "head_nut_contact": False,
        "robot_nut_contact": False,
        "nut_angle_rad": 0.0,
        "head_clearance_m": 0.03,
        "max_contact_force_n": 0.1,
        "max_penetration_m": 0.0001,
    }
    sample.update(changes)
    if "tool_nut_contact" not in changes:
        sample["tool_nut_contact"] = sample["head_nut_contact"]
    return sample


def hold(evaluator, start_tick, duration_s, **changes):
    """Supply every measured 10 ms tick, including interval endpoints."""
    final_tick = start_tick + round(duration_s / DT)
    for tick in range(start_tick, final_tick + 1):
        evaluator.update(measurement(tick * DT, **changes))
    return final_tick + 1


def state_for(gate):
    if gate == "turn":
        return {"head_nut_contact": True, "nut_angle_rad": ANGLE,
                "head_clearance_m": 0.0}
    if gate == "withdraw":
        return {"nut_angle_rad": ANGLE}
    return {}


def ready_for(evaluator, gate):
    if gate == "lift":
        return 0
    tick = hold(evaluator, 0, 2.0)
    if gate == "turn":
        return tick
    return hold(evaluator, tick, 0.5, **state_for("turn"))


def complete_flow(evaluator, start_tick=0):
    tick = hold(evaluator, start_tick, 2.0)
    tick = hold(evaluator, tick, 0.5, **state_for("turn"))
    return hold(evaluator, tick, 1.0, **state_for("withdraw"))


def test_ordered_flow_records_physical_measurements_and_gate_times():
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    final_tick = complete_flow(evaluator)

    result = evaluator.summary()
    assert result["task_id"] == "wrench_turn"
    assert result["protocol_version"] == 1
    assert result["success"] is True
    assert result["violations"] == []
    for gate, passed_at, duration in (
        ("lift", 2.0, 2.0), ("turn", 2.51, 0.5), ("withdraw", 3.52, 1.0),
    ):
        assert result["gates"][gate] == {
            "passed": True,
            "passed_at_s": pytest.approx(passed_at),
            "required_continuous_s": duration,
            "best_continuous_s": pytest.approx(duration),
        }
    assert result["metrics"]["samples"] == final_tick
    assert result["metrics"]["max_lift_m"] == pytest.approx(0.03)
    assert result["metrics"]["max_nut_turn_rad"] == pytest.approx(ANGLE)
    assert result["metrics"]["max_head_clearance_m"] == pytest.approx(0.03)
    assert result["metrics"]["max_contact_force_n"] == pytest.approx(0.1)
    assert result["metrics"]["max_penetration_m"] == pytest.approx(0.0001)


def test_empty_episode_is_not_successful_and_has_no_measured_maxima():
    result = WrenchEvaluator(initial_height_m=0.2, initial_nut_angle_rad=0.4).summary()
    assert result["task_id"] == "wrench_turn"
    assert result["success"] is False
    assert result["metrics"]["samples"] == 0
    assert result["metrics"]["last_time_s"] is None
    assert result["metrics"]["initial_height_m"] == 0.2
    assert result["metrics"]["initial_nut_angle_rad"] == 0.4
    assert result["metrics"]["max_lift_m"] is None
    assert result["metrics"]["max_nut_turn_rad"] is None
    assert result["metrics"]["first_robot_nut_contact_s"] is None
    assert all(gate["passed_at_s"] is None for gate in result["gates"].values())
    assert all(gate["best_continuous_s"] == 0.0 for gate in result["gates"].values())


@pytest.mark.parametrize("gate,duration", [("lift", 2.0), ("turn", 0.5), ("withdraw", 1.0)])
def test_each_gate_requires_its_full_continuous_duration(gate, duration):
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    tick = ready_for(evaluator, gate)
    next_tick = hold(evaluator, tick, duration - DT, **state_for(gate))
    assert evaluator.summary()["gates"][gate]["passed"] is False
    evaluator.update(measurement(next_tick * DT, **state_for(gate)))
    assert evaluator.summary()["gates"][gate]["passed"] is True


@pytest.mark.parametrize("gate", ["lift", "turn", "withdraw"])
@pytest.mark.parametrize("contacts", [[], ["index"], ["index", "index"]])
def test_every_gate_requires_two_distinct_measured_hand_contacts(gate, contacts):
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    tick = ready_for(evaluator, gate)
    state = state_for(gate)
    state["hand_tool_contacts"] = contacts
    hold(evaluator, tick, 2.1, **state)
    assert evaluator.summary()["gates"][gate]["passed"] is False


@pytest.mark.parametrize("gate", ["lift", "turn", "withdraw"])
@pytest.mark.parametrize("support", ["rack", "table", "floor"])
def test_world_or_fixture_support_blocks_each_gate(gate, support):
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    tick = ready_for(evaluator, gate)
    state = state_for(gate)
    state["tool_support_contacts"] = [support]
    hold(evaluator, tick, 2.1, **state)
    assert evaluator.summary()["gates"][gate]["passed"] is False


@pytest.mark.parametrize("gate", ["lift", "withdraw"])
@pytest.mark.parametrize("head_contact", [True, False])
def test_any_tool_nut_contact_blocks_airborne_gates(gate, head_contact):
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    tick = ready_for(evaluator, gate)
    state = state_for(gate)
    state.update(tool_nut_contact=True, head_nut_contact=head_contact)
    hold(evaluator, tick, 2.1, **state)
    assert evaluator.summary()["gates"][gate]["passed"] is False


@pytest.mark.parametrize("gate", ["lift", "withdraw"])
def test_head_contact_cannot_be_hidden_by_a_false_any_contact_flag(gate):
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    tick = ready_for(evaluator, gate)
    state = state_for(gate)
    state.update(tool_nut_contact=False, head_nut_contact=True)
    hold(evaluator, tick, 2.1, **state)
    assert evaluator.summary()["gates"][gate]["passed"] is False


@pytest.mark.parametrize("offset,expected", [(0.0, True), (-0.5e-9, True), (-2e-9, False)])
def test_lift_height_boundary_uses_center_relative_to_initial_height(offset, expected):
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    hold(evaluator, 0, 2.0, wrench_pos_m=[0.0, 0.0, 0.22 + offset])
    assert evaluator.summary()["gates"]["lift"]["passed"] is expected


@pytest.mark.parametrize("offset,expected", [(0.0, True), (-0.5e-9, True), (-2e-9, False)])
@pytest.mark.parametrize("gate", ["turn", "withdraw"])
def test_nut_angle_threshold_is_required_for_both_turn_and_withdraw(gate, offset, expected):
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    tick = ready_for(evaluator, gate)
    state = state_for(gate)
    state["nut_angle_rad"] = ANGLE + offset
    hold(evaluator, tick, 1.0, **state)
    assert evaluator.summary()["gates"][gate]["passed"] is expected


@pytest.mark.parametrize("offset,expected", [(0.0, True), (-0.5e-9, True), (-2e-9, False)])
def test_withdrawal_requires_thirty_mm_head_surface_clearance(offset, expected):
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    tick = ready_for(evaluator, "withdraw")
    hold(evaluator, tick, 1.0, nut_angle_rad=ANGLE, head_clearance_m=0.03 + offset)
    assert evaluator.summary()["gates"]["withdraw"]["passed"] is expected


@pytest.mark.parametrize("angle", [-ANGLE, -2 * math.pi + ANGLE, ANGLE - 0.001])
def test_wrong_direction_or_insufficient_nut_turn_cannot_pass(angle):
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    tick = ready_for(evaluator, "turn")
    hold(evaluator, tick, 0.6, head_nut_contact=True, nut_angle_rad=angle)
    assert evaluator.summary()["gates"]["turn"]["passed"] is False


def test_nut_angle_is_relative_to_initial_unwrapped_angle():
    evaluator = WrenchEvaluator(initial_height_m=0.2, initial_nut_angle_rad=6.1)
    tick = hold(evaluator, 0, 2.0, nut_angle_rad=6.1)
    tick = hold(evaluator, tick, 0.5, head_nut_contact=True, nut_angle_rad=6.1 + ANGLE)
    hold(evaluator, tick, 1.0, nut_angle_rad=6.1 + ANGLE)
    assert evaluator.summary()["success"] is True
    assert evaluator.summary()["metrics"]["max_nut_turn_rad"] == pytest.approx(ANGLE)


def test_handle_contact_and_nut_motion_cannot_substitute_for_head_contact():
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    tick = ready_for(evaluator, "turn")
    tick = hold(evaluator, tick, 0.6, tool_nut_contact=True, head_nut_contact=False,
                nut_angle_rad=ANGLE)
    hold(evaluator, tick, 1.1, nut_angle_rad=ANGLE)
    result = evaluator.summary()
    assert result["gates"]["lift"]["passed"] is True
    assert result["gates"]["turn"]["passed"] is False
    assert result["gates"]["withdraw"]["passed"] is False
    assert result["success"] is False


def test_turn_and_clearance_before_their_previous_gates_receive_no_credit():
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    tick = hold(evaluator, 0, 0.6, **state_for("turn"))
    tick = hold(evaluator, tick, 2.0, **state_for("withdraw"))
    result = evaluator.summary()
    assert result["gates"]["lift"]["passed"] is True
    assert result["gates"]["turn"]["best_continuous_s"] == 0.0
    assert result["gates"]["withdraw"]["best_continuous_s"] == 0.0
    tick = hold(evaluator, tick, 0.5, **state_for("turn"))
    tick = hold(evaluator, tick, 0.99, **state_for("withdraw"))
    assert evaluator.summary()["success"] is False
    evaluator.update(measurement(tick * DT, **state_for("withdraw")))
    assert evaluator.summary()["success"] is True


@pytest.mark.parametrize("contact_stage", ["before", "turn", "after"])
def test_robot_nut_contact_is_a_permanent_deduplicated_violation(contact_stage):
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    if contact_stage == "before":
        evaluator.update(measurement(0.0, robot_nut_contact=True))
        evaluator.update(measurement(DT, robot_nut_contact=True))
        first_contact = 0.0
        complete_flow(evaluator, 2)
    elif contact_stage == "turn":
        tick = ready_for(evaluator, "turn")
        first_contact = tick * DT
        tick = hold(evaluator, tick, 0.6, robot_nut_contact=True, **state_for("turn"))
        assert evaluator.summary()["gates"]["turn"]["passed"] is False
        tick = hold(evaluator, tick, 0.5, **state_for("turn"))
        hold(evaluator, tick, 1.0, **state_for("withdraw"))
    else:
        tick = complete_flow(evaluator)
        assert evaluator.summary()["success"] is True
        first_contact = tick * DT
        evaluator.update(measurement(first_contact, robot_nut_contact=True))
        evaluator.update(measurement(first_contact + DT, robot_nut_contact=True))
    result = evaluator.summary()
    assert all(gate["passed"] for gate in result["gates"].values())
    assert result["violations"] == ["robot_nut_contact"]
    assert result["metrics"]["first_robot_nut_contact_s"] == pytest.approx(first_contact)
    assert result["success"] is False


@pytest.mark.parametrize("gate,duration", [("lift", 2.0), ("turn", 0.5), ("withdraw", 1.0)])
def test_drop_and_regrasp_restart_the_current_gate_without_erasing_passed_gates(gate, duration):
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    tick = ready_for(evaluator, gate)
    tick = hold(evaluator, tick, duration * 0.6, **state_for(gate))
    evaluator.update(measurement(tick * DT, hand_tool_contacts=[]))
    tick = hold(evaluator, tick + 1, duration * 0.6, **state_for(gate))
    assert evaluator.summary()["gates"][gate]["passed"] is False
    hold(evaluator, tick, duration * 0.4, **state_for(gate))
    result = evaluator.summary()
    assert result["gates"][gate]["passed"] is True
    if gate != "lift":
        assert result["gates"]["lift"]["passed"] is True


def test_angle_recoil_during_withdrawal_restarts_the_full_hold():
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    tick = ready_for(evaluator, "withdraw")
    tick = hold(evaluator, tick, 0.6, nut_angle_rad=ANGLE)
    evaluator.update(measurement(tick * DT, nut_angle_rad=ANGLE - 0.01))
    tick = hold(evaluator, tick + 1, 0.6, nut_angle_rad=ANGLE)
    assert evaluator.summary()["gates"]["turn"]["passed"] is True
    assert evaluator.summary()["gates"]["withdraw"]["passed"] is False
    hold(evaluator, tick, 0.4, nut_angle_rad=ANGLE)
    assert evaluator.summary()["success"] is True


@pytest.mark.parametrize("gate,duration", [("lift", 2.0), ("turn", 0.5), ("withdraw", 1.0)])
def test_gap_above_twenty_ms_resets_continuity_for_each_gate(gate, duration):
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    tick = ready_for(evaluator, gate)
    tick = hold(evaluator, tick, duration * 0.6, **state_for(gate))
    tick = hold(evaluator, tick + 2, duration * 0.6, **state_for(gate))
    result = evaluator.summary()
    assert result["gates"][gate]["passed"] is False
    assert result["gates"][gate]["best_continuous_s"] == pytest.approx(duration * 0.6)


@pytest.mark.parametrize("gap,expected", [(0.02, True), (0.02 + 0.5e-9, True),
                                               (0.02 + 2e-9, False)])
def test_gap_boundary_absorbs_only_declared_floating_point_epsilon(gap, expected):
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    hold(evaluator, 0, 1.99)
    evaluator.update(measurement(1.99 + gap))
    assert evaluator.summary()["gates"]["lift"]["passed"] is expected


@pytest.mark.parametrize("time_s", [0.0, -DT, math.inf, -math.inf, math.nan])
def test_invalid_timestamps_are_rejected_without_advancing_the_episode(time_s):
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    evaluator.update(measurement(0.0))
    before = evaluator.summary()
    with pytest.raises(ValueError, match="time_s"):
        evaluator.update(measurement(time_s, robot_nut_contact=True))
    assert evaluator.summary() == before


def test_claims_and_phase_labels_cannot_establish_a_physical_gate():
    evaluator = WrenchEvaluator(initial_height_m=0.2)
    hold(evaluator, 0, 4.0, hand_tool_contacts=[], phase="withdraw",
         sourceclaim="success", success=True, gates={"lift": True, "turn": True,
                                                     "withdraw": True})
    assert evaluator.summary()["success"] is False
    assert not any(gate["passed"] for gate in evaluator.summary()["gates"].values())


def test_metrics_report_measured_maxima_independently_of_gate_credit():
    evaluator = WrenchEvaluator(initial_height_m=0.2, initial_nut_angle_rad=0.3)
    evaluator.update(measurement(0.0, wrench_pos_m=[0, 0, 0.25], nut_angle_rad=0.7,
                                 head_clearance_m=0.08, max_contact_force_n=12.0,
                                 max_penetration_m=0.002, hand_tool_contacts=[]))
    evaluator.update(measurement(DT, wrench_pos_m=[0, 0, 0.21], nut_angle_rad=0.2))
    result = evaluator.summary()
    assert result["success"] is False
    assert result["metrics"]["max_lift_m"] == pytest.approx(0.05)
    assert result["metrics"]["max_nut_turn_rad"] == pytest.approx(0.4)
    assert result["metrics"]["max_head_clearance_m"] == pytest.approx(0.08)
    assert result["metrics"]["max_contact_force_n"] == 12.0
    assert result["metrics"]["max_penetration_m"] == pytest.approx(0.002)
