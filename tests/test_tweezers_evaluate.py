import math

import pytest

from dexhand_agent.tweezers_evaluate import TweezersEvaluator


DT = 0.01
DURATIONS = {"pickup": 2.0, "pinch_lift": 1.0, "transport": 0.5, "release": 1.0}


def measurement(time_s, **changes):
    sample = {
        "time_s": time_s,
        "tool_pos_m": [0.0, 0.0, 0.23],
        "object_pos_m": [0.0, 0.0, 0.10],
        "hand_tool_contacts": ["thumb", "index"],
        "tool_support_contacts": [],
        "tool_object_contact": False,
        "object_tip_contacts": [],
        "object_other_tool_contact": False,
        "object_fixture_contacts": [],
        "object_goal_support": False,
        "robot_object_contact": False,
        "object_to_goal_xy_m": 0.10,
        "object_bottom_clearance_m": 0.02,
        "object_speed_m_s": 0.0,
        "tip_object_clearance_m": 0.03,
        "max_contact_force_n": 0.1,
        "max_penetration_m": 0.0001,
    }
    sample.update(changes)
    return sample


def state_for(gate):
    if gate in ("pinch_lift", "transport"):
        state = {
            "object_pos_m": [0.0, 0.0, 0.13],
            "tool_object_contact": True,
            "object_tip_contacts": ["left", "right"],
            "tip_object_clearance_m": 0.0,
        }
        if gate == "transport":
            state["object_to_goal_xy_m"] = 0.01
        return state
    if gate == "release":
        return {"object_pos_m": [0.0, 0.0, 0.11], "object_goal_support": True,
                "object_to_goal_xy_m": 0.01, "object_bottom_clearance_m": 0.0}
    return {}


def hold(evaluator, start_tick, duration_s, **changes):
    """Supply each 10 ms observation, including both interval endpoints."""
    final_tick = start_tick + round(duration_s / DT)
    for tick in range(start_tick, final_tick + 1):
        evaluator.update(measurement(tick * DT, **changes))
    return final_tick + 1


def ready_for(evaluator, gate):
    tick = 0
    for predecessor, duration in DURATIONS.items():
        if predecessor == gate:
            return tick
        tick = hold(evaluator, tick, duration, **state_for(predecessor))
    raise AssertionError(gate)


def complete_flow(evaluator, start_tick=0):
    tick = start_tick
    for gate, duration in DURATIONS.items():
        tick = hold(evaluator, tick, duration, **state_for(gate))
    return tick


def test_ordered_physical_flow_records_gate_times_and_diagnostics():
    evaluator = TweezersEvaluator(0.2, 0.1)
    final_tick = complete_flow(evaluator)
    result = evaluator.summary()
    assert result["task_id"] == "tweezers_transfer"
    assert result["protocol_version"] == 1
    assert result["success"] is True
    assert result["violations"] == []
    for gate, passed_at in (("pickup", 2.0), ("pinch_lift", 3.01),
                            ("transport", 3.52), ("release", 4.53)):
        measured = result["gates"][gate]
        assert measured["passed"] is True
        assert measured["passed_at_s"] == pytest.approx(passed_at)
        assert measured["required_continuous_s"] == DURATIONS[gate]
        assert measured["best_continuous_s"] >= DURATIONS[gate] - 1e-9
        assert not any(key.startswith("_") for key in measured)
    assert result["metrics"]["samples"] == final_tick
    assert result["metrics"]["last_time_s"] == pytest.approx(4.53)
    assert result["metrics"]["max_tool_lift_m"] == pytest.approx(0.03)
    assert result["metrics"]["max_object_lift_m"] == pytest.approx(0.03)
    assert result["metrics"]["min_goal_distance_m"] == pytest.approx(0.01)


def test_empty_episode_has_no_measured_extrema_or_success():
    result = TweezersEvaluator(0.2, 0.1).summary()
    assert result["success"] is False
    assert result["violations"] == []
    assert result["metrics"]["samples"] == 0
    assert result["metrics"]["last_time_s"] is None
    assert result["metrics"]["initial_tool_height_m"] == 0.2
    assert result["metrics"]["initial_object_height_m"] == 0.1
    for key in ("max_tool_lift_m", "max_object_lift_m", "min_goal_distance_m",
                "max_object_bottom_clearance_m", "max_tip_object_clearance_m",
                "max_contact_force_n", "max_penetration_m", "first_robot_object_contact_s"):
        assert result["metrics"][key] is None
    assert all(gate["passed_at_s"] is None for gate in result["gates"].values())
    assert all(gate["best_continuous_s"] == 0.0 for gate in result["gates"].values())


@pytest.mark.parametrize("gate", DURATIONS)
def test_each_gate_requires_its_full_continuous_duration(gate):
    evaluator = TweezersEvaluator(0.2, 0.1)
    tick = ready_for(evaluator, gate)
    tick = hold(evaluator, tick, DURATIONS[gate] - DT, **state_for(gate))
    assert evaluator.summary()["gates"][gate]["passed"] is False
    evaluator.update(measurement(tick * DT, **state_for(gate)))
    assert evaluator.summary()["gates"][gate]["passed"] is True


@pytest.mark.parametrize("gate", DURATIONS)
@pytest.mark.parametrize("contacts", [[], ["index"], ["index", "index"]])
def test_all_gates_require_two_distinct_hand_parts(gate, contacts):
    evaluator = TweezersEvaluator(0.2, 0.1)
    tick = ready_for(evaluator, gate)
    state = state_for(gate)
    state["hand_tool_contacts"] = contacts
    hold(evaluator, tick, DURATIONS[gate], **state)
    assert evaluator.summary()["gates"][gate]["passed"] is False


@pytest.mark.parametrize("gate", DURATIONS)
@pytest.mark.parametrize("support", ["rack", "source", "goal", "floor"])
def test_all_gates_require_tool_free_of_world_or_fixture_support(gate, support):
    evaluator = TweezersEvaluator(0.2, 0.1)
    tick = ready_for(evaluator, gate)
    state = state_for(gate)
    state["tool_support_contacts"] = [support]
    hold(evaluator, tick, DURATIONS[gate], **state)
    assert evaluator.summary()["gates"][gate]["passed"] is False


@pytest.mark.parametrize("gate", ["pickup", "release"])
@pytest.mark.parametrize("contact", [
    {"tool_object_contact": True},
    {"tool_object_contact": False, "object_tip_contacts": ["left"]},
    {"tool_object_contact": False, "object_tip_contacts": ["right"]},
    {"tool_object_contact": False, "object_other_tool_contact": True},
])
def test_pickup_and_release_exclude_every_tool_object_contact(gate, contact):
    evaluator = TweezersEvaluator(0.2, 0.1)
    tick = ready_for(evaluator, gate)
    state = state_for(gate)
    state.update(contact)
    hold(evaluator, tick, DURATIONS[gate], **state)
    assert evaluator.summary()["gates"][gate]["passed"] is False


@pytest.mark.parametrize("gate", ["pinch_lift", "transport"])
@pytest.mark.parametrize("tips", [[], ["left"], ["right"], ["left", "left"],
                                  ["right", "right"], ["left", "handle"]])
def test_two_actual_distinct_tip_sides_are_required(gate, tips):
    evaluator = TweezersEvaluator(0.2, 0.1)
    tick = ready_for(evaluator, gate)
    state = state_for(gate)
    state["object_tip_contacts"] = tips
    hold(evaluator, tick, DURATIONS[gate], **state)
    assert evaluator.summary()["gates"][gate]["passed"] is False


@pytest.mark.parametrize("gate", ["pinch_lift", "transport"])
@pytest.mark.parametrize("support", [
    {"object_other_tool_contact": True},
    {"object_fixture_contacts": ["source"]},
    {"object_fixture_contacts": ["floor"]},
    {"object_goal_support": True},
    {"robot_object_contact": True},
])
def test_only_the_two_tips_may_support_object_during_pinch_and_transport(gate, support):
    evaluator = TweezersEvaluator(0.2, 0.1)
    tick = ready_for(evaluator, gate)
    state = state_for(gate)
    state.update(support)
    hold(evaluator, tick, DURATIONS[gate], **state)
    assert evaluator.summary()["gates"][gate]["passed"] is False


@pytest.mark.parametrize("gate", ["pickup", "pinch_lift"])
@pytest.mark.parametrize("offset,expected", [(0.0, True), (-0.5e-9, True), (-2e-9, False)])
def test_lift_boundaries_use_the_respective_measured_initial_center(gate, offset, expected):
    evaluator = TweezersEvaluator(0.2, 0.1)
    tick = ready_for(evaluator, gate)
    state = state_for(gate)
    key, initial = ("tool_pos_m", 0.2) if gate == "pickup" else ("object_pos_m", 0.1)
    state[key] = [0.0, 0.0, initial + 0.020 + offset]
    hold(evaluator, tick, DURATIONS[gate], **state)
    assert evaluator.summary()["gates"][gate]["passed"] is expected


@pytest.mark.parametrize("gate", ["transport", "release"])
@pytest.mark.parametrize("offset,expected", [(0.0, True), (0.5e-9, True), (2e-9, False)])
def test_goal_horizontal_distance_boundary(gate, offset, expected):
    evaluator = TweezersEvaluator(0.2, 0.1)
    tick = ready_for(evaluator, gate)
    state = state_for(gate)
    state["object_to_goal_xy_m"] = 0.020 + offset
    hold(evaluator, tick, DURATIONS[gate], **state)
    assert evaluator.summary()["gates"][gate]["passed"] is expected


@pytest.mark.parametrize("clearance,expected", [(0.01, True), (0.01 - 0.5e-9, True),
                                                (0.01 - 2e-9, False), (0.0, False)])
def test_transport_requires_clearance_above_destination_top_plane(clearance, expected):
    evaluator = TweezersEvaluator(0.2, 0.1)
    tick = ready_for(evaluator, "transport")
    state = state_for("transport")
    state["object_bottom_clearance_m"] = clearance
    hold(evaluator, tick, 0.5, **state)
    assert evaluator.summary()["gates"]["transport"]["passed"] is expected


def test_transport_uses_destination_clearance_instead_of_initial_lift_height():
    evaluator = TweezersEvaluator(0.2, 0.1)
    tick = ready_for(evaluator, "transport")
    state = state_for("transport")
    state.update(object_pos_m=[0, 0, 0.09], object_bottom_clearance_m=0.01)
    hold(evaluator, tick, 0.5, **state)
    assert evaluator.summary()["gates"]["transport"]["passed"] is True


@pytest.mark.parametrize("state", [
    {"object_goal_support": False},
    {"object_fixture_contacts": ["source"]},
    {"object_fixture_contacts": ["floor"]},
    {"robot_object_contact": True},
])
def test_release_requires_goal_support_and_excludes_other_supports(state):
    evaluator = TweezersEvaluator(0.2, 0.1)
    tick = ready_for(evaluator, "release")
    release = state_for("release")
    release.update(state)
    hold(evaluator, tick, 1.0, **release)
    assert evaluator.summary()["gates"]["release"]["passed"] is False


@pytest.mark.parametrize("speed,expected", [(0.0199999999, True), (0.02, False),
                                            (0.0200000001, False), (0.10, False)])
def test_release_speed_is_strictly_below_twenty_mm_per_second(speed, expected):
    evaluator = TweezersEvaluator(0.2, 0.1)
    tick = ready_for(evaluator, "release")
    release = state_for("release")
    release["object_speed_m_s"] = speed
    hold(evaluator, tick, 1.0, **release)
    assert evaluator.summary()["gates"]["release"]["passed"] is expected


@pytest.mark.parametrize("clearance,expected", [(0.02, True), (0.02 - 0.5e-9, True),
                                                (0.02 - 2e-9, False), (0.0, False)])
def test_release_requires_twenty_mm_tip_to_object_surface_clearance(clearance, expected):
    evaluator = TweezersEvaluator(0.2, 0.1)
    tick = ready_for(evaluator, "release")
    release = state_for("release")
    release.update(tip_object_clearance_m=clearance, tip_gap_m=0.10)
    hold(evaluator, tick, 1.0, **release)
    assert evaluator.summary()["gates"]["release"]["passed"] is expected


def test_predecessor_activity_cannot_precredit_later_gates():
    evaluator = TweezersEvaluator(0.2, 0.1)
    tick = hold(evaluator, 0, 1.0, **state_for("transport"))
    tick = hold(evaluator, tick, 1.0, **state_for("release"))
    assert not any(gate["passed"] for gate in evaluator.summary()["gates"].values())
    assert all(evaluator.summary()["gates"][gate]["best_continuous_s"] == 0.0
               for gate in ("pinch_lift", "transport", "release"))
    complete_flow(evaluator, tick)
    assert evaluator.summary()["success"] is True


def test_pinch_passing_tick_cannot_start_transport_interval():
    evaluator = TweezersEvaluator(0.2, 0.1)
    tick = ready_for(evaluator, "pinch_lift")
    tick = hold(evaluator, tick, 1.0, **state_for("transport"))
    result = evaluator.summary()
    assert result["gates"]["pinch_lift"]["passed"] is True
    assert result["gates"]["transport"]["best_continuous_s"] == 0.0
    tick = hold(evaluator, tick, 0.49, **state_for("transport"))
    assert evaluator.summary()["gates"]["transport"]["passed"] is False
    evaluator.update(measurement(tick * DT, **state_for("transport")))
    assert evaluator.summary()["gates"]["transport"]["passed"] is True


@pytest.mark.parametrize("gate", DURATIONS)
def test_drop_and_regrasp_restart_current_interval_without_erasing_passed_gates(gate):
    evaluator = TweezersEvaluator(0.2, 0.1)
    tick = ready_for(evaluator, gate)
    state = state_for(gate)
    tick = hold(evaluator, tick, DURATIONS[gate] * 0.6, **state)
    evaluator.update(measurement(tick * DT, hand_tool_contacts=[]))
    tick = hold(evaluator, tick + 1, DURATIONS[gate] * 0.6, **state)
    assert evaluator.summary()["gates"][gate]["passed"] is False
    assert evaluator.summary()["gates"][gate]["best_continuous_s"] == pytest.approx(DURATIONS[gate] * 0.6)
    hold(evaluator, tick, DURATIONS[gate] * 0.4, **state)
    result = evaluator.summary()
    assert result["gates"][gate]["passed"] is True
    assert all(result["gates"][name]["passed"] for name in list(DURATIONS)[:list(DURATIONS).index(gate)])


@pytest.mark.parametrize("gate,interruption", [
    ("pinch_lift", {"object_tip_contacts": ["left"]}),
    ("transport", {"object_bottom_clearance_m": 0.005}),
    ("release", {"object_speed_m_s": 0.02}),
    ("release", {"object_goal_support": False}),
    ("release", {"tip_object_clearance_m": 0.0}),
])
def test_a_single_bad_physical_sample_restarts_the_current_gate(gate, interruption):
    evaluator = TweezersEvaluator(0.2, 0.1)
    tick = ready_for(evaluator, gate)
    state = state_for(gate)
    tick = hold(evaluator, tick, DURATIONS[gate] * 0.6, **state)
    invalid = dict(state)
    invalid.update(interruption)
    evaluator.update(measurement(tick * DT, **invalid))
    tick = hold(evaluator, tick + 1, DURATIONS[gate] * 0.6, **state)
    assert evaluator.summary()["gates"][gate]["passed"] is False
    hold(evaluator, tick, DURATIONS[gate] * 0.4, **state)
    assert evaluator.summary()["gates"][gate]["passed"] is True


@pytest.mark.parametrize("gate", DURATIONS)
def test_unobserved_gap_above_twenty_ms_restarts_each_gate(gate):
    evaluator = TweezersEvaluator(0.2, 0.1)
    tick = ready_for(evaluator, gate)
    state = state_for(gate)
    tick = hold(evaluator, tick, DURATIONS[gate] * 0.6, **state)
    hold(evaluator, tick + 2, DURATIONS[gate] * 0.6, **state)
    result = evaluator.summary()
    assert result["gates"][gate]["passed"] is False
    assert result["gates"][gate]["best_continuous_s"] == pytest.approx(DURATIONS[gate] * 0.6)


@pytest.mark.parametrize("gap,expected", [(0.02, True), (0.02 + 0.5e-9, True),
                                        (0.02 + 2e-9, False)])
def test_gap_boundary_absorbs_only_declared_roundoff(gap, expected):
    evaluator = TweezersEvaluator(0.2, 0.1)
    hold(evaluator, 0, 1.99)
    evaluator.update(measurement(1.99 + gap))
    assert evaluator.summary()["gates"]["pickup"]["passed"] is expected


@pytest.mark.parametrize("time_s", [0.0, -DT, math.inf, -math.inf, math.nan])
def test_invalid_timestamps_leave_the_episode_unchanged(time_s):
    evaluator = TweezersEvaluator(0.2, 0.1)
    evaluator.update(measurement(0.0))
    before = evaluator.summary()
    with pytest.raises(ValueError, match="time_s"):
        evaluator.update(measurement(time_s, robot_object_contact=True))
    assert evaluator.summary() == before


@pytest.mark.parametrize("time_s", [math.inf, -math.inf, math.nan])
def test_first_timestamp_must_also_be_finite(time_s):
    evaluator = TweezersEvaluator(0.2, 0.1)
    with pytest.raises(ValueError, match="time_s"):
        evaluator.update(measurement(time_s))
    assert evaluator.summary()["metrics"]["samples"] == 0


@pytest.mark.parametrize("contact_stage", ["before", "pinch_lift", "transport", "release", "after"])
def test_robot_object_contact_is_permanent_and_deduplicated(contact_stage):
    evaluator = TweezersEvaluator(0.2, 0.1)
    if contact_stage == "before":
        evaluator.update(measurement(0.0, robot_object_contact=True))
        evaluator.update(measurement(DT, robot_object_contact=True))
        first_contact = 0.0
        complete_flow(evaluator, 2)
    elif contact_stage == "after":
        tick = complete_flow(evaluator)
        assert evaluator.summary()["success"] is True
        first_contact = tick * DT
        evaluator.update(measurement(first_contact, robot_object_contact=True))
        evaluator.update(measurement(first_contact + DT, robot_object_contact=True))
    else:
        tick = ready_for(evaluator, contact_stage)
        first_contact = tick * DT
        invalid = state_for(contact_stage)
        invalid["robot_object_contact"] = True
        tick = hold(evaluator, tick, DURATIONS[contact_stage], **invalid)
        assert evaluator.summary()["gates"][contact_stage]["passed"] is False
        remaining = list(DURATIONS)[list(DURATIONS).index(contact_stage):]
        for gate in remaining:
            tick = hold(evaluator, tick, DURATIONS[gate], **state_for(gate))
    result = evaluator.summary()
    assert all(gate["passed"] for gate in result["gates"].values())
    assert result["success"] is False
    assert result["violations"] == ["robot_object_contact"]
    assert result["metrics"]["first_robot_object_contact_s"] == pytest.approx(first_contact)


def test_claims_phase_reward_and_quaternions_cannot_establish_a_gate():
    evaluator = TweezersEvaluator(0.2, 0.1)
    hold(evaluator, 0, 5.0, hand_tool_contacts=[], phase="release", sourceclaim="success",
         success=True, reward=100000, gates={gate: True for gate in DURATIONS},
         tool_quat_wxyz=[1, 0, 0, 0], object_quat_wxyz=[1, 0, 0, 0])
    result = evaluator.summary()
    assert result["success"] is False
    assert not any(gate["passed"] for gate in result["gates"].values())


def test_metrics_use_measured_extrema_even_without_gate_credit():
    evaluator = TweezersEvaluator(0.2, 0.1)
    evaluator.update(measurement(0.0, tool_pos_m=[0, 0, 0.25], object_pos_m=[0, 0, 0.14],
                                 hand_tool_contacts=[], object_to_goal_xy_m=0.005,
                                 object_bottom_clearance_m=0.04,
                                 tip_object_clearance_m=0.08, max_contact_force_n=12.0,
                                 max_penetration_m=0.002))
    evaluator.update(measurement(DT, tool_pos_m=[0, 0, 0.21], object_pos_m=[0, 0, 0.08]))
    metrics = evaluator.summary()["metrics"]
    assert metrics["max_tool_lift_m"] == pytest.approx(0.05)
    assert metrics["max_object_lift_m"] == pytest.approx(0.04)
    assert metrics["min_goal_distance_m"] == pytest.approx(0.005)
    assert metrics["max_object_bottom_clearance_m"] == pytest.approx(0.04)
    assert metrics["max_tip_object_clearance_m"] == pytest.approx(0.08)
    assert metrics["max_contact_force_n"] == 12.0
    assert metrics["max_penetration_m"] == pytest.approx(0.002)
    assert evaluator.summary()["success"] is False


def test_signed_lift_maxima_are_not_clamped_to_zero():
    evaluator = TweezersEvaluator(0.2, 0.1)
    evaluator.update(measurement(0.0, tool_pos_m=[0, 0, 0.19], object_pos_m=[0, 0, 0.08]))
    assert evaluator.summary()["metrics"]["max_tool_lift_m"] == pytest.approx(-0.01)
    assert evaluator.summary()["metrics"]["max_object_lift_m"] == pytest.approx(-0.02)


def test_summary_copies_cannot_alter_gate_or_violation_state():
    evaluator = TweezersEvaluator(0.2, 0.1)
    evaluator.update(measurement(0.0, robot_object_contact=True))
    result = evaluator.summary()
    result["gates"]["release"]["passed"] = True
    result["metrics"]["samples"] = 100
    result["violations"].clear()
    fresh = evaluator.summary()
    assert fresh["gates"]["release"]["passed"] is False
    assert fresh["metrics"]["samples"] == 1
    assert fresh["violations"] == ["robot_object_contact"]
