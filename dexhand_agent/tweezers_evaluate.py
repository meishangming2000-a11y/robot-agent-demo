"""Evaluate ordered tweezers transfer gates from measured physics only."""

import math


class TweezersEvaluator:
    """Accumulate one episode; timestamps must be strictly increasing seconds."""

    PROTOCOL_VERSION = 1
    _DURATIONS = {"pickup": 2.0, "pinch_lift": 1.0, "transport": 0.5, "release": 1.0}
    _EPS = 1e-9  # Only absorb floating point roundoff at inclusive boundaries.

    def __init__(self, initial_tool_height_m: float, initial_object_height_m: float):
        self.initial_tool_height_m = initial_tool_height_m
        self.initial_object_height_m = initial_object_height_m
        self._gates = {
            name: {
                "passed": False,
                "passed_at_s": None,
                "required_continuous_s": duration,
                "best_continuous_s": 0.0,
                "_started_at_s": None,
            }
            for name, duration in self._DURATIONS.items()
        }
        self._violations = []
        self._metrics = {
            "samples": 0,
            "last_time_s": None,
            "initial_tool_height_m": initial_tool_height_m,
            "initial_object_height_m": initial_object_height_m,
            "max_tool_lift_m": None,
            "max_object_lift_m": None,
            "min_goal_distance_m": None,
            "max_object_bottom_clearance_m": None,
            "max_tip_object_clearance_m": None,
            "max_contact_force_n": None,
            "max_penetration_m": None,
            "first_robot_object_contact_s": None,
        }

    def update(self, sample: dict) -> None:
        time_s = float(sample["time_s"])
        previous_time = self._metrics["last_time_s"]
        if not math.isfinite(time_s) or (
            previous_time is not None and time_s <= previous_time
        ):
            raise ValueError("time_s must be finite and strictly increasing")
        gap = previous_time is not None and time_s - previous_time > 0.02 + self._EPS

        held = len(set(sample["hand_tool_contacts"])) >= 2
        fixture_free = not sample["tool_support_contacts"]
        tip_contacts = set(sample["object_tip_contacts"])
        both_tips = {"left", "right"}.issubset(tip_contacts)
        object_tool_free = not (
            sample["tool_object_contact"] or tip_contacts or sample["object_other_tool_contact"]
        )
        robot_contact = sample["robot_object_contact"]
        object_world_free = not (
            sample["object_fixture_contacts"] or sample["object_goal_support"]
        )
        pinch_only = (
            both_tips and not sample["object_other_tool_contact"]
            and object_world_free and not robot_contact
        )
        tool_lift_m = sample["tool_pos_m"][2] - self.initial_tool_height_m
        object_lift_m = sample["object_pos_m"][2] - self.initial_object_height_m
        at_goal = sample["object_to_goal_xy_m"] <= 0.020 + self._EPS
        conditions = {
            "pickup": (
                held and fixture_free and object_tool_free
                and tool_lift_m >= 0.020 - self._EPS
            ),
            "pinch_lift": (
                held and fixture_free and pinch_only
                and object_lift_m >= 0.020 - self._EPS
            ),
            "transport": (
                held and fixture_free and pinch_only and at_goal
                and sample["object_bottom_clearance_m"] >= 0.010 - self._EPS
            ),
            "release": (
                held and fixture_free and object_tool_free and not robot_contact
                and sample["object_goal_support"] and not sample["object_fixture_contacts"]
                and at_goal and sample["object_speed_m_s"] < 0.020
                and sample["tip_object_clearance_m"] >= 0.020 - self._EPS
            ),
        }
        # Freeze eligibility first: a newly passed gate enables the next tick only.
        enabled = {
            "pickup": True,
            "pinch_lift": self._gates["pickup"]["passed"],
            "transport": self._gates["pinch_lift"]["passed"],
            "release": self._gates["transport"]["passed"],
        }
        for name, gate in self._gates.items():
            if gap or not enabled[name] or not conditions[name]:
                gate["_started_at_s"] = None
            if not enabled[name] or not conditions[name]:
                continue
            if gate["_started_at_s"] is None:
                gate["_started_at_s"] = time_s
            continuous_s = time_s - gate["_started_at_s"]
            gate["best_continuous_s"] = max(gate["best_continuous_s"], continuous_s)
            if not gate["passed"] and continuous_s + self._EPS >= self._DURATIONS[name]:
                gate["passed"] = True
                gate["passed_at_s"] = time_s

        if robot_contact and "robot_object_contact" not in self._violations:
            self._violations.append("robot_object_contact")
            self._metrics["first_robot_object_contact_s"] = time_s
        self._metrics["samples"] += 1
        self._metrics["last_time_s"] = time_s
        for key, value in (
            ("max_tool_lift_m", tool_lift_m),
            ("max_object_lift_m", object_lift_m),
            ("max_object_bottom_clearance_m", sample["object_bottom_clearance_m"]),
            ("max_tip_object_clearance_m", sample["tip_object_clearance_m"]),
            ("max_contact_force_n", sample["max_contact_force_n"]),
            ("max_penetration_m", sample["max_penetration_m"]),
        ):
            previous = self._metrics[key]
            self._metrics[key] = value if previous is None else max(previous, value)
        previous_distance = self._metrics["min_goal_distance_m"]
        distance = sample["object_to_goal_xy_m"]
        self._metrics["min_goal_distance_m"] = (
            distance if previous_distance is None else min(previous_distance, distance)
        )

    def summary(self) -> dict:
        return {
            "task_id": "tweezers_transfer",
            "protocol_version": self.PROTOCOL_VERSION,
            "success": all(gate["passed"] for gate in self._gates.values())
            and not self._violations,
            "gates": {
                name: {key: value for key, value in gate.items() if not key.startswith("_")}
                for name, gate in self._gates.items()
            },
            "violations": list(self._violations),
            "metrics": dict(self._metrics),
        }
