"""Evaluate ordered task gates from physical measurements, independently of actions."""

import math


class TaskEvaluator:
    """Accumulate one episode; timestamps must be strictly increasing seconds."""

    PROTOCOL_VERSION = 2
    _DURATIONS = {"lift": 2.0, "press": 0.5, "withdraw": 1.0}
    _EPS = 1e-9  # Only absorb floating point roundoff at the declared boundaries.

    def __init__(self, initial_height_m: float):
        self.initial_height_m = initial_height_m
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
            "initial_height_m": initial_height_m,
            "max_lift_m": None,
            "max_button_depth_m": None,
            "max_head_clearance_m": None,
            "first_robot_button_contact_s": None,
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
        airborne = fixture_free and not sample["tool_button_contact"]
        lift_m = sample["hammer_pos_m"][2] - self.initial_height_m
        button_q_m = sample["button_q_m"]
        robot_contact = sample["robot_button_contact"]
        conditions = {
            "lift": held and airborne and lift_m >= 0.020 - self._EPS,
            "press": (
                held and fixture_free and sample["head_button_contact"]
                and not robot_contact and button_q_m <= -0.008 + self._EPS
            ),
            "withdraw": (
                held and airborne
                and sample["head_clearance_m"] >= 0.030 - self._EPS
                and abs(button_q_m) <= 0.001 + self._EPS
            ),
        }
        # Freeze eligibility before updating gates: a later gate starts next tick.
        enabled = {
            "lift": True,
            "press": self._gates["lift"]["passed"],
            "withdraw": self._gates["press"]["passed"],
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

        if robot_contact and "robot_button_contact" not in self._violations:
            self._violations.append("robot_button_contact")
            self._metrics["first_robot_button_contact_s"] = time_s
        self._metrics["samples"] += 1
        self._metrics["last_time_s"] = time_s
        for key, value in (
            ("max_lift_m", lift_m),
            ("max_button_depth_m", max(0.0, -button_q_m)),
            ("max_head_clearance_m", sample["head_clearance_m"]),
        ):
            previous = self._metrics[key]
            self._metrics[key] = value if previous is None else max(previous, value)

    def summary(self) -> dict:
        return {
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
