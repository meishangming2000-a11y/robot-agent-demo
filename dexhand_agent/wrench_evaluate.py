"""Evaluate ordered wrench gates from measured physics, independently of actions."""

import math


class WrenchEvaluator:
    """Accumulate one episode; timestamps must be strictly increasing seconds."""

    PROTOCOL_VERSION = 1
    _DURATIONS = {"lift": 2.0, "turn": 0.5, "withdraw": 1.0}
    _EPS = 1e-9  # Only absorb floating point roundoff at declared boundaries.

    def __init__(self, initial_height_m: float, initial_nut_angle_rad: float = 0.0):
        self.initial_height_m = initial_height_m
        self.initial_nut_angle_rad = initial_nut_angle_rad
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
            "initial_nut_angle_rad": initial_nut_angle_rad,
            "max_lift_m": None,
            "max_nut_turn_rad": None,
            "max_head_clearance_m": None,
            "max_contact_force_n": None,
            "max_penetration_m": None,
            "first_robot_nut_contact_s": None,
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
        head_contact = sample["head_nut_contact"]
        airborne = fixture_free and not (sample["tool_nut_contact"] or head_contact)
        lift_m = sample["wrench_pos_m"][2] - self.initial_height_m
        nut_turn_rad = sample["nut_angle_rad"] - self.initial_nut_angle_rad
        turned = nut_turn_rad >= math.pi / 4 - self._EPS
        robot_contact = sample["robot_nut_contact"]
        conditions = {
            "lift": held and airborne and lift_m >= 0.020 - self._EPS,
            "turn": (
                held and fixture_free and head_contact and not robot_contact and turned
            ),
            "withdraw": (
                held and airborne and turned
                and sample["head_clearance_m"] >= 0.030 - self._EPS
            ),
        }
        # Freeze eligibility before updating gates: a later gate starts next tick.
        enabled = {
            "lift": True,
            "turn": self._gates["lift"]["passed"],
            "withdraw": self._gates["turn"]["passed"],
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

        if robot_contact and "robot_nut_contact" not in self._violations:
            self._violations.append("robot_nut_contact")
            self._metrics["first_robot_nut_contact_s"] = time_s
        self._metrics["samples"] += 1
        self._metrics["last_time_s"] = time_s
        for key, value in (
            ("max_lift_m", lift_m),
            ("max_nut_turn_rad", nut_turn_rad),
            ("max_head_clearance_m", sample["head_clearance_m"]),
            ("max_contact_force_n", sample["max_contact_force_n"]),
            ("max_penetration_m", sample["max_penetration_m"]),
        ):
            previous = self._metrics[key]
            self._metrics[key] = value if previous is None else max(previous, value)

    def summary(self) -> dict:
        return {
            "task_id": "wrench_turn",
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
