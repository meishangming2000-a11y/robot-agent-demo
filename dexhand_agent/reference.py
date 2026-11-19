"""Closed-loop engineering reference, not an online model policy.

The grasp is calibrated for the declared hammer setup from development run 007.
Decisions use only the supplied public observation, never runtime/evaluator state.
The current action protocol uses named actuator/value entries in targets.
"""


def _action(stage, duration_s, *, palm_target_m=None, palm_quat_wxyz=None, targets=None):
    operation = "move" if palm_target_m is not None or targets else "hold"
    if stage == "finish":
        operation = "finish"
    return {
        "operation": operation,
        "targets": [
            {"actuator": name, "value": value}
            for name, value in (targets or {}).items()
        ],
        "palm_target_m": palm_target_m,
        "palm_quat_wxyz": palm_quat_wxyz,
        "duration_s": duration_s,
        "memory": f"Engineering reference (calibrated hammer setup): {stage}",
    }


class ReferencePolicy:
    def __init__(self):
        self._phase = "approach"
        self._grasp = None
        self._adjustments = 0

    def decide(self, observation: dict) -> dict:
        if self._phase == "approach":
            tool = observation["tool"]["position_m"]
            self._grasp = [tool[0], tool[1] + 0.025, tool[2] + 0.082]
            target = self._grasp.copy()
            target[2] += 0.09
            self._phase = "descend"
            return _action("approach", 1.8, palm_target_m=target,
                           palm_quat_wxyz=[0, 0, 1, 0])

        if self._phase == "descend":
            self._phase = "close"
            return _action("descend", 1.2, palm_target_m=self._grasp.copy())

        if self._phase == "close":
            targets = {}
            for finger in ("FF", "MF", "RF", "LF"):
                targets[f"rh_A_{finger}J3"] = 0.8
                targets[f"rh_A_{finger}J0"] = 1.5
            targets.update({
                "rh_A_THJ5": 0.9763313232421871,
                "rh_A_THJ4": 0.9253804236340059,
                "rh_A_THJ3": 0.0,
                "rh_A_THJ2": -3.4637906367758015e-05,
                "rh_A_THJ1": 0.45,
            })
            self._phase = "lift"
            return _action("close", 1.5, targets=targets)

        if self._phase == "lift":
            target = self._grasp.copy()
            target[2] += 0.09
            self._phase = "lift_hold"
            return _action("lift", 1.5, palm_target_m=target)

        if self._phase == "lift_hold":
            self._phase = "align"
            return _action("hold", 2.2)

        if self._phase == "align":
            target = list(observation["robot"]["palm_position_m"])
            head = observation["tool"]["head_center_m"]
            button = observation["button"]["position_m"]
            for axis in (0, 1):
                target[axis] += button[axis] - head[axis]
            self._phase = "press"
            return _action("align", 2.6, palm_target_m=target)

        if self._phase == "press":
            target = list(observation["robot"]["palm_position_m"])
            target[2] -= observation["contacts"]["head_clearance_m"] + 0.011
            self._phase = "pressed_hold"
            return _action("press", 1.2, palm_target_m=target)

        if self._phase == "pressed_hold":
            self._phase = "pressure_adjust"
            return _action("pressed_hold", 1.2)

        if self._phase == "pressure_adjust":
            depth = -observation["contacts"]["button_q_m"]
            if depth < 0.0082 and self._adjustments < 4:
                target = list(observation["robot"]["palm_position_m"])
                target[2] -= min(0.006, 0.0105 - depth)
                stage = f"pressure_adjust_{self._adjustments}"
                self._adjustments += 1
                self._phase = "pressure_hold"
                return _action(stage, 0.7, palm_target_m=target)
            self._phase = "withdraw"

        if self._phase == "pressure_hold":
            self._phase = "pressure_adjust"
            return _action(f"pressure_hold_{self._adjustments - 1}", 0.7)

        if self._phase == "withdraw":
            target = list(observation["robot"]["palm_position_m"])
            target[2] += 0.14
            self._phase = "withdraw_hold"
            return _action("withdraw", 1.2, palm_target_m=target)

        if self._phase == "withdraw_hold":
            self._phase = "finish"
            return _action("withdraw_hold", 1.2)

        return _action("finish", 0.02)
