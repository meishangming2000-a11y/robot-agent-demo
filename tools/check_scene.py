"""Step-one load, named-actuator, physical tracking, and render checks."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from PIL import Image

from dexhand_agent.scene import make_model


def main():
    output = Path(__file__).resolve().parents[1] / "runs" / "scene_check"
    output.mkdir(parents=True, exist_ok=True)
    model, data = make_model()
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    initial = data.qpos.copy()
    renderer = mujoco.Renderer(model, height=720, width=960)
    renderer.update_scene(data, camera="overview")
    Image.fromarray(renderer.render()).save(output / "home.png")
    for _ in range(500):
        mujoco.mj_step(model, data)
    settled = data.qpos.copy()
    tool_settled = data.body("hammer").xpos.copy()
    tool_settled_quat = data.body("hammer").xquat.copy()
    actuator = model.actuator("rh_A_FFJ3").id
    data.ctrl[actuator] = 0.35
    for _ in range(500):
        mujoco.mj_step(model, data)
    renderer.update_scene(data, camera="overview")
    Image.fromarray(renderer.render()).save(output / "commanded.png")
    renderer.update_scene(data, camera="front")
    Image.fromarray(renderer.render()).save(output / "front.png")
    renderer.close()
    actuator_checks = []
    for name, target in (("rh_A_WRJ1", 0.1), ("rh_A_FFJ0", 0.8)):
        aid = model.actuator(name).id
        previous = float(data.ctrl[aid])
        data.ctrl[aid] = target
        for _ in range(500):
            mujoco.mj_step(model, data)
        measured = float(data.actuator_length[aid])
        actuator_checks.append({"actuator": name, "target": target, "measured": measured,
                                "error": abs(measured - target)})
        data.ctrl[aid] = previous
        for _ in range(500):
            mujoco.mj_step(model, data)
        actuator_checks[-1]["returned_error"] = abs(float(data.actuator_length[aid]) - previous)
    button_joint = model.joint("button_slide")
    button_rest = float(data.qpos[button_joint.qposadr[0]])
    data.qfrc_applied[button_joint.dofadr[0]] = -3.0
    for _ in range(250):
        mujoco.mj_step(model, data)
    button_pressed = float(data.qpos[button_joint.qposadr[0]])
    data.qfrc_applied[:] = 0
    for _ in range(500):
        mujoco.mj_step(model, data)
    button_returned = float(data.qpos[button_joint.qposadr[0]])
    joint = model.joint("rh_FFJ3").qposadr[0]
    report = {"mujoco": mujoco.__version__, "nq": model.nq, "nv": model.nv, "nu": model.nu,
              "actuators": names, "finite": bool(all(np.isfinite(x).all() for x in (data.qpos, data.qvel, data.ctrl))),
              "arm_settle_error_rad": float(np.max(np.abs(settled[:7] - initial[:7]))),
              "index_target_rad": 0.35, "index_measured_rad": float(data.qpos[joint]),
              "index_tracking_error_rad": float(abs(data.qpos[joint] - 0.35)),
              "tool_settled_position_m": tool_settled.tolist(),
              "tool_drift_m": float(np.linalg.norm(data.body("hammer").xpos - tool_settled)),
              "tool_quaternion_alignment": float(abs(np.dot(data.body("hammer").xquat, tool_settled_quat))),
              "tool_speed_m_s": float(np.linalg.norm(data.qvel[model.joint("hammer_free").dofadr[0]:model.joint("hammer_free").dofadr[0]+3])),
              "named_actuator_checks": actuator_checks,
              "button_no_contact_position_m": button_rest,
              "button_positive_control_position_m": button_pressed,
              "button_returned_position_m": button_returned,
              "palm_position_m": data.site("palm_frame").xpos.tolist(),
              "palm_rotation": data.site("palm_frame").xmat.reshape(3, 3).tolist(),
              "contacts": int(data.ncon), "time_s": data.time}
    report["numeric_pass"] = (report["finite"] and report["index_tracking_error_rad"] < 0.06
                              and report["arm_settle_error_rad"] < 0.12 and model.nu == 27
                              and tool_settled[2] > 0.235 and report["tool_drift_m"] < 0.003
                              and report["tool_quaternion_alignment"] > 0.999
                              and report["tool_speed_m_s"] < 0.001
                              and all(max(row["error"], row["returned_error"]) < 0.06 for row in actuator_checks)
                              and abs(button_rest) < 0.0005 and button_pressed < -0.006
                              and abs(button_returned) < 0.0005)
    (output / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["numeric_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
