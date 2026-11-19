# Demonstration records and future training boundary

## Online-agent records

Run the procedure in the [agent skill](../.agents/skills/dexhand-tool-use/SKILL.md).
Each decision request contains the language goal, simulator timestamp, measured
robot/tool state, permitted experience and action schema, with two fresh PNGs.
The reply names the exact request and one bounded action. Actual execution and
independent task gates are recorded separately. Wrench intents and explicit local
skill calls distinguish online choices from local feedback actions.

Boundary joint/actuator observations are not a dense qpos/qvel/ctrl trajectory.
The high-level palm/IK command is not the control applied at every physics tick.
`samples.jsonl` contains physical task measurements, not full robot-state recovery.
Preserve rejected actions, interruptions, failed outcomes and source provenance.

## Synchronized engineering-reference collector

This separate command runs calibrated reference policies, not an online agent:

```sh
.venv/Scripts/python.exe tools/collect_learning_data.py --task all --seed 0 --output runs/reference-data-001
.venv/Scripts/python.exe tools/check_learning_data.py --episode runs/reference-data-001/wrench_turn --output runs/reference-data-001/wrench-integrity.json
```

It saves `manifest.json`, `samples.npz` and `model.mjb`. The manifest declares the
task goal, seed, actor origin, camera, named joints/actuators, units and source
hashes. Arrays include synchronized RGB, qpos, qvel, ctrl, integration state,
action boundaries and applied tick controls. The checker verifies hashes,
alignment and sampled physical continuation from the saved model/state.
Its `accepted` flag establishes record integrity, not expert quality.

Use a new output directory each time. The default seed is a known development
condition. Declared train/validation/final partitions require a study id, but
those labels alone do not establish independent splits or generalization.

## Toward VLA data

Student observations would normally include RGB, language and allowed robot
proprioception; full scene/object state is privileged teacher information.
Training targets must be the declared actuator/action representation at the
chosen rate, not an unlabelled mixture of palm goals and applied controls.

Before calling these expert demonstrations, define task and contact-quality
acceptance, group splits by scene/episode, filter invalid/truncated transitions,
retain failures, and validate the trained policy in independent rollouts.
No VLA training, automatic quality filtering or dense native-agent recorder is
included in this initial release.
