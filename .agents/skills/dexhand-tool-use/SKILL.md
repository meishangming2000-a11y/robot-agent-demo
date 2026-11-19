---
name: dexhand-tool-use
description: Operate the simulated Panda and Shadow dexterous hand in this repository, using fresh observations, bounded robot actions, and measured task outcomes. Use for autonomous tool trials and reviewing or reusing their experience.
---

Use the local simulation runtime to carry out the user's tool goal. The runtime
accepts robot controls; physical task gates are evaluated separately.

For a request to review or organize existing experience, read only the relevant
existing records; do not launch quickstart, an episode or training. For a
prepare-only request, inspect the environment and published help and return the
planned command without executing it. Quickstart also runs a reference trial;
it is not a setup-only action.

Choose the requested `--task`: `hammer_button`, `wrench_turn`, or
`tweezers_transfer`; omitting it selects the hammer. For environment setup and
a reference result use `python quickstart.py --task <task>`. This runs an
engineered policy. A model export alone does not establish task success.

For the selected physical contract read only the relevant source-project file:
`docs/task-contract.md`, `docs/wrench-contract.md`, or
`docs/tweezers-contract.md`. Use the current request's goal and progress gates.
Use the current request's actual tool geometry for pose calculations; reference
shape and mass examples describe the nominal tool. Keep the assigned physical
and quality criteria.

After quickstart prepares `.venv`, use that environment's Python for online
operation: `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on Unix.
Start `<project-python> -u -m dexhand_agent agent --task <task> --output runs/<new-name>`,
replacing `<project-python>` with the appropriate path, as a background process
using the host's supported terminal tool. Use a new
directory so old observations and actions cannot be reused accidentally.
To supply existing records explicitly, repeat `--experience <record.json>` in
the intended order.
Every completed run exports `experience.json` with its outcome, action trace,
initial coordinate references, and source hashes. A successful source episode
does not verify its applicability to the next episode. Adapt from fresh measured
poses and contacts; do not assume world-coordinate targets transfer unchanged.
The default decision budget is 32 for hammer and wrench, 48 for tweezers;
the request records the actual limit. These different budgets do not constitute
a comparison of task difficulty or agent performance.
For the published autonomous wrench workflow, read these shared instructions
before choosing actions: `experiments/wrench-native-actor-v3.md`,
`experiments/wrench-memory-actor.md`, `experiments/wrench-grip-v2-actor.md`, and
`experiments/wrench-transport-v3-actor.md`. They define the ordered source
experience, fresh-observation intents, explicit withdrawal-program selection,
handoff and separate online/local counts. When an episode is already assigned,
use its declared inputs and budgets rather than launching a second runtime.

For a new wrench episode, replace the generic launch above with:

```sh
<project-python> -u -m dexhand_agent agent --task wrench_turn --output runs/<new-name> --max-decisions 160 --quality-guard --rich-feedback --experience evidence/kit-suite-v1/wrench_turn/experience.json --experience experiences/wrench-native-v2.json --experience experiences/wrench-transport-v3.json
```

When directly operating an ordinary user-requested episode, you fill the actor
role. In a supervised evaluation, the separate coordinator launches and audits
while the assigned actor chooses actions. Preserve that ownership and do not
claim an independent evaluation when you performed both roles.

This bundle declares at most 48 online choices, including skill selections and
finish; 160 bounds all robot commands. It does not change an earlier evaluation
protocol's limits. The supplied native and engineering lessons are guidance;
use measured frames and inspect their applicability. The three published pose,
withdrawal and bridge resources are programs, separate from the optional small
trained parameter adapter. Follow the four instructions' ownership and logging
rules and keep failed actions. After summary exists, retain the continuous video,
physical/trace checks, source references and actual counts. In a supervised
evaluation the coordinator performs the independent audits after the actor
reports its raw result. A local
program handoff is an observation to check, not a success certificate.

For a completed native wrench evaluation, the published checks are
`tools/audit_wrench_autonomy.py <episode> experiments/wrench-autonomy-protocol-v3.json --output <episode>/audit-physical.json`
and `tools/audit_native_skill.py <episode> --source-freeze <declared-source-freeze.json> --max-online-choices 48 --max-commands 160 --output <episode>/audit-local.json`.
Use the source snapshot declared before the run; keep historical freezes intact.
The physical report separates task/quality success from a demonstrated recovery;
a smooth task success can have no recovery event. Inspect the report rather
than treating that alone as task failure. See `docs/wrench-autonomy.md` for the
source-memory checks and retained evaluation evidence.

Keep this process alive while completing the following decision loop:

1. Read `runs/<new-name>/actor/current.json`. When waiting, read its
   `request_path`, including the goal, observation, permitted experience, action
   schema, recent actions, and the two image files listed in `images`.
   Protocol JSON is UTF-8; specify `encoding="utf-8"` when reading it in Python.
   Read JSON directly from the file; do not paste printed request contents into
   another Python/JavaScript string literal.
2. For each online choice, inspect the fresh request and both images in a
   read-only tool call and wait for its result before choosing or publishing an
   action. Publish in a separate later call. Emitting images and writing a reply
   in one call leaves visual review until after the action. The explicitly
   selected local withdrawal program instead follows its published state-feedback
   loop and returns control at handoff. The scene supplies simulator poses;
   report the run as state-assisted.
3. Select one action and atomically write its `reply_path` as
   `{"request_id":"the exact current id","action":{...}}`.
4. Wait for a new request id. Continue from the new measured state and action
   result, including rejections. Do not end the Codex turn while the episode is
   still waiting for decisions. Finish when gates have passed or further action
   is no longer justified within the trial budget. Then inspect `summary.json`.

Actuator targets are a list of `{"actuator":"name","value":number}` entries.
Omitted actuators retain their commanded targets. `palm_target_m` is a world
position; `palm_quat_wxyz` is an optional unit quaternion. Palm IK changes the
Panda arm; it is not a tool trajectory or an object-state command. Use either
arm actuator targets or a palm pose per action. Duration is simulated time;
physics pauses while you deliberate. Respect ranges and the 0.18 m/s palm
translation limit, and use shorter steps near uncertain contact.

Shadow FF/MF/RF/LF J3 actuators flex proximal fingers, J4 spread them, and J0
commands the sum of the middle and distal joint angles. LFJ5 moves the little
metacarpal. THJ5..THJ1 actuate the thumb; WRJ1/2 actuate the wrist. Public
observations distinguish measured joints, actuator coordinates, and commanded
targets. A completed command is not proof of a usable grasp. Check actual
tool retention, support contact, working-end contact, and target response.

For an evaluation trial, use only its published request, images, declared
experience, and your own earlier decisions. Do not inspect the engineered
reference policy, other episodes, private samples, or change source/physics to
obtain a successful score. Do not send scripted reply sequences in advance.
Each action must follow the current observation and exact request id.

Retain the episode's inputs, actions, physical measurements, and unsuccessful
attempts. Experience extracted from a run is a candidate until another run
verifies its applicability. Distinguish hand-authored reference guidance,
agent-proposed lessons, and experimentally verified entries. Claim only the
physical gates and conditions that the recorded evaluator actually establishes.
