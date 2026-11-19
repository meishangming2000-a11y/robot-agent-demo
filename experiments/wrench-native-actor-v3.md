# Native wrench actor — development v3

You control one assigned public Panda/Shadow simulation through its native file
mailbox. Physics pauses between robot commands. Complete the task from its
current observations. This is simulator-state-assisted control, with measured
tool/nut/palm poses, named contacts, geometry and two RGB views; it is not a
vision-only or hardware experiment.

## Access and ownership

Read this instruction and the assigned episode's current pointer, requests,
images, your own earlier actions and the reference experience embedded in each
request. You may read and use only these published control resources:

- `dexhand_agent/wrench_tools.py`: pure pose construction and observed response
  compensation; no dynamics or phase selection.
- `dexhand_agent/wrench_recovery.py`: observation-driven local withdrawal policy.
- `tools/drive_native_skill.py`: actor-invoked mailbox bridge for that policy.

Do not inspect controller/evaluator source, other episodes, training results,
inventory, original projects or diagnostics. Do not launch another simulator,
language model, remote workstation or external API. Ordinary Python/NumPy and
quaternion arithmetic are allowed. Root launches and monitors; it never chooses,
writes or replaces your robot actions. You own only actor files in your episode.

The supplied nominal reference is developer-calibrated grasp/motion guidance,
not a skill you discovered. Recompute targets using current frames. The local
withdrawal policy was engineered and physically verified from a replayed
development prefix, not learned online by you. It has limited evidence: one
development starting state passed; another lower-angle state failed. Its
55-degree entry requirement is a conservative planning margin, not evidence
that angle alone explains failure. No success-rate/generalization claim follows.

## Task, control and correction

Read the current goal and physical gates. Keep at least two distinct hand parts
on the unsupported wrench. Lift, positive contact-qualified nut turning, and
retained withdrawal must all independently pass. Never touch the nut with the
robot. Normal contact force above 25 N or global penetration above 2 mm ends
the attempt; these simulation criteria are unchanged.

Plan approach, conforming grasp, unsupported lift, working-end alignment,
engagement, turn, unloading and retained withdrawal. Actual tool/palm deflection
matters: a commanded pose is not proof of actual alignment or nut progress.
Use bounded contact-sensitive motions, new observations and measured relations.
Preserve measured tool yaw during unloading unless observation justifies a
change. Contact guidance is at most 3 mm working-end translation and 5 degrees
tool rotation per command. Stop and diagnose tracking failure or rising pressure.

After unsupported lift, use a safe-height level/align target over the nut. If a
completed motion lasting at least 0.5 s leaves actual head error at least 4 mm,
record the measured mismatch and choose an observation-based correction while
keeping the same intended head target. Check reduction to at most 2 mm and by
at least half within six decisions. Eligible targets are within 30 mm XY of
the nut and at least 10 mm above its top, with unsupported retention. Initial
approach/lift and decorative post-success motions do not count. Do not create
a fault, change the target to hide error, or claim recovery from text alone.

## Explicit local skill invocation

You may choose when to invoke `WrenchWithdrawPolicy` after observing an engaged,
retained tool and sufficient turn margin. Inspect its source and constructor.
Pass the current observation and the last **completed full palm pose action**
from your own episode. The policy uses fresh observations for every microaction,
bounded response compensation, centering, leveling and axial extraction. It
does not write robot state, call a model, finish a task or certify task success.

Invoke it yourself through `drive_policy(episode_path, policy, call_id=...)`.
The bridge records initial arguments/source hash and every fresh-observation
decision, intent and exact reply. Local iterations do not require a separate
language-model inference or image inspection. Both images are nevertheless
saved for every request. Record your online choice and its observation/reason
in `actor/skill-calls/<call_id>-selection.json` before invoking the bridge.
That selection object must contain `skill_call_id`,
`decision_origin: "online_model"`, the current `request_id` and `sequence`,
canonical `observation_sha256`, `policy: "WrenchWithdrawPolicy"`, and `diagnosis`.
The bridge requires this record and binds its digest to the initial call state.
For an 80-step policy call, `maximum_replies=81, timeout_s=600` leaves room
to observe the final result and receive a policy handoff. The timeout bounds
the whole call; it does not allow extra robot actions. Use a child process with
bounded polling so a long local call does not block your tool session.
On handoff, inspect the fresh observation and images and decide what to do.
Rejected/interrupted motions, grip loss, low angle, budget exhaustion or a
clearance-check handoff require your diagnosis. Do not relabel a handoff as
success; check independent gates before issuing finish.

You may keep calculation code in your actor directory. A local script is a
declared controller, not a series of model decisions. No unrecorded automatic
trajectory replay or alternate hidden controller is allowed in this attempt.

## Exact mailbox and records

For each online decision:

1. Read `actor/current.json` when waiting; read its fresh request and both images.
2. Before replying, write a new `actor/intents/NNN.json` containing request_id,
   observation_sha256, semantic_operation, target_head_m or null, diagnosis,
   correction_trigger or null, and `decision_origin: "online_model"`.
   Correction triggers include type, trigger_sequence and actual measurements.
   Hash the observation with
   `sha256(json.dumps(obs, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8'))`.
3. Atomically write exactly `{"request_id": "current id", "action": {...}}`
   to the indicated reply path. Supply every standard action field. Never
   overwrite replies or intents. Keep intent fields outside the strict envelope.
4. Wait briefly for the new request, or invoke the explicit bridge as above.
   Continue until summary exists. Finish honestly when complete or unrecoverable.

The attempt allows at most 160 low-level robot commands and 32 online model
choices (including skill selections and finish). Local control actions are
reported separately. Earlier failed attempts used a different 64-command budget;
this revised development run is not a matched comparison to them. Root cannot
intervene strategically while you act. Keep failed/rejected actions and all
logs. Report measured gates, corrections, online/local counts, handoffs and
errors; independent replay and audits determine acceptance.
