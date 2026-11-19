# Held wrench turns a passive hex nut

This task uses the adapted Panda/Shadow hand to lift a free wrench, turn a
passive hex nut, and withdraw the held wrench while retaining the nut's turn.
It is a separate task from the hammer/button task. Its evaluator reports
`task_id: "wrench_turn"` and `protocol_version: 1`.

## Model declaration before acceptance

The wrench is a free rigid body without a weld, attachment constraint,
actuator, or prescribed trajectory. The hex nut is passive and has no
actuator. Its angle changes through physical contact and its declared passive
joint mechanics. During manipulation only robot actuator targets may change;
neither object state nor external object forces may be written directly.

Before formal acceptance trials, the model agent must declare and freeze the
hex nut geometry, wrench head geometry and fit, joint axis and positive
direction, passive brake/friction/damping parameters, mass/inertia, fixture
geometry, contact measurement threshold, and simulation timing. The declaration
and model revision must accompany trial evidence. Tuning these parameters
creates a new model revision requiring fresh acceptance trials; evaluator tests
alone do not constitute physical task acceptance.

The current model and measurement method are frozen as
`wrench-v1-20261117` in `runs/wrench_dev/model_freeze.json`. Development episode
`007_deflection` remains a failed episode and is not relabeled. Formal acceptance
uses five new trials after this measurement freeze (`accept_seed0_r1` through
`accept_seed0_r5`).

## Ordered physical success gates

All conditions within one gate must hold together for its full measured
duration. The next gate becomes eligible only on a sample after the previous
gate passes. Earlier activity receives no later-gate credit.

1. **Lift, 2 continuous seconds:** wrench center height is at least 20 mm above
   its naturally settled initial height; at least two distinct hand parts
   contact the tool; no world/fixture supports or any wrench-to-nut contacts
   exist.
2. **Turn, 0.5 continuous seconds:** the wrench **head**, rather than merely its
   handle, contacts the nut; at least two distinct hand parts retain contact;
   the wrench has no world/fixture support; no robot body touches the nut; and
   the nut's positive angle relative to its initial angle is at least 45 degrees
   (`pi / 4` radians).
3. **Withdraw, 1 continuous second:** the minimum wrench-head to nut **surface**
   clearance is at least 30 mm; there is no wrench-to-nut contact or world/fixture
   support; at least two distinct hand parts retain contact; and the nut angle
   remains at least 45 degrees above its initial value.

A robot-to-nut contact at any time is a permanent episode violation. Repeated
contacts produce one `robot_nut_contact` entry and retain the first contact time.
Later valid gates cannot clear this violation. Dropping and regrasping is
allowed, but breaks continuity wherever the current gate conditions fail.
Previously passed gates stay recorded. The withdraw gate must measure the
retained nut angle throughout its own interval; an earlier peak is insufficient.

Timestamps must be finite and strictly increasing. An invalid timestamp raises
`ValueError` without recording the sample. A gap greater than 20 ms restarts
continuous intervals at the new sample; unobserved time receives no credit.
Threshold comparisons use an absolute `1e-9` epsilon in the relevant units to
absorb floating point roundoff, including duration and gap boundaries.

## Physical sample contract

`WrenchEvaluator(initial_height_m, initial_nut_angle_rad=0.0)` receives one
measured dictionary per call to `update(sample)`. Initial height and initial
nut angle are measured from the naturally settled episode, not inferred from
actor claims. Each sample contains:

| Field | Physical meaning |
| --- | --- |
| `time_s` | Finite, strictly increasing simulator time in seconds. |
| `wrench_pos_m` | Wrench center position `[x, y, z]` in world meters. |
| `hand_tool_contacts` | IDs of measured hand parts contacting the wrench; duplicates count once. |
| `tool_support_contacts` | List of world/fixture parts supporting or contacting the wrench; includes rack, table, and floor, excludes hand and nut. |
| `tool_nut_contact` | Boolean for any wrench geometry contacting any nut geometry. |
| `head_nut_contact` | Boolean specifically for wrench-head geometry contacting nut geometry. A true head flag also excludes lift/withdraw even if the any-contact flag is inconsistent. |
| `robot_nut_contact` | Boolean for any robot body contacting nut geometry; permanently latched as a violation. |
| `nut_angle_rad` | Measured unwrapped passive joint angle in radians, positive about the declared joint axis. |
| `head_clearance_m` | Nonnegative conservative lower bound on the separation between all wrench-head collision surfaces and the nut collision surface, in meters; overlapping bounds give zero and do not prove separation. |
| `raw_mj_head_distance_m` | Original minimum `mj_geomDistance` result for head/nut pairs, retained only for diagnostics; it does not establish gate clearance. |
| `max_contact_force_n` | Maximum measured contact normal force for the sample, in newtons. |
| `max_penetration_m` | Maximum measured contact penetration for the sample, in meters, zero when none. |

The frozen clearance method starts with each collision geometry's compiled
`model.geom_aabb` local center and half-extents. It transforms the center by
the geometry's current world position and rotation, and the half-extents by
the absolute world rotation matrix, yielding a world axis-aligned bounding
box (AABB). It takes the minimum box-to-box separation across all six head
segments and the nut's `hexnut_contact` collision mesh. The noncolliding
`nut_angle_marker` is decoration and is excluded. AABB overlap contributes
zero; this value is neither an exact surface distance nor a penetration
estimate. Because the boxes enclose the collision surfaces, a lower bound of
at least 30 mm proves the unchanged 30 mm physical clearance requirement.
The raw MuJoCo distance is retained separately and never replaces or increases
this conservative gate measurement.

Contact lists and flags come from simulator contact measurements. The frozen
method requires hand contact normal force above 0.05 N for grasp counts and
counts every solver contact for world support, tool-to-nut, and robot-to-nut
exclusions. An actor's phase, action
completion, `sourceclaim`, self-reported success, or quaternion is not a gate
input. Additional sample fields are ignored. Force and penetration are reported
as diagnostics; this protocol does not impose an undeclared force limit.

`summary()` returns `success`, `gates`, `violations`, and `metrics` alongside
the task ID and protocol version. Each gate exposes `passed`, `passed_at_s`,
`required_continuous_s`, and `best_continuous_s`. Success requires all gates and
no permanent violations. Metrics include sample count, last timestamp, initial
height/angle, `max_lift_m`, `max_nut_turn_rad`, `max_head_clearance_m`,
`max_contact_force_n`, `max_penetration_m`, and
`first_robot_nut_contact_s`. Lift and turn maxima are signed differences from
their measured initial values, even for samples that receive no gate credit.
No samples means no measured maxima (`null`), rather than evidence of success.
