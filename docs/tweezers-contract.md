# Held tweezers transfer a passive object

This task lifts a free pair of tweezers, pinches and lifts a separate passive
object, carries it above a destination platform, and releases it there while
the hand retains the tool. The evaluator reports `task_id: "tweezers_transfer"`
and `protocol_version: 1`. Reward values and actor declarations never establish
success.

## Ordered physical gates

Every condition in a gate must hold together throughout its measured interval.
A later gate starts only on a sample after its predecessor passes. Activity
before eligibility earns no credit. The hand must contact the tool with at
least two distinct measured hand parts, and the tool must have no world or
fixture contact, throughout every gate.

1. **Pickup, 2 continuous seconds:** the tool center is at least 20 mm above
   its naturally settled initial height. No part of the tool contacts the
   object. Specific tip/other-tool contact flags also exclude pickup if the
   aggregate tool-object contact flag is inconsistent.
2. **Pinch and lift, 1 continuous second:** both the left tip and the right
   tip actually contact the object, with measured normal force above 0.005 N
   for each tip. The object center is at least 20 mm above its naturally
   settled initial height. No other tool part, robot body, or world geometry
   contacts or supports the object, including the destination platform.
3. **Transport, 0.5 continuous seconds:** the same two-tip-only pinch remains,
   without robot or world support. Object-to-goal horizontal distance is at
   most 20 mm and the object's lowest collision surface is at least 10 mm
   above the destination platform's top plane. Initial object lift is not
   required again: the destination may have a different height.
4. **Release, 1 continuous second:** the destination platform supports the
   object, with no other world support or robot contact. Horizontal distance
   remains at most 20 mm and object center linear speed is strictly below
   20 mm/s. No tool geometry contacts the object, and the minimum clearance
   from either tip's collision surface to the object is at least 20 mm.
   A gap between the two tips is not a substitute for tip-to-object clearance.

Any robot-object contact anywhere in an episode permanently records a
`robot_object_contact` violation and its first timestamp. Later clean samples
cannot clear it. All four gates and no permanent violations are required for
success. A failed condition or a sample gap above 20 ms restarts the current
continuous interval. Passed gates remain recorded; dropping and regrasping
cannot combine interrupted intervals.

Timestamps must be finite and strictly increasing. Invalid timestamps raise
`ValueError` before recording the sample. Inclusive thresholds, durations, and
the 20 ms gap comparison use an absolute `1e-9` roundoff tolerance in their
respective units. The strictly-below speed threshold uses no tolerance.

## Measured sample API

`TweezersEvaluator(initial_tool_height_m, initial_object_height_m)` receives a
dictionary through `update(sample)` and returns an independent report through
`summary()`. Initial heights are the naturally settled tool and object center
heights, measured before manipulation. Positions and distances use world meters.

| Field | Physical meaning |
| --- | --- |
| `time_s` | Finite, strictly increasing simulator seconds. |
| `tool_pos_m` | Tool center world position `[x, y, z]`. |
| `object_pos_m` | Object center world position `[x, y, z]`. |
| `hand_tool_contacts` | Measured hand-part IDs contacting the tool; duplicates count once. |
| `tool_support_contacts` | World/fixture parts contacting the tool, excluding the hand and object. |
| `tool_object_contact` | Any tool geometry in contact with the object, including weak tip contacts. |
| `object_tip_contacts` | List containing `left` and/or `right` for actual tip-object contacts above 0.005 N. Repeating one side never supplies the other side. |
| `object_other_tool_contact` | Any non-tip tool geometry contacting the object. |
| `object_fixture_contacts` | All non-goal world/fixture geometries contacting or supporting the object, including its source platform and floor. |
| `object_goal_support` | Measured destination-platform support for the object. |
| `robot_object_contact` | Any robot body contacting the object; permanent violation. |
| `object_to_goal_xy_m` | Horizontal Euclidean distance from the object center to the declared goal center. |
| `object_bottom_clearance_m` | Lowest object collision-bound world height minus destination platform top-plane height. This is relative to the destination, not the initial surface. |
| `object_speed_m_s` | Magnitude of measured object center world linear velocity. |
| `tip_object_clearance_m` | Nonnegative conservative lower bound on minimum surface separation between either tip and the object. Overlapping bounds yield zero. |
| `max_contact_force_n` | Maximum measured contact normal force in the sample. |
| `max_penetration_m` | Maximum measured contact penetration, or zero if none. |

Each collision geometry's local bounding box is transformed with its measured
world center and rotation. Its world half-extents are the absolute rotation
matrix multiplied by its local half-extents. The resulting world axis-aligned
boxes enclose the collision surfaces. Minimum box-to-box distance across both
tips and all object collision geometries is a conservative lower bound for
`tip_object_clearance_m`; overlap returns zero. The lowest object box bottom
minus the declared target top plane supplies `object_bottom_clearance_m`.
These conservative quantities may reject a valid geometric separation, but
cannot award clearance using a larger, unverified distance.

Contact exclusions count any simulator contact; the 0.005 N threshold applies
only to positive left/right pinch evidence. Hand grasp force filtering and all
geometry, passive joint mechanics, masses/inertias, fixture heights, friction,
actuator limits, integration timestep, and sampling period must be declared
with the frozen model revision before formal physical acceptance. The tool
and object have no weld, grasp attachment, external state writes, external
forces, or actor-prescribed motion. Passive tool closing/opening must result
from declared mechanics and physical robot contact. Changes require a new
model revision and fresh acceptance trials; evaluator tests alone do not
establish physical task success.

Phase names, action completion, `sourceclaim`, `success`, quaternions, extra
diagnostics, and reward values are ignored. Force and penetration are reported
without adding an undeclared force/penetration acceptance threshold.

The report contains `success`, `gates`, `violations`, and `metrics`. Each gate
exposes `passed`, `passed_at_s`, `required_continuous_s`, and
`best_continuous_s`. Metrics include sample count, last timestamp, initial
heights, signed `max_tool_lift_m` and `max_object_lift_m`,
`min_goal_distance_m`, `max_object_bottom_clearance_m`,
`max_tip_object_clearance_m`, `max_contact_force_n`, `max_penetration_m`, and
`first_robot_object_contact_s`. Diagnostics include all measured samples,
including those receiving no gate credit. An empty episode has no measured
extrema (`null`) and cannot succeed. Report dictionaries are copies.
