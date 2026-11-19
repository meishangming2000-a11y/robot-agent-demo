# Fresh-agent wrench memory pilot

First read `experiments/wrench-native-actor-v3.md`. Its ownership, input limits,
robot command contract, quality limits, local skill invocation and logging rules
apply unchanged. This addendum defines the third-stage comparison.

You control only your assigned episode. Read only its current requests/images,
your own earlier actions, the explicitly embedded experience records, and the
same three published helper/skill/bridge files. Do not read other episodes,
memory-builder/auditor code, inventory, training, reference-controller source or
the successful source episode on disk. No root strategic intervention.

Your goal is complete physical task success. A smooth success is acceptable;
you do not have to reproduce an error or a recovery. Diagnose genuine errors
when observed. The simulation and all physical criteria remain unchanged.

Each request declares its available experience. All cases receive the same
developer-calibrated nominal example and control resources. Some may additionally
contain a source-backed native episode with `skill_memory`. Use only what is
actually supplied. Recompute every task relation using current observations;
the record's world positions are source references, not blind playback targets.
The additional record, if present, contains single-episode experience, not a new
neural model or evidence for new layouts, new tools or hardware.

If a supplied memory lesson informs an online choice, add to that choice's
intent an `experience_refs` list with objects containing `experience_id` (the
record's `id`) and `lesson_id`. Use an empty list when no lesson informs it.
For example, a lesson with `working_end_target.frame == "nut_axis"` specifies
the current nut position plus its measured hinge axis times the stored axial
offset. Record the actual chosen `target_head_m`; never copy source world XY.
Only cite a lesson if you actually choose to use it. Treat local response
calibration as a starting observation, then verify the new physical response.

Use these canonical intent operations where applicable: `approach`, `grasp`,
`lift`, `level`, `align`, `correct_alignment`, `insert`, `turn`, `unload`,
`withdraw`, `hold`, `finish`. This removes the earlier logging-label ambiguity.
Keep every diagnosis and optional correction trigger in the intent. The strict
reply envelope still contains only request_id and action. Local bridge intents
need no extra field; its exact selection record schema remains unchanged.

Both cases have at most 160 robot commands and 32 recorded online choices,
including skill selections and finish. These choices are not service-call or
token measurements. You may use the same published local withdrawal policy,
independently select its invocation and handle its handoff. No source or skill
change occurs between cases. Do not end while an episode is waiting for you.

After summary exists, report physical gates, errors, online/local choices,
whether any supplied lesson was used and what measured relation it informed.
Independent trace, physical and memory-adoption audits determine acceptance.
