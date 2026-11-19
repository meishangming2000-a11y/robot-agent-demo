# Recorded results and supported scope

## Native-agent example

The [longer/heavier wrench record](../evidence/wrench-geometry-v1/results.json)
comes from episode `7783f8ea3820447e9568173b96845694`. It is a fresh-actor
confirmation on an already known same-category geometry. The agent used
simulator state, declared CAD/geometry, contacts, RGB, source experience and an
explicitly selected engineered withdrawal program.

| Recorded measurement | Value |
| --- | ---: |
| Online choices, including local-skill selection | 35 |
| Local robot actions | 32 |
| Total low-level records, including finish | 66 |
| Saved physical samples | 18,750 |
| Final nut rotation | 65.149° |
| Final head clearance | 32.616 mm |
| Peak individual normal contact force | 3.850 N |
| Maximum global penetration | 0.499 mm |
| Simulation clock / elapsed wall time | 38.5 s / 953.953 s |

The physical gates and retained trace audits passed in that source experiment.
The 66 records comprise 34 online robot actions and 32 local actions; the separate
skill selection makes 35 online choices. These counts are not API/token counts.
Historical input hashes identify the original experiment; this demo-only source
snapshot has a different file set and does not retroactively change that freeze.

The [initial run](../evidence/wrench-geometry-v1/primary-result.json) physically
passed but was procedurally unaccepted because some image outputs arrived after
actions were published. The [first confirmation](../evidence/wrench-geometry-v1/confirmation-001-interruption.json)
was interrupted by a Windows mailbox replacement error and did not pass withdrawal.
Both remain failures/nonacceptance in their original categories.

The [comparison video](../media/wrench-geometry-v1.mp4) retains continuous source
frames. The shorter side holds its last frame after ending. Video duration and
simulation time do not represent model deliberation time. Original media source
paths are historical identifiers; large raw study episodes are retained in the
research workspace, not included in this small source distribution.

## Scope

The local quickstart has three engineered references. It is useful for setup and
interface verification, not evidence of three autonomous learned policies.
Experience reuse has not established lower interaction cost; the earlier paired
pilot completed both tasks with 27 versus 30 online choices.

All tasks are simulation-only. New hands, arbitrary task synthesis, unseen tool
categories, vision-only control and VLA training are not established results.
The passive forceps and task/contact gates have explicit approximations and
limits in the individual physical contracts.

## Checks on this release

The fresh packaged quickstart completed all three nominal references with 17, 20 and 41 decisions. Independent recomputation matched every saved physical result. The separate hammer synchronized-data probe passed integrity checks for 199 frames and 9,900 ticks. [Machine-readable release validation](../evidence/release-validation.json). These task and data-integrity passes do not certify expert quality; the forceps reference retained a 2.683 mm maximum global penetration diagnostic.
