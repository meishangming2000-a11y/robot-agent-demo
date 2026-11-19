# First task: held hammer presses a spring button

This is an original, deliberately small simulation task. It does not reproduce
egg fracture and does not inherit the original Isaac experiment's results.

## Initial conditions and public information

- Panda with the adapted Shadow wrist/hand; no private arm.
- A free 165 g short hammer rests naturally on three supports. It has no weld,
  attachment constraint, actuator, or prescribed trajectory.
- A gravity-compensated passive spring button has a 12 mm downward travel,
  stiffness 200 N/m and damping 0.5 N s/m. Its compensation makes the unpressed
  rest position independent of the cap's weight; the agent cannot actuate it.
- The first runtime uses declared simulator-state observations, robot
  proprioception/contact measurements, and rendered images. It is not a claim
  of perception from images alone. All participants receive the same fields,
  including independently computed goal-check progress. The evaluator uses
  physical samples and does not accept an actor's self-reported success.

## Ordered physical success gates (protocol 2)

1. Lift the hammer center at least 20 mm from its naturally settled height,
   with at least two distinct hand parts in tool contact and no support from
   any fixture, including the button, for 2 continuous seconds. Only pressure
   intervals after this gate count toward the next gate; an early attempt may
   be recovered from but receives no pressure credit.
2. Press the button down at least 8 mm for 0.5 continuous seconds. The hammer
   head must contact the cap while the hand retains the tool. No robot body may
   touch the cap. The hammer must remain unsupported by rack/table/floor.
3. Withdraw the hammer head at least 30 mm above the cap's resting top, keep
   holding the tool clear of all fixture contacts for 1 second, and let the button return to
   within 1 mm of its unpressed position.

Record violations and unsuccessful runs. Action completion alone never satisfies
a task gate. Dropping the tool onto the cap is not a held-tool success.
Contact labels require measured normal force above 0.05 N. A direct robot-to-cap
contact is a permanent violation for that episode.

The independent evaluator operates on physical samples; agent phase names,
prompts, and self-reported success do not establish any gate. During a task,
only robot actuator targets may change; object state and external forces are
not directly written. A separate fixture-only positive-control test may apply
a documented load to check button travel and return before manipulation.

## Scope of initial evidence

A fixed reference establishes scene/controls viability. An online Codex run
establishes autonomous execution under its declared observation and experience
conditions. Changed layouts and new tools require separate trials; neither is
established by one reference success.
