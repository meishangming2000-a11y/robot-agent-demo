# Contributing

Start with `python quickstart.py --task wrench_turn`, then install `.[test]`
using the generated project environment and run `python -m pytest tests -q`.
Use the same environment for tools and runtime commands.

For changes to physics, controls or evaluators, preserve task contracts and add
meaningful regression coverage. Record source hashes and failed episodes alongside
successes. Keep changes to a task/evaluator outside a frozen evaluation run.
Do not commit downloaded robot assets, environments, credentials or bulk runs.

New tasks need a scene, named actuator mapping, bounded runtime actions, independent
physical gates and a documented assistance level. Preserve third-party licenses.
Describe whether a result used an agent, reference control or an explicitly invoked
local program; report state assistance and limits.
