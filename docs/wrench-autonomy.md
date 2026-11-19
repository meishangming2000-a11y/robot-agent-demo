# Online wrench workflow

Follow the [project skill](../.agents/skills/dexhand-tool-use/SKILL.md) and its
four ordered actor instructions. Use fresh measured frames and the supplied
experience. Online selection and the explicitly invoked local withdrawal
program have separate records. [Results and limitations](results.md).

For new development runs, declare a source freeze before launching and supply
that same JSON to `tools/audit_native_skill.py --source-freeze`. The historical
geometry freeze belongs to its historical source tree, not this demo snapshot.
The independent physical audit uses `experiments/wrench-autonomy-protocol-v3.json`.
Never certify a local-controller handoff or a data-integrity pass as task success.
