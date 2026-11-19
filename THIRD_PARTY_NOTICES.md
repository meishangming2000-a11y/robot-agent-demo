# Third-party notices

The project-authored code, task fixtures, and documentation use the [MIT
License](LICENSE). Third-party robot descriptions and meshes retain their
upstream licenses; the project license does not replace those terms.
Installed runtime dependencies also retain their respective licenses.

## Robot assets

Both models come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
at commit `370c43965323c0944b433b202aac5058b5f907a9`. The downloader uses this
fixed revision. [assets/manifest.json](assets/manifest.json) records every
downloaded file's URL, size, and SHA256. Robot meshes are fetched separately
rather than bundled in the source release.

| Model used | Upstream attribution and license | Complete local license copy |
| --- | --- | --- |
| Franka Emika Panda, `panda_nohand.xml` and its meshes | The upstream [README](https://github.com/google-deepmind/mujoco_menagerie/blob/370c43965323c0944b433b202aac5058b5f907a9/franka_emika_panda/README.md) describes a simplified MJCF derived from the publicly available [Franka URDF description](https://github.com/frankaemika/franka_ros/tree/develop/franka_description), and states Apache-2.0. [Upstream LICENSE](https://github.com/google-deepmind/mujoco_menagerie/blob/370c43965323c0944b433b202aac5058b5f907a9/franka_emika_panda/LICENSE). | [Panda LICENSE](third_party/licenses/franka_emika_panda/LICENSE) |
| Shadow Hand E3M5, `right_hand.xml` and its meshes | The upstream [README](https://github.com/google-deepmind/mujoco_menagerie/blob/370c43965323c0944b433b202aac5058b5f907a9/shadow_hand/README.md) credits the original URDF and assets to [Shadow Robot Company](https://www.shadowrobot.com/) under Apache-2.0. Its [LICENSE](https://github.com/google-deepmind/mujoco_menagerie/blob/370c43965323c0944b433b202aac5058b5f907a9/shadow_hand/LICENSE) includes **Copyright 2022 Shadow Robot Company Ltd**. | [Shadow LICENSE](third_party/licenses/shadow_hand/LICENSE) |

The two local license copies preserve the upstream files byte for byte,
including their differing endings. The asset downloader also retains each
model's original `LICENSE` and `README.md` beside the downloaded files.

## Composition changes

[scene.py](dexhand_agent/scene.py) creates a modified, in-memory composition of
the two descriptions. It mounts the Shadow wrist/hand subtree on Panda's
attachment, omits the Shadow forearm and its associated contact exclusions,
adds a simple mount and palm reference site, and merges the hand's actuator,
tendon, and contact definitions. It removes Panda's source keyframe and uses
named-joint initialization, aliases mesh filenames for MuJoCo's asset filesystem,
and adds the project's primitive
hammer, supports, button, cameras, and simulation settings. Downloaded source
XML and mesh files remain unchanged.

Demo renders show these public robot models with the project's task fixtures.
The adaptation is a simulation setup; it is not a validated physical mounting
design or a claim of ownership of the original robots. Details are in
[robot provenance](docs/robot-provenance.md), with the simulation evidence in
[validation](docs/validation.md).

The selected assets exclude the prior private arm and Isaac/GPL asset sets.
Those files and prior private-arm media are not part of this release.
