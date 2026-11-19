"""Export the complete public robot/task model with a standalone VFS loader."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dexhand_agent.scene import make_model, scene_xml


TASKS = {
    "hammer_button": {"module": "dexhand_agent.scene", "model_factory": "make_model",
                      "scene_factory": "scene_xml", "label": "hammer/button",
                      "reference": "dexhand_agent.reference:ReferencePolicy",
                      "runtime": "dexhand_agent.runtime:HandRuntime",
                      "evaluator": "dexhand_agent.evaluate:TaskEvaluator",
                      "contract": "docs/task-contract.md"},
    "wrench_turn": {"module": "dexhand_agent.wrench", "model_factory": "make_wrench_model",
                    "scene_factory": "wrench_scene_xml", "label": "wrench/hex-nut",
                    "reference": "dexhand_agent.wrench:WrenchReference",
                    "runtime": "dexhand_agent.wrench:WrenchRuntime",
                    "evaluator": "dexhand_agent.wrench_evaluate:WrenchEvaluator",
                    "contract": "docs/wrench-contract.md"},
    "tweezers_transfer": {"module": "dexhand_agent.tweezers", "model_factory": "make_tweezers_model",
                          "scene_factory": "tweezers_scene_xml", "label": "tweezers/object-transfer",
                          "reference": "dexhand_agent.tweezers:TweezersReference",
                          "runtime": "dexhand_agent.tweezers:TweezersRuntime",
                          "evaluator": "dexhand_agent.tweezers_evaluate:TweezersEvaluator",
                          "contract": "docs/tweezers-contract.md"},
}

WRENCH_VARIANT = "wrench-long-heavy-v1"
GEOMETRY_PROTOCOL = "experiments/wrench-geometry-v1-protocol.json"
GEOMETRY_PROTOCOL_SHA256 = "31486d15ea02fd0e773c8a60e2c393ea6fd21c816374905e2b091014665d8de3"


def task_spec(task, wrench_variant=None):
    if not isinstance(task, str) or task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    if wrench_variant is None:
        return TASKS[task]
    if task != "wrench_turn":
        raise ValueError("wrench_variant is only valid for wrench_turn")
    if wrench_variant != WRENCH_VARIANT:
        raise ValueError(f"Unknown wrench_variant: {wrench_variant}")
    return {**TASKS[task], "module": "dexhand_agent.wrench_geometry",
            "model_factory": "make_long_wrench_model", "scene_factory": "long_wrench_scene_xml",
            "runtime": "dexhand_agent.wrench_geometry:LongWrenchRuntime", "reference": None}


def task_components(task, wrench_variant=None):
    spec = task_spec(task, wrench_variant)
    if task == "hammer_button":
        return make_model, scene_xml
    module = importlib.import_module(spec["module"])
    return getattr(module, spec["model_factory"]), getattr(module, spec["scene_factory"])


LOADER = '''"""Load this model and its home state without the source repository or GL."""

import hashlib
import json
from pathlib import Path

import mujoco


def load_model(directory=None):
    directory = Path(directory) if directory is not None else Path(__file__).resolve().parent
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    if mujoco.__version__ != manifest["mujoco_version"]:
        raise ValueError("MuJoCo version mismatch: requires " + manifest["mujoco_version"]
                         + ", found " + mujoco.__version__)
    for item in manifest["files"]:
        payload = (directory / item["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError("SHA256 mismatch: " + item["path"])
    assets = {Path(item["path"]).name: (directory / item["path"]).read_bytes()
              for item in manifest["files"] if item["path"].startswith("meshes/")}
    xml = (directory / "model.xml").read_text(encoding="utf8")
    # Reading XML/meshes with Python also supports Windows non-ASCII directories.
    model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    contract = manifest["contract"]
    if (model.nq, model.nv, model.nu) != (contract["nq"], contract["nv"], contract["nu"]):
        raise ValueError("Model state dimensions differ from the exported contract")
    home = json.loads((directory / "home.json").read_text(encoding="utf8"))
    data = mujoco.MjData(model)
    data.time = home["time_s"]
    data.qpos[:] = home["qpos"]
    data.qvel[:] = home["qvel"]
    data.ctrl[:] = home["ctrl"]
    data.act[:] = home["act"]
    mujoco.mj_forward(model, data)
    return model, data


if __name__ == "__main__":
    model, data = load_model()
    print(json.dumps({"nq": model.nq, "nv": model.nv, "nu": model.nu,
                      "time_s": data.time, "mujoco": mujoco.__version__}))
'''

USAGE = '''# Panda + Shadow + {label} model bundle

Task: `{task_id}`.
{variant_note}

This folder contains the complete composed MJCF, mesh files, initial state,
joint/actuator contract, pinned source manifest, and original robot licenses.
It can be moved independently of the source repository. Install the tested
runtime dependencies, then run:

```sh
python -m pip install -r requirements.txt
python load_model.py
```

For Python use `from load_model import load_model; model, data = load_model()`.
The loader verifies file hashes and reads XML and mesh bytes into MuJoCo's VFS;
this supports Windows paths containing Chinese characters without relying on
MuJoCo's native filesystem path decoding. The XML also contains relative mesh
paths for viewers whose native filesystem loader supports the directory.
It rejects engine versions differing from the tested version in the manifest.

The home state is the original `{model_factory}` initialization at time zero, before
the repository runtime's one-second gravity settling. Only named robot joints
are initialized; task bodies and passive joints retain the selected scene's
initial conditions. The tool retains its free joint.

`controller.json` identifies {controller_description}, runtime,
independent evaluator, protocol version, task contract, robot asset revision,
and SHA256 hashes of their source dependencies. This metadata requires the
source project to run {controller_name}; it is not a standalone control program.
From a matching source project checkout, replace the output placeholder in:

```sh
{reference_command}
```

That command reconstructs the task scene from the source project; it does not
load this bundle's XML. Match the source hashes in `controller.json` before
running it. The portable `load_model.py` only loads the model and home state.
This bundle is a simulation model and starting state, not neural weights or a
new trained policy. No hardware validation is implied.

The project loader and original task code use the [MIT license](LICENSE).
Panda and Shadow robot descriptions and meshes retain their Apache-2.0 terms:
[Panda license](provenance/franka_emika_panda/LICENSE) and
[Shadow license](provenance/shadow_hand/LICENSE). The MIT license does not
replace or relicense those upstream robot components. The modified composition
is identified by the notice at the beginning of `model.xml`.

Unmodified upstream XML/README files are in `provenance/`. The source revision
and all bundled-file SHA256 hashes are in `manifest.json`.
'''


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf8")


def export_model(output, seed=0, task="hammer_button", wrench_variant=None):
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    spec = task_spec(task, wrench_variant)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite model bundle: {output}")
    model_factory, scene_factory = (task_components(task) if wrench_variant is None
                                    else task_components(task, wrench_variant))
    model, data = model_factory(seed)
    document = ET.fromstring(scene_factory(seed))
    document.find("compiler").set("meshdir", "meshes")
    upstream_bytes = (ROOT / "assets/manifest.json").read_bytes()
    upstream = json.loads(upstream_bytes)
    task_source = spec["module"].replace(".", "/") + ".py"
    composition_paths = {"dexhand_agent/scene.py", task_source}
    if wrench_variant is not None:
        composition_paths.add("dexhand_agent/wrench.py")
    composition_sources = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                           for path in composition_paths}
    source_paths = {"dexhand_agent/__init__.py", "dexhand_agent/__main__.py",
                    "dexhand_agent/run.py", "dexhand_agent/runtime.py", "dexhand_agent/scene.py",
                    "dexhand_agent/reference.py", "dexhand_agent/evaluate.py",
                    "dexhand_agent/codex_backend.py", "dexhand_agent/file_backend.py",
                    "dexhand_agent/experience.py", "dexhand_agent/reporting.py", task_source}
    for entrypoint in (spec["reference"], spec["runtime"], spec["evaluator"]):
        if entrypoint is not None:
            source_paths.add(entrypoint.split(":")[0].replace(".", "/") + ".py")
    variant_metadata = {}
    protocol_bytes = None
    if wrench_variant is not None:
        source_paths.add("dexhand_agent/wrench.py")
        protocol_bytes = (ROOT / GEOMETRY_PROTOCOL).read_bytes()
        if hashlib.sha256(protocol_bytes).hexdigest() != GEOMETRY_PROTOCOL_SHA256:
            raise ValueError("Geometry protocol SHA256 mismatch")
        protocol = json.loads(protocol_bytes)
        geometry_module = importlib.import_module(spec["module"])
        geometry = geometry_module.GEOMETRY_SPEC
        if geometry != protocol["geometry"]:
            raise ValueError("Geometry specification differs from the declared protocol")
        for group in ("experience", "actor_instructions", "frozen_control_resources"):
            source_paths.update(protocol[group])
        variant_metadata = {"wrench_variant": wrench_variant, "geometry_spec": geometry,
                            "geometry_spec_sha256": geometry_module.GEOMETRY_SPEC_SHA256,
                            "geometry_protocol": {"path": GEOMETRY_PROTOCOL, "sha256": GEOMETRY_PROTOCOL_SHA256},
                            "minimum_nut_turn_degrees": 45, "task_success_claimed": False}
    evaluator_module, evaluator_class = spec["evaluator"].split(":")
    protocol_version = getattr(importlib.import_module(evaluator_module), evaluator_class).PROTOCOL_VERSION
    upstream_hash = hashlib.sha256(upstream_bytes).hexdigest()
    controller = {
        "schema_version": 1, "task_id": task, "model_id": document.get("model"),
        "reference_kind": "engineering_reference", "requires_source_project": True,
        "standalone": False, "neural_weights": None,
        "reference": spec["reference"], "runtime": spec["runtime"], "evaluator": spec["evaluator"],
        "model_factory": f"{spec['module']}:{spec['model_factory']}",
        "entrypoint": ["python", "-m", "dexhand_agent", "demo", "--task", task,
                       "--seed", str(seed), "--output", "<new-output-directory>", "--no-video"],
        "protocol_version": protocol_version,
        "task_version": {"composition_source_sha256": composition_sources[task_source],
                         "contract_path": spec["contract"],
                         "contract_sha256": hashlib.sha256((ROOT / spec["contract"]).read_bytes()).hexdigest()},
        "robot_assets": {"repository": upstream["repository"], "revision": upstream["revision"],
                         "asset_manifest_sha256": upstream_hash},
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                          for path in sorted(source_paths)},
        "source_project_requirements": {"path": "pyproject.toml",
                                        "sha256": hashlib.sha256((ROOT / "pyproject.toml").read_bytes()).hexdigest()},
    }
    if wrench_variant is not None:
        controller.update(variant_metadata)
        controller.update({"reference_kind": "native_agent", "reference_validation": "not_validated_for_geometry",
                           "entrypoint": [".venv/Scripts/python.exe", "-u", "-m", "dexhand_agent", "agent",
                                          "--task", task, "--backend", "native", "--wrench-variant", wrench_variant,
                                          "--seed", str(seed), "--max-decisions", "160", "--quality-guard", "--rich-feedback",
                                          "--experience", "evidence/kit-suite-v1/wrench_turn/experience.json",
                                          "--experience", "experiences/wrench-native-v2.json",
                                          "--experience", "experiences/wrench-transport-v3.json",
                                          "--output", "<new-output-directory>"]})
    document.insert(0, ET.Comment(
        " Modified by Dexhand Agent contributors, 2026.\n"
        f"Upstream: MuJoCo Menagerie revision {upstream['revision']}.\n"
        f"Selected task: {task}.\n"
        "Composition source SHA256: " + "; ".join(f"{path}={digest}" for path, digest in sorted(composition_sources.items())) + ".\n"
        "Changes: join Panda and Shadow wrist/hand; remove Shadow forearm and related contact exclusions; "
        "merge hand defaults, actuators, tendons and contacts; replace the source keyframe with separate home initialization; "
        f"add mount, palm site, original {spec['label']} task fixtures, cameras and lighting; "
        "change physics settings and alias mesh filenames with a relative meshes directory.\n"
        "Robot descriptions and meshes retain their upstream Apache-2.0 licenses in provenance/. "
        "The project loader and original task code use the root MIT LICENSE, which does not relicense upstream components. "
    ))
    known = {item["path"]: item for item in upstream["files"]}
    mesh_sources = {}
    for folder in ("franka_emika_panda", "shadow_hand"):
        for path in (ROOT / "assets/vendor" / folder / "assets").iterdir():
            mesh_sources[f"{folder}_{path.name}"] = f"{folder}/assets/{path.name}"
    contents = {"model.xml": ET.tostring(document, encoding="utf-8", xml_declaration=True),
                "home.json": json_bytes({"time_s": float(data.time), "qpos": data.qpos.tolist(),
                                         "qvel": data.qvel.tolist(), "ctrl": data.ctrl.tolist(),
                                         "act": data.act.tolist()}),
                "load_model.py": LOADER.encode("utf8"),
                "README.md": USAGE.format(label=spec["label"], task_id=task, model_factory=spec["model_factory"],
                                          variant_note=("\nVariant: `" + wrench_variant + "`. The original engineering reference is not validated "
                                                        "for this geometry. The unchanged wrench evaluator requires at least 45 degrees "
                                                        "of nut turn along with the other ordered gates; this export claims no task success. "
                                                        "The native agent command requires Codex to process the mailbox using "
                                                        "`.agents/skills/dexhand-tool-use/SKILL.md`; it is not an unattended reference replay.\n"
                                                        if wrench_variant is not None else ""),
                                          controller_description=("the native agent entrypoint" if wrench_variant is not None
                                                                  else "the selected engineering reference"),
                                          controller_name=("the native agent" if wrench_variant is not None else "the reference"),
                                          reference_command=" ".join(controller["entrypoint"])).encode("utf8"),
                "controller.json": json_bytes(controller),
                "LICENSE": (ROOT / "LICENSE").read_bytes(),
                "requirements.txt": f"mujoco=={mujoco.__version__}\nnumpy=={np.__version__}\n".encode("utf8"),
                "provenance/asset-manifest.json": upstream_bytes}
    if protocol_bytes is not None:
        contents["provenance/wrench-geometry-v1-protocol.json"] = protocol_bytes
    origins = {}
    for mesh in document.findall("./asset/mesh"):
        alias = mesh.get("file")
        if alias is None:
            continue  # Original inline task geometry stays embedded in the XML.
        origins[f"meshes/{alias}"] = mesh_sources[alias]
    for folder, source_xml in (("franka_emika_panda", "panda_nohand.xml"), ("shadow_hand", "right_hand.xml")):
        for filename in (source_xml, "LICENSE", "README.md"):
            origins[f"provenance/{folder}/{filename}"] = f"{folder}/{filename}"
    for destination, source in origins.items():
        payload = (ROOT / "assets/vendor" / source).read_bytes()
        if hashlib.sha256(payload).hexdigest() != known[source]["sha256"]:
            raise ValueError(f"Upstream asset SHA256 mismatch: {source}")
        contents[destination] = payload
    joint_kinds = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    joints = [{"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
               "kind": joint_kinds[int(model.jnt_type[i])], "limited": bool(model.jnt_limited[i]),
               "qpos_address": int(model.jnt_qposadr[i]),
               "dof_address": int(model.jnt_dofadr[i]), "range": model.jnt_range[i].tolist()}
              for i in range(model.njnt)]
    actuators = []
    for i in range(model.nu):
        tendon = model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON
        target_type = mujoco.mjtObj.mjOBJ_TENDON if tendon else mujoco.mjtObj.mjOBJ_JOINT
        actuators.append({"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
                          "transmission": "tendon" if tendon else "joint",
                          "target": mujoco.mj_id2name(model, target_type, int(model.actuator_trnid[i, 0])),
                          "control_limited": bool(model.actuator_ctrllimited[i]),
                          "control_range": model.actuator_ctrlrange[i].tolist(),
                          "force_limited": bool(model.actuator_forcelimited[i]),
                          "force_range": model.actuator_forcerange[i].tolist()})
    manifest = {"schema_version": 1, "model_id": document.get("model"), "task_id": task,
                "seed": seed, "mujoco_version": mujoco.__version__,
                "numpy_version": np.__version__,
                "protocol_version": protocol_version,
                "initialization": f"{spec['model_factory']} at time zero, before runtime gravity settling",
                "source": {"upstream_repository": upstream["repository"], "revision": upstream["revision"],
                           "asset_manifest_sha256": upstream_hash,
                           "composition_source_sha256": composition_sources[task_source],
                           "composition_sources_sha256": composition_sources},
                "contract": {"nq": model.nq, "nv": model.nv, "nu": model.nu,
                             "joints": joints, "actuators": actuators},
                "files": [{"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                           **({"upstream_path": origins[name]} if name in origins else {})}
                          for name, payload in sorted(contents.items())]}
    manifest.update(variant_metadata)
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in contents.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", choices=TASKS, default="hammer_button")
    parser.add_argument("--wrench-variant", choices=[WRENCH_VARIANT])
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be a nonnegative integer")
    if args.wrench_variant is not None and args.task != "wrench_turn":
        parser.error("--wrench-variant is only valid for wrench_turn")
    manifest = export_model(args.output, args.seed, args.task, args.wrench_variant)
    size = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
    print(json.dumps({"output": str(args.output), "bytes": size, "files": len(manifest["files"]) + 1,
                      "task_id": manifest["task_id"], "model_id": manifest["model_id"],
                      "mujoco": manifest["mujoco_version"],
                      "nq": manifest["contract"]["nq"], "nv": manifest["contract"]["nv"],
                      "nu": manifest["contract"]["nu"]}))


if __name__ == "__main__":
    main()
