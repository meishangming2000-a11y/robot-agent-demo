import hashlib
import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from dexhand_agent.scene import ROOT, make_model


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("task", ["not_a_task", None, [], 42])
def test_unknown_task_is_rejected_before_creating_output(tmp_path, task):
    exporter = import_file(ROOT / "tools/export_model.py", "exporter_unknown_task_test")
    output = tmp_path / "unknown-task"
    with pytest.raises(ValueError, match="Unknown task"):
        exporter.export_model(output, task=task)
    assert not output.exists()


@pytest.mark.parametrize("seed", [-1, 0.5, True, None, "0"])
def test_invalid_seed_is_rejected_before_model_creation(tmp_path, monkeypatch, seed):
    exporter = import_file(ROOT / "tools/export_model.py", "exporter_invalid_seed_test")
    def unexpected_compile(*args, **kwargs):
        pytest.fail("Invalid export parameters must be rejected before model compilation")
    monkeypatch.setattr(exporter, "make_model", unexpected_compile)
    output = tmp_path / "invalid-seed"
    with pytest.raises(ValueError, match="seed must be a nonnegative integer"):
        exporter.export_model(output, seed=seed)
    assert not output.exists()


def test_cli_rejects_unknown_task_before_creating_output(tmp_path):
    output = tmp_path / "unknown-cli-task"
    result = subprocess.run([sys.executable, str(ROOT / "tools/export_model.py"),
                             "--output", str(output), "--task", "not_a_task"],
                            capture_output=True, text=True, encoding="utf8", timeout=30)
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert not output.exists()


def test_cli_rejects_negative_seed_before_creating_output(tmp_path):
    output = tmp_path / "invalid-cli-seed"
    result = subprocess.run([sys.executable, str(ROOT / "tools/export_model.py"),
                             "--output", str(output), "--seed", "-1"],
                            capture_output=True, text=True, encoding="utf8", timeout=30)
    assert result.returncode == 2
    assert "seed must be a nonnegative integer" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("task,variant", [
    ("hammer_button", "wrench-long-heavy-v1"),
    ("tweezers_transfer", "wrench-long-heavy-v1"),
    ("wrench_turn", "unknown"),
    ("wrench_turn", []),
])
def test_invalid_wrench_variant_is_rejected_before_model_creation(tmp_path, monkeypatch, task, variant):
    exporter = import_file(ROOT / "tools/export_model.py", "exporter_invalid_variant_test")
    def unexpected_compile(*args, **kwargs):
        pytest.fail("Invalid variants must be rejected before selecting or compiling a model")
    monkeypatch.setattr(exporter, "task_components", unexpected_compile)
    output = tmp_path / "invalid-variant"
    with pytest.raises(ValueError, match="wrench.variant"):
        exporter.export_model(output, task=task, wrench_variant=variant)
    assert not output.exists()


@pytest.mark.parametrize("task,variant,error", [
    ("hammer_button", "wrench-long-heavy-v1", "only valid for wrench_turn"),
    ("wrench_turn", "unknown", "invalid choice"),
])
def test_cli_rejects_invalid_wrench_variant_before_creating_output(tmp_path, task, variant, error):
    output = tmp_path / "invalid-cli-variant"
    result = subprocess.run([sys.executable, str(ROOT / "tools/export_model.py"),
                             "--output", str(output), "--task", task, "--wrench-variant", variant],
                            capture_output=True, text=True, encoding="utf8", timeout=30)
    assert result.returncode == 2
    assert error in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("task", ["hammer_button", "wrench_turn", "tweezers_transfer"])
def test_controller_hashes_reporting_dependency_without_simulation(tmp_path, monkeypatch, task):
    exporter = import_file(ROOT / "tools/export_model.py", "exporter_reporting_dependency_test")
    model = SimpleNamespace(nq=0, nv=0, nu=0, njnt=0)
    data = SimpleNamespace(time=0.0, qpos=np.array([]), qvel=np.array([]),
                           ctrl=np.array([]), act=np.array([]))
    monkeypatch.setattr(exporter, "task_components", lambda task: (
        lambda seed: (model, data), lambda seed: '<mujoco model="metadata-test"><compiler/><asset/></mujoco>'))
    output = tmp_path / task
    exporter.export_model(output, task=task)
    controller = json.loads((output / "controller.json").read_text(encoding="utf8"))
    source = "dexhand_agent/reporting.py"
    assert controller["source_sha256"][source] == hashlib.sha256((ROOT / source).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    exporter = import_file(ROOT / "tools/export_model.py", "model_exporter")
    output = tmp_path_factory.mktemp("model-roundtrip") / "完整模型包"
    exporter.export_model(output, seed=0)
    loader = import_file(output / "load_model.py", "exported_model_loader")
    return output, exporter, loader


def test_roundtrip_preserves_model_state_and_free_tool(exported):
    output, _, loader = exported
    actual, state = loader.load_model(output)
    expected, original = make_model(seed=0)
    assert (actual.nq, actual.nv, actual.nu, actual.neq) == (expected.nq, expected.nv, expected.nu, expected.neq)
    assert actual.joint("hammer_free").type == mujoco.mjtJoint.mjJNT_FREE
    for kind, count in ((mujoco.mjtObj.mjOBJ_JOINT, expected.njnt), (mujoco.mjtObj.mjOBJ_ACTUATOR, expected.nu)):
        assert [mujoco.mj_id2name(actual, kind, i) for i in range(count)] == [mujoco.mj_id2name(expected, kind, i) for i in range(count)]
    np.testing.assert_array_equal(actual.actuator_ctrlrange, expected.actuator_ctrlrange)
    np.testing.assert_array_equal(actual.actuator_trntype, expected.actuator_trntype)
    np.testing.assert_array_equal(state.qpos, original.qpos)
    np.testing.assert_array_equal(state.qvel, original.qvel)
    np.testing.assert_array_equal(state.ctrl, original.ctrl)
    np.testing.assert_allclose(state.body("hammer").xpos, original.body("hammer").xpos, atol=1e-12, rtol=0)
    tree = ET.fromstring((output / "model.xml").read_text(encoding="utf8"))
    mesh_dir = tree.find("compiler").get("meshdir", "")
    assert all((output / mesh_dir / node.get("file")).is_file() for node in tree.findall("./asset/mesh"))


def test_short_actuated_dynamics_match_original(exported):
    output, _, loader = exported
    actual, state = loader.load_model(output)
    expected, original = make_model(seed=0)
    actuator = actual.actuator("rh_A_FFJ3").id
    state.ctrl[actuator] = original.ctrl[expected.actuator("rh_A_FFJ3").id] = 0.35
    for _ in range(150):
        mujoco.mj_step(actual, state)
        mujoco.mj_step(expected, original)
    mujoco.mj_forward(actual, state)
    mujoco.mj_forward(expected, original)
    assert all(np.isfinite(x).all() for x in (state.qpos, state.qvel, state.ctrl))
    assert state.actuator_length[actuator] > 0.1
    np.testing.assert_allclose(state.qpos, original.qpos, atol=1e-11, rtol=0)
    np.testing.assert_allclose(state.qvel, original.qvel, atol=1e-10, rtol=0)
    np.testing.assert_allclose(state.body("hammer").xpos, original.body("hammer").xpos, atol=1e-12, rtol=0)


def test_standalone_loader_runs_from_chinese_directory(exported):
    output, _, _ = exported
    result = subprocess.run([sys.executable, "-I", str(output / "load_model.py")], cwd=output,
                            capture_output=True, text=True, encoding="utf8", timeout=30)
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    expected, _ = make_model(seed=0)
    assert (loaded["nq"], loaded["nv"], loaded["nu"]) == (expected.nq, expected.nv, expected.nu)


@pytest.mark.parametrize("task", ["hammer_button", "wrench_turn", "tweezers_transfer"])
def test_existing_output_is_not_changed(tmp_path, monkeypatch, task):
    exporter = import_file(ROOT / "tools/export_model.py", "exporter_existing_output_test")
    def unexpected_compile(*args, **kwargs):
        pytest.fail("Existing output must be rejected before selecting or compiling a model")
    monkeypatch.setattr(exporter, "task_components", unexpected_compile)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve this version", encoding="utf8")
    with pytest.raises(FileExistsError):
        exporter.export_model(output, seed=0, task=task)
    assert list(output.iterdir()) == [marker]
    assert marker.read_text(encoding="utf8") == "preserve this version"


def test_changed_home_state_fails_integrity_check(exported):
    output, _, loader = exported
    path = output / "home.json"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="SHA256"):
            loader.load_model(output)
    finally:
        path.write_bytes(original)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf8"))
    assert next(f["sha256"] for f in manifest["files"] if f["path"] == "home.json") == hashlib.sha256(original).hexdigest()


def test_unvalidated_engine_is_rejected_before_loading_files(tmp_path):
    exporter = import_file(ROOT / "tools/export_model.py", "exporter_compatibility_test")
    (tmp_path / "load_model.py").write_text(exporter.LOADER, encoding="utf8")
    (tmp_path / "manifest.json").write_text(json.dumps({"mujoco_version": "unvalidated-version"}), encoding="utf8")
    loader = import_file(tmp_path / "load_model.py", "loader_compatibility_test")
    with pytest.raises(ValueError, match="MuJoCo version"):
        loader.load_model(tmp_path)


def test_bundle_keeps_licenses_and_marks_the_modified_xml(exported):
    output, _, _ = exported
    assert (output / "LICENSE").is_file()
    assert (output / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    for robot in ("franka_emika_panda", "shadow_hand"):
        assert (output / "provenance" / robot / "LICENSE").read_bytes() == (ROOT / "assets/vendor" / robot / "LICENSE").read_bytes()
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    document = ET.fromstring((output / "model.xml").read_text(encoding="utf8"), parser=parser)
    notice = list(document)[0]
    assert notice.tag is ET.Comment
    for detail in ("Modified", "370c43965323c0944b433b202aac5058b5f907a9", "Panda", "Shadow", "forearm", "task", "cameras", "physics", "mesh"):
        assert detail in notice.text
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf8"))
    assert manifest["source"]["revision"] in notice.text
    license_hash = next(f["sha256"] for f in manifest["files"] if f["path"] == "LICENSE")
    assert license_hash == hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest()


TASK_CASES = [
    ("hammer_button", "dexhand_agent.scene", "make_model", "panda_shadow_hammer_button", "hammer", 2),
    ("wrench_turn", "dexhand_agent.wrench", "make_wrench_model", "panda_shadow_passive_hexnut", "wrench", 1),
    ("tweezers_transfer", "dexhand_agent.tweezers", "make_tweezers_model",
     "panda_shadow_passive_spring_tweezers", "tweezers", 1),
]


@pytest.fixture(scope="module", params=TASK_CASES, ids=lambda case: case[0])
def selected_exported(request, tmp_path_factory):
    task, module, factory, model_id, tool, protocol = request.param
    if task == "hammer_button":
        output, exporter, loader = request.getfixturevalue("exported")
    else:
        exporter = import_file(ROOT / "tools/export_model.py", "selected_model_exporter")
        output = tmp_path_factory.mktemp(task) / "完整模型包"
        exporter.export_model(output, seed=1, task=task)
        loader = import_file(output / "load_model.py", "selected_model_loader")
    return output, loader, request.param


def test_selected_task_roundtrip_preserves_complete_state_and_meshes(selected_exported):
    output, loader, case = selected_exported
    task, module, factory, model_id, tool, _ = case
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf8"))
    assert manifest["task_id"] == task
    assert manifest["model_id"] == model_id
    actual, state = loader.load_model(output)
    expected, original = getattr(importlib.import_module(module), factory)(seed=manifest["seed"])
    assert (actual.nq, actual.nv, actual.nu, actual.neq) == (expected.nq, expected.nv, expected.nu, expected.neq)
    assert actual.joint(f"{tool}_free").type == mujoco.mjtJoint.mjJNT_FREE
    for kind, count in ((mujoco.mjtObj.mjOBJ_JOINT, expected.njnt), (mujoco.mjtObj.mjOBJ_ACTUATOR, expected.nu)):
        assert [mujoco.mj_id2name(actual, kind, i) for i in range(count)] == [mujoco.mj_id2name(expected, kind, i) for i in range(count)]
    for field in ("actuator_ctrlrange", "actuator_trntype", "jnt_range"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    for field in ("qpos", "qvel", "ctrl", "act"):
        np.testing.assert_array_equal(getattr(state, field), getattr(original, field))
    np.testing.assert_allclose(state.body(tool).xpos, original.body(tool).xpos, atol=1e-12, rtol=0)
    tree = ET.fromstring((output / "model.xml").read_text(encoding="utf8"))
    external = [mesh for mesh in tree.findall("./asset/mesh") if mesh.get("file") is not None]
    assert all((output / "meshes" / mesh.get("file")).is_file() for mesh in external)
    assert len(external) == len([entry for entry in manifest["files"] if entry["path"].startswith("meshes/")])
    if task == "wrench_turn":
        inline = tree.find("./asset/mesh[@name='original_hexnut']")
        assert inline.get("file") is None
        assert len(inline.get("vertex").split()) == 36
        assert actual.nmesh == expected.nmesh
    actuator = actual.actuator("rh_A_FFJ3").id
    state.ctrl[actuator] = original.ctrl[expected.actuator("rh_A_FFJ3").id] = 0.35
    for _ in range(50):
        mujoco.mj_step(actual, state)
        mujoco.mj_step(expected, original)
    np.testing.assert_allclose(state.qpos, original.qpos, atol=1e-11, rtol=0)
    np.testing.assert_allclose(state.qvel, original.qvel, atol=1e-10, rtol=0)


def test_selected_task_has_accurate_source_and_controller_provenance(selected_exported):
    output, _, case = selected_exported
    task, module, factory, model_id, _, protocol = case
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf8"))
    controller = json.loads((output / "controller.json").read_text(encoding="utf8"))
    assert controller["task_id"] == task
    assert controller["model_id"] == model_id
    assert controller["reference_kind"] == "engineering_reference"
    assert controller["requires_source_project"] is True
    assert controller["standalone"] is False
    assert controller["neural_weights"] is None
    assert controller["protocol_version"] == protocol
    assert controller["model_factory"] == f"{module}:{factory}"
    assert controller["entrypoint"][:4] == ["python", "-m", "dexhand_agent", "demo"]
    assert controller["entrypoint"][controller["entrypoint"].index("--task") + 1] == task
    assert controller["robot_assets"]["revision"] == manifest["source"]["revision"]
    task_path = module.replace(".", "/") + ".py"
    for name in (task_path, "dexhand_agent/scene.py", "dexhand_agent/runtime.py", "dexhand_agent/reporting.py",
                 controller["reference"].split(":")[0].replace(".", "/") + ".py",
                 controller["evaluator"].split(":")[0].replace(".", "/") + ".py"):
        assert controller["source_sha256"][name] == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    assert manifest["source"]["composition_source_sha256"] == controller["source_sha256"][task_path]
    assert controller["task_version"]["composition_source_sha256"] == controller["source_sha256"][task_path]
    contract = controller["task_version"]["contract_path"]
    assert controller["task_version"]["contract_sha256"] == hashlib.sha256((ROOT / contract).read_bytes()).hexdigest()
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    notice = list(ET.fromstring((output / "model.xml").read_text(encoding="utf8"), parser=parser))[0].text
    assert task in notice
    for source in {task_path, "dexhand_agent/scene.py"}:
        assert source in notice
        assert controller["source_sha256"][source] in notice
    readme = (output / "README.md").read_text(encoding="utf8")
    assert task in readme
    assert "source project" in readme
    for robot in ("franka_emika_panda", "shadow_hand"):
        assert (output / "provenance" / robot / "LICENSE").read_bytes() == (ROOT / "assets/vendor" / robot / "LICENSE").read_bytes()
    for item in manifest["files"]:
        payload = (output / item["path"]).read_bytes()
        assert len(payload) == item["bytes"]
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]


def test_selected_task_loader_runs_without_source_project(selected_exported):
    output, _, _ = selected_exported
    result = subprocess.run([sys.executable, "-I", str(output / "load_model.py")], cwd=output,
                            capture_output=True, text=True, encoding="utf8", timeout=30)
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf8"))
    assert (loaded["nq"], loaded["nv"], loaded["nu"]) == tuple(manifest["contract"][key] for key in ("nq", "nv", "nu"))


@pytest.fixture(scope="module")
def long_wrench_exported(tmp_path_factory):
    from dexhand_agent.runtime import HandRuntime

    exporter = import_file(ROOT / "tools/export_model.py", "long_wrench_exporter")
    output = tmp_path_factory.mktemp("long-wrench-roundtrip") / "加长扳手完整模型包"
    def unexpected_runtime(*args, **kwargs):
        pytest.fail("Export must not construct a runtime or settle the simulation")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(HandRuntime, "__init__", unexpected_runtime)
        exporter.export_model(output, seed=5, task="wrench_turn", wrench_variant="wrench-long-heavy-v1")
    loader = import_file(output / "load_model.py", "long_wrench_loader")
    return output, loader


def test_long_wrench_roundtrip_preserves_declared_geometry_without_runtime(long_wrench_exported, monkeypatch):
    from dexhand_agent.runtime import HandRuntime
    from dexhand_agent.wrench_geometry import GEOMETRY_SPEC, make_long_wrench_model

    def unexpected_runtime(*args, **kwargs):
        pytest.fail("Export and portable loading must not construct a runtime")
    monkeypatch.setattr(HandRuntime, "__init__", unexpected_runtime)
    output, loader = long_wrench_exported
    actual, state = loader.load_model(output)
    expected, original = make_long_wrench_model(seed=5)
    assert (actual.nq, actual.nv, actual.nu, actual.neq) == (expected.nq, expected.nv, expected.nu, expected.neq)
    assert actual.joint("wrench_free").type == mujoco.mjtJoint.mjJNT_FREE
    assert actual.body("wrench").mass[0] == pytest.approx(GEOMETRY_SPEC["total_wrench_mass_kg"])
    assert actual.body("wrench").mass[0] == pytest.approx(0.165)
    np.testing.assert_array_equal(actual.site("wrench_head_center").pos, [0.135, 0, 0])
    for field in ("actuator_ctrlrange", "actuator_trntype", "jnt_range"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    for field in ("qpos", "qvel", "ctrl", "act"):
        np.testing.assert_array_equal(getattr(state, field), getattr(original, field))
    assert state.time == original.time == 0


def test_long_wrench_metadata_pins_geometry_and_excludes_legacy_reference_claim(long_wrench_exported):
    from dexhand_agent.wrench_geometry import GEOMETRY_SPEC

    output, _ = long_wrench_exported
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf8"))
    controller = json.loads((output / "controller.json").read_text(encoding="utf8"))
    protocol_path = "experiments/wrench-geometry-v1-protocol.json"
    protocol_digest = "31486d15ea02fd0e773c8a60e2c393ea6fd21c816374905e2b091014665d8de3"
    for metadata in (manifest, controller):
        assert metadata["task_id"] == "wrench_turn"
        assert metadata["wrench_variant"] == "wrench-long-heavy-v1"
        assert metadata["geometry_spec"] == GEOMETRY_SPEC
        assert metadata["geometry_spec_sha256"] == hashlib.sha256(json.dumps(
            GEOMETRY_SPEC, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf8")).hexdigest()
        assert metadata["geometry_protocol"]["path"] == protocol_path
        assert metadata["geometry_protocol"]["sha256"] == protocol_digest
        assert metadata["minimum_nut_turn_degrees"] == 45
        assert metadata["task_success_claimed"] is False
    assert (output / "provenance/wrench-geometry-v1-protocol.json").read_bytes() == (ROOT / protocol_path).read_bytes()
    assert controller["reference"] is None
    assert controller["reference_kind"] == "native_agent"
    assert controller["reference_validation"] == "not_validated_for_geometry"
    assert controller["runtime"] == "dexhand_agent.wrench_geometry:LongWrenchRuntime"
    assert controller["model_factory"] == "dexhand_agent.wrench_geometry:make_long_wrench_model"
    assert controller["evaluator"] == "dexhand_agent.wrench_evaluate:WrenchEvaluator"
    assert "agent" in controller["entrypoint"]
    assert "demo" not in controller["entrypoint"]
    assert controller["entrypoint"][controller["entrypoint"].index("--wrench-variant") + 1] == "wrench-long-heavy-v1"
    assert controller["entrypoint"][controller["entrypoint"].index("--backend") + 1] == "native"
    for source in ("dexhand_agent/wrench_geometry.py", "dexhand_agent/wrench.py", "dexhand_agent/run.py"):
        assert controller["source_sha256"][source] == hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
    assert set(manifest["source"]["composition_sources_sha256"]) == {
        "dexhand_agent/scene.py", "dexhand_agent/wrench.py", "dexhand_agent/wrench_geometry.py"}
    protocol = json.loads((ROOT / protocol_path).read_bytes())
    for group in ("experience", "actor_instructions", "frozen_control_resources"):
        for source in protocol[group]:
            assert controller["source_sha256"][source] == hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
    readme = (output / "README.md").read_text(encoding="utf8")
    for detail in ("wrench-long-heavy-v1", "45", "not validated", "Codex", "mailbox"):
        assert detail in readme
    for robot in ("franka_emika_panda", "shadow_hand"):
        assert (output / "provenance" / robot / "LICENSE").read_bytes() == (ROOT / "assets/vendor" / robot / "LICENSE").read_bytes()
    for item in manifest["files"]:
        assert hashlib.sha256((output / item["path"]).read_bytes()).hexdigest() == item["sha256"]


def test_long_wrench_loader_runs_without_source_project(long_wrench_exported):
    output, _ = long_wrench_exported
    result = subprocess.run([sys.executable, "-I", str(output / "load_model.py")], cwd=output,
                            capture_output=True, text=True, encoding="utf8", timeout=30)
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf8"))
    assert (loaded["nq"], loaded["nv"], loaded["nu"]) == tuple(manifest["contract"][key] for key in ("nq", "nv", "nu"))
