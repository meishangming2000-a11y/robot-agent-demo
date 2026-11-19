"""Preinteraction placement conservation checks; no robot action trajectories."""

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from dexhand_agent.scene import make_model
from dexhand_agent.wrench import WRENCH_SPEC, WrenchRuntime, make_wrench_model, wrench_scene_xml
from dexhand_agent.wrench_sensing import read_wrench_feedback
from dexhand_agent.wrench_placement import (
    VARIANT_ID, PLACEMENT_SPEC, PLACEMENT_SPEC_SHA256, PROTOCOL_SOURCE_SHA256,
    PlacementWrenchRuntime, placement_scene_xml,
)


SEED = 5
PROTOCOLS = {VARIANT_ID: "experiments/wrench-placement-protocol.json",
             "placement-minus20deg-v2": "experiments/wrench-placement-v2-protocol.json",
             "placement-plus15deg-v3": "experiments/wrench-placement-v3-protocol.json"}


@pytest.fixture(scope="module", params=list(PROTOCOLS))
def variant_id(request):
    return request.param


def descriptor(variant_id):
    path = Path(__file__).resolve().parents[1] / PROTOCOLS[variant_id]
    return json.loads(path.read_text(encoding="utf8"))["placement"]


def transform(variant_id):
    spec = descriptor(variant_id)
    angle = np.deg2rad(spec["wrench_rack_yaw_deg"])
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return angle, rotation, np.asarray(spec["wrench_rack_translation_m"]), np.asarray(spec["nut_pedestal_translation_m"])


def vector(element, name="pos"):
    return np.fromstring(element.get(name), sep=" ")


def test_frozen_spec_matches_the_predeclared_protocol_and_canonical_hash():
    protocol_path = Path(__file__).resolve().parents[1] / "experiments/wrench-placement-protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf8"))
    assert VARIANT_ID == protocol["variant_id"] == "placement-30deg-v1"
    assert PLACEMENT_SPEC == protocol["placement"]
    assert PROTOCOL_SOURCE_SHA256 == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    canonical = json.dumps(PLACEMENT_SPEC, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert PLACEMENT_SPEC_SHA256 == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize("seed,expected", [
    (0, "e0eac77307c9f7af4968e8d90c349db5472f0aafca7c642a34035154ba12b8f1"),
    (5, "4a1df141928fcd7b731b54bf4895352a9fb5cc9a487934b5d9dbceabca989eb7"),
])
def test_default_xml_retains_exact_pre_v2_bytes(seed, expected):
    xml = placement_scene_xml(seed)
    assert hashlib.sha256(xml.encode("utf8")).hexdigest() == expected
    assert placement_scene_xml(seed, variant=VARIANT_ID) == xml


@pytest.mark.parametrize("seed,expected", [
    (0, "541735d10644b5faf26679855f2232c13da927bfb4adbfa4a18726a0cd352bae"),
    (5, "5c8dd8446ad5f53679d560b002db454921630a769b2b62a4e41473b137d8de58"),
])
def test_v2_xml_retains_exact_pre_v3_bytes(seed, expected):
    xml = placement_scene_xml(seed, variant="placement-minus20deg-v2")
    assert hashlib.sha256(xml.encode("utf8")).hexdigest() == expected


def test_variant_registry_binds_selected_specs_to_immutable_protocol_sources(variant_id):
    from dexhand_agent import wrench_placement

    protocol_path = Path(__file__).resolve().parents[1] / PROTOCOLS[variant_id]
    entry = wrench_placement.VARIANTS[variant_id]
    assert entry["spec"] == descriptor(variant_id)
    assert entry["protocol_source_sha256"] == hashlib.sha256(protocol_path.read_bytes()).hexdigest()


def test_xml_changes_only_the_ten_whitelisted_pose_attributes(variant_id):
    base = list(ET.fromstring(wrench_scene_xml(SEED)).iter())
    variant = list(ET.fromstring(placement_scene_xml(SEED, variant=variant_id)).iter())
    assert len(base) == len(variant)
    whitelist = {
        ("body", "wrench"): {"pos", "quat"},
        ("body", "hexnut"): {"pos"},
        ("geom", "nut_pedestal"): {"pos"},
        **{("geom", name): {"pos", "quat"} for name in (
            "wrench_rack_left", "wrench_rack_right", "wrench_rack_head")},
    }
    changed = []
    for first, second in zip(base, variant):
        assert first.tag == second.tag
        assert first.get("name") == second.get("name")
        allowed = whitelist.get((first.tag, first.get("name")), set())
        for attribute in set(first.attrib) | set(second.attrib):
            if first.get(attribute) != second.get(attribute):
                assert attribute in allowed
                changed.append((first.tag, first.get("name"), attribute))
        assert {k: v for k, v in first.attrib.items() if k not in allowed} == {
            k: v for k, v in second.attrib.items() if k not in allowed}
    assert len(changed) == 10


def test_seeded_pivot_and_all_supports_rotate_together_while_nut_yaw_stays_zero(variant_id):
    angle, rotation, translation, nut_translation = transform(variant_id)
    base = ET.fromstring(wrench_scene_xml(SEED)).find("worldbody")
    variant = ET.fromstring(placement_scene_xml(SEED, variant=variant_id)).find("worldbody")
    pivot = vector(base.find("body[@name='wrench']"))
    moved = variant.find("body[@name='wrench']")
    np.testing.assert_allclose(vector(moved), pivot + translation, atol=1e-15, rtol=0)
    expected_quaternion = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
    np.testing.assert_allclose(vector(moved, "quat"), expected_quaternion, atol=1e-15, rtol=0)
    for name in ("wrench_rack_left", "wrench_rack_right", "wrench_rack_head"):
        old, new = base.find(f"geom[@name='{name}']"), variant.find(f"geom[@name='{name}']")
        np.testing.assert_allclose(vector(new), pivot + translation + rotation @ (vector(old) - pivot), atol=1e-15, rtol=0)
        np.testing.assert_allclose(vector(new, "quat"), expected_quaternion, atol=1e-15, rtol=0)
    for tag, name in (("body", "hexnut"), ("geom", "nut_pedestal")):
        old, new = base.find(f"{tag}[@name='{name}']"), variant.find(f"{tag}[@name='{name}']")
        np.testing.assert_allclose(vector(new), vector(old) + nut_translation, atol=1e-15, rtol=0)
        assert new.get("quat") == old.get("quat")


@pytest.fixture(scope="module")
def compiled(variant_id):
    # Compilation and the base factory's normal initialization only.
    return make_wrench_model(SEED), make_model(SEED, scene_factory=lambda seed: placement_scene_xml(seed, variant=variant_id))


def test_compiled_mechanics_controls_and_contact_parameters_are_conserved(compiled):
    (base, _), (variant, _) = compiled
    assert (base.nq, base.nv, base.nu, base.nbody, base.ngeom, base.njnt, base.neq) == (
        variant.nq, variant.nv, variant.nu, variant.nbody, variant.ngeom, variant.njnt, variant.neq)
    arrays = (
        "body_mass", "body_ipos", "body_iquat", "body_inertia", "body_parentid", "body_jntadr", "body_jntnum",
        "geom_type", "geom_size", "geom_bodyid", "geom_dataid", "geom_friction", "geom_solref", "geom_solimp",
        "geom_contype", "geom_conaffinity", "geom_condim", "geom_margin", "geom_gap",
        "jnt_type", "jnt_range", "jnt_pos", "jnt_axis", "jnt_stiffness",
        "dof_damping", "dof_frictionloss", "dof_armature",
        "actuator_trntype", "actuator_trnid", "actuator_gear", "actuator_gainprm", "actuator_biasprm",
        "actuator_ctrlrange", "actuator_forcerange", "actuator_dyntype", "actuator_dynprm",
        "pair_geom1", "pair_geom2", "pair_friction", "pair_solref", "pair_solimp",
    )
    for name in arrays:
        np.testing.assert_allclose(getattr(base, name), getattr(variant, name), atol=1e-14, rtol=0, err_msg=name)
    assert base.opt.timestep == variant.opt.timestep == .002
    np.testing.assert_array_equal(base.opt.gravity, variant.opt.gravity)
    assert base.opt.solver == variant.opt.solver
    assert base.opt.iterations == variant.opt.iterations
    for kind, count in ((mujoco.mjtObj.mjOBJ_BODY, base.nbody), (mujoco.mjtObj.mjOBJ_JOINT, base.njnt),
                        (mujoco.mjtObj.mjOBJ_ACTUATOR, base.nu)):
        assert [mujoco.mj_id2name(base, kind, i) for i in range(count)] == [
            mujoco.mj_id2name(variant, kind, i) for i in range(count)]


def test_compiled_pose_changes_match_the_descriptor_and_preserve_robot_local_geometry(compiled, variant_id):
    _, rotation, translation, nut_translation = transform(variant_id)
    (base, base_data), (variant, data) = compiled
    pivot = base_data.body("wrench").xpos
    np.testing.assert_allclose(data.body("wrench").xpos, pivot + translation, atol=1e-15, rtol=0)
    np.testing.assert_allclose(data.body("hexnut").xpos,
                               base_data.body("hexnut").xpos + nut_translation, atol=1e-15, rtol=0)
    np.testing.assert_allclose(data.body("hexnut").xquat, [1, 0, 0, 0], atol=1e-15, rtol=0)
    for name in ("wrench_rack_left", "wrench_rack_right", "wrench_rack_head"):
        np.testing.assert_allclose(data.geom(name).xpos,
                                   pivot + translation + rotation @ (base_data.geom(name).xpos - pivot),
                                   atol=1e-15, rtol=0)
    for name in ("rh_palm", "link7", "link0"):
        np.testing.assert_array_equal(data.body(name).xpos, base_data.body(name).xpos)
        np.testing.assert_array_equal(data.body(name).xquat, base_data.body(name).xquat)
    np.testing.assert_array_equal(variant.site("wrench_head_center").pos, base.site("wrench_head_center").pos)
    assert np.all(data.xfrc_applied == 0) and np.all(data.qfrc_applied == 0)


def test_runtime_only_changes_construction_and_exposes_truthful_geometry_metadata(variant_id):
    assert PlacementWrenchRuntime.execute is WrenchRuntime.execute
    assert PlacementWrenchRuntime.observe is WrenchRuntime.observe
    assert PlacementWrenchRuntime.physical_sample is WrenchRuntime.physical_sample
    before = copy.deepcopy(WRENCH_SPEC)
    runtime = PlacementWrenchRuntime(SEED, variant=variant_id)
    try:
        assert runtime.data.time == pytest.approx(1.)
        assert runtime.samples == [] and runtime.last_result["status"] == "ready"
        expected = np.array(before["nut_position_m"]) + descriptor(variant_id)["nut_pedestal_translation_m"]
        np.testing.assert_allclose(runtime.wrench_geometry_spec["nut_position_m"], expected, atol=1e-15, rtol=0)
        assert WRENCH_SPEC == before
        metadata = runtime.variant_metadata
        assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata
        assert metadata["variant_id"] == variant_id
        assert metadata["spec"] == descriptor(variant_id)
        assert metadata["source_protocol_sha256"] == hashlib.sha256(
            (Path(__file__).resolve().parents[1] / PROTOCOLS[variant_id]).read_bytes()).hexdigest()
        assert metadata["spec_sha256"] == hashlib.sha256(json.dumps(
            descriptor(variant_id), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        assert metadata["xml_sha256"] == hashlib.sha256(placement_scene_xml(SEED, variant=variant_id).encode("utf8")).hexdigest()
        assert all(metadata["compiled_proof"].values())
        observed = read_wrench_feedback(runtime)["geometry"]["spec"]
        np.testing.assert_allclose(observed["nut_position_m"], expected, atol=1e-15, rtol=0)
    finally:
        runtime.close()


@pytest.mark.parametrize("factory", [placement_scene_xml, PlacementWrenchRuntime])
def test_unknown_variant_rejected_before_model_construction(monkeypatch, factory):
    from dexhand_agent import wrench_placement

    def forbidden(*args):
        pytest.fail("Unknown variants must not build a source scene")

    monkeypatch.setattr(wrench_placement, "wrench_scene_xml", forbidden)
    with pytest.raises(ValueError, match="Unknown wrench variant"):
        factory(SEED, variant="unknown")

