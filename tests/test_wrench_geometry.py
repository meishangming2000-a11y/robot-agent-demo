"""Compile-only checks for the declared longer, heavier wrench; no actions."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from dexhand_agent.wrench import WrenchRuntime, make_wrench_model, wrench_scene_xml
from dexhand_agent.wrench_geometry import (
    VARIANT_ID, GEOMETRY_SPEC, GEOMETRY_SPEC_SHA256, PROTOCOL_SOURCE_SHA256,
    LongWrenchRuntime, long_wrench_scene_xml, make_long_wrench_model,
)


SEED = 5


def test_geometry_spec_binds_the_predeclared_protocol():
    protocol_path = Path(__file__).resolve().parents[1] / "experiments/wrench-geometry-v1-protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf8"))
    assert VARIANT_ID == protocol["variant_id"] == "wrench-long-heavy-v1"
    assert GEOMETRY_SPEC == protocol["geometry"]
    assert PROTOCOL_SOURCE_SHA256 == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    canonical = json.dumps(GEOMETRY_SPEC, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert GEOMETRY_SPEC_SHA256 == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize("seed", [0, SEED])
def test_xml_changes_only_declared_handle_head_and_head_support_attributes(seed):
    base = list(ET.fromstring(wrench_scene_xml(seed)).iter())
    variant = list(ET.fromstring(long_wrench_scene_xml(seed)).iter())
    assert len(base) == len(variant)
    whitelist = {
        ("geom", "wrench_handle"): {"fromto", "mass", "rgba"},
        ("geom", "wrench_rack_head"): {"pos"},
        ("site", "wrench_head_center"): {"pos"},
        **{("geom", f"wrench_head_{index}"): {"pos"} for index in range(6)},
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
    assert len(changed) == 11
    tool = ET.fromstring(long_wrench_scene_xml(seed)).find("worldbody/body[@name='wrench']")
    handle = tool.find("geom[@name='wrench_handle']")
    np.testing.assert_array_equal(np.fromstring(handle.get("fromto"), sep=" "), [-.100, 0, 0, .100, 0, 0])
    assert float(handle.get("mass")) == .065
    np.testing.assert_array_equal(np.fromstring(handle.get("rgba"), sep=" "), [.85, .4, .12, 1])
    for first, second in zip(base, variant):
        if "pos" in whitelist.get((first.tag, first.get("name")), set()):
            np.testing.assert_allclose(np.fromstring(second.get("pos"), sep=" ") -
                                       np.fromstring(first.get("pos"), sep=" "),
                                       [.022, 0, 0], atol=1e-15, rtol=0)


@pytest.fixture(scope="module")
def compiled():
    # Both factories compile and call mj_forward only, with zero integration steps.
    return make_wrench_model(SEED), make_long_wrench_model(SEED)


def test_compiled_real_mass_inertia_and_socket_interface(compiled):
    (base, _), (variant, data) = compiled
    tool_id = variant.body("wrench").id
    assert data.time == 0
    assert variant.body_mass[tool_id] == pytest.approx(.165, abs=1e-14)
    assert base.body_mass[tool_id] == pytest.approx(.135, abs=1e-14)
    assert np.all(variant.body_inertia[tool_id] > 0)
    assert not np.allclose(variant.body_inertia[tool_id], base.body_inertia[tool_id], atol=1e-8, rtol=0)
    # Analytic uniform capsule plus six unchanged .100/6 kg socket walls,
    # including their parallel-axis contributions about the combined center.
    np.testing.assert_allclose(variant.body_ipos[tool_id], [.08181818181818182, 0, 0], atol=1e-14, rtol=0)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, variant.body_iquat[tool_id])
    rotation = rotation.reshape(3, 3)
    inertia = rotation @ np.diag(variant.body_inertia[tool_id]) @ rotation.T
    np.testing.assert_allclose(inertia, np.diag([
        .000033139930555555554, .0009963679135101013, .0010196318023989901,
    ]), atol=1e-14, rtol=0)
    handle_id = variant.geom("wrench_handle").id
    assert variant.geom_type[handle_id] == mujoco.mjtGeom.mjGEOM_CAPSULE
    np.testing.assert_allclose(variant.geom_size[handle_id, :2], [.010, .100], atol=1e-15, rtol=0)
    np.testing.assert_array_equal(variant.geom_pos[handle_id], [0, 0, 0])
    site_id = variant.site("wrench_head_center").id
    np.testing.assert_allclose(variant.site_pos[site_id], [.135, 0, 0], atol=1e-15, rtol=0)
    assert variant.site_bodyid[site_id] == tool_id
    for index in range(6):
        geom_id = variant.geom(f"wrench_head_{index}").id
        for name in ("geom_size", "geom_quat", "geom_friction", "geom_solref", "geom_solimp"):
            np.testing.assert_array_equal(getattr(variant, name)[geom_id], getattr(base, name)[geom_id])
        np.testing.assert_allclose(variant.geom_pos[geom_id] - base.geom_pos[geom_id],
                                   [.022, 0, 0], atol=1e-15, rtol=0)


def test_compiled_robot_nut_controls_and_non_tool_physics_are_unchanged(compiled):
    (base, base_data), (variant, data) = compiled
    assert (base.nq, base.nv, base.nu, base.nbody, base.ngeom, base.njnt, base.neq) == (
        variant.nq, variant.nv, variant.nu, variant.nbody, variant.ngeom, variant.njnt, variant.neq)
    arrays = (
        "body_pos", "body_quat", "body_parentid", "body_jntadr", "body_jntnum",
        "geom_type", "geom_bodyid", "geom_dataid", "geom_friction", "geom_solref", "geom_solimp",
        "geom_contype", "geom_conaffinity", "geom_condim", "geom_margin", "geom_gap",
        "jnt_type", "jnt_range", "jnt_pos", "jnt_axis", "jnt_stiffness",
        "dof_damping", "dof_frictionloss", "dof_armature",
        "actuator_trntype", "actuator_trnid", "actuator_gear", "actuator_gainprm", "actuator_biasprm",
        "actuator_ctrlrange", "actuator_forcerange", "actuator_dyntype", "actuator_dynprm",
        "pair_geom1", "pair_geom2", "pair_friction", "pair_solref", "pair_solimp",
        "cam_pos", "cam_quat", "cam_fovy", "eq_type", "eq_obj1id", "eq_obj2id", "eq_data",
    )
    for name in arrays:
        np.testing.assert_array_equal(getattr(base, name), getattr(variant, name), err_msg=name)
    tool_id = base.body("wrench").id
    other_bodies = np.arange(base.nbody) != tool_id
    for name in ("body_mass", "body_ipos", "body_iquat", "body_inertia"):
        np.testing.assert_array_equal(getattr(base, name)[other_bodies],
                                      getattr(variant, name)[other_bodies], err_msg=name)
    changed_geoms = [base.geom("wrench_handle").id, base.geom("wrench_rack_head").id,
                     *[base.geom(f"wrench_head_{index}").id for index in range(6)]]
    other_geoms = ~np.isin(np.arange(base.ngeom), changed_geoms)
    for name in ("geom_size", "geom_pos", "geom_quat", "geom_rgba"):
        np.testing.assert_array_equal(getattr(base, name)[other_geoms],
                                      getattr(variant, name)[other_geoms], err_msg=name)
    np.testing.assert_array_equal(data.qpos, base_data.qpos)
    np.testing.assert_array_equal(data.ctrl, base_data.ctrl)
    assert base.opt.timestep == variant.opt.timestep == .002
    np.testing.assert_array_equal(base.opt.gravity, variant.opt.gravity)
    assert base.opt.solver == variant.opt.solver
    assert base.opt.iterations == variant.opt.iterations
    assert np.all(data.xfrc_applied == 0) and np.all(data.qfrc_applied == 0)


def test_wrench_is_free_nut_is_passive_and_no_tool_actuation_or_weld_is_added(compiled):
    (base, _), (variant, _) = compiled
    free = variant.joint("wrench_free")
    nut = variant.joint("nut_hinge")
    assert free.type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert nut.type[0] == mujoco.mjtJoint.mjJNT_HINGE
    assert variant.body_jntnum[variant.body("wrench").id] == 1
    assert not {free.id, nut.id}.intersection(variant.actuator_trnid[:, 0])
    assert variant.neq == base.neq == 0
    assert list(LongWrenchRuntime.__dict__.keys()) == ["__module__", "__doc__", "__init__"]
    assert LongWrenchRuntime.execute is WrenchRuntime.execute
    assert LongWrenchRuntime.observe is WrenchRuntime.observe
    assert LongWrenchRuntime.physical_sample is WrenchRuntime.physical_sample


def test_unknown_runtime_variant_is_rejected_before_scene_or_initialization(monkeypatch):
    from dexhand_agent import wrench_geometry

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid variants must not build or initialize a model")

    monkeypatch.setattr(wrench_geometry, "long_wrench_scene_xml", forbidden)
    monkeypatch.setattr(wrench_geometry.HandRuntime, "__init__", forbidden)
    with pytest.raises(ValueError, match="Unknown wrench variant"):
        LongWrenchRuntime(SEED, variant="unknown")
