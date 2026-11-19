"""Mechanical invariants of the passive spring forceps approximation."""

import mujoco
import numpy as np

from dexhand_agent.tweezers import make_tweezers_model, TweezersRuntime, TWEEZERS_SPEC


def test_free_tool_free_object_and_passive_leaf_only():
    model, data = make_tweezers_model()
    assert model.joint("tweezers_free").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.joint("object_free").type == mujoco.mjtJoint.mjJNT_FREE
    leaf = model.joint("tweezer_leaf_hinge")
    assert leaf.type == mujoco.mjtJoint.mjJNT_HINGE
    assert model.jnt_stiffness[leaf.id] == TWEEZERS_SPEC["leaf_stiffness_nm_rad"]
    assert model.dof_damping[leaf.dofadr[0]] > 0
    assert model.neq == 0 and model.nu == 27
    assert all(model.actuator_trnid[:, 0] != leaf.id)
    assert np.all(data.xfrc_applied == 0) and np.all(data.qfrc_applied == 0)
    assert model.body("small_object").mass[0] == 0.001


def test_two_separate_working_tips_and_positive_inertia():
    model, _ = make_tweezers_model()
    assert model.geom("tweezer_tip_left").bodyid != model.geom("tweezer_tip_right").bodyid
    for body in ("tweezers", "tweezer_leaf", "small_object"):
        assert model.body(body).mass[0] > 0
        assert np.all(model.body(body).inertia > 0)
    assert model.geom("tweezer_tip_left").type == mujoco.mjtGeom.mjGEOM_BOX
    assert model.geom("tweezer_tip_right").type == mujoco.mjtGeom.mjGEOM_BOX


def test_frozen_spec_matches_tip_and_support_geometry():
    model, _ = make_tweezers_model()
    for label in ("left", "right"):
        assert np.array_equal(model.geom(f"tweezer_tip_{label}").size,
                              TWEEZERS_SPEC[f"{label}_tip_half_size_m"])
    for name in ("source_platform", "source_platform_right", "goal_platform", "goal_platform_right"):
        rail = model.geom(name)
        assert np.isclose(rail.pos[2] + rail.size[2], TWEEZERS_SPEC["support_rail_top_m"])
        assert np.isclose(2 * rail.size[0], TWEEZERS_SPEC["support_rail_width_m"])
    assert np.isclose(model.body("tweezers").mass[0], TWEEZERS_SPEC["backbone_mass_kg"])
    assert np.isclose(model.body("tweezer_leaf").mass[0], TWEEZERS_SPEC["leaf_mass_kg"])


def test_goal_post_contact_is_a_fixture_not_goal_top_support():
    runtime = TweezersRuntime()
    try:
        # Place a test object against a post below the two top rails. This
        # fixture-only setup verifies measurement classification, not a policy.
        adr = runtime.model.joint("object_free").qposadr[0]
        runtime.data.qpos[adr:adr + 3] = [0.3302, 0.182, 0.210]
        mujoco.mj_forward(runtime.model, runtime.data)
        sample = runtime.physical_sample()
        assert "goal_platform_post" in sample["object_fixture_contacts"]
        assert not sample["object_goal_support"]
    finally:
        runtime.close()
