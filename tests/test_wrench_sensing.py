import copy
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from dexhand_agent.wrench import WRENCH_SPEC
from dexhand_agent.wrench_sensing import read_wrench_feedback


@pytest.fixture
def contact_runtime():
    # Small contact controls; no main-task scene, runtime initialization or step.
    model = mujoco.MjModel.from_xml_string("""+    <mujoco><worldbody>
      <geom name="rh_ff_misleading_world" type="plane" size="1 1 .1"/>
      <body name="wrench" pos="0 0 .049"><freejoint/>
        <geom name="geometry_a" type="sphere" size=".05" mass=".135"/>
        <site name="wrench_head_center" pos=".113 0 0"/>
      </body>
      <body name="hexnut" pos=".095 0 .049"><joint name="nut_hinge" axis="0 0 1"/>
        <geom name="geometry_b" type="sphere" size=".05" mass=".08"/>
      </body>
      <body name="rh_ffproximal" pos="-.095 0 .049"><freejoint/>
        <geom name="hexnut_fake" type="sphere" size=".05" mass=".02"/>
      </body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(model=model, data=data)


def test_real_contact_controls_retain_pair_identity_frame_and_force(contact_runtime):
    runtime = contact_runtime
    feedback = read_wrench_feedback(runtime)
    assert feedback["contact_count"] == runtime.data.ncon
    assert feedback["contact_count"] > 0
    assert len(feedback["contacts"]["all"]) == runtime.data.ncon
    for relation in ("tool_nut", "hand_tool", "tool_world"):
        assert feedback["contacts"][relation]
        assert any(c["normal_force_n"] > 0 for c in feedback["contacts"][relation])
    for entry in feedback["contacts"]["all"]:
        contact = runtime.data.contact[entry["contact_index"]]
        force = np.zeros(6)
        mujoco.mj_contactForce(runtime.model, runtime.data, entry["contact_index"], force)
        assert entry["geom1"]["id"] == int(contact.geom1)
        assert entry["geom2"]["id"] == int(contact.geom2)
        assert entry["signed_distance_m"] == float(contact.dist)
        assert entry["penetration_m"] == max(0, -float(contact.dist))
        assert entry["normal_force_n"] == force[0]
        np.testing.assert_array_equal(entry["force_contact_frame"], force)
        np.testing.assert_array_equal(entry["raw_frame_rows"], contact.frame.reshape(3, 3))
        np.testing.assert_array_equal(entry["position_world_m"], contact.pos)
    tool_nut = feedback["contacts"]["tool_nut"][0]
    assert {tool_nut[g]["body_name"] for g in ("geom1", "geom2")} == {"wrench", "hexnut"}
    hand_tool = feedback["contacts"]["hand_tool"][0]
    assert {hand_tool[g]["body_name"] for g in ("geom1", "geom2")} == {"wrench", "rh_ffproximal"}
    assert all(any(c[g]["body_id"] == 0 for g in ("geom1", "geom2"))
               for c in feedback["contacts"]["tool_world"])
    # The world geom's misleading hand prefix must not create a hand relation.
    assert all("rh_ff_misleading_world" not in {c[g]["name"] for g in ("geom1", "geom2")}
               for c in feedback["contacts"]["hand_tool"])
    entries = feedback["contacts"]["all"]
    worst_pen = feedback["global_extrema"]["worst_penetration"]
    worst_force = feedback["global_extrema"]["worst_normal_force"]
    assert worst_pen["penetration_m"] == max(c["penetration_m"] for c in entries)
    assert worst_force["normal_force_n"] == max(c["normal_force_n"] for c in entries)
    assert worst_pen in entries and worst_force in entries
    assert "raw" in feedback["contact_semantics"]["frame"]
    assert "not asserted" in feedback["contact_semantics"]["force_direction"]
    json.dumps(feedback, allow_nan=False)


def test_no_contact_negative_control_has_empty_lists_and_no_extrema(contact_runtime):
    runtime = contact_runtime
    runtime.data.qpos[:3] = [0, 0, .3]
    runtime.data.qpos[8:11] = [-.5, 0, .3]
    mujoco.mj_forward(runtime.model, runtime.data)
    feedback = read_wrench_feedback(runtime)
    assert runtime.data.ncon == 1  # Only the nut/world control remains.
    assert feedback["contacts"]["tool_nut"] == []
    assert feedback["contacts"]["hand_tool"] == []
    assert feedback["contacts"]["tool_world"] == []
    # Move the nut's compiled body above the plane, then refresh this test model.
    runtime.model.body_pos[runtime.model.body("hexnut").id] = [.5, 0, .3]
    mujoco.mj_forward(runtime.model, runtime.data)
    empty = read_wrench_feedback(runtime)
    assert empty["contact_count"] == 0
    assert all(not contacts for contacts in empty["contacts"].values())
    assert empty["global_extrema"] == {"worst_penetration": None, "worst_normal_force": None}


def test_global_extrema_include_contacts_unrelated_to_tool():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1"/>
      <body name="wrench" pos="0 0 .3"><freejoint/>
        <geom type="sphere" size=".05" mass=".135"/>
        <site name="wrench_head_center" pos=".113 0 0"/>
      </body>
      <body name="hexnut" pos=".5 0 .3"><joint name="nut_hinge" axis="0 0 1"/>
        <geom type="sphere" size=".05" mass=".08"/>
      </body>
      <body name="unrelated" pos="-.5 0 .03"><freejoint/>
        <geom name="background_sphere" type="sphere" size=".05" mass=".1"/>
      </body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    feedback = read_wrench_feedback(SimpleNamespace(model=model, data=data))
    assert not any(feedback["contacts"][name] for name in ("tool_nut", "hand_tool", "tool_world"))
    for name in ("worst_penetration", "worst_normal_force"):
        contact = feedback["global_extrema"][name]
        assert {contact[g]["body_name"] for g in ("geom1", "geom2")} == {"world", "unrelated"}
        assert contact["penetration_m"] == pytest.approx(.02)
        assert contact["normal_force_n"] > 0


def test_compiled_geometry_and_measured_axis_offsets_are_explicit(contact_runtime):
    runtime = contact_runtime
    runtime.data.qpos[3:7] = [np.sqrt(.5), np.sqrt(.5), 0, 0]
    mujoco.mj_forward(runtime.model, runtime.data)
    feedback = read_wrench_feedback(runtime)
    geometry = feedback["geometry"]
    assert geometry["spec"] == WRENCH_SPEC
    assert geometry["compiled"]["wrench_mass_kg"] == pytest.approx(.135)
    assert geometry["compiled"]["head_site_local_m"] == [.113, 0, 0]
    assert geometry["consistency"]["wrench_mass_matches_declared"] is True
    assert geometry["consistency"]["head_site_matches_declared"] is True
    assert geometry["consistency"]["head_site_on_tool_body"] is True
    assert "shape" in geometry["consistency"]["scope"]
    measured = feedback["head_nut"]
    np.testing.assert_allclose(measured["head_center_world_m"], [.113, 0, .049], atol=1e-15)
    np.testing.assert_allclose(measured["head_minus_nut_world_m"], [.018, 0, 0], atol=1e-15)
    np.testing.assert_allclose(measured["socket_axis_world"], [0, -1, 0], atol=1e-15)
    assert measured["axis_angle_deg"] == pytest.approx(90)
    assert measured["axial_offset_m"] == pytest.approx(0)
    assert measured["lateral_offset_m"] == pytest.approx(.018)


def test_runtime_geometry_override_is_copied_and_mismatch_is_measured(contact_runtime):
    runtime = contact_runtime
    runtime.wrench_geometry_spec = {"wrench_mass_kg": .2, "head_center_local_m": [.12, .01, 0]}
    before = copy.deepcopy(runtime.wrench_geometry_spec)
    feedback = read_wrench_feedback(runtime)
    geometry = feedback["geometry"]
    assert geometry["spec_source"] == "runtime.wrench_geometry_spec"
    assert geometry["spec"]["socket_apothem_m"] == WRENCH_SPEC["socket_apothem_m"]
    assert geometry["consistency"]["wrench_mass_matches_declared"] is False
    assert geometry["consistency"]["head_site_matches_declared"] is False
    assert geometry["consistency"]["wrench_mass_error_kg"] == pytest.approx(-.065)
    np.testing.assert_allclose(geometry["consistency"]["head_site_error_m"], [-.007, -.01, 0])
    geometry["spec"]["head_center_local_m"][0] = 99
    assert runtime.wrench_geometry_spec == before
    assert WRENCH_SPEC["head_center_local_m"] == [.113, 0, 0]


def test_read_does_not_step_forward_or_mutate_model_and_data(monkeypatch, contact_runtime):
    runtime = contact_runtime
    data_fields = ["qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied", "qacc", "xpos"]
    model_fields = ["body_mass", "body_pos", "site_pos", "geom_size"]
    before_data = {name: getattr(runtime.data, name).copy() for name in data_fields}
    before_model = {name: getattr(runtime.model, name).copy() for name in model_fields}
    before_time = runtime.data.time

    def forbidden(*args, **kwargs):
        pytest.fail("Sensing must only read the existing state")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    monkeypatch.setattr(mujoco, "mj_forward", forbidden)
    feedback = read_wrench_feedback(runtime)
    assert feedback["time_s"] == before_time
    assert runtime.data.time == before_time
    for name, before in before_data.items():
        np.testing.assert_array_equal(getattr(runtime.data, name), before)
    for name, before in before_model.items():
        np.testing.assert_array_equal(getattr(runtime.model, name), before)
