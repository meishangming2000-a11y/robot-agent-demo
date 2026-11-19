"""Mechanical invariants of the original public wrench scene."""

import xml.etree.ElementTree as ET
from itertools import product

import mujoco
import numpy as np

from dexhand_agent.wrench import (make_wrench_model, wrench_scene_xml, WRENCH_SPEC,
                                 _world_geom_aabb, _aabb_distance_lower_bound)


def test_free_rigid_wrench_and_unactuated_passive_nut():
    model, data = make_wrench_model()
    wrench = model.body("wrench")
    nut = model.body("hexnut")
    assert model.joint("wrench_free").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.joint("nut_hinge").type == mujoco.mjtJoint.mjJNT_HINGE
    assert model.neq == 0
    assert model.nu == 27  # Seven public arm and twenty public hand controls.
    assert all(model.actuator_trnid[:, 0] != model.joint("nut_hinge").id)
    assert model.dof_frictionloss[model.joint("nut_hinge").dofadr[0]] == WRENCH_SPEC["nut_frictionloss_nm"]
    assert np.all(data.xfrc_applied == 0)
    assert np.all(data.qfrc_applied == 0)
    assert wrench.mass[0] > 0 and nut.mass[0] > 0


def test_closed_hex_socket_and_hexnut_are_original_geometry():
    root = ET.fromstring(wrench_scene_xml())
    world = root.find("worldbody")
    assert world.find("body[@name='hammer']") is None
    assert world.find("body[@name='button']") is None
    head = world.findall("body[@name='wrench']/geom")
    assert len([geom for geom in head if geom.get("name", "").startswith("wrench_head_")]) == 6
    assert all(geom.get("type") in {"capsule", "box"} for geom in head)
    vertices = root.find("asset/mesh[@name='original_hexnut']").get("vertex").split()
    assert len(vertices) == 12 * 3
    assert WRENCH_SPEC["socket_apothem_m"] > WRENCH_SPEC["nut_apothem_m"]


def test_clearance_lower_bound_known_box_gap_and_overlap():
    box = (np.zeros(3), np.ones(3))
    other = (np.array([0.0, 0.0, 2.04]), np.ones(3))
    assert abs(_aabb_distance_lower_bound(box, other) - 0.04) < 1e-12
    overlap = (np.array([0.2, 0.3, 0.4]), np.ones(3))
    assert _aabb_distance_lower_bound(box, overlap) == 0


def test_rotated_world_bounds_contain_all_mesh_vertices_and_box_corners():
    model, data = make_wrench_model()
    # Isolated static geometry fixture, not an episode or task policy.
    data.qpos[model.joint("nut_hinge").qposadr[0]] = 0.73
    adr = model.joint("wrench_free").qposadr[0]
    data.qpos[adr+3:adr+7] = np.array([0.7, 0.2, -0.4, 0.5]) / np.linalg.norm([0.7, 0.2, -0.4, 0.5])
    mujoco.mj_forward(model, data)
    for name in ("hexnut_contact", *(f"wrench_head_{i}" for i in range(6))):
        gid = model.geom(name).id
        center, extent = _world_geom_aabb(model, data, gid)
        if name == "hexnut_contact":
            mid = model.geom_dataid[gid]
            start = model.mesh_vertadr[mid]
            vertices = model.mesh_vert[start:start + model.mesh_vertnum[mid]]
        else:
            vertices = np.array(list(product((-1, 1), repeat=3))) * model.geom_size[gid]
        world_vertices = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        assert np.all(np.abs(world_vertices - center) <= extent + 1e-8)
