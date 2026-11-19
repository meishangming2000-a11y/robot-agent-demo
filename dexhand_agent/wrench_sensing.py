"""Read compiled wrench geometry and attributed contacts without advancing state."""

import copy
import math

import mujoco
import numpy as np

from .wrench import WRENCH_SPEC


def read_wrench_feedback(runtime):
    """Return JSON-compatible measurements from already refreshed model/data."""
    model, data = runtime.model, runtime.data
    tool_id, nut_id = model.body("wrench").id, model.body("hexnut").id
    site_id, hinge_id = model.site("wrench_head_center").id, model.joint("nut_hinge").id
    spec = copy.deepcopy(WRENCH_SPEC)
    override = getattr(runtime, "wrench_geometry_spec", None)
    if override is not None:
        spec.update(copy.deepcopy(override))
    local_head = model.site_pos[site_id].copy()
    head_error = local_head - np.asarray(spec["head_center_local_m"])
    mass = float(model.body_mass[tool_id])
    mass_error = mass - spec["wrench_mass_kg"]
    head, nut = data.site_xpos[site_id], data.xpos[nut_id]
    delta = head - nut
    axis = data.xaxis[hinge_id]
    socket_axis = data.site_xmat[site_id].reshape(3, 3)[:, 2]
    axial = float(delta @ axis)
    hand_ids = {i for i in range(model.nbody) if (model.body(i).name or "").startswith("rh_")}

    def geom_ref(geom_id):
        body_id = int(model.geom_bodyid[geom_id])
        return {"id": geom_id, "name": model.geom(geom_id).name,
                "body_id": body_id, "body_name": model.body(body_id).name}

    contacts = {name: [] for name in ("all", "tool_nut", "hand_tool", "tool_world")}
    for index, contact in enumerate(data.contact):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        first, second = geom_ref(int(contact.geom1)), geom_ref(int(contact.geom2))
        bodies = {first["body_id"], second["body_id"]}
        entry = {"contact_index": index, "geom1": first, "geom2": second,
                 "position_world_m": contact.pos.tolist(),
                 "raw_frame_rows": contact.frame.reshape(3, 3).tolist(),
                 "signed_distance_m": float(contact.dist),
                 "penetration_m": max(0.0, -float(contact.dist)),
                 "normal_force_n": float(force[0]), "force_contact_frame": force.tolist()}
        contacts["all"].append(entry)
        if bodies == {tool_id, nut_id}:
            contacts["tool_nut"].append(entry)
        if tool_id in bodies and bodies.intersection(hand_ids):
            contacts["hand_tool"].append(entry)
        if bodies == {tool_id, 0}:
            contacts["tool_world"].append(entry)
    return {
        "schema": "WRENCH_RICH_FEEDBACK_V1", "time_s": float(data.time),
        "geometry": {
            "spec": spec,
            "spec_source": "runtime.wrench_geometry_spec" if override is not None else "wrench.WRENCH_SPEC",
            "compiled": {"tool_body_id": tool_id, "nut_body_id": nut_id,
                         "wrench_mass_kg": mass, "nut_mass_kg": float(model.body_mass[nut_id]),
                         "head_site_id": site_id, "head_site_local_m": local_head.tolist()},
            "consistency": {"wrench_mass_error_kg": mass_error,
                            "wrench_mass_matches_declared": abs(mass_error) <= 1e-12,
                            "head_site_error_m": head_error.tolist(),
                            "head_site_matches_declared": bool(np.linalg.norm(head_error) <= 1e-12),
                            "head_site_on_tool_body": bool(model.site_bodyid[site_id] == tool_id),
                            "scope": "Tool mass/head site only; socket and nut shape dimensions remain declarations."}},
        "head_nut": {
            "head_center_world_m": head.tolist(), "nut_center_world_m": nut.tolist(),
            "head_minus_nut_world_m": delta.tolist(),
            "head_minus_nut_body_m": (data.xmat[nut_id].reshape(3, 3).T @ delta).tolist(),
            "socket_axis_world": socket_axis.tolist(), "nut_hinge_axis_world": axis.tolist(),
            "axis_angle_deg": math.degrees(math.acos(float(np.clip(socket_axis @ axis, -1, 1)))),
            "axial_offset_m": axial, "lateral_offset_m": float(np.linalg.norm(delta - axial * axis))},
        "contact_count": int(data.ncon), "contacts": contacts,
        "global_extrema": {
            "worst_penetration": max(contacts["all"], key=lambda c: c["penetration_m"], default=None),
            "worst_normal_force": max(contacts["all"], key=lambda c: c["normal_force_n"], default=None)},
        "contact_semantics": {
            "frame": "raw MuJoCo contact.frame rows, copied verbatim",
            "force_component_order": ["normal_N", "tangent1_N", "tangent2_N",
                                      "torsion_Nm", "bend1_Nm", "bend2_Nm"],
            "force_direction": "Contact-frame components; geom/body world-force sign is not asserted.",
            "hand_membership": "Actual body IDs whose body names start with rh_; geom names do not classify."},
    }
