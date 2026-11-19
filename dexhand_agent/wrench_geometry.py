"""Predeclared longer, heavier wrench; only initial XML construction changes."""

import copy
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .runtime import HandRuntime
from .scene import make_model
from .wrench import WRENCH_SPEC, WrenchRuntime, wrench_scene_xml


VARIANT_ID = "wrench-long-heavy-v1"
GEOMETRY_SPEC = {
    "handle_endpoints_m": [-.100, .100],
    "handle_radius_m": .010,
    "handle_mass_kg": .065,
    "head_center_local_m": [.135, 0., 0.],
    "head_mass_kg": .100,
    "total_wrench_mass_kg": .165,
    "handle_rgba": [.85, .40, .12, 1.],
    "socket_dimensions": "Unchanged aperture, wall thickness and head height; same passive nut interface",
    "head_support": "Move wrench_rack_head +0.022m along world X at the unchanged base placement so it remains below the shifted head; other supports unchanged",
}
GEOMETRY_SPEC_SHA256 = hashlib.sha256(json.dumps(
    GEOMETRY_SPEC, sort_keys=True, separators=(",", ":"), allow_nan=False,
).encode("utf8")).hexdigest()
PROTOCOL_SOURCE_SHA256 = "31486d15ea02fd0e773c8a60e2c393ea6fd21c816374905e2b091014665d8de3"


def long_wrench_scene_xml(seed=0):
    """Apply the declared handle and head changes before model compilation."""
    root = ET.fromstring(wrench_scene_xml(seed))
    world = root.find("worldbody")
    wrench = world.find("body[@name='wrench']")
    handle = wrench.find("geom[@name='wrench_handle']")
    handle.set("fromto", "-0.100 0 0 0.100 0 0")
    handle.set("mass", "0.065")
    handle.set("rgba", "0.85 0.40 0.12 1")
    for element in [*[wrench.find(f"geom[@name='wrench_head_{index}']") for index in range(6)],
                    wrench.find("site[@name='wrench_head_center']"),
                    world.find("geom[@name='wrench_rack_head']")]:
        position = np.fromstring(element.get("pos"), sep=" ")
        position[0] += .022
        element.set("pos", " ".join(str(float(value)) for value in position))
    return ET.tostring(root, encoding="unicode")


def make_long_wrench_model(seed=0):
    return make_model(seed, scene_factory=long_wrench_scene_xml)


class LongWrenchRuntime(WrenchRuntime):
    """Construction only; retain base actions, observations and physical sampling."""

    def __init__(self, seed=0, variant=VARIANT_ID):
        if variant != VARIANT_ID:
            raise ValueError(f"Unknown wrench variant: {variant}")
        HandRuntime.__init__(self, seed, model_factory=make_long_wrench_model, tool_body="wrench")
        nut_id = self.model.body("hexnut").id
        tool_id = self.model.body("wrench").id
        site_id = self.model.site("wrench_head_center").id
        handle_id = self.model.geom("wrench_handle").id
        self.initial_nut_angle_rad = float(self.data.qpos[self.model.joint("nut_hinge").qposadr[0]])
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, self.model.geom_quat[handle_id])
        offset = rotation.reshape(3, 3)[:, 2] * self.model.geom_size[handle_id, 1]
        center = self.model.geom_pos[handle_id]
        endpoints = sorted(float(point[0]) for point in (center - offset, center + offset))
        self.wrench_geometry_spec = copy.deepcopy(WRENCH_SPEC)
        self.wrench_geometry_spec.update({
            "handle_endpoints_m": endpoints,
            "handle_radius_m": float(self.model.geom_size[handle_id, 0]),
            "head_center_local_m": self.model.site_pos[site_id].tolist(),
            "wrench_mass_kg": float(self.model.body_mass[tool_id]),
            "nut_position_m": self.model.body_pos[nut_id].tolist(),
        })
        self.variant_metadata = {
            "variant_id": variant, "seed": seed,
            "spec": copy.deepcopy(GEOMETRY_SPEC), "spec_sha256": GEOMETRY_SPEC_SHA256,
            "source_protocol_sha256": PROTOCOL_SOURCE_SHA256,
            "xml_sha256": hashlib.sha256(long_wrench_scene_xml(seed).encode("utf8")).hexdigest(),
            "compiled": {
                "wrench_mass_kg": float(self.model.body_mass[tool_id]),
                "nut_mass_kg": float(self.model.body_mass[nut_id]),
                "wrench_body_inertia_kg_m2": self.model.body_inertia[tool_id].tolist(),
                "nut_body_inertia_kg_m2": self.model.body_inertia[nut_id].tolist(),
                "handle_endpoints_m": endpoints,
                "handle_radius_m": float(self.model.geom_size[handle_id, 0]),
                "head_site_local_m": self.model.site_pos[site_id].tolist(),
                "initial_wrench_position_m": self.model.body_pos[tool_id].tolist(),
                "initial_wrench_quat_wxyz": self.model.body_quat[tool_id].tolist(),
                "nut_position_m": self.model.body_pos[nut_id].tolist(),
                "settled_wrench_position_m": self.data.xpos[tool_id].tolist(),
                "nq": self.model.nq, "nv": self.model.nv, "nu": self.model.nu,
                "nbody": self.model.nbody, "ngeom": self.model.ngeom,
                "timestep_s": self.model.opt.timestep,
            },
            "compiled_proof": {
                "wrench_mass_matches_declared": bool(abs(
                    self.model.body_mass[tool_id] - GEOMETRY_SPEC["total_wrench_mass_kg"]) <= 1e-12),
                "nut_mass_matches_declared": bool(abs(
                    self.model.body_mass[nut_id] - WRENCH_SPEC["nut_mass_kg"]) <= 1e-12),
                "handle_endpoints_match_declared": bool(np.allclose(
                    endpoints, GEOMETRY_SPEC["handle_endpoints_m"], atol=1e-12, rtol=0)),
                "handle_radius_matches_declared": bool(abs(
                    self.model.geom_size[handle_id, 0] - GEOMETRY_SPEC["handle_radius_m"]) <= 1e-12),
                "head_site_on_wrench_body": bool(self.model.site_bodyid[site_id] == tool_id),
                "head_site_matches_declared": bool(np.allclose(
                    self.model.site_pos[site_id], GEOMETRY_SPEC["head_center_local_m"], atol=1e-12, rtol=0)),
            },
        }
