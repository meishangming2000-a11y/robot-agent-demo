"""Predeclared same-wrench placement change; task mechanics stay unchanged."""
import copy
import hashlib
import json
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .runtime import HandRuntime
from .scene import make_model
from .wrench import WRENCH_SPEC, WrenchRuntime, wrench_scene_xml


VARIANT_ID = "placement-30deg-v1"
PLACEMENT_SPEC = {
    "wrench_rack_translation_m": [-.03, .025, 0.],
    "wrench_rack_yaw_deg": 30.,
    "rotation_pivot": "seeded base wrench body center; translate that pivot, rotate wrench and all three supports together",
    "nut_pedestal_translation_m": [.02, .03, 0.],
}
PLACEMENT_SPEC_SHA256 = hashlib.sha256(json.dumps(
    PLACEMENT_SPEC, sort_keys=True, separators=(",", ":"), allow_nan=False,
).encode("utf8")).hexdigest()
PROTOCOL_SOURCE_SHA256 = "74e19612f4d97a6b90f4648a0240e97dc2510d70c66dbc1025548000c52d9898"
VARIANTS = {
    VARIANT_ID: {"spec": PLACEMENT_SPEC, "protocol_source_sha256": PROTOCOL_SOURCE_SHA256},
    "placement-minus20deg-v2": {
        "spec": {
            "wrench_rack_translation_m": [.015, -.020, 0.],
            "wrench_rack_yaw_deg": -20.,
            "rotation_pivot": PLACEMENT_SPEC["rotation_pivot"],
            "nut_pedestal_translation_m": [-.025, .015, 0.],
        },
        "protocol_source_sha256": "98f0bc5b01c115571d9aec2e0a67d70dc4f6e8ff2a72de174d0e1e31da9ee45c",
    },
    "placement-plus15deg-v3": {
        "spec": {
            "wrench_rack_translation_m": [-.02, -.015, 0.],
            "wrench_rack_yaw_deg": 15.,
            "rotation_pivot": PLACEMENT_SPEC["rotation_pivot"],
            "nut_pedestal_translation_m": [.015, .02, 0.],
        },
        "protocol_source_sha256": "44ab8c2401dc7ecd4b2dfdc80baa9f35415f8ed38577dae9fcf493ea05f24e0f",
    },
}


def _text(vector):
    return " ".join(str(float(value)) for value in vector)


def placement_scene_xml(seed=0, variant=VARIANT_ID):
    """Transform declared initial XML poses before normal model compilation."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown wrench variant: {variant}")
    spec = VARIANTS[variant]["spec"]
    root = ET.fromstring(wrench_scene_xml(seed))
    world = root.find("worldbody")
    wrench = world.find("body[@name='wrench']")
    pivot = np.fromstring(wrench.get("pos"), sep=" ")
    shift = np.asarray(spec["wrench_rack_translation_m"])
    angle = math.radians(spec["wrench_rack_yaw_deg"])
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    yaw = np.array([math.cos(angle / 2), 0, 0, math.sin(angle / 2)])
    for element in [wrench, *[world.find(f"geom[@name='{name}']") for name in (
            "wrench_rack_left", "wrench_rack_right", "wrench_rack_head")]]:
        position = np.fromstring(element.get("pos"), sep=" ")
        element.set("pos", _text(pivot + shift + rotation @ (position - pivot)))
        quaternion = np.zeros(4)
        mujoco.mju_mulQuat(quaternion, yaw, np.fromstring(element.get("quat", "1 0 0 0"), sep=" "))
        element.set("quat", _text(quaternion))
    nut_shift = np.asarray(spec["nut_pedestal_translation_m"])
    for element in (world.find("body[@name='hexnut']"), world.find("geom[@name='nut_pedestal']")):
        element.set("pos", _text(np.fromstring(element.get("pos"), sep=" ") + nut_shift))
    return ET.tostring(root, encoding="unicode")


class PlacementWrenchRuntime(WrenchRuntime):
    """Change construction only; execution, observations and sampling are inherited."""

    def __init__(self, seed=0, variant=VARIANT_ID):
        xml = placement_scene_xml(seed, variant=variant)
        descriptor = VARIANTS[variant]
        spec = descriptor["spec"]
        def model_factory(current_seed):
            return make_model(current_seed, scene_factory=lambda _: xml)
        HandRuntime.__init__(self, seed, model_factory=model_factory, tool_body="wrench")
        nut_id = self.model.body("hexnut").id
        tool_id = self.model.body("wrench").id
        site_id = self.model.site("wrench_head_center").id
        self.initial_nut_angle_rad = float(self.data.qpos[self.model.joint("nut_hinge").qposadr[0]])
        self.wrench_geometry_spec = copy.deepcopy(WRENCH_SPEC)
        self.wrench_geometry_spec["nut_position_m"] = self.model.body_pos[nut_id].tolist()
        self.variant_metadata = {
            "variant_id": variant, "seed": seed,
            "spec": copy.deepcopy(spec), "spec_sha256": hashlib.sha256(json.dumps(
                spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest(),
            "source_protocol_sha256": descriptor["protocol_source_sha256"],
            "xml_sha256": hashlib.sha256(xml.encode("utf8")).hexdigest(),
            "compiled": {
                "wrench_mass_kg": float(self.model.body_mass[tool_id]),
                "nut_mass_kg": float(self.model.body_mass[nut_id]),
                "wrench_body_inertia_kg_m2": self.model.body_inertia[tool_id].tolist(),
                "nut_body_inertia_kg_m2": self.model.body_inertia[nut_id].tolist(),
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
                "wrench_mass_matches_declared": abs(self.model.body_mass[tool_id] - WRENCH_SPEC["wrench_mass_kg"]) <= 1e-12,
                "nut_mass_matches_declared": abs(self.model.body_mass[nut_id] - WRENCH_SPEC["nut_mass_kg"]) <= 1e-12,
                "head_site_on_wrench_body": bool(self.model.site_bodyid[site_id] == tool_id),
                "head_site_matches_declared": bool(np.allclose(
                    self.model.site_pos[site_id], WRENCH_SPEC["head_center_local_m"], atol=1e-12, rtol=0)),
            },
        }
        self.variant_metadata["compiled_proof"] = {
            key: bool(value) for key, value in self.variant_metadata["compiled_proof"].items()}

