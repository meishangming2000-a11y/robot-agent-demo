"""Compose public robot descriptions with an original primitive tool fixture.

The Shadow forearm is omitted. Its wrist/hand subtree is mounted on Panda's
attachment body. Source robot descriptions are preserved in assets/vendor.
"""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PANDA_HOME = [0.0, -0.35, 0.0, -1.9, 0.0, 1.65, -0.785398]


def _description(folder, filename):
    directory = ROOT / "assets" / "vendor" / folder
    document = ET.parse(directory / filename).getroot()
    for mesh in document.findall("./asset/mesh"):
        filename = mesh.get("file")
        mesh.set("name", mesh.get("name", Path(filename).stem))
        mesh.set("file", f"{folder}_{Path(filename).name}")
    return document


def scene_xml(seed=0):
    panda = _description("franka_emika_panda", "panda_nohand.xml")
    hand = _description("shadow_hand", "right_hand.xml")
    panda.set("model", "panda_shadow_hammer_button")
    panda.find("compiler").attrib.pop("meshdir", None)
    panda.find("option").attrib.update(timestep="0.002", gravity="0 0 -9.81", cone="elliptic")
    ET.SubElement(panda, "visual").append(ET.Element("global", offwidth="960", offheight="720"))
    panda.find("visual").append(ET.Element("map", znear="0.01"))
    panda.find("default").extend(deepcopy(list(hand.find("default"))))
    panda.find("asset").extend(deepcopy(list(hand.find("asset"))))
    wrist = deepcopy(hand.find(".//body[@name='rh_wrist']"))
    wrist.set("pos", "0 0 0.012")
    wrist.set("childclass", "right_hand")
    attachment = panda.find(".//body[@name='attachment']")
    attachment.append(ET.Element("geom", name="mount_visual", type="cylinder", size="0.03 0.012",
                                 pos="0 0 0", rgba="0.2 0.35 0.4 1", contype="0", conaffinity="0", mass="0.05"))
    attachment.append(wrist)
    wrist.find(".//body[@name='rh_palm']").append(
        ET.Element("site", name="palm_frame", pos="0 0 0.05", size="0.004", rgba="0 1 0 0.3"))
    for section in ("tendon", "actuator", "contact"):
        destination = panda.find(section)
        if destination is None:
            destination = ET.SubElement(panda, section)
        for element in hand.find(section):
            if "rh_forearm" not in element.attrib.values():
                destination.append(deepcopy(element))
    keyframe = panda.find("keyframe")
    if keyframe is not None:
        panda.remove(keyframe)
    world = panda.find("worldbody")
    world.append(ET.Element("light", pos="0.4 -0.3 1.8", dir="0 0 -1", diffuse="0.8 0.8 0.8"))
    world.append(ET.Element("light", pos="0 1 1", dir="0 -1 -1", diffuse="0.5 0.5 0.5"))
    world.append(ET.Element("camera", name="overview", pos="1.5 -1.3 1.15", xyaxes="0.72 0.69 0 -0.23 0.24 0.94"))
    world.append(ET.Element("camera", name="front", pos="0.6 -1.0 0.6", xyaxes="1 0 0 0 0.35 0.94"))
    world.append(ET.Element("geom", name="floor", type="plane", size="2 2 0.1", rgba="0.12 0.15 0.19 1"))
    world.append(ET.Element("geom", name="table", type="box", pos="0.55 0 0.07", size="0.3 0.38 0.07",
                            rgba="0.58 0.49 0.36 1", friction="0.8 0.02 0.001"))
    rng = np.random.default_rng(seed)
    offset = rng.uniform(-0.01, 0.01, 2) if seed else np.zeros(2)
    x, y = np.array([0.53, -0.13]) + offset
    for dx in (-0.055, 0.055):
        world.append(ET.Element("geom", name=f"rack_{'left' if dx < 0 else 'right'}", type="box",
                                pos=f"{x+dx} {y} 0.185", size="0.009 0.025 0.045", rgba="0.25 0.3 0.36 1"))
    world.append(ET.Element("geom", name="rack_head", type="box", pos=f"{x+0.105} {y} 0.1785",
                            size="0.018 0.035 0.0385", rgba="0.25 0.3 0.36 1"))
    hammer = ET.SubElement(world, "body", name="hammer", pos=f"{x} {y} 0.242")
    ET.SubElement(hammer, "freejoint", name="hammer_free")
    ET.SubElement(hammer, "geom", name="hammer_handle", type="capsule", fromto="-0.085 0 0 0.085 0 0",
                  size="0.01", mass="0.035", rgba="0.65 0.28 0.07 1", friction="1.1 0.02 0.002")
    ET.SubElement(hammer, "geom", name="hammer_head", type="box", pos="0.105 0 0", size="0.023 0.04 0.023",
                  mass="0.13", rgba="0.22 0.25 0.3 1", friction="0.8 0.01 0.001")
    ET.SubElement(hammer, "site", name="hammer_work", pos="0.105 0 -0.023", size="0.004", rgba="1 0.4 0 0.4")
    world.append(ET.Element("geom", name="button_base", type="cylinder", pos="0.53 0.16 0.17", size="0.048 0.035",
                            rgba="0.2 0.3 0.35 1"))
    button = ET.SubElement(world, "body", name="button", pos="0.53 0.16 0.23", gravcomp="1")
    ET.SubElement(button, "joint", name="button_slide", type="slide", axis="0 0 1", range="-0.012 0",
                  stiffness="200", damping="0.5")
    ET.SubElement(button, "geom", name="button_cap", type="cylinder", size="0.031 0.01", mass="0.025",
                  rgba="0.1 0.65 0.4 1", friction="0.6 0.01 0.001")
    return ET.tostring(panda, encoding="unicode")


def make_model(seed=0, scene_factory=None):
    # VFS also supports Windows project directories containing non-ASCII text.
    assets = {}
    for folder in ("franka_emika_panda", "shadow_hand"):
        for path in (ROOT / "assets" / "vendor" / folder / "assets").iterdir():
            assets[f"{folder}_{path.name}"] = path.read_bytes()
    model = mujoco.MjModel.from_xml_string((scene_factory or scene_xml)(seed), assets=assets)
    data = mujoco.MjData(model)
    for index, value in enumerate(PANDA_HOME, 1):
        joint = model.joint(f"joint{index}")
        data.qpos[joint.qposadr[0]] = value
    mujoco.mj_forward(model, data)
    data.ctrl[:] = data.actuator_length
    return model, data
