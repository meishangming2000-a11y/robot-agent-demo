"""Small NumPy policy for bounded offsets to the frozen wrench reference.

The network selects five parameters once from initial tool XY. It only adjusts
existing robot command targets; the reference retains its complete phase logic.
"""

from collections.abc import Mapping
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .scene import ROOT
from .wrench import WrenchReference


PARAMETER_NAMES = ["grasp_y_offset_m", "grasp_z_offset_m", "tool_pose_x_offset_m",
                   "tool_pose_y_offset_m", "tool_pose_z_offset_m"]
PARAMETER_BOUNDS_M = [0.003, 0.003, 0.004, 0.004, 0.004]
CONTEXT_CENTER_M = [0.53, -0.13]
CONTEXT_SCALE_M = [0.01, 0.01]
_WEIGHT_SHAPES = {"W0": (16, 2), "b0": (16,), "W1": (16, 16),
                  "b1": (16,), "W2": (5, 16), "b2": (5,)}
_BASE_SOURCES = ("dexhand_agent/wrench.py", "dexhand_agent/runtime.py",
                 "dexhand_agent/scene.py", "dexhand_agent/wrench_evaluate.py",
                 "assets/manifest.json")


def _finite_vector(value, length, name):
    vector = np.array(value, dtype=np.float64, copy=True)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return vector


def context_from_observation(observation):
    """Normalize initial tool XY, in meters, without clipping the context."""
    xy = _finite_vector(observation["tool"]["position_m"][:2], 2, "tool XY")
    return (xy - CONTEXT_CENTER_M) / CONTEXT_SCALE_M


class ParameterWrenchReference(WrenchReference):
    """Apply validated meter offsets to existing reference command targets."""

    def __init__(self, parameters):
        parameters = _finite_vector(parameters, 5, "parameters")
        if np.any(np.abs(parameters) > PARAMETER_BOUNDS_M):
            raise ValueError("parameters exceed their meter bounds")
        parameters.setflags(write=False)
        self.parameters_m = parameters
        super().__init__()

    def decide(self, observation):
        approach = self._phase == "approach"
        action = super().decide(observation)
        if approach and np.any(self.parameters_m[:2]):
            for axis, offset in zip((1, 2), self.parameters_m[:2]):
                self._grasp[axis] += float(offset)
                action["palm_target_m"][axis] += float(offset)
        return action

    def _tool_pose_action(self, observation, center, yaw, stage, duration):
        action = super()._tool_pose_action(observation, center, yaw, stage, duration)
        if np.any(self.parameters_m[2:]):
            action["palm_target_m"] = (
                np.asarray(action["palm_target_m"]) + self.parameters_m[2:]).tolist()
        return action


def _validated_weights(weights):
    if not isinstance(weights, Mapping) or set(weights) != set(_WEIGHT_SHAPES):
        raise ValueError("weights must contain exactly W0, b0, W1, b1, W2, b2")
    result = {}
    for name, shape in _WEIGHT_SHAPES.items():
        array = np.asarray(weights[name])
        if array.dtype != np.dtype(np.float32) or array.shape != shape:
            raise ValueError(f"{name} must have float32 dtype and shape {shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite weights")
        result[name] = array.copy()
        result[name].setflags(write=False)
    return result


def _metadata_contract():
    if version("mujoco") != "3.5.0":
        raise ValueError("Policy requires MuJoCo 3.5.0")
    return {
        "format_version": 1,
        "architecture": [2, 16, 16, 5],
        "activation": "tanh",
        "normalization": {"observation": "tool.position_m[:2]",
                          "center_m": CONTEXT_CENTER_M, "scale_m": CONTEXT_SCALE_M},
        "units": "m",
        "parameter_names": PARAMETER_NAMES,
        "parameter_bounds_m": PARAMETER_BOUNDS_M,
        "weights_file": "weights.npz",
        "mujoco_version": "3.5.0",
        "base_source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in _BASE_SOURCES},
    }


def _reject_json_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _finite_json_float(value):
    number = float(value)
    if not np.isfinite(number):
        _reject_json_constant(value)
    return number


def _verify_vendor_assets():
    manifest = json.loads((ROOT / "assets/manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        try:
            payload = (ROOT / "assets/vendor" / item["path"]).read_bytes()
        except OSError as error:
            raise ValueError(f"Missing vendor asset: {item['path']}") from error
        if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError(f"Vendor asset hash mismatch: {item['path']}")


class TinyParameterPolicy:
    """Fixed 2→16→16→5 tanh network; inference requires only NumPy."""

    def __init__(self, weights):
        self._weights = _validated_weights(weights)
        self.metadata = {}

    def predict(self, context):
        context = _finite_vector(context, 2, "context")
        with np.errstate(over="raise", invalid="raise"):
            hidden = context.astype(np.float32)
            for layer in range(3):
                hidden = np.tanh(self._weights[f"W{layer}"] @ hidden
                                 + self._weights[f"b{layer}"])
        return np.asarray(PARAMETER_BOUNDS_M) * hidden

    @classmethod
    def load(cls, policy_dir):
        directory = Path(policy_dir)
        metadata = json.loads((directory / "policy.json").read_text(encoding="utf-8"),
                              parse_constant=_reject_json_constant, parse_float=_finite_json_float)
        if not isinstance(metadata, dict):
            raise ValueError("policy.json must be an object")
        for field, expected in _metadata_contract().items():
            if metadata.get(field) != expected or (field == "format_version"
                                                  and type(metadata[field]) is not int):
                raise ValueError(f"Incompatible policy metadata: {field}")
        if not isinstance(metadata.get("training_provenance"), dict):
            raise ValueError("Policy requires training_provenance")
        payload = (directory / "weights.npz").read_bytes()
        if hashlib.sha256(payload).hexdigest() != metadata.get("weights_sha256"):
            raise ValueError("Policy weights hash mismatch")
        _verify_vendor_assets()
        try:
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                if len(archive.files) != len(_WEIGHT_SHAPES) or set(archive.files) != set(_WEIGHT_SHAPES):
                    raise ValueError("Unexpected weight array names")
                policy = cls({name: archive[name] for name in _WEIGHT_SHAPES})
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid policy weights: {error}") from error
        policy.metadata = metadata
        return policy


def load_policy(policy_dir):
    return TinyParameterPolicy.load(policy_dir)


def predict_parameters(policy_dir, context):
    return load_policy(policy_dir).predict(context)


def save_policy(policy_dir, weights, provenance):
    """Publish a complete JSON/NPZ bundle atomically to a new directory."""
    destination = Path(policy_dir)
    if destination.exists():
        raise FileExistsError(destination)
    weights = _validated_weights(weights)
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be an object")
    metadata = _metadata_contract()
    metadata["training_provenance"] = provenance
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        np.savez(staging / "weights.npz", **weights)
        metadata["weights_sha256"] = hashlib.sha256((staging / "weights.npz").read_bytes()).hexdigest()
        (staging / "policy.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(staging, destination)
    return destination


class NeuralWrenchReference(ParameterWrenchReference):
    """Infer a fixed parameter vector once from the initial observation."""

    def __init__(self, policy_dir, initial_observation):
        self.policy = load_policy(policy_dir)
        self.context = context_from_observation(initial_observation)
        super().__init__(self.policy.predict(self.context))
