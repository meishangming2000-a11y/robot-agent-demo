"""Bounded policy/controller contracts, without constructing a physics model."""

import copy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from dexhand_agent import wrench_policy
from dexhand_agent.scene import ROOT
from dexhand_agent.wrench import WrenchReference
from dexhand_agent.wrench_policy import (
    CONTEXT_CENTER_M, CONTEXT_SCALE_M, PARAMETER_BOUNDS_M, PARAMETER_NAMES,
    NeuralWrenchReference, ParameterWrenchReference, TinyParameterPolicy,
    context_from_observation, load_policy, predict_parameters, save_policy,
)


@pytest.fixture
def observation():
    return {
        "tool": {"position_m": [0.53, -0.13, 0.242],
                 "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                 "head_center_m": [0.643, -0.13, 0.242]},
        "robot": {"palm_position_m": [0.53, -0.105, 0.414],
                  "palm_axes_world": np.eye(3).tolist()},
        "nut": {"position_m": [0.53, 0.16, 0.245], "angle_rad": 0.0},
    }


@pytest.fixture
def weights():
    return {"W0": np.zeros((16, 2), dtype=np.float32),
            "b0": np.zeros(16, dtype=np.float32),
            "W1": np.zeros((16, 16), dtype=np.float32),
            "b1": np.zeros(16, dtype=np.float32),
            "W2": np.zeros((5, 16), dtype=np.float32),
            "b2": np.zeros(5, dtype=np.float32)}


def test_zero_offsets_match_every_reference_action_exactly(observation):
    before = copy.deepcopy(observation)
    reference = WrenchReference()
    policy = ParameterWrenchReference([0.0] * 5)
    for _ in range(24):
        assert policy.decide(observation) == reference.decide(observation)
        assert policy._phase == reference._phase
        assert policy._grasp == reference._grasp
    assert policy._phase == "finish"
    assert observation == before


def test_offsets_affect_only_declared_existing_command_targets(observation):
    parameters = [0.001, -0.002, 0.003, -0.004, 0.002]
    reference = WrenchReference()
    policy = ParameterWrenchReference(parameters)
    grasp_stages = {"wrench_approach", "wrench_descend", "wrench_lift"}
    tool_pose_stages = {
        "wrench_level_1", "wrench_level_2", "wrench_align_level_1",
        "wrench_align_level_2", "wrench_insert", "wrench_turn_1",
        "wrench_turn_2", "wrench_turn_3", "wrench_unload", "wrench_withdraw",
    }
    seen = set()
    for _ in range(24):
        expected = reference.decide(observation)
        actual = policy.decide(observation)
        stage = expected["memory"].split(": ")[1]
        seen.add(stage)
        expected_target = expected.pop("palm_target_m")
        actual_target = actual.pop("palm_target_m")
        assert actual == expected
        assert policy._phase == reference._phase
        if stage in grasp_stages:
            np.testing.assert_allclose(np.subtract(actual_target, expected_target),
                                       [0.0, *parameters[:2]], atol=1e-16)
        elif stage in tool_pose_stages:
            np.testing.assert_allclose(np.subtract(actual_target, expected_target),
                                       parameters[2:], atol=1e-16)
        else:
            assert actual_target == expected_target
    assert grasp_stages | tool_pose_stages <= seen


@pytest.mark.parametrize("parameters", [
    [], [0.0] * 4, [0.0] * 6, [[0.0] * 5],
    [np.nan, 0, 0, 0, 0], [0, np.inf, 0, 0, 0],
    [0.003001, 0, 0, 0, 0], [0, -0.003001, 0, 0, 0],
    [0, 0, 0.004001, 0, 0], [0, 0, 0, -0.004001, 0],
    [0, 0, 0, 0, 0.004001],
])
def test_parameter_adapter_rejects_invalid_vectors(parameters):
    with pytest.raises(ValueError):
        ParameterWrenchReference(parameters)


def test_inclusive_parameter_bounds_and_input_copy():
    parameters = np.array(PARAMETER_BOUNDS_M)
    policy = ParameterWrenchReference(parameters)
    parameters[:] = 0
    np.testing.assert_array_equal(policy.parameters_m, PARAMETER_BOUNDS_M)
    ParameterWrenchReference(-np.array(PARAMETER_BOUNDS_M))


def test_context_uses_initial_tool_xy_in_meters(observation):
    np.testing.assert_array_equal(context_from_observation(observation), [0, 0])
    observation["tool"]["position_m"][:2] = [0.54, -0.14]
    np.testing.assert_allclose(context_from_observation(observation), [1, -1])
    assert CONTEXT_CENTER_M == [0.53, -0.13]
    assert CONTEXT_SCALE_M == [0.01, 0.01]


@pytest.mark.parametrize("context", [[], [0], [0, 0, 0], [[0, 0]], [np.nan, 0], [0, np.inf]])
def test_prediction_rejects_invalid_context(weights, context):
    with pytest.raises(ValueError):
        TinyParameterPolicy(weights).predict(context)


def test_known_network_roundtrip_and_metadata(tmp_path, weights):
    weights["W0"][0, 0] = 0.5
    weights["W0"][1, 1] = -0.25
    weights["W1"][0, 0] = 0.75
    weights["W1"][1, 1] = 0.5
    weights["W2"][:, 0] = np.arange(1, 6, dtype=np.float32) / 5
    weights["W2"][:, 1] = -0.25
    weights["b2"][:] = 0.1
    directory = tmp_path / "policy"
    provenance = {"kind": "unit_test_fixture", "training_claim": False}
    save_policy(directory, weights, provenance)
    expected = np.array(PARAMETER_BOUNDS_M) * np.tanh(
        np.arange(1, 6, dtype=np.float32) / 5 * np.tanh(0.75 * np.tanh(0.5 * 0.8))
        - 0.25 * np.tanh(0.5 * np.tanh(-0.25 * -0.6)) + 0.1)
    loaded = load_policy(directory)
    np.testing.assert_allclose(loaded.predict([0.8, -0.6]), expected, rtol=2e-7)
    np.testing.assert_array_equal(predict_parameters(directory, [0.8, -0.6]),
                                  loaded.predict([0.8, -0.6]))
    metadata = json.loads((directory / "policy.json").read_text(encoding="utf-8"))
    assert metadata["architecture"] == [2, 16, 16, 5]
    assert metadata["activation"] == "tanh"
    assert metadata["parameter_names"] == PARAMETER_NAMES
    assert metadata["parameter_bounds_m"] == PARAMETER_BOUNDS_M
    assert metadata["normalization"] == {
        "observation": "tool.position_m[:2]", "center_m": CONTEXT_CENTER_M,
        "scale_m": CONTEXT_SCALE_M}
    assert metadata["units"] == "m"
    assert metadata["mujoco_version"] == "3.5.0"
    assert metadata["training_provenance"] == provenance
    assert metadata["weights_sha256"] == hashlib.sha256((directory / "weights.npz").read_bytes()).hexdigest()
    assert set(metadata["base_source_sha256"]) == {
        "dexhand_agent/wrench.py", "dexhand_agent/runtime.py", "dexhand_agent/scene.py",
        "dexhand_agent/wrench_evaluate.py", "assets/manifest.json"}
    for name, digest in metadata["base_source_sha256"].items():
        assert digest == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def test_neural_reference_predicts_once_from_initial_observation(tmp_path, weights, observation):
    weights["W0"][0, 0] = 1
    weights["W1"][0, 0] = 1
    weights["W2"][:, 0] = 1
    directory = tmp_path / "policy"
    save_policy(directory, weights, {"kind": "unit_test_fixture"})
    observation["tool"]["position_m"][0] += 0.005
    reference = NeuralWrenchReference(directory, observation)
    expected = predict_parameters(directory, [0.5, 0])
    np.testing.assert_allclose(reference.parameters_m, expected, rtol=2e-7)
    observation["tool"]["position_m"][0] += 0.005
    reference.decide(observation)
    np.testing.assert_array_equal(reference.parameters_m, reference.policy.predict([0.5, 0]))


def test_saturated_network_stays_within_meter_bounds(weights):
    weights["b2"][:] = [100, -100, 100, -100, 100]
    parameters = TinyParameterPolicy(weights).predict([0, 0])
    np.testing.assert_array_equal(abs(parameters), PARAMETER_BOUNDS_M)
    ParameterWrenchReference(parameters)


def test_save_refuses_existing_destination(tmp_path, weights):
    directory = tmp_path / "policy"
    directory.mkdir()
    (directory / "keep.txt").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        save_policy(directory, weights, {})
    assert list(directory.iterdir()) == [directory / "keep.txt"]


def test_save_publishes_only_after_both_files_are_complete(tmp_path, weights, monkeypatch):
    directory = tmp_path / "policy"
    rename = wrench_policy.os.rename

    def publish(source, destination):
        assert not directory.exists()
        assert set(path.name for path in Path(source).iterdir()) == {"weights.npz", "policy.json"}
        load_policy(source)
        return rename(source, destination)

    monkeypatch.setattr(wrench_policy.os, "rename", publish)
    assert save_policy(directory, weights, {}) == directory
    assert list(tmp_path.iterdir()) == [directory]


def test_loader_rejects_weights_hash_mismatch(tmp_path, weights):
    directory = tmp_path / "policy"
    save_policy(directory, weights, {})
    with (directory / "weights.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        load_policy(directory)


@pytest.mark.parametrize("field, value", [
    ("format_version", 2), ("architecture", [2, 8, 5]),
    ("activation", "relu"), ("normalization", {}), ("units", "mm"),
    ("parameter_names", ["different"] * 5), ("parameter_bounds_m", [1] * 5),
    ("weights_file", "other.npz"), ("mujoco_version", "3.4.0"),
    ("base_source_sha256", {}),
])
def test_loader_rejects_incompatible_metadata(tmp_path, weights, field, value):
    directory = tmp_path / "policy"
    save_policy(directory, weights, {})
    path = directory / "policy.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[field] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        load_policy(directory)


def test_loader_rejects_incompatible_installed_mujoco(tmp_path, weights, monkeypatch):
    directory = tmp_path / "policy"
    save_policy(directory, weights, {})
    monkeypatch.setattr(wrench_policy, "version", lambda name: "3.4.0")
    with pytest.raises(ValueError, match="MuJoCo"):
        load_policy(directory)


@pytest.mark.parametrize("tamper", [False, True])
def test_loader_checks_vendor_payloads_against_sealed_manifest(tmp_path, weights, monkeypatch, tamper):
    source_root = tmp_path / "source"
    for name in ("wrench.py", "runtime.py", "scene.py", "wrench_evaluate.py"):
        target = source_root / "dexhand_agent" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("source fixture", encoding="utf-8")
    asset = source_root / "assets/vendor/robot/assets/mesh.obj"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"original")
    manifest = {"files": [{"path": "robot/assets/mesh.obj", "bytes": 8,
                           "sha256": hashlib.sha256(b"original").hexdigest()}]}
    (source_root / "assets/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(wrench_policy, "ROOT", source_root)
    directory = tmp_path / "policy"
    save_policy(directory, weights, {})
    if tamper:
        asset.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="asset"):
            load_policy(directory)
    else:
        np.testing.assert_array_equal(load_policy(directory).predict([0, 0]), [0] * 5)


def test_loader_rejects_overflowed_json_number(tmp_path, weights):
    directory = tmp_path / "policy"
    save_policy(directory, weights, {"metric": 1.25})
    path = directory / "policy.json"
    path.write_text(path.read_text(encoding="utf-8").replace("1.25", "1e999"), encoding="utf-8")
    with pytest.raises(ValueError, match="Nonfinite"):
        load_policy(directory)


def test_loader_rejects_duplicate_npz_members_with_matching_hash(tmp_path, weights):
    directory = tmp_path / "policy"
    save_policy(directory, weights, {})
    payload = io.BytesIO()
    np.save(payload, weights["W0"], allow_pickle=False)
    with zipfile.ZipFile(directory / "weights.npz", "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("W0.npy", payload.getvalue())
    path = directory / "policy.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["weights_sha256"] = hashlib.sha256((directory / "weights.npz").read_bytes()).hexdigest()
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        load_policy(directory)


@pytest.mark.parametrize("bad_value", [
    np.zeros((16, 3), dtype=np.float32), np.zeros((16, 2), dtype=np.float64),
    np.full((16, 2), np.nan, dtype=np.float32),
    np.full((16, 2), np.inf, dtype=np.float32),
    np.full((16, 2), None, dtype=object),
])
def test_loader_rejects_invalid_arrays_even_with_matching_hash(tmp_path, weights, bad_value):
    directory = tmp_path / "policy"
    save_policy(directory, weights, {})
    weights["W0"] = bad_value
    np.savez(directory / "weights.npz", **weights)
    metadata_path = directory / "policy.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["weights_sha256"] = hashlib.sha256((directory / "weights.npz").read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        load_policy(directory)


@pytest.mark.parametrize("change", ["missing", "extra", "wrong_shape", "wrong_dtype", "nonfinite"])
def test_in_memory_network_rejects_malformed_weights(weights, change):
    if change == "missing":
        weights.pop("b2")
    elif change == "extra":
        weights["extra"] = np.zeros(1, dtype=np.float32)
    elif change == "wrong_shape":
        weights["b2"] = np.zeros(6, dtype=np.float32)
    elif change == "wrong_dtype":
        weights["b2"] = np.zeros(5, dtype=np.float64)
    else:
        weights["b2"][0] = np.nan
    with pytest.raises(ValueError):
        TinyParameterPolicy(weights)
