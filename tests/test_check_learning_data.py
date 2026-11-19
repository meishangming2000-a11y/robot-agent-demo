import hashlib
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


@pytest.fixture
def episode(tmp_path, monkeypatch):
    from dexhand_agent.learning_data import SynchronizedRecorder

    class Renderer:
        def __init__(self, model, height, width):
            self.shape = (height, width, 3)

        def update_scene(self, data, camera):
            pass

        def render(self):
            return np.zeros(self.shape, dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(mujoco, "Renderer", Renderer)

    def make(*, incomplete=False, unavailable_final=False, directory_name="episode"):
        model = mujoco.MjModel.from_xml_string("""
            <mujoco><option timestep=".01" gravity="0 0 0"/>
              <worldbody><body><joint name="slide" type="slide" damping=".1"/>
                <geom type="sphere" size=".03" mass="1"/></body></worldbody>
              <actuator><motor name="drive" joint="slide"/></actuator>
            </mujoco>""")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        runtime = SimpleNamespace(model=model, data=data, on_sample=None)
        binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, None, binary)
        path = tmp_path / directory_name
        recorder = SynchronizedRecorder(
            runtime, path, width=4, height=3, sample_period_s=.02,
            provenance={"model_binary": {"file": "model.mjb", "bytes": binary.size,
                         "sha256": hashlib.sha256(binary).hexdigest()}})
        (path / "model.mjb").write_bytes(binary.tobytes())
        for index, target in enumerate((.8, -.4)):
            recorder.begin_action({"operation": "move", "duration_s": .04,
                                   "targets": [{"actuator": "drive", "value": target}]})
            before = float(data.ctrl[0])
            for tick in range(4):
                data.ctrl[0] = before + ((tick + 1) / 4) * (target - before)
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                runtime.on_sample({"time_s": float(data.time)})
                if incomplete and index == 1 and tick == 1:
                    if unavailable_final:
                        data.qpos[:] = np.nan
                    recorder.finalize(status="failed", outcome={"error": "interrupted"})
                    return path
            recorder.end_action({"status": "completed", "time_s": float(data.time)})
        recorder.finalize(status="completed", outcome={"success": False})
        return path

    return make


def change_arrays(path, mutation):
    with np.load(path / "samples.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    mutation(arrays)
    np.savez_compressed(path / "samples.npz", **arrays)


def change_manifest(path, mutation):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    mutation(manifest)
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_checks_saved_controls_without_a_policy_or_task_success(episode):
    from tools.check_learning_data import validate_episode

    report = validate_episode(episode())
    assert report["accepted"], report["errors"]
    assert report["counts"]["frames"] == 5
    assert report["counts"]["ticks"] == 8
    assert report["counts"]["complete_actions"] == 2
    assert report["counts"]["complete_frame_intervals"] == 4
    assert report["counts"]["replayed_intervals"] >= 3
    assert report["max_deviations"]["qpos"] < 1e-12
    assert report["max_deviations"]["qvel"] < 1e-12
    assert set(report["input_sha256"]) == {"manifest.json", "samples.npz", "model.mjb"}


def test_absolute_unicode_model_path_loads_verified_bytes_and_replays(episode):
    from tools.check_learning_data import validate_episode

    path = episode(directory_name="\u4e2d\u6587")
    report = validate_episode(path.resolve())
    assert report["accepted"], report["errors"]
    assert report["input_sha256"]["model.mjb"] == hashlib.sha256(
        (path / "model.mjb").read_bytes()).hexdigest()
    assert report["counts"]["replayed_intervals"] == 4
    assert report["max_deviations"] == {"qpos": 0., "qvel": 0., "ctrl": 0.}


@pytest.mark.parametrize("mutation, message", [
    (lambda a: a["qpos"].__setitem__((1, 0), np.nan), "finite"),
    (lambda a: a.__setitem__("rgb", a["rgb"].astype(np.float64)), "dtype"),
    (lambda a: a["tick_time_s"].__setitem__(2, a["tick_time_s"][2] + .001), "tick"),
    (lambda a: a["action_index"].__setitem__(2, 1), "incoming"),
    (lambda a: a["qpos"].__setitem__((1, 0), a["qpos"][1, 0] + .001), "state slice"),
])
def test_rejects_corrupt_arrays(episode, mutation, message):
    from tools.check_learning_data import validate_episode

    path = episode()
    change_arrays(path, mutation)
    report = validate_episode(path)
    assert not report["accepted"]
    assert any(message in error for error in report["errors"])


def test_replay_detects_wrong_control_between_saved_frames(episode):
    from tools.check_learning_data import validate_episode

    path = episode()
    change_arrays(path, lambda a: a["tick_ctrl"].__setitem__((0, 0), 1.7))
    report = validate_episode(path)
    assert not report["accepted"]
    assert any("replay" in error for error in report["errors"])


@pytest.mark.parametrize("mutation, message", [
    (lambda m: m.__setitem__("mujoco_version", "0.0.0"), "version"),
    (lambda m: m["provenance"]["model_binary"].__setitem__("sha256", "0" * 64), "model binary"),
    (lambda m: m["actions"][0].__setitem__("end_time_s", .03), "boundary"),
    (lambda m: m["actions"][0]["action"].__setitem__("duration_s", .05), "elapsed"),
])
def test_rejects_wrong_model_version_or_action_labels(episode, mutation, message):
    from tools.check_learning_data import validate_episode

    path = episode()
    change_manifest(path, mutation)
    report = validate_episode(path)
    assert not report["accepted"]
    assert any(message in error for error in report["errors"])


@pytest.mark.parametrize("unavailable_final", [False, True])
def test_failed_incomplete_action_is_retained_but_excluded(episode, unavailable_final):
    from tools.check_learning_data import validate_episode

    path = episode(incomplete=True, unavailable_final=unavailable_final)
    report = validate_episode(path)
    assert report["accepted"], report["errors"]
    assert report["counts"]["complete_actions"] == 1
    assert report["counts"]["incomplete_actions"] == 1
    assert report["counts"]["complete_frame_intervals"] == 2
    assert report["counts"]["excluded_frame_intervals"] == 1
    change_manifest(path, lambda m: m.__setitem__("status", "completed"))
    assert not validate_episode(path)["accepted"]


def test_cli_records_invalid_result_and_refuses_overwrite(episode, tmp_path):
    from tools.check_learning_data import main

    path = episode()
    output = tmp_path / "report.json"
    assert main(["--episode", str(path), "--output", str(output)]) == 0
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(["--episode", str(path), "--output", str(output)])
    assert output.read_bytes() == saved
    change_arrays(path, lambda a: a["tick_ctrl"].__setitem__((0, 0), 1.7))
    invalid = tmp_path / "invalid.json"
    assert main(["--episode", str(path), "--output", str(invalid)]) == 1
    assert not json.loads(invalid.read_text(encoding="utf-8"))["accepted"]
