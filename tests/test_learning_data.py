import json
import mujoco
import numpy as np
import pytest

from dexhand_agent.runtime import HandRuntime


class TinyRuntime:
    """Real MuJoCo dynamics and the public execute loop; graphics isolated below."""

    execute = HandRuntime.execute
    _validate = HandRuntime._validate

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string("""
            <mujoco><option timestep="0.01" gravity="0 0 0"/>
              <size nuserdata="2"/>
              <worldbody><camera name="overview" pos="0 -2 1"/>
                <body><joint name="hinge" damping="0.1"/>
                  <geom type="capsule" size="0.03" fromto="0 0 0 .2 0 0"/>
                </body><body name="marker" mocap="true" pos="0 0 .3"/>
              </worldbody>
              <actuator><general name="drive" joint="hinge" ctrllimited="true"
                ctrlrange="-2 2" dyntype="filterexact" dynprm=".03"
                gainprm="8" biasprm="0 -8 -.5" biastype="affine"/>
              </actuator>
            </mujoco>""")
        self.data = mujoco.MjData(self.model)
        self.data.qfrc_applied[:] = .01
        self.data.userdata[:] = [2., 3.]
        mujoco.mj_forward(self.model, self.data)
        self.on_sample = None
        self.samples = []
        self.actuator_ids = {"drive": 0}
        self.last_result = {}
        self.closed = False

    def physical_sample(self):
        return {"time_s": float(self.data.time)}

    def close(self):
        self.closed = True


@pytest.fixture
def graphics(monkeypatch):
    instances = []

    class Renderer:
        def __init__(self, model, height, width):
            self.height, self.width = height, width
            self.closed = False
            instances.append(self)

        def update_scene(self, data, camera):
            self.channels = np.rint([data.time * 100, data.ctrl[0] * 100,
                                     data.qpos[0] * 100]).astype(np.uint8)

        def render(self):
            return np.broadcast_to(self.channels, (self.height, self.width, 3)).copy()

        def close(self):
            self.closed = True

    monkeypatch.setattr(mujoco, "Renderer", Renderer)
    return instances


def recorder(runtime, path, **kwargs):
    from dexhand_agent.learning_data import SynchronizedRecorder

    return SynchronizedRecorder(runtime, path, width=8, height=6,
                                sample_period_s=.02, **kwargs)


def move(value=1., duration=.04):
    return {"operation": "move", "duration_s": duration,
            "targets": [{"actuator": "drive", "value": value}]}


def load(path):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return manifest, np.load(path / manifest["arrays_file"], allow_pickle=False)


def test_synchronous_frames_tick_controls_action_boundaries_and_callback(tmp_path, graphics):
    runtime = TinyRuntime()
    seen = []
    previous = seen.append
    runtime.on_sample = previous
    rec = recorder(runtime, tmp_path / "episode", provenance={"source": "unit-test"})
    action = move()
    assert rec.begin_action(action) == 0
    runtime.execute(action)
    rec.end_action(runtime.last_result)
    action["targets"][0]["value"] = -1.  # Original command must be retained.
    assert rec.begin_action({"operation": "hold", "duration_s": .02}) == 1
    runtime.execute({"operation": "hold", "duration_s": .02})
    rec.end_action(runtime.last_result)
    rec.finalize(status="completed", outcome={"passed": False, "evaluator": "test"})
    rec.close()

    manifest, arrays = load(tmp_path / "episode")
    assert len(seen) == 6
    assert runtime.on_sample is previous
    assert graphics[0].closed
    assert not runtime.closed
    np.testing.assert_allclose(arrays["time_s"], [0., .02, .04, .06])
    np.testing.assert_allclose(arrays["tick_time_s"], np.arange(1, 7) * .01)
    np.testing.assert_allclose(arrays["tick_start_time_s"], np.arange(6) * .01)
    np.testing.assert_allclose(arrays["tick_ctrl"][:, 0], [.15625, .5, .84375, 1, 1, 1])
    np.testing.assert_array_equal(arrays["tick_action_index"], [0, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(arrays["action_index"], [-1, 0, 0, 1])
    assert arrays["rgb"].dtype == np.uint8
    assert arrays["rgb"].shape == (4, 6, 8, 3)
    expected = np.rint(np.c_[arrays["time_s"], arrays["ctrl"][:, 0],
                             arrays["qpos"][:, 0]] * 100).astype(np.uint8)
    np.testing.assert_array_equal(arrays["rgb"][:, 0, 0], expected)
    assert manifest["actions"][0]["action"]["targets"][0]["value"] == 1.
    assert manifest["actions"][0]["start_time_s"] == 0.
    assert manifest["actions"][1]["start_sample_index"] == 2
    assert manifest["outcome"]["passed"] is False
    assert manifest["provenance"]["source"] == "unit-test"
    assert manifest["state_spec"]["name"] == "mjSTATE_INTEGRATION"
    assert manifest["state_spec"]["value"] == int(mujoco.mjtState.mjSTATE_INTEGRATION)
    assert manifest["joints"][0]["name"] == "hinge"
    assert manifest["actuators"][0]["name"] == "drive"
    assert manifest["arrays"]["integration_state"]["role"] == "privileged_teacher"
    assert manifest["arrays"]["rgb"]["role"] == "visual_student_input"
    for name in arrays.files:
        assert manifest["arrays"][name]["shape"] == list(arrays[name].shape)
        assert manifest["arrays"][name]["dtype"] == str(arrays[name].dtype)
        assert np.isfinite(arrays[name]).all()


def test_saved_integration_state_restores_and_continues(tmp_path, graphics):
    runtime = TinyRuntime()
    rec = recorder(runtime, tmp_path / "episode")
    rec.begin_action(move())
    runtime.execute(move())
    rec.end_action()
    rec.finalize(status="completed", outcome=None)
    _, arrays = load(tmp_path / "episode")
    restored = mujoco.MjData(runtime.model)
    mujoco.mj_setState(runtime.model, restored, arrays["integration_state"][-1],
                       mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(runtime.model, restored)
    for _ in range(5):
        mujoco.mj_step(runtime.model, runtime.data)
        mujoco.mj_forward(runtime.model, runtime.data)
        mujoco.mj_step(runtime.model, restored)
        mujoco.mj_forward(runtime.model, restored)
    np.testing.assert_allclose(restored.qpos, runtime.data.qpos, rtol=0, atol=1e-13)
    np.testing.assert_allclose(restored.qvel, runtime.data.qvel, rtol=0, atol=1e-13)
    np.testing.assert_array_equal(restored.userdata, [2., 3.])
    rec.close()


def test_failure_boundary_and_callback_failure_can_be_finalized(tmp_path, graphics):
    runtime = TinyRuntime()

    def failing_callback(sample):
        raise RuntimeError("observer failed")

    runtime.on_sample = failing_callback
    rec = recorder(runtime, tmp_path / "episode")
    rec.begin_action(move())
    with pytest.raises(RuntimeError, match="observer failed"):
        runtime.execute(move())
    rec.snapshot("failure")
    rec.finalize(status="failed", outcome={"error": "observer failed"})
    manifest, arrays = load(tmp_path / "episode")
    assert manifest["status"] == "failed"
    np.testing.assert_allclose(arrays["time_s"], [0., .01])
    np.testing.assert_allclose(arrays["tick_time_s"], [.01])
    assert manifest["actions"][0]["end_time_s"] is None
    assert any(event["reason"] == "failure" for event in manifest["events"])
    assert runtime.on_sample is failing_callback


def test_never_overwrites_an_existing_output_directory(tmp_path, graphics):
    runtime = TinyRuntime()
    target = tmp_path / "episode"
    target.mkdir()
    marker = target / "manifest.json"
    marker.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        recorder(runtime, target)
    assert marker.read_text(encoding="utf-8") == "existing"
    assert runtime.on_sample is None


@pytest.mark.parametrize("mutation", ["nonfinite", "backwards"])
def test_rejects_nonfinite_and_nonmonotonic_state(tmp_path, graphics, mutation):
    runtime = TinyRuntime()
    rec = recorder(runtime, tmp_path / "episode")
    if mutation == "nonfinite":
        runtime.data.qpos[0] = np.nan
    else:
        runtime.data.time = -.01
    with pytest.raises(ValueError):
        rec.snapshot("invalid")
    rec.close()


def test_budget_stops_capture_and_keeps_failure_finalizable(tmp_path, graphics):
    runtime = TinyRuntime()
    rec = recorder(runtime, tmp_path / "episode", max_bytes=16000)
    rec.begin_action(move(duration=.3))
    with pytest.raises(BufferError, match="budget"):
        runtime.execute(move(duration=.3))
    rec.finalize(status="failed", outcome={"reason": "budget"})
    manifest, arrays = load(tmp_path / "episode")
    assert manifest["status"] == "failed"
    assert manifest["truncated"] is True
    assert arrays["time_s"][-1] == pytest.approx(runtime.data.time)
    assert sum(path.stat().st_size for path in (tmp_path / "episode").iterdir()) < 100 * 1024**2


def test_does_not_replace_a_new_callback_on_close(tmp_path, graphics):
    runtime = TinyRuntime()
    rec = recorder(runtime, tmp_path / "episode")
    later = lambda sample: None
    runtime.on_sample = later
    rec.close()
    rec.close()
    assert runtime.on_sample is later
    assert not runtime.closed


def test_oversized_command_preserves_failure_finalize_capacity(tmp_path, graphics):
    runtime = TinyRuntime()
    rec = recorder(runtime, tmp_path / "episode", max_bytes=16000)
    with pytest.raises(BufferError, match="budget"):
        rec.begin_action({"memory": "x" * 16000})
    rec.snapshot("failure")
    rec.finalize(status="failed", outcome={"reason": "oversized command"})
    manifest, arrays = load(tmp_path / "episode")
    assert manifest["truncated"] is True
    assert manifest["actions"] == []
    assert arrays["time_s"].tolist() == [0.]


@pytest.mark.parametrize("failure", ["nonfinite", "renderer"])
def test_failed_finalize_preserves_valid_rows_when_final_frame_unavailable(tmp_path, graphics, failure):
    runtime = TinyRuntime()
    rec = recorder(runtime, tmp_path / "episode")
    rec.begin_action(move())
    if failure == "renderer":
        def broken_render():
            raise RuntimeError("graphics unavailable")

        graphics[0].render = broken_render
        with pytest.raises(RuntimeError, match="graphics unavailable"):
            runtime.execute(move())
    else:
        runtime.execute(move())
        runtime.data.qpos[0] = np.nan
    rec.finalize(status="failed", outcome={"error": failure})
    manifest, arrays = load(tmp_path / "episode")
    assert manifest["status"] == "failed"
    assert manifest["final_snapshot"]["available"] is False
    assert manifest["final_snapshot"]["error"]
    assert manifest["actions"][0]["end_time_s"] is None
    assert np.isfinite(arrays["integration_state"]).all()
    assert arrays["time_s"][0] == 0.
    assert runtime.on_sample is None
    assert graphics[0].closed


def test_completed_finalize_rejects_unavailable_final_state(tmp_path, graphics):
    runtime = TinyRuntime()
    rec = recorder(runtime, tmp_path / "episode")
    runtime.data.qpos[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        rec.finalize(status="completed", outcome=None)
    assert not (tmp_path / "episode" / "manifest.json").exists()
    rec.close()


def test_real_public_runtime_rendered_smoke(tmp_path):
    from dexhand_agent.learning_data import SynchronizedRecorder

    runtime = HandRuntime()
    try:
        rec = SynchronizedRecorder(runtime, tmp_path / "public-smoke", width=64,
                                   height=48, sample_period_s=.02)
        try:
            rec.begin_action({"operation": "hold", "duration_s": .02})
            runtime.execute({"operation": "hold", "duration_s": .02})
            rec.end_action()
            rec.finalize(status="completed", outcome={"scope": "render_smoke_only"})
        finally:
            rec.close()
        _, arrays = load(tmp_path / "public-smoke")
        assert arrays["rgb"].shape == (2, 48, 64, 3)
        assert arrays["rgb"].dtype == np.uint8
        assert np.ptp(arrays["rgb"][0].astype(int)) > 20
        assert arrays["time_s"][1] - arrays["time_s"][0] == pytest.approx(.02)
    finally:
        runtime.close()
