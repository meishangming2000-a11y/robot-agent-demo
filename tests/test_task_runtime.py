import pytest

from dexhand_agent import scene
from dexhand_agent.runtime import HandRuntime


def renamed_tool_scene(seed):
    return scene.scene_xml(seed).replace('name="hammer"', 'name="test_tool"')


def test_scene_factory_preserves_robot_and_builds_selected_tool():
    model, data = scene.make_model(0, scene_factory=renamed_tool_scene)
    assert model.body("test_tool").id >= 0
    assert model.joint("hammer_free").type == 0
    assert model.nu == 27
    assert data.time == 0.0


def test_runtime_factory_and_tool_anchor_use_selected_model():
    calls = []

    def factory(seed):
        calls.append(seed)
        return scene.make_model(seed, scene_factory=renamed_tool_scene)

    runtime = HandRuntime(7, model_factory=factory, tool_body="test_tool")
    try:
        assert calls == [7]
        assert runtime.initial_height_m == pytest.approx(runtime.data.body("test_tool").xpos[2])
        assert runtime.data.time == pytest.approx(1.0)
    finally:
        runtime.close()


def test_common_robot_observation_matches_default_public_observation():
    runtime = HandRuntime()
    try:
        assert runtime.robot_observation() == runtime.observe()["robot"]
        assert len(runtime.robot_observation()["actuators"]) == 27
    finally:
        runtime.close()
