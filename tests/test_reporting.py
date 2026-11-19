import json

from dexhand_agent.reporting import write_episode_report, write_suite_report


def test_episode_reports_physical_success_separately_from_runtime_error(tmp_path):
    (tmp_path / "metadata.json").write_text("{}", encoding="utf8")
    (tmp_path / "experience.json").write_text("{}", encoding="utf8")
    summary = {"task_id": "hammer_button", "success": True,
               "error": "RuntimeError: <unsafe>&\"message\"", "backend": "native",
               "gates": {"lift": {"passed": True, "best_continuous_s": 2.1}},
               "metrics": {"max_lift_m": 0.031}, "violations": ["<contact>"]}
    report = write_episode_report(tmp_path, summary)
    html = report.read_text(encoding="utf8")
    assert report == tmp_path / "report.html"
    assert "Physical success: PASS" in html
    assert "RuntimeError: &lt;unsafe&gt;&amp;&quot;message&quot;" in html
    assert "<unsafe>" not in html and "&lt;contact&gt;" in html
    assert "native" in html and "max_lift_m" in html and "0.031" in html
    assert "lift" in html and "best_continuous_s" in html
    assert 'href="metadata.json"' in html and 'href="experience.json"' in html
    assert "episode.mp4" not in html and "<video" not in html
    assert "<script" not in html and "https://" not in html


def test_episode_embeds_only_existing_local_video(tmp_path):
    (tmp_path / "episode.mp4").write_bytes(b"local video placeholder")
    html = write_episode_report(tmp_path, {"success": False, "error": None,
                                          "backend": "reference"}).read_text(encoding="utf8")
    assert "Physical success: FAIL" in html
    assert '<video controls' in html and 'src="episode.mp4"' in html


def test_missing_summary_never_creates_physical_evidence(tmp_path):
    html = write_episode_report(tmp_path, None).read_text(encoding="utf8")
    assert "No physical record available" in html
    assert "Physical success: PASS" not in html and "best_continuous_s" not in html
    assert "metadata.json" not in html and "experience.json" not in html


def test_suite_links_existing_reports_video_and_failed_tasks(tmp_path):
    passed = tmp_path / "hammer_button"
    passed.mkdir()
    (passed / "episode.mp4").write_bytes(b"local video placeholder")
    summary = {"task_id": "hammer_button", "success": True, "error": None,
               "backend": "reference", "gates": {"lift": {"passed": True}}, "metrics": {}}
    (passed / "summary.json").write_text(json.dumps(summary), encoding="utf8")
    write_episode_report(passed, summary)
    failed = tmp_path / "wrench_turn"
    write_episode_report(failed, None)
    runs = [{"task_id": "hammer_button", "output": "hammer_button", "returncode": 0,
             "summary": summary, "error": None},
            {"task_id": "wrench_turn", "output": "wrench_turn", "returncode": 7,
             "summary": None, "error": "Process <failed>"}]
    html = write_suite_report(tmp_path, runs).read_text(encoding="utf8")
    assert 'href="hammer_button/report.html"' in html
    assert 'href="hammer_button/episode.mp4"' in html
    assert 'href="wrench_turn/report.html"' in html
    assert "wrench_turn/episode.mp4" not in html
    assert "Process &lt;failed&gt;" in html and "Exit status: 7" in html
    assert "No physical record available" in html


def test_learned_episode_identifies_parameters_and_reference_control(tmp_path):
    (tmp_path / "metadata.json").write_text("{}", encoding="utf8")
    summary = {"task_id": "wrench_turn", "success": True, "error": None,
               "backend": "reference", "controller": {"kind": "neural_context_parameters",
               "policy_dir": '<script>alert("policy")</script>', "weights_sha256": "a" * 64}}
    html = write_episode_report(tmp_path, summary).read_text(encoding="utf8")
    assert "Learned parameters + reference control" in html
    assert "Controller backend: reference" in html
    assert 'href="metadata.json"' in html
    assert "<script>" not in html and "alert" not in html


def test_suite_identifies_only_the_learned_wrench_controller(tmp_path):
    runs = [{"task_id": task, "output": task, "returncode": 0, "error": None,
             "summary": {"task_id": task, "success": True, "error": None, "backend": "reference"}}
            for task in ("hammer_button", "wrench_turn", "tweezers_transfer")]
    runs[1]["summary"]["controller"] = {"kind": "neural_context_parameters"}
    html = write_suite_report(tmp_path, runs).read_text(encoding="utf8")
    assert html.count("Learned parameters + reference control") == 1
    assert html.count("Controller backend: reference") == 3


def test_other_controller_metadata_does_not_claim_learned_parameters(tmp_path):
    summary = {"success": False, "error": '<unsafe> & "message"',
               "backend": "<reference>", "controller": {"kind": "<script>"}}
    html = write_episode_report(tmp_path, summary).read_text(encoding="utf8")
    assert "Learned parameters + reference control" not in html
    assert "&lt;reference&gt;" in html and "&lt;unsafe&gt; &amp; &quot;message&quot;" in html
    assert "<script>" not in html
