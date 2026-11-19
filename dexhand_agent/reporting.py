"""Small local HTML reports; all evidence comes from recorded episode outputs."""

from html import escape
import json
from pathlib import Path
from urllib.parse import quote


def _text(value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return escape(str(value))


def _table(values):
    return "<table>" + "".join(
        f"<tr><th>{_text(key)}</th><td>{_text(value)}</td></tr>"
        for key, value in values.items()) + "</table>"


def _physical_details(summary):
    if summary is None:
        return "<p>Physical success: UNKNOWN — No physical record available.</p>"
    status = "PASS" if summary.get("success") is True else (
        "FAIL" if summary.get("success") is False else "UNKNOWN")
    details = (f"<p>Physical success: {status}</p>"
               f"<p>Controller backend: {_text(summary.get('backend', 'Not recorded'))}</p>"
               f"<p>Runtime error: {_text(summary.get('error') or 'None reported')}</p>")
    if summary.get("controller", {}).get("kind") == "neural_context_parameters":
        details += "<p>Controller: Learned parameters + reference control</p>"
    gates = summary.get("gates", {})
    details += "<h3>Physical gates</h3>"
    if gates:
        details += "<table><tr><th>Gate</th><th>Status</th><th>Measurements</th></tr>"
        for name, gate in gates.items():
            status = "PASS" if gate.get("passed") is True else (
                "FAIL" if gate.get("passed") is False else "UNKNOWN")
            measurements = {key: value for key, value in gate.items() if key != "passed"}
            details += f"<tr><th>{_text(name)}</th><td>{status}</td><td>{_text(measurements)}</td></tr>"
        details += "</table>"
    else:
        details += "<p>No gate measurements recorded.</p>"
    details += "<h3>Metrics</h3>" + (_table(summary["metrics"]) if summary.get("metrics")
                                     else "<p>No metrics recorded.</p>")
    if summary.get("violations"):
        details += f"<h3>Violations</h3><p>{_text(summary['violations'])}</p>"
    return details


def _links(output, prefix="", include_report=False):
    names = ["summary.json", "metadata.json", "experience.json", "error.txt", "episode.mp4"]
    if include_report:
        names.insert(0, "report.html")
    links = [f'<a href="{quote(prefix + name, safe="/")}">{name}</a>'
             for name in names if (output / name).is_file()]
    return "<p>Local evidence: " + " | ".join(links) + "</p>" if links else ""


def _write(output, title, body):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "report.html"
    path.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{_text(title)}</title>"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<style>body{font:16px system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;'
        'color:#202832;background:#fafafa}table{border-collapse:collapse;width:100%;margin:1rem 0}'
        'th,td{border:1px solid #bcc4cc;padding:.5rem;text-align:left;overflow-wrap:anywhere}'
        'section{border-top:2px solid #bcc4cc;margin-top:2rem}video{width:100%;max-height:600px}'
        'p{overflow-wrap:anywhere}a{color:#174fa0}</style></head>'
        f"<body><h1>{_text(title)}</h1>{body}</body></html>\n", encoding="utf8")
    return path


def write_episode_report(output, summary):
    """Write and return report.html; summary=None means no physical record exists."""
    output = Path(output)
    title = "Episode: " + str(summary.get("task_id", output.name) if summary else output.name)
    body = _physical_details(summary) + _links(output)
    if (output / "episode.mp4").is_file():
        body += '<video controls preload="metadata" src="episode.mp4"></video>'
    return _write(output, title, body)


def write_suite_report(output, task_runs):
    """Render task_id/output/returncode/summary/error records; output is relative to this suite."""
    output = Path(output)
    body = "<p>Command status and measured physical success are reported separately.</p>"
    if (output / "suite.json").is_file():
        body += '<p><a href="suite.json">suite.json</a></p>'
    for run in task_runs:
        folder = output / run["output"]
        prefix = folder.relative_to(output).as_posix().rstrip("/") + "/"
        body += (f"<section><h2>{_text(run['task_id'])}</h2>"
                 f"<p>Exit status: {_text(run['returncode'])}</p>")
        if run.get("error"):
            body += f"<p>Launcher error: {_text(run['error'])}</p>"
        body += _physical_details(run.get("summary"))
        body += _links(folder, prefix, include_report=True) + "</section>"
    return _write(output, "Local tool-use suite", body)
