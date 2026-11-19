"""Download just the two public robot models at a pinned Menagerie revision."""

import hashlib
import json
from pathlib import Path
import time
import urllib.request
import xml.etree.ElementTree as ET


REVISION = "370c43965323c0944b433b202aac5058b5f907a9"
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}"
ROOT = Path(__file__).resolve().parents[1]
MODELS = {"shadow_hand": "right_hand.xml", "franka_emika_panda": "panda_nohand.xml"}


def download(url):
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dexhand-agent-asset-fetcher"})
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def main():
    manifest_path = ROOT / "assets" / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    known = {item["path"]: item for item in previous.get("files", [])}
    files = []
    for model, xml_name in MODELS.items():
        xml = download(f"{BASE}/{model}/{xml_name}")
        tree = ET.fromstring(xml)
        compiler = tree.find("compiler")
        mesh_dir = compiler.get("meshdir", "")
        names = [xml_name, "LICENSE", "README.md"]
        names += [f"{mesh_dir}/{node.get('file')}" for node in tree.findall("./asset/mesh")]
        for name in dict.fromkeys(names):
            relative = f"{model}/{name}"
            target = ROOT / "assets" / "vendor" / relative
            expected = known.get(relative, {}).get("sha256")
            if target.exists() and expected and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
                payload = target.read_bytes()
            else:
                payload = xml if name == xml_name else download(f"{BASE}/{relative}")
                digest = hashlib.sha256(payload).hexdigest()
                if expected and digest != expected:
                    raise ValueError(f"Upstream digest differs from manifest: {relative}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            files.append({"path": relative, "sha256": hashlib.sha256(payload).hexdigest(),
                          "bytes": len(payload), "url": f"{BASE}/{relative}"})
        print(f"Fetched {model}: {len(names)} required files", flush=True)
    manifest = {"repository": "https://github.com/google-deepmind/mujoco_menagerie",
                "revision": REVISION, "license_by_model": {name: "Apache-2.0" for name in MODELS},
                "files": files}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(f"Verified {len(files)} files; {sum(row['bytes'] for row in files) / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
