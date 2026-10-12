import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from stapipe import media, models  # noqa: E402
from stapipe.config import SPEC_FILES  # noqa: E402

SCENES = [("testsrc2", 3.75), ("mandelbrot", 4.25), ("smptehdbars", 3.75), ("testsrc", 4.25), ("rgbtestsrc", 3.75)]


@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    """合成 workspace：5 个硬切场景（30 FPS，需变换到 24 FPS）+ 粉噪声音轨。"""
    ws = tmp_path_factory.mktemp("workspace")
    for name in SPEC_FILES:
        (ws / name).parent.mkdir(parents=True, exist_ok=True)
        (ws / name).write_text(f"snapshot of {name}\n", encoding="utf-8")
    (ws / "inputs").mkdir()
    graph = ";".join(f"{src}=s=640x360:r=30,trim=duration={d},setpts=PTS-STARTPTS[v{i}]" for i, (src, d) in enumerate(SCENES))
    graph += ";" + "".join(f"[v{i}]" for i in range(len(SCENES))) + f"concat=n={len(SCENES)}:v=1,format=yuv420p[v]"
    total = sum(d for _, d in SCENES)
    subprocess.run([media.ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"anoisesrc=d={total}:c=pink:a=0.4:seed=7:r=44100", "-filter_complex", graph, "-map", "[v]",
                    "-map", "0:a", "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-ac", "2", "-shortest",
                    str(ws / "inputs" / "synthetic.mp4")], check=True)
    cfg_dir = ws / "pipeline"
    cfg_dir.mkdir()
    lock = models.default_lock()
    lock["backends"] = {"mock": {"type": "python_callable", "callable": "mock_backend:respond",
                                 "modalities": ["text", "image"]}}
    for node, spec in lock["nodes"].items():
        spec["template"] = str(ROOT / spec["template"])
        if spec["backend"]:
            spec["backend"], spec["model"] = "mock", "mock-1"
    (cfg_dir / "models.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    models.refresh_lock(cfg_dir / "models.lock.json")
    (cfg_dir / "config.yaml").write_text(
        "workspace: ..\ninput_video: inputs/synthetic.mp4\nmodels_lock: models.lock.json\n"
        "audio: {asr_python: missing/python}\n"
        "search: {target_evaluation_budget: 4, shortlist_targets: 2, reference_pool_per_type: 3,\n"
        "         reference_bundles_per_target: 1, export_pair_budget: 1}\n", encoding="utf-8")
    return ws
