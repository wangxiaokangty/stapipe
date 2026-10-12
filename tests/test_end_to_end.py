import json
from pathlib import Path

import soundfile as sf

from stapipe.pipeline import run


def test_pipeline_end_to_end_with_mock_models(workspace):
    config = workspace / "pipeline" / "config.yaml"
    result = run(config)
    assert result["status"] == "needs_review", result
    sample = Path(result["samples"][0]["path"])
    record = json.loads((sample / "pure" / "dataset.jsonl").read_text(encoding="utf-8"))
    assert record["id"] == result["samples"][0]["sample_id"] and record["split_group"].startswith("sha256:")
    frames = record["target"]["num_frames"]
    assert (frames - 5) % 17 == 0 and 5 <= frames / 24 <= 15
    samples = round(frames * 32000 / 24)
    assert record["target"]["waveform_samples_per_channel"] == samples
    assert sf.info(sample / "pure" / record["target"]["audio"]).frames == samples
    assert record["needs_audio_review"] is True
    assert record["uncertain_audio_spans_seconds"] == [[0.0, int(frames / 24 * 1000) / 1000]]
    assert (sample / "work" / "prompt.md").read_text(encoding="utf-8") == record["prompt"]
    verification = json.loads((sample / "work" / "verification.json").read_text(encoding="utf-8"))
    by_name = {c["check"]: c["status"] for c in verification["checks"]}
    assert by_name.pop("audio_facts_dialogue_and_speakers") == "not_checked"
    assert set(by_name.values()) == {"passed"}, {k: v for k, v in by_name.items() if v != "passed"}
    assert not list((Path(result["run_dir"]).parent / ".staging").iterdir())

    again = run(config)                                   # 相同指纹：复用已提交结果，不重复发布
    assert again.get("reused") and again["samples"] == result["samples"]

    replay = run(config, ["mode=replay", f"replay_run={Path(result['run_dir']).parent}"])
    assert replay["status"] == "needs_review", replay
    replayed = json.loads((Path(replay["samples"][0]["path"]) / "pure" / "dataset.jsonl").read_text(encoding="utf-8"))
    assert replayed["prompt"] == record["prompt"] and replayed["references"] == record["references"]
    assert all(v["invoked"] == 0 for v in replay["model_calls"].values())


def test_missing_input_is_blocked_not_skipped(workspace):
    result = run(workspace / "pipeline" / "config.yaml", ["input_video=inputs/absent.mp4"])
    assert result["status"] == "blocked" and "input_missing" in result["reason"]
