"""节点 08 的程序检查：只读取已落盘的包（pure/ + work/）与源媒体，可在发布前后重复执行。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

from . import media
from .media import AUDIO_CH, AUDIO_SR, FPS
from .planning import canvas_geometry, legal_frames, wav_samples
from .prompt import check_prompt
from .util import read_json, sha256_file

RECORD_TYPES = {"id": str, "split_group": str, "path_base": str, "prompt": str, "references": list, "target": dict,
                "needs_audio_review": bool, "uncertain_audio_spans_seconds": list}
TARGET_KEYS = ["video", "audio", "width", "height", "num_frames", "fps", "audio_sample_rate", "audio_channels",
               "waveform_samples_per_channel"]


def _tiny_of_image(path) -> np.ndarray:
    # 与 frame_features 相同的灰度定义（RGB 均值），否则彩色画面会出现系统性偏差
    gray = np.asarray(Image.open(path).convert("RGB").resize((media.FEATURE_W, media.FEATURE_H)),
                      np.float32).mean(axis=2)
    return gray.reshape(9, media.FEATURE_H // 9, 16, media.FEATURE_W // 16).mean(axis=(1, 3))


def frame_map_error(m, a: int, b: int, geo: dict, exported_tiny: np.ndarray, mapping: list[dict]) -> dict:
    """导出帧与其映射到的源帧逐帧比对（同一缩放补边后的 16×9 灰度平均绝对差）。"""
    source = media.frame_features(m.path, media.source_clip_prefilter(m, a, b, geo))["tiny"]
    if len(source) != b - a or len(exported_tiny) != len(mapping):
        return {"max": float("inf"), "detail": f"frame count mismatch: source {len(source)} vs {b - a}, "
                                               f"exported {len(exported_tiny)} vs {len(mapping)}"}
    diffs = [media.tiny_diff(exported_tiny[k], source[item["source_frame"] - a]) for k, item in enumerate(mapping)]
    return {"max": round(max(diffs), 3), "mean": round(float(np.mean(diffs)), 3),
            "first": round(diffs[0], 3), "last": round(diffs[-1], 3)}


def verify_package(pure: Path, work: Path, m, cfg, checks) -> dict:
    """执行全部程序类必需检查，结果写入 checks；返回过程中测得的媒体参数。"""
    th, limits = cfg.thresholds, cfg.limits
    prov = read_json(work / "provenance.json")
    facts = read_json(work / "facts.json")
    selection = read_json(work / "selection.json")
    measured: dict = {"videos": {}}

    # 1. 本地 JSONL 字段/类型与相对路径
    c = "schema_and_paths"
    lines = (pure / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    checks.record(c, "jsonl_single_line", len(lines) == 1, f"{len(lines)} line(s)")
    rec = json.loads(lines[0])
    checks.record(c, "fields_and_types", set(rec) == set(RECORD_TYPES) and all(
        isinstance(rec.get(k), t) for k, t in RECORD_TYPES.items()), sorted(rec))
    checks.record(c, "path_base_value", rec.get("path_base") == "directory containing dataset.jsonl")
    checks.record(c, "target_fields", sorted(rec["target"]) == sorted(TARGET_KEYS), sorted(rec["target"]))
    prompt, refs, tg = rec["prompt"], rec["references"], rec["target"]
    checks.record(c, "prompt_not_placeholder", len(prompt) > 400 and not re.search(
        r"TODO|TBD|<placeholder>|rendered_prompt|\.\.\.\s*$", prompt), f"{len(prompt)} chars")
    paths = [r["path"] for r in refs] + [tg["video"], tg["audio"]]
    checks.record(c, "relative_paths_resolve_inside_package", all(
        not Path(p).is_absolute() and ".." not in Path(p).parts and (pure / p).is_file() for p in paths), paths)
    checks.record(c, "references_only_under_inputs", all(r["path"].startswith("inputs/") for r in refs))
    on_disk = sorted(str(p.relative_to(pure)) for p in (pure / "inputs").iterdir())
    checks.record(c, "inputs_exactly_the_referenced_files", on_disk == sorted(r["path"] for r in refs), on_disk)
    extra = sorted(str(p.relative_to(pure)) for p in pure.rglob("*") if p.is_file()
                   and str(p.relative_to(pure)) not in set(paths) | {"dataset.jsonl", "oringin.mp4"})
    checks.record(c, "no_unexpected_files_in_pure", not extra, extra)
    checks.record(c, "reference_objects_wellformed", all(
        r.get("type") in ("image", "video", "audio") and re.fullmatch(
            r"<(Picture|Video|Audio) \d+>", r.get("label", "")) for r in refs), "")

    # 2. 全部媒体完整解码
    c = "decode_all_media"
    counts = {"image": 0, "video": 0, "audio": 0}
    video_total = audio_total = 0.0
    sync_audio = 0
    for r in refs:
        counts[r["type"]] += 1
        path = pure / r["path"]
        try:
            if r["type"] == "image":
                with Image.open(path) as im:
                    im.load()
                    checks.record(c, f"image:{r['path']}", True, f"{im.size} {im.mode}")
            elif r["type"] == "video":
                info = media.probe_and_decode(path)
                measured["videos"][r["path"]] = info
                checks.record(c, f"video:{r['path']}", not info.data["decode_errors"],
                              f"{info.n} frames; errors={info.data['decode_errors'][:2]}")
            else:
                data = sf.info(path)
                audio_total += data.frames / data.samplerate
                checks.record(c, f"audio:{r['path']}", True, f"{data.frames} frames @ {data.samplerate}")
        except Exception as error:  # noqa: BLE001 解码失败即该项失败
            checks.record(c, f"{r['type']}:{r['path']}", False, error)
    target = media.probe_and_decode(pure / tg["video"])
    measured["target"] = target
    checks.record(c, f"video:{tg['video']}", not target.data["decode_errors"], f"{target.n} frames")
    wav_info = sf.info(pure / tg["audio"])
    wav, _ = sf.read(pure / tg["audio"], dtype="float32", always_2d=True)
    checks.record(c, f"audio:{tg['audio']}", len(wav) == wav_info.frames, f"{wav_info.frames} samples/channel")

    # 3. 媒体配置与参考上限
    c = "media_profile_and_reference_limits"
    geo = canvas_geometry(m.width, m.height)
    n, total = target.n, target.n / FPS

    def cfr24(info) -> bool:
        return info.n > 1 and all((b - a) * info.tb == media.Fraction(1, FPS) for a, b in zip(info.pts, info.pts[1:]))

    checks.record(c, "target_matches_record", n == tg["num_frames"] and (target.width, target.height) == (
        tg["width"], tg["height"]) and tg["fps"] == FPS, f"decoded {n} frames {target.width}x{target.height}")
    checks.record(c, "target_24fps_constant", cfr24(target), f"time_base={target.tb}")
    checks.record(c, "target_frames_17k_plus_5", legal_frames(n), n)
    checks.record(c, "target_duration_range", cfg.search.min_target_seconds <= total <= cfg.search.max_target_seconds,
                  f"{total:.4f}s")
    checks.record(c, "canvas_matches_profile", (tg["width"], tg["height"]) == (geo["canvas_w"], geo["canvas_h"])
                  and tg["width"] % 32 == 0 and tg["height"] % 32 == 0, f"{tg['width']}x{tg['height']}")
    checks.record(c, "target_mp4_has_audio_track", target.has_audio, target.audio)
    for r in refs:
        if r["type"] != "video":
            continue
        info = measured["videos"].get(r["path"])
        if info is None:
            continue
        duration = info.n / FPS
        video_total += duration
        include = bool(r.get("include_audio"))
        sync_audio += include
        audio_total += duration if include and cfg.loading.sync_audio_counts_toward_audio_budget else 0.0
        checks.record(c, f"reference_video_profile:{r['path']}", legal_frames(info.n) and cfr24(info) and (
            info.width, info.height) == (geo["canvas_w"], geo["canvas_h"]) and cfg.search.min_reference_seconds
            <= duration <= cfg.search.max_reference_seconds, f"{info.n} frames {duration:.3f}s {info.width}x{info.height}")
        checks.record(c, f"reference_video_audio_flag:{r['path']}", include == info.has_audio and (
            ("audio_label" in r) == include), f"include_audio={include} stream={info.has_audio}")
    audio_slots = counts["audio"] + (sync_audio if cfg.loading.sync_audio_counts_toward_audio_budget else 0)
    checks.record(c, "reference_limits", counts["image"] <= limits.max_images and counts["video"] <= limits.max_videos
                  and video_total <= limits.max_video_total_seconds + 1e-6 and audio_slots <= limits.max_audios
                  and audio_total <= limits.max_audio_total_seconds + 1e-6 and len(refs) <= limits.max_files
                  and counts["image"] + counts["video"] > 0,
                  f"images={counts['image']} videos={counts['video']} video_total={video_total:.3f}s "
                  f"audio_slots={audio_slots} audio_total={audio_total:.3f}s files={len(refs)} "
                  f"(loading adapter {cfg.loading.adapter_version})")

    # 4. WAV 采样数与音画对齐
    c = "wav_samples_and_av_alignment"
    expected = wav_samples(n)
    checks.record(c, "wav_format", (wav_info.samplerate, wav_info.channels, wav_info.subtype, wav_info.format) == (
        AUDIO_SR, AUDIO_CH, "PCM_16", "WAV") and tg["audio_sample_rate"] == AUDIO_SR
        and tg["audio_channels"] == AUDIO_CH, f"{wav_info.samplerate} Hz {wav_info.channels} ch {wav_info.subtype}")
    checks.record(c, "wav_samples_per_channel", wav_info.frames == expected == tg["waveform_samples_per_channel"],
                  f"wav={wav_info.frames} expected={expected} record={tg['waveform_samples_per_channel']}")
    checks.record(c, "wav_vs_video_within_one_sample", abs(media.Fraction(wav_info.frames, AUDIO_SR)
                                                           - media.Fraction(n, FPS)) <= media.Fraction(1, AUDIO_SR),
                  f"{wav_info.frames / AUDIO_SR:.6f}s vs {total:.6f}s")
    mono = wav.mean(axis=1)
    mp4_audio = media.decode_audio(pure / tg["video"], AUDIO_SR, AUDIO_CH).mean(axis=1)
    lag, corr = media.best_lag(mp4_audio, mono, 1600)
    quiet = float(np.sqrt((mono ** 2).mean())) < 1e-4       # 源局部静音不因低音量而失败
    checks.record(c, "mp4_audio_matches_wav", quiet or (abs(lag) <= 16 and corr > th.mp4_wav_corr),
                  f"lag {lag} samples, corr={corr:.4f}, mp4 decoded {len(mp4_audio)} vs wav {len(mono)} "
                  f"(difference is codec padding){' ; near-silent target' if quiet else ''}")
    tv = prov["target"]
    source_audio = media.decode_audio(m.path, AUDIO_SR, AUDIO_CH, tv["source_start_seconds"],
                                      tv["source_end_seconds"]).mean(axis=1)
    lag, corr = media.best_lag(mono, source_audio, 1600)
    checks.record(c, "wav_matches_source_interval", quiet or (abs(lag) <= 16 and corr > th.wav_source_corr),
                  f"lag {lag} samples, corr={corr:.4f} against source [{tv['source_start_seconds']:.3f}, "
                  f"{tv['source_end_seconds']:.3f})")
    real = int(round((tv["source_end_seconds"] - tv["source_start_seconds"]) * AUDIO_SR))
    tail = np.abs(wav[min(len(wav), real + 64):]).max() if real + 64 < len(wav) else 0.0
    checks.record(c, "tail_padding_is_silence", tail == 0.0, f"max |sample| after source end = {tail}")

    # 5. 来源区间与所有媒体首尾
    c = "source_intervals_and_all_export_boundaries"
    a, b = tv["start_frame"], tv["end_frame_excl"]
    detail, ok = [], True
    for v in prov["videos"]:
        disjoint = v["end_frame_excl"] <= a or v["start_frame"] >= b
        detail.append(f"{v['path']} [{v['start_frame']},{v['end_frame_excl']}) disjoint={disjoint}")
        ok &= disjoint
    for im in prov["images"]:
        outside = not a <= im["source_frame"] < b
        detail.append(f"{im['path']} frame {im['source_frame']} outside_target={outside}")
        ok &= outside
    checks.record(c, "source_intervals_disjoint", ok, detail)
    search = cfg.search
    for item in [tv] + prov["videos"]:
        pad, dur = item["padding_seconds"], item["source_end_seconds"] - item["source_start_seconds"]
        checks.record(c, f"tail_padding_within_limits:{item['path']}", -1e-9 <= pad <= search.max_tail_padding_seconds
                      + 1e-9 and pad <= dur * search.max_tail_padding_ratio + 1e-9,
                      f"pad {pad:.5f}s ({pad / dur:.3%}) of source {dur:.5f}s; no tail truncation")
        info = target if item is tv else measured["videos"].get(item["path"])
        if info is None:
            continue
        mapping = media.expected_frame_map(m, item["start_frame"], item["end_frame_excl"], info.n)
        tiny = media.frame_features(pure / item["path"])["tiny"]
        error = frame_map_error(m, item["start_frame"], item["end_frame_excl"], geo, tiny, mapping)
        checks.record(c, f"every_frame_maps_to_selected_source_frames:{item['path']}",
                      error["max"] < th.boundary_frame_diff, error)
        measured.setdefault("tiny", {})[item["path"]] = tiny
    source_tiny = None
    for im in prov["images"]:
        if source_tiny is None:
            source_tiny = media.frame_features(m.path)["tiny"]
        diff = media.tiny_diff(_tiny_of_image(pure / im["path"]), source_tiny[im["source_frame"]])
        checks.record(c, f"image_is_selected_source_frame:{im['path']}", diff < th.boundary_frame_diff, f"diff {diff:.3f}")
        measured.setdefault("image_tiny", {})[im["path"]] = _tiny_of_image(pure / im["path"])

    # 6. 重复内容与条件隔离（程序部分 + 选择阶段记录的模型复核）
    c = "duplicate_content_and_condition_isolation"
    target_tiny = measured["tiny"][tv["path"]].reshape(n, -1).astype(np.float32)
    for r in refs:
        if r["type"] == "audio":
            continue
        ref_tiny = (measured["image_tiny"][r["path"]][None] if r["type"] == "image"
                    else measured["tiny"][r["path"]][::2])
        diffs = np.array([np.abs(target_tiny - row.reshape(-1)).mean(axis=1).min() for row in ref_tiny.astype(np.float32)])
        coverage = float((diffs < th.duplicate_tiny_diff).mean())
        checks.record(c, f"exported_reference_not_duplicating_target:{r['path']}", coverage < th.duplicate_coverage,
                      f"min diff {diffs.min():.2f}, near-identical coverage {coverage:.2f}")
    reviews = {ref["name"]: ref["duplicate_check"]["result"] for ref in selection["references"]}
    checks.record(c, "selection_duplicate_checks_clear", all(v == "clear" for v in reviews.values()), reviews)
    ref_paths = [r["path"] for r in refs]
    checks.record(c, "target_and_origin_not_in_references", not any(
        p.startswith("targets/") or "oringin" in p for p in ref_paths) and len(set(ref_paths)) == len(ref_paths),
        ref_paths)
    hashes = {p: sha256_file(pure / p) for p in ref_paths}
    checks.record(c, "reference_bytes_differ_from_target_and_origin", not set(hashes.values()) & {
        sha256_file(pure / tg["video"]), sha256_file(pure / tg["audio"]), sha256_file(pure / "oringin.mp4")}, "")

    # 7. 六段、标签映射、切镜时间
    c = "prompt_structure_labels_and_cut_times"
    for name, ok, detail in check_prompt(prompt, rec, facts["cuts"]["confirmed_frames"], th.cut_tolerance_frames):
        checks.record(c, name, ok, detail)
    labels = read_json(work / "labels.json")
    checks.record(c, "references_match_label_table", refs == labels["reference_records"], "")

    # 8. 原片溯源与提示词副本
    c = "origin_provenance_and_prompt_copy"
    origin = prov["origin"]
    if origin["method"] == "byte_copy":
        checks.record(c, "oringin_is_byte_identical_copy", sha256_file(pure / "oringin.mp4") == m.sha256, "")
    else:
        kept = work / Path(origin["original_kept_at"]).name
        converted = media.probe_and_decode(pure / "oringin.mp4")
        checks.record(c, "non_mp4_original_kept_and_converted", kept.is_file() and sha256_file(kept) == m.sha256
                      and sha256_file(pure / "oringin.mp4") == origin["oringin_sha256"] and converted.n == m.n,
                      f"{origin['method']}; frames {converted.n} vs {m.n}")
    checks.record(c, "prompt_md_identical_to_record", (work / "prompt.md").read_text(encoding="utf-8") == prompt, "")
    log = read_json(work / "export_log.json")
    checks.record(c, "export_commands_and_source_map_present", bool(log) and bool(prov.get("frame_maps")), f"{len(log)} commands")
    checks.record(c, "recorded_output_hashes_match", all(
        sha256_file(pure / item["path"]) == item["sha256"] for item in [tv] + prov["videos"] + prov["images"])
        and sha256_file(pure / tg["audio"]) == tv["audio_sha256"], "")

    # 9. 不确定性标记与区间
    c = "uncertainty_flags_and_ranges"
    spans = rec["uncertain_audio_spans_seconds"]
    checks.record(c, "spans_wellformed_sorted_merged_within_target", all(
        isinstance(s, list) and len(s) == 2 and 0 <= s[0] < s[1] <= total + 1e-6 for s in spans) and all(
        x[1] < y[0] for x, y in zip(spans, spans[1:])), spans)
    checks.record(c, "spans_imply_needs_audio_review", rec["needs_audio_review"] or not spans,
                  f"needs_audio_review={rec['needs_audio_review']}")
    omitted = facts.get("omitted_speech", [])
    checks.record(c, "omitted_speech_is_flagged_with_spans", not omitted or (rec["needs_audio_review"] and all(
        any(s[0] <= o["start"] + 1e-6 and o["end"] <= s[1] + 1e-6 for s in spans) for o in omitted)),
        f"{len(omitted)} utterance(s) left out of the prompt")
    observed = facts["audio"]["observed_by_model"]
    checks.record(c, "unheard_audio_is_flagged", observed or rec["needs_audio_review"],
                  f"audio observed by a model: {observed}")
    measured["record"] = rec
    return measured
