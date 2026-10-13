"""节点 05–09：导出到暂存目录、在导出媒体上重新标注、写作、校验、审查与至多一次修复。"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

from . import media, schemas
from .context import NOT_DONE, Checks
from .media import AUDIO_SR, FPS
from .models import audio as audio_block
from .models import image, text
from .planning import canvas_geometry, legal_frames, wav_samples
from .prompt import allocate_labels, render_six_sections, validate_draft
from .source import LANGUAGES
from .util import Blocked, RejectPair, merge_spans, sha256_file, write_json
from .verify import verify_package

KNOWN_ERROR_KINDS = {"wrong_fact", "relation_overclaim", "unconfirmed_dialogue", "format"}
SPEAKER_SPECIAL = ("off_screen", "unknown")
# 听不清的对白整句不写入提示词（不写 <d>，不用占位符）；只保留区间供听辨复核。
OMITTED_SPEECH_NOTE = "speech here could not be transcribed reliably and is left out of the prompt"


def tid(frame: int) -> str:
    return f"t{frame:04d}"


# ---------------------------------------------------------------- 节点 05

def compile_export_plan(pair: dict, m, cfg) -> dict:
    target = pair["target"]
    plan = {"geometry": canvas_geometry(m.width, m.height), "images": [], "videos": [], "audios": [],
            "target": {"path": "targets/target.mp4", "audio_path": "targets/target_audio.wav",
                       "start_frame": target["start_frame"], "end_frame_excl": target["end_frame_excl"],
                       "num_frames": target["plan"]["num_frames"],
                       "padding_seconds": target["plan"]["padding_seconds"],
                       "wav_samples_per_channel": wav_samples(target["plan"]["num_frames"]),
                       "forbid_tail_truncation": True}}
    for ref in pair["references"]:
        if ref["type"] == "image":
            name = f"picture_{len(plan['images']) + 1}"
            plan["images"].append({"name": name, "path": f"inputs/{name}.png", "reference_id": ref["id"],
                                   "source_frame": ref["source_frame"], "crop": None})
        else:
            name = f"video_{len(plan['videos']) + 1}"
            s, e = ref["source_interval"]
            plan["videos"].append({"name": name, "path": f"inputs/{name}.mp4", "reference_id": ref["id"],
                                   "start_frame": s, "end_frame_excl": e, "num_frames": ref["plan"]["num_frames"],
                                   "padding_seconds": ref["plan"]["padding_seconds"], "include_audio": False})
    return plan


def check_plan_constraints(plan: dict, m, cfg) -> None:
    limits, tg = cfg.limits, plan["target"]
    a, b = tg["start_frame"], tg["end_frame_excl"]
    problems = []
    if not legal_frames(tg["num_frames"]) or not (cfg.search.min_target_seconds <= tg["num_frames"] / FPS
                                                  <= cfg.search.max_target_seconds):
        problems.append("target_frames_or_duration")
    if tg["num_frames"] / FPS < float(m.t(b) - m.t(a)) - 1e-9:
        problems.append("target_would_be_truncated")
    if len(plan["images"]) > limits.max_images or len(plan["videos"]) > limits.max_videos:
        problems.append("reference_count")
    if sum(v["num_frames"] / FPS for v in plan["videos"]) > limits.max_video_total_seconds:
        problems.append("reference_video_total_duration")
    if len(plan["images"]) + len(plan["videos"]) + len(plan["audios"]) > limits.max_files or not (
            plan["images"] or plan["videos"]):
        problems.append("reference_file_count")
    for v in plan["videos"]:
        if not legal_frames(v["num_frames"]) or max(v["start_frame"], a) < min(v["end_frame_excl"], b):
            problems.append(f"{v['name']}_illegal_or_overlapping")
    for im in plan["images"]:
        if a <= im["source_frame"] < b:
            problems.append(f"{im['name']}_inside_target")
    if problems:
        raise RejectPair("plan_constraints:" + ",".join(problems))


def export_media(plan: dict, m, pure: Path, work: Path, with_reference_audio: bool) -> dict:
    """目标 WAV 由源区间直接提取；目标 MP4 的 AAC 由该 WAV 编码。参考视频默认导出为无声。"""
    (pure / "inputs").mkdir(parents=True)
    (pure / "targets").mkdir()
    (work / "intermediate").mkdir(parents=True, exist_ok=True)
    exporter, geo, tg = media.Exporter(m), plan["geometry"], plan["target"]
    prov = {"source_path": str(m.path), "source_sha256": m.sha256, "source_frames": m.n,
            "source_time_base": m.data["time_base"], "source_constant_frame_rate": m.data["constant_frame_rate"],
            "source_audio": m.audio, "geometry": geo,
            "frame_numbering": "zero-based source frames, end-exclusive; seconds are decoded PTS",
            "spatial_transform": f"{m.width}x{m.height} -> {geo['scaled_w']}x{geo['scaled_h']} (lanczos, aspect "
                                 f"preserved), pad to {geo['canvas_w']}x{geo['canvas_h']} at ({geo['pad_x']},"
                                 f"{geo['pad_y']}) with black; no crop",
            "temporal_transform": f"exact PTS trim, then {FPS} fps constant rate (nearest frame); only the last "
                                  "selected frame is repeated at the tail; audio resampled to 32 kHz stereo on the "
                                  "shared source time origin and padded with silence at the tail only",
            "images": [], "videos": [], "audios": [], "frame_maps": {}}
    for im in plan["images"]:
        exporter.image(im["source_frame"], pure / im["path"])
        prov["images"].append({"path": im["path"], "source_frame": im["source_frame"],
                               "source_seconds": float(m.t(im["source_frame"])),
                               "transform": "full frame at source resolution; no crop, no synthesis",
                               "sha256": sha256_file(pure / im["path"])})

    def interval(item):
        return {"start_frame": item["start_frame"], "end_frame_excl": item["end_frame_excl"],
                "source_start_seconds": float(m.t(item["start_frame"])),
                "source_end_seconds": float(m.t(item["end_frame_excl"])),
                "num_frames": item["num_frames"], "duration_seconds": item["num_frames"] / FPS,
                "padding_seconds": item["padding_seconds"], "tail_truncated_seconds": 0.0}

    for v in plan["videos"]:
        if with_reference_audio:            # 仅供声音节点判断用途；是否随视频提交由事实决定
            exporter.wav(v["start_frame"], v["end_frame_excl"], wav_samples(v["num_frames"]),
                         work / "intermediate" / f"{v['name']}_audio.wav")
        exporter.mp4(v["start_frame"], v["end_frame_excl"], v["num_frames"], geo, pure / v["path"], None)
        prov["videos"].append({"path": v["path"], **interval(v), "include_audio": False,
                               "sha256": sha256_file(pure / v["path"])})
        prov["frame_maps"][v["path"]] = media.expected_frame_map(m, v["start_frame"], v["end_frame_excl"],
                                                                 v["num_frames"])
    wav = pure / tg["audio_path"]
    exporter.wav(tg["start_frame"], tg["end_frame_excl"], tg["wav_samples_per_channel"], wav)
    exporter.mp4(tg["start_frame"], tg["end_frame_excl"], tg["num_frames"], geo, pure / tg["path"], wav)
    track = m.audio_interval()
    t0, t1 = float(m.t(tg["start_frame"])), float(m.t(tg["end_frame_excl"]))
    prov["target"] = {"path": tg["path"], "audio_path": tg["audio_path"], **interval(tg),
                      "waveform_samples_per_channel": tg["wav_samples_per_channel"],
                      "audio_mono_source_duplicated_to_stereo": m.audio.get("channels") == 1,
                      "source_audio_uncovered_seconds": {"head": round(max(0.0, track[0] - t0), 5),
                                                         "tail": round(max(0.0, t1 - track[1]), 5)},
                      "sha256": sha256_file(pure / tg["path"]), "audio_sha256": sha256_file(wav)}
    prov["frame_maps"][tg["path"]] = media.expected_frame_map(m, tg["start_frame"], tg["end_frame_excl"],
                                                              tg["num_frames"])
    prov["origin"] = exporter.origin(pure / "oringin.mp4", work)
    write_json(work / "provenance.json", prov)
    write_json(work / "export_log.json", exporter.log)
    return {"provenance": prov, "exporter": exporter}


# ---------------------------------------------------------------- 节点 06

def _jpeg(path, width: int) -> bytes:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((width, width * 4))
        buffer = io.BytesIO()
        im.save(buffer, format="JPEG", quality=88)
        return buffer.getvalue()


def _utterances_from_asr(evidence: dict, cfg, total: float) -> tuple[list[dict], list[dict], list[dict]]:
    """没有声音理解能力时：转录只是证据，全部标为未经听辨。含低置信词的整句不写入提示词，只记录区间。"""
    utterances, omitted, uncertain = [], [], []
    if not evidence.get("available"):
        return utterances, omitted, [{"text": f"ASR unavailable: {evidence.get('reason')}", "span": [0.0, total]}]
    language = LANGUAGES.get(evidence["asr"].get("language"), "Unknown")
    threshold = cfg.audio.confident_word_probability
    for seg in evidence["asr"]["segments"]:
        words = [w for w in seg["words"] if w["w"] and w["s"] < total]
        if not words:
            continue
        if seg["no_speech_prob"] > 0.6 and seg["avg_logprob"] < -1.0:
            uncertain.append({"text": "possible speech or lyrics that ASR could not transcribe reliably",
                              "span": [seg["start"], min(total, seg["end"])]})
            continue
        start, end = words[0]["s"], min(total, words[-1]["e"])
        if not all(w["p"] >= threshold for w in words):
            omitted.append({"start": start, "end": end, "reason": "low_confidence_asr_words"})
            uncertain.append({"text": OMITTED_SPEECH_NOTE, "span": [max(0.0, start - 0.1), min(total, end + 0.1)]})
            continue
        utterances.append({"utterance_id": f"u{len(utterances) + 1}", "start": start, "end": end,
                           "language": language, "status": "asr_confident_unverified", "words": words,
                           "asr_text": seg["text"], "voice_description": None})
    return utterances, omitted, uncertain


def _words_text(words: list[dict]) -> str:
    return " ".join(w["w"] for w in words)


def _split_parts(utterance: dict, shots: list[dict], cutoff: bool) -> list[dict]:
    """跨切镜的同一句话拆成两段，连接处由程序加 <scenetrans>；没有逐词时间时不拆。"""
    def shot_of(seconds):
        frame = int(seconds * FPS + 1e-6)
        return next(s["shot_id"] for s in reversed(shots) if s["start_frame"] <= frame)

    groups: list[tuple[str, list[dict]]] = []
    if utterance.get("words"):
        for w in utterance["words"]:
            sid = shot_of((w["s"] + w["e"]) / 2)          # 词按其中点归属镜头，贴着切镜开始的词不误归前一镜头
            if groups and groups[-1][0] == sid:
                groups[-1][1].append(w)
            else:
                groups.append((sid, [w]))
        texts = [(sid, _words_text(ws)) for sid, ws in groups]
    else:
        texts = [(shot_of((utterance["start"] + utterance["end"]) / 2), utterance["text"])]
    return [{"part_id": f"{utterance['utterance_id']}.{k + 1}", "shot_id": sid, "text": body,
             "lead_scenetrans": k > 0, "trail_scenetrans": k < len(texts) - 1,
             "cutoff": cutoff and k == len(texts) - 1} for k, (sid, body) in enumerate(texts)]


def build_export_facts(ctx, m, pair: dict, plan: dict, pure: Path, work: Path, timeline: dict, scope: str,
                       checks: Checks) -> dict:
    """只用导出目标和参考重新建立目标时间线；切镜定位到导出帧，不沿用源绝对时间。"""
    cfg, tg = ctx.cfg, plan["target"]
    n, total = tg["num_frames"], tg["num_frames"] / FPS
    target_path = pure / tg["path"]
    feats = media.frame_features(target_path)
    if len(feats["tiny"]) != n:
        raise Blocked(f"exported_target_frame_count:{len(feats['tiny'])}!={n}")
    detected = media.detect_cuts(feats["cut_score"])
    can_hear = bool(ctx.capabilities.get("can_observe_audio"))

    # 声音证据（目标相对时间）
    evidence = ctx.audio_evidence(pure / tg["audio_path"])
    write_json(work / "transcripts" / "target_audio_evidence.json", evidence)
    relations = []
    for ref in pair["references"]:
        name = next(x["name"] for x in plan["images"] + plan["videos"] if x["reference_id"] == ref["id"])
        for rel in ref["relations"]:
            relations.append({"relation_id": f"rel{len(relations) + 1}", "reference_name": name,
                              "reference_id": ref["id"], "role": rel["role"],
                              "target_subject_key": rel["target_subject_or_event_id"],
                              "promised_attributes": rel["promised_attributes"],
                              "preserved_attributes": rel["preserved_attributes"],
                              "changed_attributes": rel["changed_attributes"],
                              "relation_marker": rel["relation_marker"], "selection_uncertainties": rel["uncertainties"]})
    audio_model, reference_audio = None, []
    if can_hear:
        content = [text(f"TARGET AUDIO ({total:.3f}s, seconds from 0). Unverified ASR for cross-checking: "
                        f"{json.dumps((evidence.get('asr') or {}).get('segments', []), ensure_ascii=False)[:6000]}"),
                   audio_block((pure / tg["audio_path"]).read_bytes(), "target_audio")]
        valid_refs = []
        if cfg.loading.video_sync_audio_verified:
            for v in plan["videos"]:
                content += [text(f"REFERENCE AUDIO {v['name']}: synchronized track of reference clip {v['name']}"),
                            audio_block((work / "intermediate" / f"{v['name']}_audio.wav").read_bytes(), v["name"])]
                valid_refs.append(v["name"])

        def validate_audio(value):
            got = [c["reference_id"] for c in value["reference_audio_checks"]]
            errors = [] if sorted(got) == sorted(valid_refs) else [
                f"reference_audio_checks must list exactly {valid_refs}"]
            errors += [f"time outside target: {u}" for u in value["utterances"]
                       if not 0 <= u["start"] < u["end"] <= total + 0.2]
            return errors

        audio_model = ctx.models.call("exported_audio_facts", content, schemas.EXPORTED_AUDIO_FACTS, validate_audio,
                                      scope)
        heard = sorted(audio_model["utterances"], key=lambda u: (u["start"], u["end"]))
        utterances = [{"utterance_id": f"u{k + 1}", "start": u["start"], "end": min(total, u["end"]),
                       "language": u["language"], "kind": u["kind"], "voice_description": u["voice_description"],
                       "status": "heard_by_model", "text": u["text"], "words": None}
                      for k, u in enumerate(u for u in heard if u["intelligible"] == "yes")]
        omitted = [{"start": u["start"], "end": min(total, u["end"]), "kind": u["kind"],
                    "reason": f"intelligible={u['intelligible']}"} for u in heard if u["intelligible"] != "yes"]
        uncertain = [{"text": u["text"], "span": [u["start"], u["end"]]} for u in audio_model["uncertainties"]]
        uncertain += [{"text": OMITTED_SPEECH_NOTE, "span": [o["start"], o["end"]]} for o in omitted]
        for check in audio_model["reference_audio_checks"]:
            if check["useful"] == "yes" and check["relation_marker"] != "none" and check["role"] != "none":
                marker = check["relation_marker"]
                if marker in ("fully_copy", "partially_copy"):          # 需要对齐的信号匹配证据
                    ref_wav, _ = sf.read(work / "intermediate" / f"{check['reference_id']}_audio.wav",
                                         dtype="float32", always_2d=True)
                    tgt_wav, _ = sf.read(pure / tg["audio_path"], dtype="float32", always_2d=True)
                    _, corr = media.best_lag(ref_wav.mean(axis=1), tgt_wav.mean(axis=1), AUDIO_SR // 2, 16)
                    if not corr > 0.6:
                        marker = "reference"       # 不把“像同一首歌”升级成信号复制
                reference_audio.append({"reference_name": check["reference_id"], "role": check["role"],
                                        "relation_marker": marker, "note": check["note"]})
    else:
        utterances, omitted, uncertain = _utterances_from_asr(evidence, cfg, total)

    # 证据帧：固定步长 + 候选切镜两侧 + 末帧；参考取导出文件本身
    step = max(1, round(cfg.evidence.export_step_seconds * FPS))
    ref_images = [(im["name"], _jpeg(pure / im["path"], cfg.evidence.reference_width)) for im in plan["images"]]
    ref_videos = []
    for v in plan["videos"]:
        idx = sorted({round(k * (v["num_frames"] - 1) / 5) for k in range(6)})
        frames = media.extract_frames(pure / v["path"], idx, cfg.evidence.reference_width)
        ref_videos.append((v["name"], [(f"{v['name']}:r{i:04d}", frames[i], i / FPS) for i in idx]))
    reference_ids = {name: [name] for name, _ in ref_images} | {name: [x[0] for x in fr] for name, fr in ref_videos}
    budget = cfg.evidence.max_images_per_call - len(ref_images) - sum(len(fr) for _, fr in ref_videos)

    def evidence_frames(cuts):
        keep = {0, n - 1} | {c - 1 for c in cuts} | set(cuts)
        grid = [k for k in range(0, n, step) if k not in keep]
        room = max(0, budget - len(keep))
        if len(grid) > room:
            grid = [grid[int(k * len(grid) / room)] for k in range(room)] if room else []
        return sorted(keep | set(grid))

    asr_note = ("heard by the audio node" if can_hear else
                "UNVERIFIED ASR only; nobody listened. Attribute a speaker only from visible mouth movement")
    utter_lines = "\n".join(f"- {u['utterance_id']} {u['start']:.2f}-{u['end']:.2f}s: "
                            f"{u.get('text') or _words_text(u['words'])}" for u in utterances) or "(none)"
    relation_lines = "\n".join(
        f"- {r['relation_id']}: reference {r['reference_name']} -> target `{r['target_subject_key']}` as "
        f"{r['role']} ({r['relation_marker']}); promised: {r['promised_attributes']}" for r in relations)
    subject_lines = "\n".join(f"- key `{s['key']}` ({s['kind']}, {s['importance']}): {s['description']}"
                              for s in pair["target_subjects"])

    def call(cuts):
        frames = evidence_frames(cuts)
        jpegs = media.extract_frames(target_path, frames, cfg.evidence.dense_width)
        content = [text(
            f"EXPORTED TARGET: {n} frames at {FPS} fps, {total:.3f}s. Frame tNNNN is at NNNN/{FPS} seconds.\n"
            "CANDIDATE CUT FRAMES (a new shot may start exactly at these frames; verify each): "
            f"{[tid(c) for c in cuts] or 'none'}\n"
            f"TARGET SUBJECT KEYS fixed at selection (reuse these keys; add new keys for other visible subjects):\n"
            f"{subject_lines}\nREFERENCE RELATIONS to re-check on the exported media:\n{relation_lines}\n"
            f"SPEECH in the target ({asr_note}):\n{utter_lines}")]
        for k in frames:
            content += [text(f"target frame {tid(k)} t={k / FPS:.3f}s"), image(jpegs[k], tid(k))]
        for name, data in ref_images:
            content += [text(f"reference {name} (image)"), image(data, name)]
        for name, items in ref_videos:
            content.append(text(f"reference {name} (video clip, sampled frames in time order)"))
            for ident, data, seconds in items:
                content += [text(f"{ident} t={seconds:.2f}s"), image(data, ident)]
        valid_t, cut_ids = {tid(k) for k in frames}, [tid(c) for c in cuts]

        def validate(value):
            errors = []
            starts = [s["start_frame_id"] for s in value["shots"]]
            if not starts or starts[0] != "t0000":
                errors.append("the first shot must start at t0000")
            if set(starts[1:]) - set(cut_ids) or starts != sorted(set(starts)):
                errors.append(f"later shots must start at distinct candidate cut frames in order: {cut_ids}")
            keys = {s["key"] for s in value["subjects"]}
            for s in value["shots"]:
                if set(s["evidence_frame_ids"]) - valid_t or not s["evidence_frame_ids"]:
                    errors.append(f"shot {s['start_frame_id']}: evidence_frame_ids must be provided target frames")
                if set(s["subject_keys"]) - keys:
                    errors.append(f"shot {s['start_frame_id']}: subject_keys not listed in subjects")
            for s in value["subjects"]:
                if s["first_frame_id"] not in valid_t:
                    errors.append(f"subject {s['key']}: first_frame_id must be a provided target frame")
            if sorted(c["reference_id"] for c in value["relation_checks"]) != sorted(r["relation_id"] for r in relations):
                errors.append(f"relation_checks.reference_id must list each of {[r['relation_id'] for r in relations]} once")
            by_id = {r["relation_id"]: r for r in relations}
            for c in value["relation_checks"]:
                rel = by_id.get(c["reference_id"])
                if rel and c["holds"] == "yes":
                    if not c["target_evidence_ids"] or set(c["target_evidence_ids"]) - valid_t:
                        errors.append(f"{c['reference_id']}: target_evidence_ids must be provided target frames")
                    if not c["reference_evidence_ids"] or set(c["reference_evidence_ids"]) - set(
                            reference_ids[rel["reference_name"]]):
                        errors.append(f"{c['reference_id']}: reference_evidence_ids must be ids of {rel['reference_name']}")
                    if rel["role"] in schemas.SUBJECT_ROLES and rel["target_subject_key"] not in keys:
                        errors.append(f"subjects must include key {rel['target_subject_key']} for {c['reference_id']}")
            if sorted(a["utterance_id"] for a in value["speech_attribution"]) != sorted(
                    u["utterance_id"] for u in utterances):
                errors.append(f"speech_attribution must list each of {[u['utterance_id'] for u in utterances]} once")
            for a in value["speech_attribution"]:
                if a["speaker"] not in keys | set(SPEAKER_SPECIAL):
                    errors.append(f"speaker for {a['utterance_id']} must be a subject key, off_screen or unknown")
            for miss in value["missed_cuts"]:
                if not {miss["last_frame_id_before"], miss["first_frame_id_after"]} <= valid_t or (
                        miss["last_frame_id_before"] >= miss["first_frame_id_after"]):
                    errors.append("missed_cuts must name two provided target frames in time order")
            return errors

        return ctx.models.call("exported_visual_facts", content, schemas.EXPORTED_VISUAL_FACTS, validate, scope)

    cuts, model_reported = list(detected), []
    visual = call(cuts)
    if visual["missed_cuts"]:                  # 固定且至多一次的补充：把模型指出的区间细化为候选切镜后重新标注
        for miss in visual["missed_cuts"]:
            lo, hi = int(miss["last_frame_id_before"][1:]), int(miss["first_frame_id_after"][1:])
            frame = lo + 1 + int(np.argmax(feats["cut_score"][lo + 1:hi + 1]))
            if frame not in cuts:
                cuts.append(frame)
                model_reported.append(frame)
        cuts.sort()
        visual = call(cuts)
        if visual["missed_cuts"]:
            checks.review("visual", "the visual node still reports a possible undetected cut after one refinement")

    confirmed = sorted(int(s["start_frame_id"][1:]) for s in visual["shots"][1:])
    bounds = [0] + confirmed + [n]
    shots = [{"shot_id": f"shot{k + 1}", "index": k + 1, "start_frame": bounds[k], "end_frame_excl": bounds[k + 1],
              "start_seconds": round(bounds[k] / FPS, 6),
              **{f: s[f] for f in ("framing", "setting_and_lighting", "subjects_and_positions", "action",
                                   "camera_movement", "visible_text", "subject_keys", "evidence_frame_ids",
                                   "uncertainties")}}
             for k, s in enumerate(visual["shots"])]
    holds = {c["reference_id"]: c for c in visual["relation_checks"]}
    for rel in relations:
        rel.update(holds=holds[rel["relation_id"]]["holds"], check_note=holds[rel["relation_id"]]["note"],
                   target_evidence_ids=holds[rel["relation_id"]]["target_evidence_ids"],
                   reference_evidence_ids=holds[rel["relation_id"]]["reference_evidence_ids"])
    attribution = {a["utterance_id"]: a for a in visual["speech_attribution"]}
    end_s = float(m.t(tg["end_frame_excl"]))
    source_words = [w for s in (timeline["audio"]["asr"] or {}).get("segments", []) for w in s["words"]]
    cut_by_end = any(w["s"] < end_s - 0.1 and w["e"] > end_s + 0.1 for w in source_words) or bool(
        audio_model and audio_model["truncated_at_end"] == "yes")
    for k, u in enumerate(utterances):
        who = attribution[u["utterance_id"]]
        u.update(speaker_key=who["speaker"] if who["speaker"] not in SPEAKER_SPECIAL else f"{who['speaker']}_voice",
                 speaker_description=who["speaker_description"], on_screen=who["on_screen"],
                 speaker_evidence_frame_ids=who["evidence_frame_ids"])
        u["parts"] = _split_parts(u, shots, cut_by_end and k == len(utterances) - 1)
        u["text"] = u.get("text") or _words_text(u["words"])
        if who["speaker"] in SPEAKER_SPECIAL or who["on_screen"] != "yes":
            uncertain.append({"text": f"speaker of {u['utterance_id']} is not visually confirmed ({who['speaker']})",
                              "span": [u["start"], u["end"]]})

    facts = {
        "coordinate_system": f"exported target; frame k is at k/{FPS} seconds from 0",
        "target": {"num_frames": n, "duration_seconds": total, "source_interval_frames": [
            tg["start_frame"], tg["end_frame_excl"]]},
        "style": visual["style"], "shots": shots,
        "cuts": {"detected_frames": detected, "model_reported_then_refined_frames": model_reported,
                 "confirmed_frames": confirmed, "rejected_frames": [c for c in cuts if c not in confirmed],
                 "method": "pixel/histogram detector on the exported target, each candidate confirmed by the visual node"},
        "subjects": [{"key": s["key"], "description": s["description"], "first_frame": int(s["first_frame_id"][1:])}
                     for s in visual["subjects"]],
        "relations": relations, "utterances": utterances,
        "omitted_speech": omitted,       # 听不清而未写入提示词的对白；只有时间区间，没有文字
        "audio": {"observed_by_model": can_hear,
                  "unobserved_reason": None if can_hear else (ctx.capabilities.get("audio_reason")
                                                              or "audio_capability_missing"),
                  "model_facts": audio_model, "reference_audio": reference_audio,
                  "asr_available": evidence.get("available", False),
                  "sound_tag_hints": (evidence.get("tags") or {}).get("windows"),
                  "sound_tag_note": "relative scores over a fixed label set; hints only, never ground truth"},
        "boundaries": {"start": visual["start_boundary"], "end": visual["end_boundary"]},
        "uncertainties": {"visual": visual["uncertainties"] + [f"{s['shot_id']}: {u}" for s in shots
                                                                for u in s["uncertainties"]],
                          "audio": uncertain},
    }
    return facts


# ---------------------------------------------------------------- 节点 07–09

def _writer_content(facts: dict, labels: dict, pair: dict) -> list[dict]:
    slim = {**facts, "audio": {k: v for k, v in facts["audio"].items() if k != "model_facts" or v},
            "utterances": [{k: v for k, v in u.items() if k not in ("words", "speaker_evidence_frame_ids")}
                           for u in facts["utterances"]]}
    table = {"media_labels": labels["media"], "subject_labels": labels["subjects"],
             "video_structure_labels": labels["video_structures"], "audio_labels": labels["audios"],
             "labels_needing_definition_and_retention_lines_in_order": labels["definitions_required"],
             "speaker_ids": labels["speakers"], "speaker_subject_labels": labels["speaker_subject_labels"]}
    tokens = [{"dialogue_token": "{{D:%s}}" % p["part_id"], "speaker_token": "{{S:%s}}" % u["utterance_id"],
               "shot_id": p["shot_id"], "status": u["status"], "speaker_key": u["speaker_key"]}
              for u in facts["utterances"] for p in u["parts"]]
    return [text("FACTS (the only source of truth; ids are fact_ids):\n" + json.dumps(slim, ensure_ascii=False)),
            text("LABEL TABLE (assigned by the program; use these labels verbatim):\n"
                 + json.dumps(table, ensure_ascii=False)),
            text("DIALOGUE TOKENS to place inside the matching shot text:\n" + json.dumps(tokens, ensure_ascii=False)),
            text("SELECTION-TIME SUBJECT DESCRIPTIONS:\n" + json.dumps(pair["target_subjects"], ensure_ascii=False))]


def _audit_content(ctx, plan, pure: Path, facts: dict, labels: dict, prompt: str) -> tuple[list[dict], set]:
    cfg, n = ctx.cfg, plan["target"]["num_frames"]
    cuts = facts["cuts"]["confirmed_frames"]
    step = max(1, round(cfg.evidence.export_step_seconds * FPS))
    keep = sorted({0, n - 1} | {c - 1 for c in cuts} | set(cuts) | set(range(0, n, step)))
    room = cfg.evidence.max_images_per_call - len(plan["images"]) - 6 * len(plan["videos"])
    if len(keep) > room:
        must = {0, n - 1} | {c - 1 for c in cuts} | set(cuts)
        rest = [k for k in keep if k not in must]
        take = max(0, room - len(must))
        keep = sorted(must | {rest[int(k * len(rest) / take)] for k in range(take)}) if take else sorted(must)
    jpegs = media.extract_frames(pure / plan["target"]["path"], keep, cfg.evidence.dense_width)
    content = [text("PROMPT UNDER REVIEW:\n" + prompt),
               text("REFERENCE RECORDS: " + json.dumps(labels["reference_records"]) + "\nRELATIONS CLAIMED: "
                    + json.dumps([{k: r[k] for k in ("relation_id", "reference_name", "role", "target_subject_key",
                                                      "promised_attributes", "relation_marker")}
                                  for r in facts["relations"]], ensure_ascii=False)
                    + "\nSPEECH EVIDENCE (" + ("heard by an audio model" if facts["audio"]["observed_by_model"]
                                                else "UNVERIFIED ASR; you cannot hear the audio") + "): "
                    + json.dumps([{k: u[k] for k in ("utterance_id", "start", "end", "text", "status", "speaker_key")}
                                  for u in facts["utterances"]], ensure_ascii=False)
                    + "\nSPEECH LEFT OUT OF THE PROMPT ON PURPOSE (could not be transcribed reliably): "
                    + json.dumps([[o["start"], o["end"]] for o in facts["omitted_speech"]])),
               text(f"EXPORTED TARGET frames (frame tNNNN is at NNNN/{FPS}s):")]
    ids = set()
    for k in keep:
        content += [text(f"target frame {tid(k)} t={k / FPS:.3f}s"), image(jpegs[k], tid(k))]
        ids.add(tid(k))
    for im in plan["images"]:
        content += [text(f"reference {im['name']} = {im['path']}"),
                    image(_jpeg(pure / im["path"], cfg.evidence.reference_width), im["name"])]
        ids.add(im["name"])
    for v in plan["videos"]:
        idx = sorted({round(k * (v["num_frames"] - 1) / 5) for k in range(6)})
        frames = media.extract_frames(pure / v["path"], idx, cfg.evidence.reference_width)
        content.append(text(f"reference {v['name']} = {v['path']} (sampled frames in time order)"))
        for i in idx:
            content += [text(f"{v['name']}:r{i:04d} t={i / FPS:.2f}s"), image(frames[i], f"{v['name']}:r{i:04d}")]
            ids.add(f"{v['name']}:r{i:04d}")
    return content, ids


def build_pair(ctx, m, pair: dict, group: dict, timeline: dict, attempt: dict) -> dict:
    cfg = ctx.cfg
    checks = Checks()
    stage = Path(attempt["stage"])
    pure, work = stage / "pure", stage / "work"
    work.mkdir(parents=True)
    scope = attempt["id"]
    plan = compile_export_plan(pair, m, cfg)
    check_plan_constraints(plan, m, cfg)

    can_hear = bool(ctx.capabilities.get("can_observe_audio"))
    exported = export_media(plan, m, pure, work, can_hear and cfg.loading.video_sync_audio_verified)
    facts = build_export_facts(ctx, m, pair, plan, pure, work, timeline, scope, checks)

    # 核心参考关系在导出媒体上重新核对：未知也不能靠 needs_review 冒充有效配对
    names = {x["reference_id"]: x["name"] for x in plan["images"] + plan["videos"]}
    for ref in pair["references"]:
        if not any(r["holds"] == "yes" for r in facts["relations"] if r["reference_id"] == ref["id"]):
            write_json(work / "facts.json", facts)
            raise RejectPair(f"exported_pair_not_semantically_supported:{names[ref['id']]}")
    if "problem" in facts["boundaries"].values():
        write_json(work / "facts.json", facts)
        raise RejectPair("invalid_semantic_boundaries_on_export")
    for side, value in facts["boundaries"].items():
        if value == "unknown":
            checks.review("visual", f"target {side} boundary could not be confirmed as clean on the exported clip")

    # 声音参考确有用途时启用同步音轨（重新封装该参考视频）；否则保持无声
    for check in facts["audio"]["reference_audio"]:
        v = next(v for v in plan["videos"] if v["name"] == check["reference_name"])
        v["include_audio"] = True
        exported["exporter"].mp4(v["start_frame"], v["end_frame_excl"], v["num_frames"], plan["geometry"],
                                 pure / v["path"], work / "intermediate" / f"{v['name']}_audio.wav")
        entry = next(x for x in exported["provenance"]["videos"] if x["path"] == v["path"])
        entry.update(include_audio=True, sha256=sha256_file(pure / v["path"]))
    write_json(work / "provenance.json", exported["provenance"])
    write_json(work / "export_log.json", exported["exporter"].log)
    write_json(work / "export_plan.json", plan)

    references = [{"id": x["reference_id"], "type": "image" if x in plan["images"] else "video", "name": x["name"],
                   "path": x["path"], "include_audio": x.get("include_audio", False)}
                  for x in plan["images"] + plan["videos"]]
    labels = allocate_labels(references, facts)
    write_json(work / "facts.json", facts)
    write_json(work / "labels.json", labels)
    by_id = {r["id"]: r for r in pair["references"]}
    write_json(work / "selection.json", {
        "pair_id": pair["id"], "pair_score": pair["score"], "bundle_metrics": pair["bundle"],
        "target": {k: pair["target"][k] for k in ("id", "start_frame", "end_frame_excl", "start_s", "end_s",
                                                  "source_duration_s", "plan", "boundary_kinds", "verdict",
                                                  "score_components", "target_score")},
        "target_subjects": pair["target_subjects"],
        "references": [{"name": x["name"], "path": x["path"], "type": by_id[x["reference_id"]]["type"],
                        "source_interval_frames": by_id[x["reference_id"]]["source_interval"], "crop": None,
                        "relations": by_id[x["reference_id"]]["relations"],
                        "duplicate_check": by_id[x["reference_id"]]["duplicate_check"],
                        "reason": "; ".join(f"{r['role']} for `{r['target_subject_or_event_id']}`: "
                                            f"{', '.join(r['promised_attributes'])}"
                                            for r in by_id[x["reference_id"]]["relations"])}
                       for x in plan["images"] + plan["videos"]],
        "selection_rule": "fixed beam search over references with a supported relation; see run/pairs.json"})

    total = plan["target"]["num_frames"] / FPS
    target_record = {"video": plan["target"]["path"], "audio": plan["target"]["audio_path"],
                     "width": plan["geometry"]["canvas_w"], "height": plan["geometry"]["canvas_h"],
                     "num_frames": plan["target"]["num_frames"], "fps": FPS, "audio_sample_rate": AUDIO_SR,
                     "audio_channels": media.AUDIO_CH,
                     "waveform_samples_per_channel": plan["target"]["wav_samples_per_channel"]}
    writer_content = _writer_content(facts, labels, pair)
    draft = ctx.models.call("prompt_writer", writer_content, schemas.PROMPT_DRAFT,
                            lambda v: validate_draft(v, facts, labels), scope)

    # 最多原稿 + 一次修复；格式重试与语义修复分别计数。
    revisions = []
    for revision in range(cfg.retry.prompt_repairs + 1):
        prompt = render_six_sections(draft, facts, labels)
        spans = [u["span"] for u in facts["uncertainties"]["audio"]]
        if not facts["audio"]["observed_by_model"]:
            spans.append([0.0, total])                  # 无法精确定位的声音不确定性标记整个目标区间
        spans = merge_spans(spans, 0.0, total)
        needs_audio_review = bool(spans)
        record = {"id": attempt["provisional_sample_id"], "split_group": group["id"],
                  "path_base": "directory containing dataset.jsonl", "prompt": prompt,
                  "references": labels["reference_records"], "target": target_record,
                  "needs_audio_review": needs_audio_review, "uncertain_audio_spans_seconds": spans}
        (pure / "dataset.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        (work / "prompt.md").write_text(prompt, encoding="utf-8")
        # 重新检查全部文本相关项目，不沿用上一版草稿的 passed。
        checks.reset(list(checks.items))
        try:
            verify_package(pure, work, m, cfg, checks)
        except (Blocked, RejectPair):
            raise
        except Exception as error:  # noqa: BLE001 校验实现或读写异常属于技术错误
            raise Blocked(f"artifact_or_validation_implementation_error:{type(error).__name__}:{error}") from error
        content, ids = _audit_content(ctx, plan, pure, facts, labels, prompt)
        locations = {"definitions", "summary", "retention", "style", "soundscape", "music"} | {
            s["shot_id"] for s in facts["shots"]}
        review = ctx.models.call(
            "semantic_auditor", content, schemas.SEMANTIC_REVIEW,
            lambda v: [f"issue location must be one of {sorted(locations)}" for i in v["issues"]
                       if i["location"] not in locations][:1]
            + [f"unknown evidence ids {sorted(set(i['evidence_frame_ids']) - ids)[:6]}" for i in v["issues"]
               if set(i["evidence_frame_ids"]) - ids][:1], scope)
        known = [i for i in review["issues"] if i["kind"] in KNOWN_ERROR_KINDS]
        soft = [i for i in review["issues"] if i["kind"] not in KNOWN_ERROR_KINDS]
        revisions.append({"revision": revision, "draft": draft, "prompt": prompt, "review": review,
                          "program_checks_failed": checks.failed()})
        write_json(work / "prompt_draft.json", {"final_revision": revision, "revisions": revisions})
        if review["core_relation_valid"] != "yes" or review["boundaries_valid"] == "no":
            raise RejectPair("semantic_relation_or_boundary_failed")
        program_failed = [c for c in checks.failed() if c not in ("visual_facts_and_reference_roles",
                                                                   "audio_facts_dialogue_and_speakers")]
        if program_failed and program_failed != ["prompt_structure_labels_and_cut_times"]:
            # 导出器违反其已验证计划属于技术错误；保留逐项结果供排查
            write_json(work / "verification.json", verification_report("node_error", checks, record["id"]))
            raise Blocked(f"export_does_not_match_verified_plan:{program_failed}")
        format_issues = [{"location": "summary", "kind": "format", "quote": d["name"], "explanation": d["detail"],
                          "evidence_frame_ids": [], "repairable_by_text": True, "suggestion": ""}
                         for d in checks.items["prompt_structure_labels_and_cut_times"]["details"]
                         if d["status"] == "failed"]
        if not known and not soft and not format_issues:
            break
        final = revision == cfg.retry.prompt_repairs
        if final or not all(i["repairable_by_text"] for i in known):
            if known or format_issues:               # 未解决的已知事实错误不能作为 needs_review 发布
                raise RejectPair("unresolved_known_annotation_error")
            break
        draft = ctx.models.call(
            "prompt_repair", writer_content + [
                text("CURRENT DRAFT (JSON):\n" + json.dumps(draft, ensure_ascii=False)),
                text("ISSUES TO FIX (the only changes allowed):\n"
                     + json.dumps(known + soft + format_issues, ensure_ascii=False))],
            schemas.PROMPT_DRAFT, lambda v: validate_draft(v, facts, labels), scope)

    # 审查结果只汇总为检查项与质量标记；正文不在最终审查之后再变化。
    c = "visual_facts_and_reference_roles"
    checks.record(c, "auditor_saw_exported_media_and_found_prompt_consistent",
                  True if review["visual_consistent"] == "yes" else (None if review["visual_consistent"] == "unknown"
                                                                    else False),
                  f"visual_consistent={review['visual_consistent']} core_relation_valid={review['core_relation_valid']}")
    checks.record(c, "relations_hold_on_exported_media", all(
        r["holds"] == "yes" for r in facts["relations"] if r["relation_id"] in {
            i for s in list(labels["subjects"].values()) + list(labels["video_structures"].values())
            for i in s["relation_ids"]}), [(r["relation_id"], r["holds"]) for r in facts["relations"]])
    for issue in soft:
        checks.review("visual", f"auditor ({issue['kind']}) at {issue['location']}: {issue['explanation']}")
    for item in facts["uncertainties"]["visual"] + review["visual_uncertainties"]:
        checks.review("visual", item)
    if review["boundaries_valid"] == "unknown":
        checks.review("visual", "auditor could not confirm the semantic boundaries")
    if any(i["kind"] == "visual" for i in checks.review_items):
        checks.record(c, "no_open_visual_uncertainty", None, f"{sum(i['kind'] == 'visual' for i in checks.review_items)} "
                      "visual item(s) need review")
    c = "audio_facts_dialogue_and_speakers"
    if facts["audio"]["observed_by_model"]:
        open_audio = facts["uncertainties"]["audio"] + [{"text": t} for t in review["audio_uncertainties"]]
        checks.record(c, "audio_heard_by_model_without_open_uncertainty", None if open_audio else True,
                      f"{len(open_audio)} open audio item(s); single listening pass by exported_audio_facts")
    else:
        checks.record(c, "audio_heard_by_a_model", None,
                      f"not checked: {facts['audio']['unobserved_reason']}; dialogue, speakers, soundscape and "
                      "music rest on ASR/sound-tag evidence only")
        checks.review("audio", "no model listened to the target audio; verify dialogue, speakers, soundscape and "
                      "music by ear", [0.0, round(total, 3)])
    for item in facts["uncertainties"]["audio"]:
        checks.review("audio", item["text"], item.get("span"))
    for item in review["audio_uncertainties"]:
        checks.review("audio", f"auditor: {item}")

    if review["audio_uncertainties"] and not record["needs_audio_review"]:
        # 审查新增声音不确定性：只刷新布尔值与区间，不改正文
        record.update(needs_audio_review=True, uncertain_audio_spans_seconds=[[0.0, round(total, 3)]])
        (pure / "dataset.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    status = classify_sample(checks)
    write_json(work / "verification.json", verification_report(status, checks, attempt["provisional_sample_id"]))
    if not any((work / "intermediate").iterdir()):
        (work / "intermediate").rmdir()
    return {"stage": str(stage), "pair_id": pair["id"], "score": pair["score"], "status": status,
            "target_interval": [pair["target"]["start_frame"], pair["target"]["end_frame_excl"]],
            "checks": checks, "attempt": attempt}


def verification_report(status: str, checks: Checks, sample_id: str) -> dict:
    return {"status": status, "sample_id": sample_id, "review_items": checks.review_items,
            "checks": checks.to_list(), "not_done": NOT_DONE,
            "note": "status describes data construction and annotation checks only, not training quality"}


def classify_sample(checks: Checks) -> str:
    hard = [n for n, item in checks.items.items() if item["hard"]]
    if any(checks.status(n) == "failed" for n in hard):
        raise Blocked(f"required_check_failed:{[n for n in hard if checks.status(n) == 'failed']}")
    if checks.failed():
        raise RejectPair(f"required_check_failed:{checks.failed()}")
    if any(checks.status(n) == "not_checked" for n in hard):
        raise Blocked("required_hard_validation_missing")
    if checks.review_items or any(item["status"] != "passed" for item in checks.items.values()):
        return "needs_review"
    return "completed"
