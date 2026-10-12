"""节点 01–02：来源分组、固定抽样与视听理解，产出可引用的源时间线。"""
from __future__ import annotations

import io

import numpy as np

from . import media, schemas
from .models import audio as audio_block
from .models import image, text
from .util import Blocked, merge_spans

LANGUAGES = {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "fr": "French", "de": "German",
             "es": "Spanish", "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic"}


def fid(index: int) -> str:
    return f"f{index:06d}"


def assign_source_group(ctx, m, feats) -> dict:
    """优先作品标识；否则查版本化登记表（哈希 → 画面指纹）；再否则以源哈希建临时组。"""
    fingerprint = [bytes(feats["dhash"][m.frame_at(t)]).hex() for t in range(0, int(float(m.t(m.n - 1))) + 1)]
    if ctx.cfg.optional_work_id:
        return {"id": ctx.cfg.optional_work_id, "method": "configured_work_id", "video_fingerprint": fingerprint}
    for group in ctx.registry.get("groups", []):
        if m.sha256 in group.get("sha256", []):
            return {"id": group["id"], "method": "registry_sha256", "video_fingerprint": fingerprint}
    for group in ctx.registry.get("groups", []):
        for known in group.get("video_fingerprints", []):
            pairs = list(zip(fingerprint, known))
            if len(pairs) >= 5 and abs(len(fingerprint) - len(known)) <= 2:
                distance = np.mean([bin(int(a, 16) ^ int(b, 16)).count("1") for a, b in pairs])
                if distance <= 6:
                    return {"id": group["id"], "method": "registry_video_fingerprint",
                            "mean_hamming": float(distance), "video_fingerprint": fingerprint}
    return {"id": f"sha256:{m.sha256[:16]}", "method": "provisional_source_hash",
            "video_fingerprint": fingerprint,
            "note": "provisional group; merge through the source registry when another encoding is identified"}


def _windows(m, cfg) -> list[tuple[float, float]]:
    total = float(m.t(m.n)) if m.last_end_known else float(m.t(m.n - 1))
    size, stride = cfg.search.understanding_window_seconds, (
        cfg.search.understanding_window_seconds - cfg.search.understanding_overlap_seconds)
    out, start = [], 0.0
    while True:
        end = min(total, start + size)
        out.append((start, end))
        if end >= total:
            return out
        start += stride


def _speech_and_pauses(asr: dict | None, rms: np.ndarray, hop: float, total: float, cfg) -> dict:
    """VAD（语音活动检测）近似：ASR 词间停顿 + 低能量区间。只是边界候选的证据。"""
    words = [w for s in (asr or {}).get("segments", []) for w in s["words"] if w["e"] > w["s"]]
    speech = merge_spans([[w["s"], w["e"]] for w in words], 0.0, total)
    need = cfg.audio.pause_seconds
    pauses = []
    edges = [0.0] + [x for span in speech for x in span] + [total]
    for start, end in zip(edges[0::2], edges[1::2]):
        if end - start >= need:
            pauses.append({"start": round(start, 3), "end": round(end, 3), "source": "asr_word_gap"})
    quiet = rms < max(cfg.thresholds.silence_db, float(rms.max()) - 35.0)
    run = None
    for i, flag in enumerate(list(quiet) + [False]):
        if flag and run is None:
            run = i
        elif not flag and run is not None:
            if (i - run) * hop >= need:
                pauses.append({"start": round(run * hop, 3), "end": round(i * hop, 3), "source": "low_energy"})
            run = None
    return {"speech_spans": speech, "pauses": sorted(pauses, key=lambda p: (p["start"], p["end"], p["source"]))}


def _wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, samples, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def build_source_timeline(ctx, m, feats) -> dict:
    cfg = ctx.cfg
    cuts = media.detect_cuts(feats["cut_score"])
    ctx.log("cuts_detected", count=len(cuts))
    windows = _windows(m, cfg)
    step = cfg.search.overview_step_seconds
    registry: dict[str, dict] = {}
    cut_votes: dict[int, list[str]] = {c: [] for c in cuts}
    events, visible_text, quality, uncertainties, window_log = [], [], [], [], []

    # 固定顺序窗口；主体登记表按窗口顺序传递，保证跨窗口使用同一主体键。
    for w, (ws, we) in enumerate(windows, start=1):
        overview = sorted({m.frame_at(ws + k * step) for k in range(int((we - ws) / step) + 1)
                           if ws + k * step < we})
        local_cuts = [c for c in cuts if ws <= float(m.t(c)) < we]
        indices = sorted(set(overview) | {c - 1 for c in local_cuts} | set(local_cuts))
        known = sorted(registry.values(), key=lambda s: (-len(s["clear_frames"]), s["key"]))
        known = known[:cfg.evidence.known_subject_thumbnails]
        budget = cfg.evidence.max_images_per_call - len(known)
        if len(indices) > budget:                    # 确定性降采样；保留切镜两侧帧
            keep = {c - 1 for c in local_cuts} | set(local_cuts)
            rest = [i for i in indices if i not in keep]
            take = max(0, budget - len(keep))
            rest = [rest[int(k * len(rest) / take)] for k in range(take)] if take else []
            indices = sorted(keep | set(rest))[:budget]
        jpegs = ctx.frames(indices, cfg.evidence.overview_width)
        content = [text(f"SOURCE WINDOW w{w:02d}: source seconds [{ws:.3f}, {we:.3f}). "
                        f"Frames are real decoded frames in time order; sampling step is about {step}s, so "
                        "anything between two frames is unobserved.")]
        if known:
            content.append(text("KNOWN SUBJECTS from earlier windows. Reuse the exact key when the same subject "
                                "appears again; invent a new snake_case key only for a genuinely new subject."))
            thumbs = ctx.frames([s["thumbnail_frame"] for s in known], cfg.evidence.overview_width)
            for s in known:
                content += [text(f"known subject key={s['key']} kind={s['kind']}: {s['description']}"),
                            image(thumbs[s["thumbnail_frame"]], f"known_{s['key']}")]
        content.append(text("CANDIDATE CUTS from a pixel-difference heuristic (flashes, fast motion and fades can "
                            "trigger it):\n" + ("\n".join(
                                f"- c{c:06d}: new shot would start at {fid(c)}, previous frame {fid(c - 1)}"
                                for c in local_cuts) or "(none in this window)")))
        for i in indices:
            content += [text(f"frame {fid(i)} t={float(m.t(i)):.3f}s"), image(jpegs[i], fid(i))]
        valid_frames = {fid(i) for i in indices}
        valid_cuts = {f"c{c:06d}" for c in local_cuts}

        def validate(value, valid_frames=valid_frames, valid_cuts=valid_cuts):
            errors = []
            if {c["cut_id"] for c in value["cuts"]} != valid_cuts:
                errors.append(f"cuts must list exactly these cut_ids once each: {sorted(valid_cuts)}")
            used = [f for s in value["subjects"] for f in s["clear_frame_ids"]]
            used += [f for e in value["events"] for f in (e["start_frame_id"], e["end_frame_id"])]
            used += [f for t in value["visible_text"] + value["quality_issues"] for f in t["frame_ids"]]
            bad = sorted(set(used) - valid_frames)
            if bad:
                errors.append(f"unknown frame ids: {bad[:8]}")
            keys = {s["key"] for s in value["subjects"]} | set(registry)
            for e in value["events"]:
                if set(e["subject_keys"]) - keys:
                    errors.append(f"event uses undefined subject keys: {sorted(set(e['subject_keys']) - keys)}")
                if e["start_frame_id"] > e["end_frame_id"]:
                    errors.append("event start_frame_id is after end_frame_id")
            return errors

        facts = ctx.models.call("source_visual_facts", content, schemas.VISUAL_FACTS, validate)
        for c in facts["cuts"]:
            cut_votes[int(c["cut_id"][1:])].append(c["verdict"])
        for s in facts["subjects"]:
            frames = sorted(int(f[1:]) for f in s["clear_frame_ids"])
            entry = registry.setdefault(s["key"], {"key": s["key"], "kind": s["kind"],
                                                   "description": s["description"], "clear_frames": []})
            entry["clear_frames"] = sorted(set(entry["clear_frames"]) | set(frames))
            if entry["clear_frames"]:
                entry["thumbnail_frame"] = max(entry["clear_frames"], key=lambda i: (feats["sharpness"][i], -i))
        registry = {k: v for k, v in registry.items() if v["clear_frames"]}
        for e in facts["events"]:
            start, end = int(e["start_frame_id"][1:]), int(e["end_frame_id"][1:])
            later = [i for i in indices if i > end]
            events.append({"window": w, "start_frame": start, "last_observed_frame": end,
                           "next_observed_frame": later[0] if later else None,
                           "start_s": float(m.t(start)), "last_observed_s": float(m.t(end)),
                           "description": e["description"], "subject_keys": e["subject_keys"],
                           "action_complete_at_end": e["action_complete_at_end"]})
        visible_text += [{"text": t["text"], "frames": [int(f[1:]) for f in t["frame_ids"]]}
                         for t in facts["visible_text"]]
        quality += [{"issue": q["issue"], "frames": [int(f[1:]) for f in q["frame_ids"]]}
                    for q in facts["quality_issues"]]
        uncertainties += [f"w{w:02d}: {u}" for u in facts["uncertainties"]]
        window_log.append({"window_id": f"w{w:02d}", "start_s": ws, "end_s": we, "frames": indices,
                           "candidate_cuts": local_cuts})
        ctx.log("source_window_done", window=w, of=len(windows), subjects=len(registry))

    # 重叠窗口的冲突按固定规则保留为 unknown；不以多数投票消除矛盾。
    cut_records = []
    for c in cuts:
        votes = set(cut_votes[c])
        verdict = "confirmed" if votes == {"confirmed"} else ("rejected" if votes == {"rejected"} else "unknown")
        cut_records.append({"cut_id": f"c{c:06d}", "frame": c, "seconds": float(m.t(c)),
                            "score": round(float(feats["cut_score"][c]), 2), "verdict": verdict,
                            "votes": cut_votes[c]})
    confirmed = [c["frame"] for c in cut_records if c["verdict"] == "confirmed"]
    bounds = [0] + confirmed + [m.n]
    shots = [{"shot_id": f"s{k + 1:03d}", "start_frame": a, "end_frame_excl": b, "start_s": float(m.t(a)),
              "end_s": float(m.t(b)) if b < m.n or m.last_end_known else None}
             for k, (a, b) in enumerate(zip(bounds, bounds[1:]))]
    for k, e in enumerate(sorted(events, key=lambda e: (e["start_frame"], e["last_observed_frame"])), start=1):
        e["event_id"] = f"e{k:03d}"
    events.sort(key=lambda e: e["event_id"])

    timeline = {
        "coordinate_system": "source seconds from decoded PTS; frames zero-based, intervals end-exclusive",
        "sampling": {"overview_step_seconds": step, "windows": window_log,
                     "note": "details between sampled frames were not observed"},
        "cuts": cut_records, "shots": shots, "subjects": registry, "events": events,
        "visible_text": visible_text, "quality_issues": quality, "uncertainties": uncertainties,
        "audio": _source_audio(ctx, m, windows),
    }
    ctx.save("timeline.json", timeline)
    return timeline


def _source_audio(ctx, m, windows) -> dict:
    cfg = ctx.cfg
    total = float(m.t(m.n)) if m.last_end_known else float(m.t(m.n - 1))
    hop = 0.05
    mono = media.decode_audio(m.path, 16000, 1)
    rms = media.rms_db(mono, 16000, hop)
    evidence = ctx.audio_evidence(m.path)
    asr = evidence.get("asr") if evidence["available"] else None
    offset = m.audio.get("start_offset_seconds", 0.0)
    if offset > 0:          # 音轨晚于画面开始：把证据移回共同的源时间原点，不抹掉真实延迟
        for seg in (asr or {}).get("segments", []):
            seg["start"], seg["end"] = round(seg["start"] + offset, 3), round(seg["end"] + offset, 3)
            for word in seg["words"]:
                word["s"], word["e"] = round(word["s"] + offset, 3), round(word["e"] + offset, 3)
        for window in (evidence.get("tags") or {}).get("windows", []):
            window["start"], window["end"] = round(window["start"] + offset, 3), round(window["end"] + offset, 3)
        rms = np.concatenate([np.full(int(offset / hop), -90.0), rms])
    out = {
        "track_start_offset_seconds": offset,
        "asr_available": evidence["available"], "asr_unavailable_reason": evidence.get("reason"),
        "asr": asr, "tags": evidence.get("tags"), "tags_error": evidence.get("tags_error"),
        "language": LANGUAGES.get((asr or {}).get("language"), (asr or {}).get("language")),
        "rms_db_hop_seconds": hop, "rms_db": [round(float(x), 1) for x in rms],
        "effectively_silent": bool(rms.max() < cfg.thresholds.silence_db),
        **_speech_and_pauses(asr, rms, hop, total, cfg),
        "observed_by_model": False, "model_facts": None,
        "note": "ASR and sound tags are evidence, not ground truth",
    }
    if ctx.capabilities.get("can_observe_audio"):
        def one(job):
            w, (ws, we) = job
            clip = mono[int(ws * 16000):int(we * 16000), 0]
            words = [x for s in (asr or {}).get("segments", []) for x in s["words"] if ws <= x["s"] < we]
            return ctx.models.call("source_audio_facts", [
                text(f"SOURCE AUDIO WINDOW w{w:02d}: source seconds [{ws:.3f}, {we:.3f}). Report times in source "
                     f"seconds (clip second 0 equals source second {ws:.3f}).\nUnverified ASR words: {words}"),
                audio_block(_wav_bytes(clip, 16000), f"audio_w{w:02d}")], schemas.AUDIO_FACTS)
        out["model_facts"] = ctx.models.map(list(enumerate(windows, start=1)), one)
        out["observed_by_model"] = True
    else:
        out["unobserved_reason"] = ctx.capabilities.get("audio_reason") or "audio_capability_missing"
    return out


def check_feature_count(m, feats) -> None:
    if len(feats["luma"]) != m.n:
        raise Blocked(f"frame_feature_count_mismatch:{len(feats['luma'])}!={m.n}")
