"""节点 03：枚举并排序目标候选。网格端点只是召回手段，不证明语义完整。"""
from __future__ import annotations

import numpy as np

from . import schemas
from .models import image, text
from .planning import plan_legal_video
from .source import fid


def boundary_set(m, timeline, cfg, with_grid: bool) -> dict[int, str]:
    """帧边界 → 边界类别（取权重最高的类别）。"""
    weights = cfg.scoring.boundary_kind
    out: dict[int, str] = {}

    def add(frame: int, kind: str) -> None:
        if 0 <= frame <= m.n and (frame < m.n or m.last_end_known):
            if frame not in out or weights[kind] > weights[out[frame]]:
                out[frame] = kind

    add(0, "source_edge")
    add(m.n, "source_edge")
    for cut in timeline["cuts"]:
        if cut["verdict"] == "confirmed":
            add(cut["frame"], "confirmed_cut")
    for event in timeline["events"]:
        if event["action_complete_at_end"] == "yes" and event["next_observed_frame"] is not None:
            add((event["last_observed_frame"] + event["next_observed_frame"] + 1) // 2, "action_end")
    for pause in timeline["audio"]["pauses"]:
        add(m.nearest_boundary((pause["start"] + pause["end"]) / 2), "vad_pause")
    if with_grid:
        total = float(m.t(m.n)) if m.last_end_known else float(m.t(m.n - 1))
        step = cfg.search.candidate_step_seconds
        for k in range(int(total / step) + 1):
            add(m.nearest_boundary(k * step), "grid")
    return out


def in_speech(timeline, seconds: float, margin: float = 0.05) -> bool:
    return any(s + margin < seconds < e - margin for s, e in timeline["audio"]["speech_spans"])


def visual_defect(m, feats, timeline, a: int, b: int, cfg) -> str | None:
    """黑屏、严重模糊、片尾占比等程序初筛。"""
    th = cfg.thresholds
    if float((feats["luma"][a:b] < th.black_luma).mean()) > th.black_fraction:
        return "black_frames"
    if float((feats["sharpness"][a:b] < th.blur_sharpness).mean()) > th.blur_fraction:
        return "severe_blur"
    sampled = [i for w in timeline["sampling"]["windows"] for i in w["frames"] if a <= i < b]
    flagged = {i for q in timeline["quality_issues"] if q["issue"] in ("credits_or_endcard", "black")
               for i in q["frames"] if a <= i < b}
    if sampled and len(flagged) / len(set(sampled)) > th.endcard_fraction:
        return "credits_or_endcard"
    return None


def audio_state(m, timeline, a: int, b: int, cfg) -> dict:
    audio = timeline["audio"]
    hop = audio["rms_db_hop_seconds"]
    t0, t1 = float(m.t(a)), float(m.t(b))
    rms = np.array(audio["rms_db"][int(t0 / hop):max(int(t0 / hop) + 1, int(t1 / hop))] or [-90.0])
    words = [w for s in (audio["asr"] or {}).get("segments", []) for w in s["words"] if t0 <= w["s"] < t1]
    track = m.audio_interval() or (0.0, 0.0)
    return {"silent": bool(rms.max() < cfg.thresholds.silence_db), "words": words,
            "uncovered_seconds": round(max(0.0, track[0] - t0) + max(0.0, t1 - track[1]), 4)}


def enumerate_target_candidates(ctx, m, feats, timeline) -> list[dict]:
    cfg, search = ctx.cfg, ctx.cfg.search
    weights = cfg.scoring.boundary_kind
    bounds = boundary_set(m, timeline, cfg, with_grid=True)
    frames = sorted(bounds)
    lo, hi = search.min_target_seconds, search.max_target_seconds
    feasible, rejected = [], {}
    sharp_ref = float(np.percentile(feats["sharpness"], 90)) + 1e-6
    for i, a in enumerate(frames):
        for b in frames[i + 1:]:
            duration = m.t(b) - m.t(a)
            if duration > hi:
                break
            if duration < lo - search.max_tail_padding_seconds:
                continue
            plan = plan_legal_video(duration, lo, hi, search)
            if not plan:
                rejected["no_legal_frame_plan"] = rejected.get("no_legal_frame_plan", 0) + 1
                continue
            reason = visual_defect(m, feats, timeline, a, b, cfg)
            state = audio_state(m, timeline, a, b, cfg)
            if not reason and state["silent"]:
                reason = "no_audible_source_sound"
            if not reason and state["uncovered_seconds"] > 0.5 * float(duration):
                reason = "source_audio_does_not_cover_interval"
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            t0, t1 = float(m.t(a)), float(m.t(b))
            speech_ok = 0.5 * (not in_speech(timeline, t0)) + 0.5 * (not in_speech(timeline, t1))
            coarse = (0.35 * weights[bounds[a]] + 0.35 * weights[bounds[b]] + 0.20 * speech_ok
                      + 0.10 * min(1.0, float(feats["sharpness"][a:b].mean()) / sharp_ref))
            feasible.append({"id": f"t_{a:06d}_{b:06d}", "start_frame": a, "end_frame_excl": b,
                             "start_s": t0, "end_s": t1, "source_duration_s": float(duration), "plan": plan,
                             "boundary_kinds": [bounds[a], bounds[b]], "coarse_score": round(coarse, 4),
                             "audio_uncovered_seconds": state["uncovered_seconds"]})
    # 固定粗排 + 时间覆盖去冗余；完整性模型调用也有显式预算。
    feasible.sort(key=lambda c: (-c["coarse_score"], c["start_frame"], c["end_frame_excl"]))
    shortlist, truncated = [], []
    for c in feasible:
        redundant = False
        for kept in shortlist:
            inter = min(c["end_s"], kept["end_s"]) - max(c["start_s"], kept["start_s"])
            union = max(c["end_s"], kept["end_s"]) - min(c["start_s"], kept["start_s"])
            if inter / union > search.target_coverage_max_iou:
                redundant = True
                break
        if redundant or len(shortlist) >= search.target_evaluation_budget:
            truncated.append([c["id"], c["coarse_score"], "temporal_redundancy" if redundant else "budget"])
        else:
            shortlist.append(c)
    ctx.log("target_candidates", boundaries=len(frames), feasible=len(feasible), evaluated=len(shortlist),
            rejected=rejected)

    levels = cfg.scoring.levels
    confident = cfg.audio.confident_word_probability

    def evaluate(c):
        a, b = c["start_frame"], c["end_frame_excl"]
        t0, t1 = c["start_s"], c["end_s"]
        steps = int((t1 - t0) / search.dense_step_seconds) + 1
        inside = sorted({m.frame_at(t0 + k * search.dense_step_seconds) for k in range(steps)} & set(range(a, b))
                        | {a, b - 1})
        budget = cfg.evidence.max_images_per_call - 2
        if len(inside) > budget:
            inside = sorted({inside[round(k * (len(inside) - 1) / (budget - 1))] for k in range(budget)})
        before, after = (a - 1 if a > 0 else None), (b if b < m.n else None)
        indices = [i for i in (before, *inside, after) if i is not None]
        jpegs = ctx.frames(indices, cfg.evidence.dense_width)
        words = [w for s in (timeline["audio"]["asr"] or {}).get("segments", []) for w in s["words"]
                 if t0 - 1.5 <= w["s"] < t1 + 1.5]
        word_lines = "\n".join(
            f"- {w['s'] - t0:+.2f}s..{w['e'] - t0:+.2f}s p={w['p']:.2f} "
            f"[{'INSIDE' if t0 <= w['s'] and w['e'] <= t1 else 'CROSSES BOUNDARY' if w['s'] < t1 and w['e'] > t0 else 'outside'}] "
            f"{w['w']}" for w in words) or "(no ASR words near this interval)"
        content = [text(
            f"CANDIDATE TARGET {c['id']}: source seconds [{t0:.3f}, {t1:.3f}), {t1 - t0:.3f}s long. Boundary "
            f"kinds: start={c['boundary_kinds'][0]}, end={c['boundary_kinds'][1]}.\n"
            f"Unverified ASR words near the interval (times relative to the candidate start; nobody listened to "
            f"this audio):\n{word_lines}")]
        for i in indices:
            where = "BEFORE the candidate (context only)" if i == before else (
                "AFTER the candidate (context only)" if i == after else f"t={float(m.t(i)) - t0:.3f}s")
            content += [text(f"frame {fid(i)} {where}"), image(jpegs[i], fid(i))]
        valid = {fid(i) for i in indices}
        verdict = ctx.models.call(
            "target_completeness", content, schemas.CANDIDATE_VERDICT,
            lambda v: [f"unknown frame ids: {sorted(set(v['evidence_frame_ids']) - valid)[:8]}"]
            if set(v["evidence_frame_ids"]) - valid else [])
        inside_words = [w for w in words if t0 <= w["s"] < t1]
        if not timeline["audio"]["asr_available"]:
            annotatable = "low"
        elif not inside_words:
            annotatable = "medium"
        else:
            annotatable = "high" if min(w["p"] for w in inside_words) >= confident else "low"
        clean = (verdict["start_boundary"], verdict["end_boundary"])
        complete = verdict["completeness"] == "complete" and not any(x.startswith("mid_") for x in clean)
        components = {"semantic": 1.0 if clean == ("clean", "clean") else 0.6,
                      "visual": min(levels[verdict["visual_quality"]], levels[verdict["describability"]]),
                      "audio": levels[annotatable], "information": levels[verdict["information"]]}
        w = cfg.scoring.target
        return {**c, "verdict": verdict, "complete": complete, "audio_annotatability": annotatable,
                "score_components": components, "evidence_frames": inside,
                "target_score": round(sum(w[k] * components[k] for k in w), 4)}

    evaluated = ctx.models.map(shortlist, evaluate)
    # 完整性是硬条件；排序键 = (-target_score, 源开始PTS, 源结束PTS, candidate_id)。
    candidates = sorted((c for c in evaluated if c["complete"]),
                        key=lambda c: (-c["target_score"], c["start_frame"], c["end_frame_excl"], c["id"]))
    ctx.save("candidates.json", {
        "boundaries": {str(k): v for k, v in sorted(bounds.items()) if v != "grid"},
        "grid_boundary_count": sum(v == "grid" for v in bounds.values()),
        "enumerated_feasible": len(feasible), "program_rejections": rejected,
        "evaluated": evaluated, "ranked_complete_ids": [c["id"] for c in candidates],
        "rejected_incomplete_or_unknown": [c["id"] for c in evaluated if not c["complete"]],
        "truncated_by_budget_or_redundancy": {"count": len(truncated), "items": truncated[:2000]},
        "shortlist_limit": search.shortlist_targets,
    })
    ctx.log("targets_ranked", complete=len(candidates), evaluated=len(evaluated))
    return candidates
