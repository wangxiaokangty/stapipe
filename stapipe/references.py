"""节点 04：目标外参考检索、近重复检查、配对判定与参考集合选择。"""
from __future__ import annotations

import numpy as np

from . import schemas
from .models import image, text
from .planning import plan_legal_video
from .source import fid
from .targets import boundary_set, visual_defect
from .util import content_hash


def _tiny(feats, indices) -> np.ndarray:
    return feats["tiny"][indices].reshape(len(indices), -1).astype(np.float32)


def _min_diffs(feats, ref_frames, target_frames) -> np.ndarray:
    """每个参考帧到任一目标帧的最小平均绝对差（16×9 灰度）。"""
    ref, tgt = _tiny(feats, ref_frames), _tiny(feats, target_frames)
    return np.array([np.abs(tgt - row).mean(axis=1).min() for row in ref])


def reference_pool(ctx, m, feats, timeline, target) -> dict:
    """图片取目标外清晰真实帧；视频取目标外自然边界片段。区间左闭右开，跨越目标的候选直接排除。"""
    cfg, search = ctx.cfg, ctx.cfg.search
    a, b = target["start_frame"], target["end_frame_excl"]
    per_type = search.reference_pool_per_type
    subjects = timeline["subjects"]
    target_keys = sorted(k for k, s in subjects.items() if any(a <= f < b for f in s["clear_frames"]))
    sharp_ref = float(np.percentile(feats["sharpness"], 90)) + 1e-6

    def outside(i):
        return not a <= i < b

    # 图片：按目标主体轮流取其在目标外最清晰的帧；同分按源时间排序。
    per_subject = {k: sorted((f for f in subjects[k]["clear_frames"] if outside(f)),
                             key=lambda f: (-feats["sharpness"][f], f)) for k in target_keys}
    images: list[int] = []
    while len(images) < per_type and any(per_subject.values()):
        for key in target_keys:
            while per_subject[key]:
                f = per_subject[key].pop(0)
                near = any(np.abs(_tiny(feats, [f]) - _tiny(feats, [g])).mean() < cfg.thresholds.duplicate_tiny_diff
                           for g in images)
                if f not in images and not near:
                    images.append(f)
                    break
            if len(images) >= per_type:
                break
    image_refs = [{"id": f"img_{f:06d}", "type": "image", "source_frame": f, "source_seconds": float(m.t(f)),
                   "source_interval": [f, f + 1],
                   "timeline_subjects": sorted(k for k in subjects if f in subjects[k]["clear_frames"]),
                   "sharpness": float(feats["sharpness"][f])} for f in sorted(images)]

    # 视频：自然边界（不含网格）之间、可完成 17k+5 与时长规划的片段。
    bounds = boundary_set(m, timeline, cfg, with_grid=False)
    weights = cfg.scoring.boundary_kind
    frames = sorted(bounds)
    lo, hi = search.min_reference_seconds, search.max_reference_seconds
    videos = []
    for i, s in enumerate(frames):
        for e in frames[i + 1:]:
            duration = m.t(e) - m.t(s)
            if duration > hi:
                break
            if max(s, a) < min(e, b) or duration < lo - search.max_tail_padding_seconds:
                continue
            plan = plan_legal_video(duration, lo, hi, search)
            if plan is None or visual_defect(m, feats, timeline, s, e, cfg):
                continue
            keys = sorted(k for k in target_keys if any(s <= f < e for f in subjects[k]["clear_frames"]))
            score = (len(keys) / max(1, len(target_keys)) + 0.25 * (weights[bounds[s]] + weights[bounds[e]])
                     + 0.1 * min(1.0, float(feats["sharpness"][s:e].mean()) / sharp_ref))
            videos.append({"id": f"vid_{s:06d}_{e:06d}", "type": "video", "source_interval": [s, e],
                           "start_s": float(m.t(s)), "end_s": float(m.t(e)), "plan": plan,
                           "boundary_kinds": [bounds[s], bounds[e]], "timeline_subjects": keys,
                           "retrieval_score": round(score, 4)})
    videos.sort(key=lambda v: (-v["retrieval_score"], v["source_interval"]))
    kept = []
    for v in videos:
        s, e = v["source_interval"]
        if all((min(e, k["source_interval"][1]) - max(s, k["source_interval"][0]))
               / (max(e, k["source_interval"][1]) - min(s, k["source_interval"][0])) <= 0.6 for k in kept):
            kept.append(v)
        if len(kept) >= per_type:
            break
    return {"target_timeline_subjects": target_keys, "images": image_refs, "videos": kept,
            "audios": [], "audio_note": "standalone audio references are not retrieved in stapipe-v1"}


def measure_near_duplicate(ctx, feats, ref, target) -> dict:
    """逐帧感知特征 + 覆盖率；有歧义时交给固定的 duplicate_review 节点。抽样检查不宣称穷尽证明。"""
    th = ctx.cfg.thresholds
    s, e = ref["source_interval"]
    ref_frames = list(range(s, e, 2 if e - s > 2 else 1))
    target_frames = list(range(target["start_frame"], target["end_frame_excl"]))
    diffs = _min_diffs(feats, ref_frames, target_frames)
    dup_cov = float((diffs < th.duplicate_tiny_diff).mean())
    amb_cov = float((diffs < th.ambiguous_tiny_diff).mean())
    need = th.duplicate_coverage if ref["type"] == "video" else 0.5
    result = "duplicate" if dup_cov >= need else ("ambiguous" if amb_cov >= 0.1 else "clear")
    worst = int(np.argmin(diffs))
    return {"result": result, "method": "tiny_gray_min_abs_diff", "min_diff": round(float(diffs.min()), 2),
            "duplicate_coverage": round(dup_cov, 3), "ambiguous_coverage": round(amb_cov, 3),
            "thresholds": [th.duplicate_tiny_diff, th.ambiguous_tiny_diff, need],
            "closest_reference_frame": ref_frames[worst], "sampled_reference_frames": len(ref_frames),
            "_order": [ref_frames[i] for i in np.argsort(diffs)[:4]]}


def _duplicate_review(ctx, m, feats, ref, target, measured) -> dict:
    target_frames = list(range(target["start_frame"], target["end_frame_excl"]))
    content = [text("Decide whether the REFERENCE repeats footage of the TARGET (same shot replayed or a "
                    "near-identical take) or shows the same subjects in genuinely different footage.")]
    ids = []
    for r in sorted(measured["_order"]):
        match = target_frames[int(np.abs(_tiny(feats, target_frames) - _tiny(feats, [r])).mean(axis=1).argmin())]
        jpegs = ctx.frames([r, match], ctx.cfg.evidence.reference_width)
        content += [text(f"reference frame {fid(r)}"), image(jpegs[r], fid(r)),
                    text(f"closest target frame {fid(match)}"), image(jpegs[match], fid(match))]
        ids += [fid(r), fid(match)]
    review = ctx.models.call("duplicate_review", content, schemas.DUPLICATE_REVIEW,
                             lambda v: ["unknown evidence ids"] if set(v["evidence_ids"]) - set(ids) else [])
    return {**measured, "result": review["result"], "model_review": review}


def judge_relations(ctx, m, timeline, target, refs) -> dict:
    """固定配对节点观察目标与全部候选参考的真实内容，对每条关系返回属性与证据。"""
    cfg = ctx.cfg
    n_target = min(16, len(target["evidence_frames"]))
    t_frames = sorted({target["evidence_frames"][round(k * (len(target["evidence_frames"]) - 1) / max(1, n_target - 1))]
                       for k in range(n_target)})
    vids = [r for r in refs if r["type"] == "video"]
    imgs = [r for r in refs if r["type"] == "image"]
    per_video = max(3, min(8, (cfg.evidence.max_images_per_call - len(t_frames) - len(imgs)) // max(1, len(vids))))
    ref_frames: dict[str, list[int]] = {}
    for r in refs:
        s, e = r["source_interval"]
        ref_frames[r["id"]] = [s] if r["type"] == "image" else sorted(
            {s + round(k * (e - 1 - s) / (per_video - 1)) for k in range(per_video)})
    jpegs = ctx.frames(t_frames, cfg.evidence.dense_width)
    content = [text(f"TARGET {target['id']} ({target['source_duration_s']:.2f}s). Target frames in time order:")]
    for i in t_frames:
        content += [text(f"target frame {fid(i)} t={float(m.t(i)) - target['start_s']:.2f}s"),
                    image(jpegs[i], fid(i))]
    content.append(text("CANDIDATE REFERENCES. All come from the same source video, outside the target interval."))
    for r in refs:
        frames = ctx.frames(ref_frames[r["id"]], cfg.evidence.reference_width)
        head = (f"reference {r['id']} (single image)" if r["type"] == "image" else
                f"reference {r['id']} (video clip, {r['end_s'] - r['start_s']:.2f}s; sampled frames in time order)")
        content.append(text(head))
        for i in ref_frames[r["id"]]:
            content += [text(f"{r['id']} frame {fid(i)}"), image(frames[i], fid(i))]
    target_ids = {fid(i) for i in t_frames}

    def validate(value):
        errors = []
        keys = [s["key"] for s in value["target_subjects"]]
        if len(set(keys)) != len(keys):
            errors.append("target_subjects keys must be unique")
        for s in value["target_subjects"]:
            if set(s["target_frame_ids"]) - target_ids:
                errors.append(f"target subject {s['key']} cites unknown target frame ids")
        seen = set()
        for rel in value["relations"]:
            rid = rel["reference_id"]
            seen.add(rid)
            if rid not in ref_frames:
                errors.append(f"unknown reference_id {rid}")
                continue
            if rel["supported"] != "yes":
                continue
            own = {fid(i) for i in ref_frames[rid]}
            if rel["role"] == "none" or rel["relation_marker"] == "none" or not rel["promised_attributes"]:
                errors.append(f"{rid}: a supported relation needs role, relation_marker and promised_attributes")
            if not rel["target_evidence_ids"] or set(rel["target_evidence_ids"]) - target_ids:
                errors.append(f"{rid}: target_evidence_ids must be non-empty target frame ids")
            if not rel["reference_evidence_ids"] or set(rel["reference_evidence_ids"]) - own:
                errors.append(f"{rid}: reference_evidence_ids must be non-empty frame ids of that reference")
            if rel["role"] in schemas.SUBJECT_ROLES and rel["target_subject_or_event_id"] not in keys:
                errors.append(f"{rid}: target_subject_or_event_id must be a target_subjects key")
            if rel["role"] in schemas.STRUCTURE_ROLES and next(r for r in refs if r["id"] == rid)["type"] != "video":
                errors.append(f"{rid}: an image cannot supply {rel['role']}")
        if seen != set(ref_frames):
            errors.append(f"every reference needs at least one relations entry; missing {sorted(set(ref_frames) - seen)}")
        return errors

    return ctx.models.call("reference_relations", content, schemas.REFERENCE_RELATIONS, validate)


def _covers(ref) -> set:
    return {(rel["target_subject_or_event_id"], rel["role"]) if rel["role"] in schemas.SUBJECT_ROLES
            else ("__structure__", rel["role"]) for rel in ref["relations"]}


def beam_search(valid: list[dict], target_subjects: list[dict], cfg) -> list[dict]:
    """固定顺序束搜索。先满足硬条件，再比分数；布局偏好只作同分排序。"""
    search, limits, w, levels = cfg.search, cfg.limits, cfg.scoring.bundle, cfg.scoring.levels
    main = {s["key"] for s in target_subjects if s["importance"] == "main"} or {s["key"] for s in target_subjects}
    universe = set(main)
    if any(k == "__structure__" for r in valid for k, _ in _covers(r)):
        universe.add("__structure__")       # 池中存在成立的结构关系时，结构也是可覆盖的属性
    valid = sorted(valid, key=lambda r: r["id"])
    preferred = sorted(x.replace("_with_audio", "") for x in cfg.selection.prefer_layout)

    def evaluate(refs):
        covered = {k for r in refs for k, _ in _covers(r)} & universe
        pairs = [c for r in refs for c in _covers(r)]
        redundant = sum(1 for r in refs if any(pairs.count(c) > 1 for c in _covers(r)))
        clarity = np.mean([max(levels[x["clarity"]] for x in r["relations"]) for r in refs])
        evidence = np.mean([min(1.0, max(len(x["target_evidence_ids"]) + len(x["reference_evidence_ids"])
                                         for x in r["relations"]) / 4) for r in refs])
        score = (w.coverage * len(covered) / max(1, len(universe)) + w.clarity * clarity + w.evidence * evidence
                 - w.redundancy_penalty * redundant)
        layout = sorted(r["type"] for r in refs)
        return {"score": round(float(score), 4), "coverage": sorted(covered), "universe": sorted(universe),
                "clarity": round(float(clarity), 3), "evidence": round(float(evidence), 3),
                "redundant_references": redundant, "layout_preferred": layout == preferred}

    def allowed(refs, new):
        if not _covers(new) - {c for r in refs for c in _covers(r)}:
            return False                                   # 无用参考：不提供新的 (对象, 用途)
        both = refs + [new]
        videos = [r for r in both if r["type"] == "video"]
        if len([r for r in both if r["type"] == "image"]) > limits.max_images or len(videos) > limits.max_videos:
            return False
        if sum(r["plan"]["duration_seconds"] for r in videos) > limits.max_video_total_seconds:
            return False
        if len(both) > limits.max_files:
            return False
        s, e = new["source_interval"]                      # 参考来源互不交叠
        return all(max(s, r["source_interval"][0]) >= min(e, r["source_interval"][1]) for r in refs)

    def key(state):
        return (-state["metrics"]["score"], 0 if state["metrics"]["layout_preferred"] else 1,
                [r["id"] for r in state["refs"]])

    beam, seen = [{"refs": [], "last": -1}], []
    for _ in range(limits.max_files):
        grown = []
        for state in beam:
            for index in range(state["last"] + 1, len(valid)):
                if allowed(state["refs"], valid[index]):
                    refs = state["refs"] + [valid[index]]
                    grown.append({"refs": refs, "last": index, "metrics": evaluate(refs)})
        if not grown:
            break
        beam = sorted(grown, key=key)[:search.beam_width]
        seen += beam
    return sorted(seen, key=key)[:search.reference_bundles_per_target]


def find_reference_bundles(ctx, m, feats, timeline, target) -> tuple[list[dict], dict]:
    pool = reference_pool(ctx, m, feats, timeline, target)
    report = {"target_id": target["id"], "pool": pool, "duplicates": {}, "rejected_references": {}}
    survivors = []
    for ref in pool["images"] + pool["videos"]:
        measured = measure_near_duplicate(ctx, feats, ref, target)
        if measured["result"] == "ambiguous":
            measured = _duplicate_review(ctx, m, feats, ref, target, measured)
        measured.pop("_order", None)
        report["duplicates"][ref["id"]] = measured
        if measured["result"] != "clear":         # 重复或未解决的实质嫌疑都不进入配对
            report["rejected_references"][ref["id"]] = f"near_duplicate:{measured['result']}"
            continue
        survivors.append({**ref, "duplicate_check": measured})
    if not survivors:
        report["outcome"] = "no_reference_candidates_outside_target"
        return [], report
    judged = judge_relations(ctx, m, timeline, target, survivors)
    report["relations"] = judged
    valid = []
    for ref in survivors:
        supported = [r for r in judged["relations"] if r["reference_id"] == ref["id"] and r["supported"] == "yes"]
        if supported:
            valid.append({**ref, "relations": supported})
        else:                                     # 已知不成立或核心关系未知：拒绝，不靠 needs_review 冒充
            report["rejected_references"][ref["id"]] = "no_supported_relation"
    bundles = beam_search(valid, judged["target_subjects"], ctx.cfg)
    w = ctx.cfg.scoring.pair
    pairs = []
    for bundle in bundles:
        refs = sorted(bundle["refs"], key=lambda r: (r["type"], r["source_interval"]))
        fingerprint = content_hash([target["id"], [r["id"] for r in refs]])[:8]
        pairs.append({
            "id": f"p_{target['id'][2:]}_{fingerprint}", "target": target, "references": refs,
            "target_subjects": judged["target_subjects"], "bundle": bundle["metrics"],
            "score": round(w.target * target["target_score"] + w.bundle * bundle["metrics"]["score"], 4)})
    report["outcome"] = f"{len(pairs)} bundle(s) from {len(valid)} valid reference(s)"
    return pairs, report
