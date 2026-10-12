"""固定工作流的调度：节点、循环、重试、修复和状态转移全部由程序预定义。"""
from __future__ import annotations

import os
import traceback
import uuid

from . import media
from .build import build_pair
from .config import load_config
from .context import RunContext
from .models import probe_capabilities
from .publish import archive_attempt, finalize_and_commit, select_up_to_limit
from .references import find_reference_bundles
from .source import assign_source_group, build_source_timeline, check_feature_count
from .targets import enumerate_target_candidates
from .util import AttrDict, Blocked, RejectPair


def run(config_path, overrides=()) -> dict:
    try:
        cfg = load_config(config_path, overrides)
    except Blocked as error:                     # 配置不可用时仍返回明确的阻塞原因
        return {"status": "blocked", "reason": str(error), "samples": []}
    ctx = RunContext(cfg)
    try:
        return _run(ctx)
    except Blocked as error:
        ctx.technical_errors.append({"where": "run", "reason": str(error)})
        return ctx.finish("blocked", str(error), [])
    except Exception as error:  # noqa: BLE001 未预期的实现错误不能变成“视频不适合”或 completed
        ctx.technical_errors.append({"where": "run", "reason": "implementation_error",
                                     "error": f"{type(error).__name__}: {error}",
                                     "traceback": traceback.format_exc()[-4000:]})
        return ctx.finish("blocked", f"implementation_error:{type(error).__name__}: {error}", [])


def _run(ctx: RunContext) -> dict:
    cfg = ctx.cfg
    # 01：全量解码；不能仅凭容器头参数或文件扩展名通过。
    ctx.log("decode_source", path=cfg.paths["input_video"])
    m = media.probe_and_decode(cfg.paths["input_video"])
    ctx.bind(m)
    if ctx.reused_index is not None:
        ctx.log("reused_committed_run", run=str(ctx.run_dir))
        return {"run_dir": str(ctx.run_dir), "reused": True, **ctx.reused_index}
    ctx.save("source_manifest.json", {k: v for k, v in m.data.items() if k not in ("pts", "durations")} | {
        "pts_ticks": m.pts, "frame_duration_ticks": m.durations})
    if m.data["decode_errors"]:
        raise Blocked(f"source_decode_errors:{m.data['decode_errors'][:3]}")
    if not m.has_audio:
        return ctx.finish("skipped", "no_source_audio", [])

    ctx.capabilities = probe_capabilities(ctx.models)
    ctx.capabilities["asr_python_present"] = os.path.exists(cfg.paths["asr_python"])
    # 装载适配器有不确定规格时，禁用受影响布局，保留已知合法布局。
    ctx.capabilities["layouts"] = {
        "images": True, "silent_video": True,
        "video_with_sync_audio": bool(cfg.loading.video_sync_audio_verified and ctx.capabilities["can_observe_audio"]),
        "standalone_audio": False,
        "note": "sync audio needs a verified loading contract and a model that can hear; standalone audio "
                "references are not implemented in stapipe-v1"}
    ctx.save("capabilities.json", ctx.capabilities)
    ctx.log("capabilities", visual=ctx.capabilities["can_observe_visual_media"],
            audio=ctx.capabilities["can_observe_audio"])
    if not ctx.capabilities["can_observe_visual_media"]:
        raise Blocked("visual_understanding_unavailable")

    feats = media.frame_features(m.path)
    check_feature_count(m, feats)
    group = assign_source_group(ctx, m, feats)
    ctx.save("source_group.json", group)

    # 02 → 03 → 04：先产生目标短名单，再为每个目标在其外部检索参考。
    timeline = build_source_timeline(ctx, m, feats)
    targets = enumerate_target_candidates(ctx, m, feats, timeline)
    shortlist = targets[:cfg.search.shortlist_targets]
    results = ctx.models.map(shortlist, lambda t: find_reference_bundles(ctx, m, feats, timeline, t))
    pairs = sorted((p for found, _ in results for p in found),
                   key=lambda p: (-p["score"], p["target"]["start_frame"], p["id"]))
    budget = cfg.search.export_pair_budget
    ctx.save("pairs.json", {
        "targets_ranked": len(targets), "targets_searched": len(shortlist),
        "targets_truncated_by_shortlist": [t["id"] for t in targets[cfg.search.shortlist_targets:]],
        "pair_order": [{"id": p["id"], "score": p["score"], "target": p["target"]["id"],
                        "references": [r["id"] for r in p["references"]], "bundle": p["bundle"]} for p in pairs],
        "pairs_truncated_by_export_budget": [p["id"] for p in pairs[budget:]],
        "reference_search": [report for _, report in results]})
    ctx.log("pairs_ranked", pairs=len(pairs), export_budget=budget)

    # 即使已找到第一条也完成预算内评估，以便先选 completed 再按质量排序。
    accepted = []
    for pair in pairs[:budget]:
        attempt_id = f"{pair['id']}_{uuid.uuid4().hex[:6]}"
        attempt = {"id": attempt_id, "pair_id": pair["id"], "stage": str(ctx.staging / attempt_id),
                   "provisional_sample_id": f"{ctx.source_id}_{pair['id'].rsplit('_', 1)[-1]}_provisional"}
        ctx.log("attempt_started", attempt=attempt_id, score=pair["score"])
        try:
            accepted.append(build_pair(ctx, m, pair, group, timeline, attempt))
            ctx.log("attempt_accepted", attempt=attempt_id, status=accepted[-1]["status"])
        except RejectPair as error:
            archive_attempt(ctx, attempt, "candidate_rejected", str(error))
        except Blocked as error:
            # 本次失败不能被最终的 skipped 覆盖；仍可完成其他独立候选。
            ctx.technical_errors.append({"where": attempt_id, "reason": str(error)})
            archive_attempt(ctx, attempt, "node_error", str(error))

    selected = select_up_to_limit(accepted, cfg)
    samples = []
    for ordinal, artifact in enumerate(selected, start=1):
        try:
            samples.append(finalize_and_commit(ctx, m, artifact, ordinal))
            ctx.attempts.append({"attempt": artifact["attempt"]["id"], "pair_id": artifact["pair_id"],
                                 "outcome": "published", "reason": artifact["status"]})
        except Blocked as error:
            ctx.technical_errors.append({"where": f"publish:{artifact['pair_id']}", "reason": str(error)})
            ctx.attempts.append({"attempt": artifact["attempt"]["id"], "pair_id": artifact["pair_id"],
                                 "outcome": "node_error", "reason": str(error),
                                 "staging_kept_at": artifact["stage"]})
    for artifact in accepted:
        if artifact not in selected:
            archive_attempt(ctx, artifact["attempt"], "accepted_not_selected",
                            f"{artifact['status']}; outranked, overlapping or beyond max_samples")

    extra = {"source": {"path": str(m.path), "sha256": m.sha256, "split_group": group["id"]},
             "search": {"targets_ranked": len(targets), "pairs_found": len(pairs),
                        "pairs_exported": min(len(pairs), budget)}}
    if samples:
        status = "completed" if all(s["status"] == "completed" for s in samples) else "needs_review"
        reason = None if len(samples) == cfg.max_samples else "fewer_valid_samples_than_max_samples"
        return ctx.finish(status, reason, samples, extra)
    if ctx.technical_errors:
        return ctx.finish("blocked", "technical_failures", [], extra)
    # 不声称原片绝对不存在有效配对：只说明在本配置的搜索预算内未找到。
    return ctx.finish("skipped", "no_valid_pair_within_configured_budget", [], extra)


def summarize(result: dict) -> str:
    result = AttrDict(result)
    lines = [f"status: {result.status}" + (f" ({result.reason})" if result.get("reason") else ""),
             f"run: {result.get('run_dir')}"]
    for s in result.get("samples", []):
        lines.append(f"  sample {s['sample_id']}: {s['status']} -> {s['path']}")
    for a in result.get("attempts", []):
        lines.append(f"  attempt {a['pair_id']}: {a['outcome']} ({a['reason']})")
    return "\n".join(lines)
