"""Ref2VA 静态流水线伪代码，与 README.md 配套。

使用 Python 语法表达控制流；ctx / ops / store / schemas 等是待实现接口。
它不是可执行的数据构造器，不会调用模型、生成样本或修改已有媒体。
所有循环、模型节点、重试、修复和状态转移均由程序预定义。
"""

from __future__ import annotations

from fractions import Fraction


SECTIONS = (
    "subject_definitions", "summary", "retention_analysis",
    "detailed_description", "overall_soundscape", "non_diegetic_music",
)
VISUAL_RELATIONS = {
    "fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference",
}
AUDIO_RELATIONS = {"fully_copy", "partially_copy", "reference", "weak_reference"}

# 方法实现必须返回媒体证据，不能通过让 LLM 自填 passed 代替检查。
REQUIRED_CHECKS = {
    "schema_and_paths": "program",
    "decode_all_media": "program",
    "media_profile_and_reference_limits": "program",
    "wav_samples_and_av_alignment": "program",
    "source_intervals_and_all_export_boundaries": "program",
    "duplicate_content_and_condition_isolation": "hybrid",
    "prompt_structure_labels_and_cut_times": "program",
    "visual_facts_and_reference_roles": "model",
    "audio_facts_dialogue_and_speakers": "model",
    "origin_provenance_and_prompt_copy": "program",
    "uncertainty_flags_and_ranges": "program",
}
HARD_CHECKS = set(REQUIRED_CHECKS) - {
    "visual_facts_and_reference_roles", "audio_facts_dialogue_and_speakers",
}
# 两类语义项可存在待复核细节，但“核心参考关系成立”仍单独作硬门槛。

# 所有模型调用均来自这些固定角色；不存在 plan_next_step / execute_tool 节点。
MODEL_NODES = {
    "source_visual_facts", "source_audio_facts", "target_completeness",
    "reference_relations", "duplicate_review", "exported_visual_facts", "exported_audio_facts",
    "prompt_writer", "semantic_auditor", "prompt_repair",
}


class Blocked(Exception):
    """技术/能力阻塞；保存节点、原因与已有产物。"""


class RejectPair(Exception):
    """已知内容问题或无法建立核心配对关系；尝试下一固定候选。"""


def run(config_path, overrides):
    # bootstrap 即使输入或配置不可用也创建最小运行日志。
    ctx = bootstrap_run_log(config_path)
    try:
        cfg = resolve_and_freeze_config(config_path, overrides)
        # workspace 相对配置文件定位；业务参数全部相对 workspace 解析。
        # 读取本地 PROMPT、媒体规范、H3 指南与样例结构，校验锁定版本。
        ctx.freeze(cfg, snapshot_specs_code_dependencies_and_registry(cfg))
        source = require_source(cfg.input_video)          # 缺失 -> Blocked
        ctx.bind_source_hash(sha256(source))
        ops = locked_media_and_model_adapters(ctx)

        # 01：全量解码；不能仅凭容器头参数或文件扩展名通过。
        media = ops.probe_and_decode_entire_source(source)
        ctx.save("source_manifest.json", media)
        if not media.has_audio_stream:
            return ctx.finish_without_sample("skipped", reason="no_source_audio")
        if not ops.capabilities.can_observe_visual_media:
            raise Blocked("visual_understanding_unavailable")
        # 声音能力不足写入能力清单；可形成 needs_review，不能假装已试听。
        # 装载适配器有不确定规格时，禁用受影响布局，保留已知合法布局。
        layouts = ops.resolve_supported_reference_layouts(cfg)
        if not layouts:
            raise Blocked("no_verified_reference_loading_contract")

        group = ops.assign_source_group(
            source_hash=media.sha256,
            work_id=cfg.optional_work_id,
            fingerprints=ops.source_fingerprints(media),
            registry_snapshot=ctx.registry,
        )
        timeline = build_source_timeline(ctx, ops, media)
        targets = enumerate_target_candidates(ctx, ops, timeline)
        pairs = []

        # 03 -> 04：先产生目标短名单，再为每个目标在其外部检索参考。
        for target in targets[:cfg.search.shortlist_targets]:
            pairs.extend(find_reference_bundles(ctx, ops, timeline, target, layouts))
        pairs = stable_sort(pairs, key=lambda p: (-p.score, p.target.start, p.id))
        ctx.save_candidates_and_budget_truncation(targets, pairs)

        # 即使已找到第一条也完成预算内评估，以便先选 completed 再按质量排序。
        accepted = []
        for pair in pairs[:cfg.search.export_pair_budget]:
            attempt = ctx.new_unique_attempt(pair.id)
            try:
                artifact = build_pair(ctx, ops, media, pair, group, attempt)
                accepted.append(artifact)
            except RejectPair as error:
                attempt.finish("candidate_rejected", reason=str(error))
            except Blocked as error:
                # 本次失败不能被最终的 skipped 覆盖；仍可完成其他独立候选。
                attempt.finish("node_error", reason=str(error))

        # 多样本按确定性规则选择互不重叠的目标；共享同一个 split_group。
        ranked = stable_sort(
            accepted,
            key=lambda a: (0 if a.status == "completed" else 1, -a.score, a.pair_id),
        )
        selected = select_up_to_limit(
            ranked, cfg.max_samples,
            disallow_target_overlap=cfg.selection.disallow_overlap_between_selected_targets,
            allow_review=cfg.selection.publish_needs_review,
        )
        for ordinal, artifact in enumerate(selected, start=1):
            # 稳定样本 ID 来自源标识、配对指纹与序号；目标参数取实际文件。
            finalize_sample_identity(artifact, group, ordinal)
            # 所有模型调用已结束，以最终账本指纹封存；ID 改动后重建清单。
            ops.verify_final_package(artifact)
            artifact.package.seal_artifact_manifest(ctx.execution_fingerprint)
            commit_without_overwrite(ctx, ops, artifact)

        if not selected:
            if ctx.has_technical_errors:
                return ctx.finish_without_sample("blocked", reason="technical_failures", checks=ctx.checks)
            return ctx.finish_without_sample(
                "skipped", reason="no_valid_pair_within_configured_budget", checks=ctx.checks,
            )
        # 返回逐样本状态和计数；不足 max_samples 时说明原因，不伪造补足样本。
        return ctx.finish_run_index(selected, include_all_attempt_failures=True)
    except Blocked as error:
        return ctx.finish_with_checkpoints("blocked", reason=str(error))
    except Exception as error:
        # 未预期的实现错误不能变成“视频不适合”或“completed”。
        return ctx.finish_with_checkpoints("blocked", reason="implementation_error", error=error)


def build_source_timeline(ctx, ops, media):
    cfg = ctx.cfg
    evidence = ops.fixed_source_evidence(
        media,
        overview_step=cfg.search.overview_step_seconds,
        window=cfg.search.understanding_window_seconds,
        overlap=cfg.search.understanding_overlap_seconds,
        include_detected_cut_neighbors=True,
    )
    # 固定顺序窗口；可并发计算，但结果按 window_id 合并，不按返回顺序合并。
    visual = [model_node(ctx, "source_visual_facts", w, "VisualFacts") for w in evidence.windows]
    asr = ops.run_locked_asr_and_vad(evidence.audio)
    if ops.capabilities.can_observe_audio:
        audio = [
            model_node(ctx, "source_audio_facts", {"media": w, "asr": asr.for_window(w)}, "AudioFacts")
            for w in evidence.audio_windows
        ]
    else:
        audio = unknown_audio_facts(evidence.audio_windows, reason="audio_capability_missing")
    # 冲突按固定规则保留为 unknown；不以多数投票消除矛盾。
    timeline = ops.merge_by_evidence_and_actual_pts(media, visual, audio, asr)
    ctx.save("timeline.json", timeline)
    return timeline


def enumerate_target_candidates(ctx, ops, timeline):
    # 枚举由程序生成的候选；网格端点只是召回手段，不证明语义完整。
    boundaries = ops.union_boundaries(
        timeline, kinds=["source_edges", "confirmed_cuts", "vad_pauses", "action_ends"],
        grid_seconds=ctx.cfg.search.candidate_step_seconds,
    )
    feasible = []
    for interval in enumerate_intervals(
        boundaries, min_seconds=5 - ctx.cfg.search.max_tail_padding_seconds, max_seconds=15,
    ):
        plan = plan_legal_video(interval, minimum=5, maximum=15, cfg=ctx.cfg)
        if plan is None or ops.obvious_visual_defect(interval, timeline):
            ctx.reject_candidate(interval, "duration_alignment_or_visual_quality")
            continue
        feasible.append((interval, plan))
    # 固定粗排 + 时间覆盖去冗余；完整性模型调用也有显式预算。
    shortlist = ops.coarse_rank_with_temporal_coverage(
        feasible, timeline, limit=ctx.cfg.search.target_evaluation_budget,
    )
    ctx.save_target_search_truncation(feasible, shortlist)
    candidates = []
    for interval, plan in shortlist:
        evidence = ops.dense_evidence(interval, step=ctx.cfg.search.dense_step_seconds)
        verdict = model_node(ctx, "target_completeness", evidence, "CandidateVerdict")
        if verdict.completeness != "complete":
            ctx.reject_candidate(interval, "incomplete_or_unknown_semantic_boundary", verdict)
            continue
        candidates.append(candidate_from(interval, plan, evidence, verdict))
    # 评分分量、等级到数值的映射、权重与同分规则均锁定于 cfg。
    return stable_sort(candidates, key=lambda c: (-c.target_score, c.start, c.end, c.id))


def find_reference_bundles(ctx, ops, timeline, target, layouts):
    pool = ops.reference_pool_outside(
        timeline, target.interval,
        image_source="real_frames_and_recorded_crops",
        video_source="natural_boundary_intervals",
        audio_source="natural_boundary_intervals",
    )
    # 视频候选提前做 17k+5 与时长规划；独立音频不套用帧数公式。
    pool = ops.keep_media_feasible(pool, plan_legal_video, ctx.cfg)
    pool = ops.deterministic_retrieve(pool, target, top_per_type=ctx.cfg.search.reference_pool_per_type)
    valid = []
    for ref in pool:
        if overlap(ref.source_interval, target.interval):
            continue
        duplication = ops.measure_near_duplicate_content(ref, target)
        if duplication.result == "ambiguous":
            duplication = model_node(
                ctx, "duplicate_review",
                ops.actual_pair_evidence_with_matches(ref, target, duplication), "DuplicateReview",
            )
        if duplication.result != "clear":         # 重复或未解决的实质嫌疑都不进入配对。
            ctx.reject_reference(ref, duplication)
            continue
        relation = model_node(
            ctx, "reference_relations",
            ops.actual_target_and_reference_evidence(target, ref), "ReferenceRelation",
        )
        if not core_relation_supported(relation):
            ctx.reject_reference(ref, relation)
            continue
        if relation.audio_marker in {"fully_copy", "partially_copy"}:
            if not ops.verify_signal_reuse_evidence(ref, target, relation):
                # 不把“像同一首歌”升级成信号复制；只有独立成立的风格用途可保留。
                relation = retain_supported_style_role_or_reject(relation)
        if relation is not None:
            valid.append(reference_with_relation(ref, relation))
    bundles = fixed_beam_search(
        valid,
        supported_layouts=layouts,
        constraints=ops.reference_constraints,
        score=locked_bundle_score,
        tie_break="layout_preference_then_reference_ids",
        width=ctx.cfg.search.reference_bundles_per_target,
        discard_redundant_references=True,
    )
    # 每份参考必须实际提供有证据的属性；不强制两图一视频。
    return [pair_from(target, bundle) for bundle in bundles if bundle.has_visual_input]


def overlap(a, b):
    # 源区间均左闭右开；图片用源帧实际显示区间，而不是只有一个无时长点。
    return max(a.start, b.start) < min(a.end, b.end)


def plan_legal_video(interval, minimum, maximum, cfg):
    duration = interval.end - interval.start       # 有理数 PTS；不用平均帧率换算。
    plans = []
    for n in range(5, 24 * maximum + 1, 17):
        length = Fraction(n, 24)
        padding = length - duration
        if not minimum <= length <= maximum or padding < 0:
            continue
        if padding > cfg.search.max_tail_padding_seconds:
            continue
        if padding > duration * cfg.search.max_tail_padding_ratio:
            continue
        plans.append(video_plan(interval, n, padding, forbid_tail_truncation=True))
    return min(plans, key=lambda p: (p.padding, p.num_frames)) if plans else None


def build_pair(ctx, ops, media, pair, group, attempt):
    # 全部必需检查先注册为 not_checked，缺省清单不能意外成为通过。
    checks = check_registry(REQUIRED_CHECKS, initial="not_checked", hard_checks=HARD_CHECKS)
    plan = ops.compile_export_plan(pair, media, ctx.cfg)
    attempt.save("selection.json", pair.selection_with_evidence)
    attempt.save("export_plan.json", plan)
    ops.check_plan_constraints(plan)                 # 有内容/规格问题 -> RejectPair

    # 唯一暂存目录，与最终发布目标在同一个文件系统；不删除已有输出。
    package = attempt.new_package_stage()
    ops.export_exact_source_media(plan, package)
    # 目标 WAV 由源区间直接提取；目标 MP4 的 AAC 由该 WAV 编码。
    # 共享源时间原点；精确裁切先于重采样；只在末尾重复帧/补静音。
    ops.preserve_or_convert_origin(media, package)
    # 原 MP4 完整复制并校验哈希；非 MP4 原件存 work，真实转换并记录命令。
    measured = ops.decode_measure_and_map_all_exports(package, plan)
    checks.update(ops.check_exported_media(measured, plan, media))
    if checks.has_failed_hard_media_check:
        # 导出器违反其已验证计划属于技术错误，不谎称原片不适合。
        raise Blocked("export_does_not_match_verified_plan")

    # 06：只用导出目标和参考重新建立目标时间线，避免沿用源绝对时间。
    evidence = ops.fixed_export_evidence(package, measured)
    visual = model_node(ctx, "exported_visual_facts", evidence.visual, "ExportedVisualFacts")
    asr = ops.run_locked_asr_and_vad(package.target_audio)
    if ops.capabilities.can_observe_audio:
        audio = model_node(
            ctx, "exported_audio_facts", {"media": evidence.audio, "asr": asr}, "ExportedAudioFacts",
        )
    else:
        audio = unknown_audio_facts(evidence.audio, reason="audio_capability_missing")
    facts = ops.validate_and_join_export_facts(visual, audio, measured, pair)
    if facts.core_relation_invalid_or_unknown or facts.invalid_semantic_boundaries:
        raise RejectPair("exported_pair_not_semantically_supported")
    labels = allocate_stable_labels(pair.references, facts)
    attempt.save("facts.json", facts)
    attempt.save("labels.json", labels)
    draft = model_node(ctx, "prompt_writer", {"facts": facts, "labels": labels}, "PromptDraft")

    # 07、08、09：最多原稿 + 一次修复；格式重试与语义修复分别计数。
    for revision in range(ctx.cfg.retry.prompt_repairs + 1):
        prompt = render_six_sections(draft, facts, labels)
        audio_review = derive_audio_uncertainty(facts, draft)
        record = make_local_record(
            id=attempt.provisional_sample_id,
            split_group=group.id,
            path_base="directory containing dataset.jsonl",
            prompt=prompt,
            references=labels.reference_records,
            target=measured.target_record,
            needs_audio_review=audio_review.required,
            uncertain_audio_spans_seconds=audio_review.merged_target_spans,
        )
        package.write_jsonl("pure/dataset.jsonl", [record], utf8=True)
        package.write_text("work/prompt.md", prompt, exact=True)
        # 重新检查全部文本相关项目，不沿用上一版草稿的 passed。
        checks.reset_prompt_dependent_checks()
        checks.update(ops.verify_program_contracts(package, record, facts, labels, plan))
        review = model_node(
            ctx, "semantic_auditor",
            {"actual_exported_media": evidence, "facts": facts, "prompt": prompt,
             "reference_relations": pair.relations},
            "SemanticReview",
        )
        # 返回事实问题/证据，不让模型直接决定最终 completed。
        # 新发现的未确认对白若仍被写成确定台词，必须记为需要修复的文本错误。
        checks.update(ops.convert_review_to_checks(review, evidence, facts))
        attempt.save_revision(revision, draft, prompt, checks, review)
        if review.invalid_core_relation_or_boundary:
            raise RejectPair("semantic_relation_or_boundary_failed")
        if not checks.has_known_prompt_errors:
            break
        if revision == ctx.cfg.retry.prompt_repairs or not only_allowed_text_repairs(checks):
            raise RejectPair("unresolved_known_annotation_error")
        draft = model_node(
            ctx, "prompt_repair",
            {"immutable_facts": facts, "immutable_labels": labels,
             "draft": draft, "allowed_issues": checks.allowed_text_issues},
            "PromptDraft",
        )

    # 非对白的声音关系不确定性也需要设置 needs_audio_review；视觉问题只进报告。
    # 若审查新增声音不确定性，合并区间并刷新标记，不能保留旧 false。
    # 正文不能在最终审查后变化；仍需删去的未确认台词应在上面的修复循环解决。
    ops.apply_reviewed_uncertainty_flags_only(package, facts, review, checks)
    checks.update(ops.verify_final_record_flags_and_text(package, facts))
    status = classify_sample(checks)
    attempt.attach_all_evidence_to(package)           # selection/provenance/命令/预览/转录/调用记录
    package.write_verification(status, checks, not_done=ctx.actual_unperformed_checks)
    # 最终 ID 与执行账本尚未冻结；只保存实测与哈希草稿，发布前再封存。
    package.save_unsealed_manifest(measured)
    return accepted_artifact(package, pair, status)


def render_six_sections(draft, facts, labels):
    # 模型只写描述字段；标题、标签、对白标签及时间由程序插入。
    validate_claim_links(draft, facts)
    sections = render_content_objects(draft, facts, labels)
    shots = sections["detailed_description"].shots
    assert shots[0].start_frame == 0
    for i, shot in enumerate(shots, start=1):
        shot.prefix = "[Shot 1]" if i == 1 else (
            f"[Shot {i}] At {format_mmss_milliseconds(Fraction(shot.start_frame, 24))},"
        )
    # first shot 之前另有一两句风格描述；对白/歌词/可见文字保留原语言。
    # 说话人、scenetrans、cutoff 由事实表的发声事件及跨切镜关系渲染。
    return "\n\n".join(f"{name}:\n{sections[name].render()}" for name in SECTIONS)


def classify_sample(checks):
    if checks.has_implementation_or_io_failure:
        raise Blocked("artifact_or_validation_implementation_error")
    if checks.any_known_failure:
        raise RejectPair("required_check_failed")
    if checks.required_hard_check_not_run:
        raise Blocked("required_hard_validation_missing")
    if checks.core_reference_relation_unknown:
        raise RejectPair("core_reference_relation_unproven")
    if checks.has_review_items or checks.required_semantic_check_not_confirmed:
        return "needs_review"
    return "completed"


def model_node(ctx, node, payload, schema_name):
    assert node in MODEL_NODES
    spec = ctx.models_lock[node]
    request = build_fixed_request(spec.template, payload, schema_name, tools=[])
    key = content_hash({
        "node": node, "implementation": spec.implementation_hash,
        "model": spec.version, "template": spec.template_hash,
        "schema": ctx.schemas.hash(schema_name), "request": request,
        "media_bytes_and_order": hash_actual_media_inputs(payload),
        "seed": spec.seed, "decoding": spec.decoding,
    })
    if ctx.cfg.mode == "replay":
        # 指定历史账本缺任何节点/修复调用就阻塞，不访问线上模型。
        return ctx.archive.require_validated_result(ctx.replay_run_id, key)
    if ctx.cfg.mode == "record" and ctx.cache.has_validated(key):
        result = ctx.cache.read(key)
        ctx.ledger.attach_cached_call(key, result)
        return result

    # schema 修复也是固定模板；不允许返回工具名、路径或可执行命令。
    for schema_attempt in range(ctx.cfg.retry.schema_repairs + 1):
        response = invoke_with_fixed_transport_retries(ctx, spec, request)
        ctx.ledger.save_raw_call(key, schema_attempt, request, response)
        parsed = ctx.schemas.validate(response, schema_name)
        if parsed.valid:
            ctx.cache.append_immutable(key, parsed.value, call_id=response.call_id)
            return parsed.value
        request = fixed_schema_repair_request(spec, request, response, parsed.errors)
    raise Blocked(f"model_schema_invalid:{node}")


def invoke_with_fixed_transport_retries(ctx, spec, request):
    for attempt in range(ctx.cfg.retry.transport_retries + 1):
        try:
            return ctx.model_client.invoke(spec, request, tools_enabled=False)
        except RetryableTransportError as error:
            ctx.ledger.save_transport_error(spec, attempt, error)
            if attempt == ctx.cfg.retry.transport_retries:
                raise Blocked("model_service_retries_exhausted") from error
            wait_seconds(ctx.cfg.retry.transport_backoff_seconds[attempt])
        except NonRetryableModelError as error:
            raise Blocked("model_service_or_capability_error") from error


def commit_without_overwrite(ctx, ops, artifact):
    destination = ctx.resolve_output_paths(artifact.sample_id)
    # 完整性、内容哈希、源码/模型调用指纹均匹配才可复用。
    if destination.exists:
        if ops.same_complete_artifact(destination, artifact):
            ctx.record_reused_output(destination)
            return
        destination = ctx.allocate_new_output_paths(artifact.execution_fingerprint)
    # 自动目录整体无覆盖 rename；显式分离目录采用 work -> pure -> 提交标记。
    # 封存前验证复制后的相对路径；失败保留暂存，恢复时校验清单。
    ops.commit_staged_package_no_replace(artifact, destination)
    ops.verify_committed_hashes_and_paths(destination)
    ctx.record_committed_output(destination, artifact.status)


# 抽象接口实现约束：
# 1. ops 只封装固定解码/分析/导出/验证函数，不解释 LLM 返回的命令。
# 2. selectors 使用固定阈值、等级定义、束宽、权重与同分规则，记录拒绝原因。
# 3. store/ctx 缓存不可变，记录实际输入输出及模型调用；不得按文件名复用。
# 4. 检查 schema、条件隔离、媒体时长/计数、源区间与所有媒体首尾均为硬约束。
# 5. 内容不合格抛 RejectPair；缺能力、服务/解码/写入/程序异常抛 Blocked。
# 6. apply_reviewed_uncertainty_flags_only 只更新布尔值、区间和待复核报告，不改正文。
#    删去未确认对白等正文变更只能进入固定修复循环并重新接受审查。听不清的对白整句不写入提示词，
#    不使用 [unclear] 占位符，只保存区间并设置 needs_audio_review。
# 7. commit 后只有已提交索引对消费者可见；中断恢复不能覆盖另一有效运行。

if __name__ == "__main__":
    raise SystemExit("这是设计伪代码。请按 README.md 实现各接口后再运行流水线。")
