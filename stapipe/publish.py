"""节点 10：在固定候选预算内择优，封存清单并无覆盖地原子发布。"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .build import classify_sample, verification_report
from .context import Checks
from .util import Blocked, read_json, sha256_file, write_json
from .verify import verify_package


def select_up_to_limit(accepted: list[dict], cfg) -> list[dict]:
    """先选 completed，再按分数与稳定编号；默认多个目标互不重叠。"""
    ranked = sorted(accepted, key=lambda a: (0 if a["status"] == "completed" or not cfg.selection.prefer_completed
                                             else 1, -a["score"], a["pair_id"]))
    chosen = []
    for artifact in ranked:
        if len(chosen) >= cfg.max_samples:
            break
        if artifact["status"] != "completed" and not cfg.selection.publish_needs_review:
            continue
        a, b = artifact["target_interval"]
        if cfg.selection.disallow_overlap_between_selected_targets and any(
                max(a, c["target_interval"][0]) < min(b, c["target_interval"][1]) for c in chosen):
            continue
        chosen.append(artifact)
    return chosen


def _manifest(stage: Path) -> dict:
    return {str(p.relative_to(stage)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in sorted(stage.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"}


def finalize_and_commit(ctx, m, artifact: dict, ordinal: int) -> dict:
    """最终 ID → 重新校验 → 以执行指纹封存清单 → 整体 rename；失败时保留暂存目录。"""
    stage = Path(artifact["stage"])
    pure, work = stage / "pure", stage / "work"
    sample_id = f"{ctx.source_id}_{artifact['pair_id'].rsplit('_', 1)[-1]}_{ordinal:03d}"
    record = json.loads((pure / "dataset.jsonl").read_text(encoding="utf-8"))
    record["id"] = sample_id
    (pure / "dataset.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    # ID 改动后在最终包上重跑全部程序检查；语义项沿用构建时的模型核验结果。
    previous = artifact["checks"]
    checks = Checks()
    verify_package(pure, work, m, ctx.cfg, checks)
    for name, item in previous.items.items():
        if not item["hard"]:
            checks.items[name] = item
    checks.review_items = previous.review_items
    status = classify_sample(checks)
    if status != artifact["status"]:
        raise Blocked(f"final_package_status_changed:{artifact['status']}->{status}")
    write_json(work / "verification.json", verification_report(status, checks, sample_id))

    calls = work / "model_calls"
    calls.mkdir(exist_ok=True)
    used = [c for c in ctx.models.calls if c["scope"] in ("run", artifact["attempt"]["id"])]
    for call in used:
        shutil.copyfile(ctx.models.ledger_dir / f"{call['key']}.json", calls / f"{call['key']}.json")
    write_json(calls / "index.json", used)
    for name in ("timeline.json", "candidates.json"):
        shutil.copyfile(ctx.run_dir / name, work / name)
    write_json(work / "artifact_manifest.json", {
        "sample_id": sample_id, "status": status, "run_fingerprint": ctx.run_fingerprint,
        "execution_fingerprint": ctx.models.execution_fingerprint(), "files": _manifest(stage)})

    destination = ctx.root / sample_id
    if destination.exists():
        existing = destination / "work" / "artifact_manifest.json"
        if existing.is_file() and read_json(existing)["files"] == _manifest(stage):
            ctx.log("reused_committed_output", sample=sample_id)
            shutil.rmtree(stage)
            return {"sample_id": sample_id, "status": status, "path": str(destination), "reused": True}
        destination = ctx.root / f"{sample_id}_{ctx.models.execution_fingerprint()[:8]}"
        if destination.exists():
            raise Blocked(f"output_exists_with_different_content:{destination}")
    os.rename(stage, destination)             # 同一文件系统内的整体提交
    manifest = read_json(destination / "work" / "artifact_manifest.json")
    if _manifest(destination) != manifest["files"]:
        raise Blocked(f"committed_hash_mismatch:{destination}")
    for rel in [r["path"] for r in record["references"]] + [record["target"]["video"], record["target"]["audio"]]:
        if not (destination / "pure" / rel).is_file():
            raise Blocked(f"committed_relative_path_missing:{rel}")
    ctx.log("committed", sample=sample_id, status=status, path=str(destination))
    return {"sample_id": sample_id, "status": status, "path": str(destination), "reused": False,
            "pair_id": artifact["pair_id"], "score": artifact["score"],
            "target_source_frames": artifact["target_interval"], "review_items": checks.review_items}


def archive_attempt(ctx, attempt: dict, outcome: str, reason: str) -> None:
    """未发布的尝试：保留 work 证据，删除已导出的媒体。"""
    stage = Path(attempt["stage"])
    if (stage / "work").is_dir():
        destination = ctx.run_dir / "attempts" / attempt["id"]
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(stage / "work"), str(destination))
    shutil.rmtree(stage, ignore_errors=True)
    ctx.attempts.append({"attempt": attempt["id"], "pair_id": attempt["pair_id"], "outcome": outcome,
                         "reason": reason})
    ctx.log("attempt_finished", attempt=attempt["id"], outcome=outcome, reason=reason)
