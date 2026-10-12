"""配置解析与冻结。workspace 相对配置文件定位；业务路径全部相对 workspace 解析。"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

from . import WORKFLOW_VERSION
from .util import AttrDict, Blocked

# 候选预算、抽样密度、阈值和评分权重都是待校准的工程选择，不是 H3 规范。
DEFAULTS = {
    "workflow_version": WORKFLOW_VERSION,
    "workspace": "../ref2va-workspace",
    "input_video": "examples/input.mp4",
    "output_root": "outputs",
    "cache_dir": "outputs/.stapipe_cache",
    "max_samples": 1,
    "media_profile": "local_h3_v1",
    "mode": "record",                      # record / replay / refresh
    "replay_run": None,                    # replay 模式下指定的历史运行目录
    "seed": 0,
    "models_lock": "models.lock.json",     # 相对配置文件
    "source_registry": "source_registry.json",
    "optional_work_id": None,
    "model_concurrency": 4,
    "search": {
        "min_target_seconds": 5, "max_target_seconds": 15,
        "min_reference_seconds": 2, "max_reference_seconds": 15,
        "overview_step_seconds": 1.0,
        "understanding_window_seconds": 12.0,
        "understanding_overlap_seconds": 2.0,
        "candidate_step_seconds": 0.25,
        "target_evaluation_budget": 96,
        "target_coverage_max_iou": 0.8,
        "shortlist_targets": 24,
        "reference_pool_per_type": 12,
        "reference_bundles_per_target": 3,
        "beam_width": 8,
        "dense_step_seconds": 0.125,
        "export_pair_budget": 24,
        "max_tail_padding_seconds": 0.125,
        "max_tail_padding_ratio": 0.02,
    },
    "retry": {"transport_retries": 2, "transport_backoff_seconds": [1, 3], "schema_repairs": 1,
              "prompt_repairs": 1, "model_timeout_seconds": 600},
    "selection": {"prefer_layout": ["image", "image", "video_with_audio"], "prefer_completed": True,
                  "disallow_overlap_between_selected_targets": True, "publish_needs_review": True},
    "evidence": {"max_images_per_call": 60, "overview_width": 448, "dense_width": 384, "reference_width": 512,
                 "export_step_seconds": 0.25, "known_subject_thumbnails": 6},
    "audio": {"asr_python": ".venv-asr/bin/python", "asr_model": "mlx-community/whisper-large-v3-turbo",
              "tags": True, "offline": True, "confident_word_probability": 0.6, "pause_seconds": 0.35},
    # 版本化装载适配器：带声参考视频的同步音轨按一个物理文件计数，音轨时长计入音频预算。
    "loading": {"adapter_version": "local-jsonl-v1", "video_sync_audio_verified": True,
                "sync_audio_counts_toward_audio_budget": True, "standalone_audio": True},
    "limits": {"max_images": 9, "max_videos": 3, "max_video_total_seconds": 15, "max_audios": 3,
               "max_audio_total_seconds": 15, "max_files": 12},
    "thresholds": {"black_luma": 16.0, "black_fraction": 0.2, "blur_sharpness": 4.0, "blur_fraction": 0.5,
                   "endcard_fraction": 0.3, "silence_db": -60.0,
                   "duplicate_tiny_diff": 3.0, "ambiguous_tiny_diff": 9.0, "duplicate_coverage": 0.5,
                   "boundary_frame_diff": 6.0, "mp4_wav_corr": 0.95, "wav_source_corr": 0.98,
                   "cut_tolerance_frames": 1},
    "scoring": {
        "levels": {"high": 1.0, "medium": 0.6, "low": 0.2, "unknown": 0.0},
        "target": {"semantic": 0.40, "visual": 0.30, "audio": 0.20, "information": 0.10},
        "bundle": {"coverage": 0.50, "clarity": 0.30, "evidence": 0.20, "redundancy_penalty": 0.05},
        "pair": {"target": 0.60, "bundle": 0.40},
        "boundary_kind": {"source_edge": 1.0, "confirmed_cut": 1.0, "action_end": 0.9, "vad_pause": 0.7,
                          "grid": 0.2},
    },
}

SPEC_FILES = [
    "PROMPT.md", "reference/media-profile.md", "reference/minimax-h3/input-specs.md",
    "reference/minimax-h3/skills/h3-prompt-writing/references/ref-en.txt",
    "reference/minimax-h3/skills/h3-prompt-writing/references/base-en.txt",
    "examples/pure/dataset.jsonl",
]


def _merge(base: dict, extra: dict, path: str = "") -> dict:
    for key, value in extra.items():
        if key not in base:
            raise Blocked(f"unknown_config_key:{path}{key}")
        if isinstance(base[key], dict) and isinstance(value, dict):
            _merge(base[key], value, f"{path}{key}.")
        else:
            base[key] = value
    return base


def _apply_override(cfg: dict, item: str) -> None:
    key, _, raw = item.partition("=")
    node, parts = cfg, key.split(".")
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            raise Blocked(f"unknown_config_key:{key}")
        node = node[part]
    if parts[-1] not in node:
        raise Blocked(f"unknown_config_key:{key}")
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(config_path, overrides=()) -> AttrDict:
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise Blocked(f"config_missing:{config_path}")
    cfg = _merge(copy.deepcopy(DEFAULTS), yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    for item in overrides:
        _apply_override(cfg, item)
    if cfg["mode"] not in ("record", "replay", "refresh"):
        raise Blocked(f"invalid_mode:{cfg['mode']}")
    if cfg["media_profile"] != "local_h3_v1":          # 不默默切换媒体规格
        raise Blocked(f"unsupported_media_profile:{cfg['media_profile']}")
    if cfg["workflow_version"] != WORKFLOW_VERSION:
        raise Blocked(f"workflow_version_mismatch:{cfg['workflow_version']}")
    base = config_path.parent
    workspace = (base / cfg["workspace"]).resolve()
    if not (workspace / "PROMPT.md").is_file():
        raise Blocked(f"workspace_missing_PROMPT.md:{workspace}")

    def in_workspace(value) -> str:
        return str((workspace / value).resolve())

    cfg["paths"] = {
        "config": str(config_path), "workspace": str(workspace),
        "input_video": in_workspace(cfg["input_video"]),
        "output_root": in_workspace(cfg["output_root"]),
        "cache_dir": in_workspace(cfg["cache_dir"]),
        "models_lock": str((base / cfg["models_lock"]).resolve()),
        "source_registry": str((base / cfg["source_registry"]).resolve()),
        # 不解析符号链接：虚拟环境的 python 是指向基础解释器的链接。
        "asr_python": os.path.abspath(workspace / cfg["audio"]["asr_python"]),
        "replay_run": in_workspace(cfg["replay_run"]) if cfg["replay_run"] else None,
        "specs": {name: in_workspace(name) for name in SPEC_FILES},
    }
    return AttrDict(cfg)
