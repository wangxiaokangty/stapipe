"""运行上下文：指纹、目录、日志、检查注册表、抽帧缓存与声音证据缓存。"""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import WORKFLOW_VERSION, media
from .models import PACKAGE_DIR, ModelRuntime
from .util import Blocked, content_hash, read_json, sha256_file, slug, write_json

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
SEMANTIC_CHECKS = ("visual_facts_and_reference_roles", "audio_facts_dialogue_and_speakers")
# 本流水线从不执行的事项；如实写入每份报告。
NOT_DONE = ["H3 model inference", "training", "human viewing or listening review"]


class Checks:
    """全部必需检查先注册为 not_checked；缺省清单不能意外成为通过。"""

    def __init__(self):
        self.items = {name: {"check": name, "required": True, "hard": name not in SEMANTIC_CHECKS,
                             "method": method, "status": "not_checked", "reason": "not executed",
                             "version": WORKFLOW_VERSION, "details": []}
                      for name, method in REQUIRED_CHECKS.items()}
        self.review_items: list[dict] = []

    def record(self, check: str, name: str, ok: bool | None, detail="") -> None:
        item = self.items[check]
        status = "not_checked" if ok is None else ("passed" if ok else "failed")
        item["details"].append({"name": name, "status": status, "detail": str(detail)[:2000]})
        states = {d["status"] for d in item["details"]}
        item["status"] = "failed" if "failed" in states else ("not_checked" if "not_checked" in states else "passed")
        item["reason"] = "; ".join(f"{d['name']}: {d['detail']}"[:300] for d in item["details"]
                                   if d["status"] != "passed")[:1500]

    def reset(self, names) -> None:
        for name in names:
            self.items[name].update(status="not_checked", reason="not executed", details=[])

    def status(self, check: str) -> str:
        return self.items[check]["status"]

    def failed(self) -> list[str]:
        return [name for name, item in self.items.items() if item["status"] == "failed"]

    def review(self, kind: str, text: str, span=None) -> None:
        entry = {"kind": kind, "text": text, **({"target_span_seconds": span} if span else {})}
        if entry not in self.review_items:
            self.review_items.append(entry)

    def to_list(self) -> list[dict]:
        return list(self.items.values())


class RunContext:
    def __init__(self, cfg):
        self.cfg = cfg
        self.events: list[dict] = []
        self.technical_errors: list[dict] = []
        self.attempts: list[dict] = []
        self.run_dir: Path | None = None
        self.models: ModelRuntime | None = None
        self.capabilities: dict = {}
        self.fingerprint: dict = {}
        self.reused_index: dict | None = None
        self._frame_lock = threading.Lock()
        self.started = time.time()

    # ------------------------------------------------------------ 冻结与目录

    def bind(self, source: media.SourceMedia) -> None:
        """冻结规范快照、源码、依赖、模型锁与配置，得到运行指纹并确定运行目录。"""
        cfg = self.cfg
        self.source = source
        code = {str(p.relative_to(PACKAGE_DIR)): sha256_file(p)
                for p in sorted(PACKAGE_DIR.rglob("*")) if p.suffix in (".py", ".md") and "__pycache__" not in p.parts}
        specs = {}
        for name, path in cfg.paths["specs"].items():
            if not Path(path).is_file():
                raise Blocked(f"spec_snapshot_missing:{name}")
            specs[name] = sha256_file(path)
        registry = Path(cfg.paths["source_registry"])
        self.registry = read_json(registry) if registry.is_file() else {"version": 0, "groups": []}
        dependencies = {}
        for package in ("numpy", "pillow", "soundfile", "imageio-ffmpeg", "jsonschema", "pyyaml"):
            dependencies[package] = importlib.metadata.version(package)
        frozen_cfg = {k: v for k, v in cfg.items() if k not in ("paths", "mode", "replay_run", "model_concurrency")}
        self.fingerprint = {
            "workflow_version": WORKFLOW_VERSION, "source_sha256": source.sha256, "config": frozen_cfg,
            "specs": specs, "code": code, "dependencies": dependencies, "python": sys.version.split()[0],
            "platform": platform.platform(), "ffmpeg": media.ffmpeg_version(),
            "models_lock": content_hash(read_json(cfg.paths["models_lock"])) if Path(
                cfg.paths["models_lock"]).is_file() else None,
            "source_registry_version": self.registry.get("version"), "source_registry": content_hash(self.registry),
        }
        self.run_fingerprint = content_hash(self.fingerprint)
        self.source_id = f"{slug(source.path.stem, 24)}_{source.sha256[:8]}"
        run_id = self.run_fingerprint[:8]
        if cfg.mode != "record":
            run_id += f"_{cfg.mode}{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
        self.run_id = run_id
        self.root = Path(cfg.paths["output_root"]) / f"{self.source_id}_{run_id}"
        self.run_dir = self.root / "run"
        index = self.run_dir / "index.json"
        if index.is_file() and read_json(index).get("committed"):
            self.reused_index = read_json(index)      # 相同输入、配置、代码与模型锁的已提交结果
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.staging = self.root / ".staging"
        self.staging.mkdir(exist_ok=True)
        self.models = ModelRuntime(cfg, self.run_dir)
        self.save("config.snapshot.json", dict(cfg))
        self.save("fingerprint.json", {"run_fingerprint": self.run_fingerprint, **self.fingerprint})
        for event in self.events:
            self._append_log(event)

    def save(self, name: str, obj) -> None:
        write_json(self.run_dir / name, obj)

    def log(self, event: str, **fields) -> None:
        entry = {"t": round(time.time() - self.started, 2), "event": event, **fields}
        self.events.append(entry)
        print(f"[stapipe {entry['t']:8.1f}s] {event} " + " ".join(f"{k}={v}" for k, v in fields.items())[:300],
              file=sys.stderr, flush=True)
        if self.run_dir and self.run_dir.is_dir():
            self._append_log(entry)

    def _append_log(self, entry: dict) -> None:
        with open(self.run_dir / "log.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    # ------------------------------------------------------------ 证据

    def frames(self, indices, width: int, path=None, tag: str = "src") -> dict[int, bytes]:
        """真实解码帧（JPEG），按 (来源, 帧号, 宽度) 缓存到运行目录。"""
        path = path or self.source.path
        folder = self.run_dir / "frames" / f"{tag}_w{width}"
        folder.mkdir(parents=True, exist_ok=True)
        with self._frame_lock:                      # 并发节点共享抽帧缓存
            missing = [i for i in set(indices) if not (folder / f"{i:06d}.jpg").is_file()]
            for index, data in media.extract_frames(path, missing, width).items():
                (folder / f"{index:06d}.jpg").write_bytes(data)
        return {i: (folder / f"{i:06d}.jpg").read_bytes() for i in indices}

    def audio_evidence(self, path) -> dict:
        """锁定的 ASR/声音类别工作进程；按媒体内容哈希与参数缓存。没有环境时返回能力缺失。"""
        audio_cfg = self.cfg.audio
        python = Path(self.cfg.paths["asr_python"])
        if not python.exists():
            return {"available": False, "reason": f"asr_python_missing:{python}"}
        key = content_hash({"media": sha256_file(path), "model": audio_cfg.asr_model, "tags": audio_cfg.tags,
                            "worker": sha256_file(PACKAGE_DIR / "audio_worker.py")})
        cached = Path(self.cfg.paths["cache_dir"]) / "audio" / f"{key}.json"
        if cached.is_file() and self.cfg.mode != "refresh":
            return {"available": True, "cache_key": key, **read_json(cached)}
        if self.cfg.mode == "replay":
            raise Blocked("replay_missing_audio_evidence")
        cached.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(python), str(PACKAGE_DIR / "audio_worker.py"), "--media", str(path), "--ffmpeg",
               media.ffmpeg_exe(), "--out", str(cached), "--asr-model", audio_cfg.asr_model]
        if audio_cfg.tags:
            cmd.append("--tags")
        env = None
        if audio_cfg.offline:
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0 or not cached.is_file():
            return {"available": False, "reason": f"asr_worker_failed:{proc.stderr[-400:]}"}
        return {"available": True, "cache_key": key, **read_json(cached)}

    # ------------------------------------------------------------ 结束

    def finish(self, status: str, reason: str | None, samples: list[dict], extra: dict | None = None) -> dict:
        index = {
            "committed": True, "status": status, "reason": reason, "workflow_version": WORKFLOW_VERSION,
            "run_fingerprint": getattr(self, "run_fingerprint", None),
            "execution_fingerprint": self.models.execution_fingerprint() if self.models else None,
            "mode": self.cfg.mode, "max_samples": self.cfg.max_samples,
            "samples": samples, "sample_count": len(samples),
            "shortfall": max(0, self.cfg.max_samples - len(samples)),
            "attempts": self.attempts, "technical_errors": self.technical_errors,
            "model_calls": self._call_stats(), "not_done": NOT_DONE,
            "elapsed_seconds": round(time.time() - self.started, 1), **(extra or {}),
        }
        if self.run_dir is None:                      # 输入或配置不可用：仍留下最小运行日志
            self.run_dir = Path(self.cfg.paths["output_root"]) / f"_unbound_{int(self.started)}" / "run"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            for event in self.events:
                self._append_log(event)
        if not samples:
            checks = Checks()
            write_json(self.run_dir / "verification.json", {
                "status": status, "reason": reason, "checks": checks.to_list(), "not_done": NOT_DONE,
                "note": "no sample was published; required checks stay not_checked"})
        if self.models:
            write_json(self.run_dir / "model_calls_index.json", self.models.calls)
        write_json(self.run_dir / "index.json", index)
        self.log("finished", status=status, reason=reason, samples=len(samples))
        return {"run_dir": str(self.run_dir), **index}

    def _call_stats(self) -> dict:
        stats: dict = {}
        for call in (self.models.calls if self.models else []):
            node = stats.setdefault(call["node"], {"invoked": 0, "cache_hit": 0, "replayed": 0})
            node[call["how"]] += 1
        return stats
