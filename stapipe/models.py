"""固定模型节点：锁定配置、请求构造、缓存、账本、重试与三种运行模式。

所有模型调用都来自 MODEL_NODES；不存在 plan_next_step / execute_tool 节点。
模型只返回符合 JSON 模式的内容对象，不获得任何工具。
"""
from __future__ import annotations

import base64
import importlib
import io
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jsonschema

from .util import Blocked, content_hash, read_json, sha256_bytes, sha256_file, write_json

MODEL_NODES = (
    "capability_probe",
    "source_visual_facts", "source_audio_facts", "target_completeness",
    "reference_relations", "duplicate_review", "exported_visual_facts", "exported_audio_facts",
    "prompt_writer", "semantic_auditor", "prompt_repair",
)
AUDIO_NODES = ("source_audio_facts", "exported_audio_facts")
PACKAGE_DIR = Path(__file__).resolve().parent
IMPLEMENTATION = "1"

SCHEMA_REPAIR_TEXT = (
    "Your previous output did not pass validation. Return a corrected JSON object for the same task and the same "
    "media. Use only the identifiers that were provided. Validation errors:\n{errors}\n\nPrevious output:\n{previous}"
)


class RetryableTransportError(Exception):
    pass


class NonRetryableModelError(Exception):
    pass


def text(value: str) -> dict:
    return {"type": "text", "text": value}


def image(data: bytes, media_id: str, media_type: str = "image/jpeg") -> dict:
    return {"type": "image", "data": data, "media_type": media_type, "id": media_id}


def audio(data: bytes, media_id: str) -> dict:
    return {"type": "audio", "data": data, "format": "wav", "id": media_id}


# ---------------------------------------------------------------- 后端

class ClaudeCliBackend:
    """通过 `claude -p` 调用；禁用全部工具、MCP 与技能，在空目录中运行。只支持图像输入。"""

    modalities = ("text", "image")

    def __init__(self, spec: dict, cache_dir: Path):
        self.command = spec.get("command", "claude")
        self.cwd = cache_dir / "cli_cwd"
        self.cwd.mkdir(parents=True, exist_ok=True)

    def invoke(self, node_spec: dict, request: dict, timeout: float) -> dict:
        blocks = []
        for block in request["content"]:
            if block["type"] == "text":
                blocks.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                blocks.append({"type": "image", "source": {
                    "type": "base64", "media_type": block["media_type"],
                    "data": base64.b64encode(block["data"]).decode()}})
            else:
                raise NonRetryableModelError(f"backend_does_not_accept:{block['type']}")
        message = {"type": "user", "message": {"role": "user", "content": blocks}}
        cmd = [self.command, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
               "--tools", "", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
               "--model", node_spec["model"], "--system-prompt", request["system"],
               "--json-schema", json.dumps(request["schema"])]
        try:
            proc = subprocess.run(cmd, input=json.dumps(message) + "\n", capture_output=True, text=True,
                                  timeout=timeout, cwd=self.cwd)
        except subprocess.TimeoutExpired as error:
            raise RetryableTransportError("timeout") from error
        except FileNotFoundError as error:
            raise NonRetryableModelError(f"command_not_found:{self.command}") from error
        result = None
        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "result":
                result = event
        if result is None:
            raise RetryableTransportError(f"no_result exit={proc.returncode} {proc.stderr[-300:]}")
        if result.get("is_error"):
            status = result.get("api_error_status")
            detail = f"{result.get('subtype')} status={status} {str(result.get('result'))[:300]}"
            if status in (400, 401, 403, 404, 413):
                raise NonRetryableModelError(detail)
            raise RetryableTransportError(detail)
        output = result.get("structured_output")
        raw = result.get("result") or ""
        if output is None:
            try:
                output = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                output = None
        return {"output": output, "raw": raw if output is None else json.dumps(output, ensure_ascii=False),
                "call_id": result.get("session_id"),
                "meta": {"cost_usd": result.get("total_cost_usd"), "duration_ms": result.get("duration_ms"),
                         "stop_reason": result.get("stop_reason")}}


class OpenAICompatibleBackend:
    """Chat Completions 兼容接口，可传图像与音频。本仓库未用真实服务验证过此后端。"""

    def __init__(self, spec: dict, cache_dir: Path):
        self.base_url = os.environ.get(spec.get("base_url_env", ""), spec.get("base_url", "")).rstrip("/")
        self.api_key = os.environ.get(spec.get("api_key_env", ""), "")
        self.modalities = tuple(spec.get("modalities", ("text", "image", "audio")))
        self.extra = spec.get("extra_body", {})

    def invoke(self, node_spec: dict, request: dict, timeout: float) -> dict:
        if not self.base_url or not self.api_key:
            raise NonRetryableModelError("openai_compatible_backend_not_configured")
        parts = []
        for block in request["content"]:
            encoded = base64.b64encode(block["data"]).decode() if "data" in block else None
            if block["type"] == "text":
                parts.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                parts.append({"type": "image_url", "image_url": {"url": f"data:{block['media_type']};base64,{encoded}"}})
            else:
                parts.append({"type": "input_audio", "input_audio": {"data": encoded, "format": block["format"]}})
        body = {"model": node_spec["model"], "temperature": 0, "seed": node_spec.get("seed", 0),
                "messages": [{"role": "system", "content": request["system"]}, {"role": "user", "content": parts}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "output", "schema": request["schema"], "strict": False}}, **self.extra}
        http = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                      headers={"Authorization": f"Bearer {self.api_key}",
                                               "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(http, timeout=timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = f"http_{error.code}:{error.read()[:300]!r}"
            if error.code in (408, 409, 429) or error.code >= 500:
                raise RetryableTransportError(detail) from error
            raise NonRetryableModelError(detail) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RetryableTransportError(str(error)) from error
        raw = payload["choices"][0]["message"].get("content") or ""
        try:
            output = json.loads(raw)
        except json.JSONDecodeError:
            output = None
        return {"output": output, "raw": raw, "call_id": payload.get("id"), "meta": {"usage": payload.get("usage")}}


class PythonCallableBackend:
    """调用 `模块:函数`(node, request) -> dict；用于离线测试与自定义适配。"""

    def __init__(self, spec: dict, cache_dir: Path):
        module, _, name = spec["callable"].partition(":")
        self.fn = getattr(importlib.import_module(module), name)
        self.modalities = tuple(spec.get("modalities", ("text", "image")))

    def invoke(self, node_spec: dict, request: dict, timeout: float) -> dict:
        output = self.fn(node_spec["node"], request)
        return {"output": output, "raw": json.dumps(output, ensure_ascii=False), "call_id": None, "meta": {}}


BACKENDS = {"claude_cli": ClaudeCliBackend, "openai_compatible": OpenAICompatibleBackend,
            "python_callable": PythonCallableBackend}


# ---------------------------------------------------------------- 锁文件

def default_lock() -> dict:
    """第一版模型锁：视觉与文本节点用 claude CLI；声音节点未配置（能力缺失）。"""
    fast, strong = "claude-sonnet-5-5", "claude-opus-5-5"
    models = {"capability_probe": fast, "source_visual_facts": fast, "target_completeness": fast,
              "duplicate_review": fast, "reference_relations": strong, "exported_visual_facts": strong,
              "prompt_writer": strong, "semantic_auditor": strong, "prompt_repair": strong}
    nodes = {}
    for node in MODEL_NODES:
        template = f"stapipe/templates/{node}.md"
        nodes[node] = {"backend": None if node in AUDIO_NODES else "claude_cli", "model": models.get(node),
                       "template": template, "template_sha256": None, "implementation": IMPLEMENTATION,
                       "seed": 0, "decoding": "backend default; the claude CLI exposes no temperature or seed"}
    return {"version": 1, "backends": {"claude_cli": {"type": "claude_cli", "command": "claude"}}, "nodes": nodes}


def refresh_lock(lock_path) -> dict:
    """写入各模板的当前 SHA-256；模板或模型改动后需显式重新锁定。"""
    lock_path = Path(lock_path)
    lock = read_json(lock_path) if lock_path.is_file() else default_lock()
    for spec in lock["nodes"].values():
        spec["template_sha256"] = sha256_file(lock_path.parent / spec["template"])
    write_json(lock_path, lock)
    return lock


class ModelRuntime:
    def __init__(self, cfg, run_dir: Path):
        self.cfg = cfg
        lock_path = Path(cfg.paths["models_lock"])
        if not lock_path.is_file():
            raise Blocked(f"models_lock_missing:{lock_path} (run `python -m stapipe lock`)")
        self.lock = read_json(lock_path)
        self.lock_hash = content_hash(self.lock)
        self.templates: dict[str, str] = {}
        for node in MODEL_NODES:
            spec = self.lock["nodes"].get(node)
            if spec is None:
                raise Blocked(f"models_lock_missing_node:{node}")
            spec["node"] = node
            body = (lock_path.parent / spec["template"]).read_bytes()
            if sha256_bytes(body) != spec["template_sha256"]:
                raise Blocked(f"models_lock_template_mismatch:{node} (run `python -m stapipe lock`)")
            self.templates[node] = body.decode("utf-8")
        self.cache_dir = Path(cfg.paths["cache_dir"])
        self.call_cache = self.cache_dir / "model_calls"
        self.ledger_dir = run_dir / "model_calls"
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.call_cache.mkdir(parents=True, exist_ok=True)
        self.backends: dict[str, object] = {}
        self.lock_guard = threading.Lock()
        self.calls: list[dict] = []          # 本次运行实际使用的调用（含缓存命中）
        self.replay_dir = Path(cfg.paths["replay_run"]) / "run" / "model_calls" if cfg.mode == "replay" else None
        if cfg.mode == "replay" and not (self.replay_dir and self.replay_dir.is_dir()):
            raise Blocked("replay_run_ledger_missing")

    def spec(self, node: str) -> dict:
        return self.lock["nodes"][node]

    def backend(self, spec: dict):
        name = spec.get("backend")
        if not name:
            raise Blocked(f"model_backend_not_configured:{spec['node']}")
        with self.lock_guard:
            if name not in self.backends:
                backend_spec = self.lock["backends"][name]
                self.backends[name] = BACKENDS[backend_spec["type"]](backend_spec, self.cache_dir)
        return self.backends[name]

    def supports(self, node: str, modality: str) -> bool:
        spec = self.spec(node)
        if not spec.get("backend"):
            return False
        backend_spec = self.lock["backends"][spec["backend"]]
        declared = backend_spec.get("modalities") or BACKENDS[backend_spec["type"]].__dict__.get("modalities")
        return modality in (declared or ("text", "image", "audio"))

    def execution_fingerprint(self) -> str:
        return content_hash(sorted((c["key"], c["value_sha256"]) for c in self.calls))

    # ------------------------------------------------------------ 调用

    def call(self, node: str, content: list[dict], schema: dict, validator=None, scope: str = "run",
             spec_override: dict | None = None):
        assert node in MODEL_NODES, node
        spec = {**self.spec(node), **(spec_override or {})}
        request = {"system": self.templates[node], "content": content, "schema": schema}
        key = content_hash({
            "node": node, "implementation": spec["implementation"], "backend": spec["backend"],
            "model": spec["model"], "template": spec["template_sha256"], "schema": schema,
            "request": [b["text"] if b["type"] == "text" else {"type": b["type"], "id": b["id"],
                                                                "sha256": sha256_bytes(b["data"])}
                        for b in content],
            "seed": spec.get("seed"), "decoding": spec.get("decoding"),
        })
        if self.cfg.mode == "replay":
            path = self.replay_dir / f"{key}.json"
            if not path.is_file():
                raise Blocked(f"replay_missing_model_call:{node}:{key[:12]}")
            return self._use(node, key, read_json(path), scope, "replayed")
        cached = self.call_cache / f"{key}.json"
        if self.cfg.mode == "record" and cached.is_file():
            return self._use(node, key, read_json(cached), scope, "cache_hit")

        backend = self.backend(spec)
        attempts, current = [], request
        for schema_attempt in range(self.cfg.retry.schema_repairs + 1):
            response = self._transport(backend, spec, current, attempts)
            errors = self._validate(response["output"], schema, validator)
            attempts.append({"schema_attempt": schema_attempt, "raw": response["raw"],
                             "call_id": response["call_id"], "meta": response["meta"], "errors": errors})
            if not errors:
                record = {"key": key, "node": node, "backend": spec["backend"], "model": spec["model"],
                          "template_sha256": spec["template_sha256"], "created_unix": time.time(),
                          "request": {"system_sha256": sha256_bytes(request["system"].encode()),
                                      "content": [b if b["type"] == "text" else
                                                  {"type": b["type"], "id": b["id"],
                                                   "sha256": sha256_bytes(b["data"])} for b in content],
                                      "schema": schema},
                          "attempts": attempts, "value": response["output"]}
                write_json(cached, record)             # 以键寻址；相同键的旧记录只在 refresh 时被替换
                return self._use(node, key, record, scope, "invoked")
            current = {**request, "content": content + [text(SCHEMA_REPAIR_TEXT.format(
                errors="\n".join(f"- {e}" for e in errors[:20]), previous=response["raw"][:6000]))]}
        write_json(self.ledger_dir / f"FAILED_{key}.json", {"key": key, "node": node, "attempts": attempts})
        raise Blocked(f"model_schema_invalid:{node}")

    def _use(self, node, key, record, scope, how):
        write_json(self.ledger_dir / f"{key}.json", record)
        with self.lock_guard:
            self.calls.append({"key": key, "node": node, "scope": scope, "how": how,
                               "value_sha256": content_hash(record["value"])})
        return record["value"]

    def _transport(self, backend, spec, request, attempts):
        retries = self.cfg.retry.transport_retries
        for attempt in range(retries + 1):
            try:
                return backend.invoke(spec, request, self.cfg.retry.model_timeout_seconds)
            except RetryableTransportError as error:
                attempts.append({"transport_attempt": attempt, "error": str(error)})
                if attempt == retries:
                    raise Blocked(f"model_service_retries_exhausted:{spec['node']}:{error}") from error
                time.sleep(self.cfg.retry.transport_backoff_seconds[min(attempt, len(
                    self.cfg.retry.transport_backoff_seconds) - 1)])
            except NonRetryableModelError as error:
                raise Blocked(f"model_service_or_capability_error:{spec['node']}:{error}") from error

    @staticmethod
    def _validate(output, schema, validator) -> list[str]:
        if output is None:
            return ["output is not a JSON object"]
        errors = [f"{'/'.join(map(str, e.absolute_path))}: {e.message}"[:300]
                  for e in jsonschema.Draft202012Validator(schema).iter_errors(output)]
        if not errors and validator:
            errors = list(validator(output))
        return errors

    def map(self, jobs: list, fn):
        """并发执行固定顺序的作业；结果按输入顺序返回，不按完成顺序合并。"""
        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=max(1, int(self.cfg.model_concurrency))) as pool:
            return list(pool.map(fn, jobs))


# ---------------------------------------------------------------- 能力预检

PROBE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer"],
                "properties": {"answer": {"type": "string"}}}


def _probe_image() -> bytes:
    from PIL import Image

    canvas = Image.new("RGB", (256, 128), (220, 20, 20))
    canvas.paste((20, 40, 220), (128, 0, 256, 128))
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _probe_audio() -> bytes:
    import numpy as np
    import soundfile as sf

    sr = 16000
    beep = 0.5 * np.sin(2 * np.pi * 880 * np.arange(int(0.25 * sr)) / sr)
    gap = np.zeros(int(0.35 * sr))
    wave = np.concatenate([gap, beep, gap, beep, gap, beep, gap]).astype("float32")
    buffer = io.BytesIO()
    sf.write(buffer, wave, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def probe_capabilities(runtime: ModelRuntime) -> dict:
    """用真实媒体调用确认各节点所用模型的视觉、声音能力；不凭模型名称推定。"""
    report = {"visual": {}, "audio": {}}
    for node in MODEL_NODES:
        spec = runtime.spec(node)
        if node == "capability_probe" or not spec.get("backend"):
            continue
        ident = f"{spec['backend']}:{spec['model']}"
        override = {"backend": spec["backend"], "model": spec["model"]}
        if node in AUDIO_NODES:
            if ident not in report["audio"]:
                if not runtime.supports(node, "audio"):
                    report["audio"][ident] = {"ok": False, "reason": "backend_has_no_audio_input"}
                else:
                    answer = runtime.call("capability_probe", [
                        text("How many separate beeps are in this audio clip? Answer with a single digit."),
                        audio(_probe_audio(), "probe_audio")], PROBE_SCHEMA, spec_override=override)["answer"]
                    report["audio"][ident] = {"ok": "3" in answer or "three" in answer.lower(), "answer": answer}
        elif ident not in report["visual"] and node not in ("prompt_writer", "prompt_repair"):
            answer = runtime.call("capability_probe", [
                text("The image has two solid halves. Answer exactly as `<left color>,<right color>` in lowercase."),
                image(_probe_image(), "probe_image")], PROBE_SCHEMA, spec_override=override)["answer"].lower()
            report["visual"][ident] = {"ok": "red" in answer.split(",")[0] and "blue" in answer.split(",")[-1],
                                       "answer": answer}
    audio_nodes_ready = all(runtime.spec(n).get("backend") for n in AUDIO_NODES)
    return {
        "can_observe_visual_media": bool(report["visual"]) and all(v["ok"] for v in report["visual"].values()),
        "can_observe_audio": audio_nodes_ready and bool(report["audio"]) and all(
            v["ok"] for v in report["audio"].values()),
        "audio_reason": None if audio_nodes_ready else "audio_nodes_have_no_backend_in_models_lock",
        "probes": report,
    }
