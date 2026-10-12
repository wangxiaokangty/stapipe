"""通用工具：哈希、JSON 读写、异常类型与小型数据容器。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from fractions import Fraction
from pathlib import Path


class Blocked(Exception):
    """技术/能力阻塞；保存节点、原因与已有产物。"""


class RejectPair(Exception):
    """已知内容问题或无法建立核心配对关系；尝试下一固定候选。"""


class AttrDict(dict):
    """可用属性访问的字典；嵌套字典在读取时自动包装。"""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as error:
            raise AttributeError(name) from error
        return AttrDict(value) if isinstance(value, dict) and not isinstance(value, AttrDict) else value

    def __setattr__(self, name, value):
        self[name] = value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, bytes):
        return {"sha256": sha256_bytes(value), "bytes": len(value)}
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_jsonable)


def content_hash(obj) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def write_json(path, obj) -> None:
    """先写临时文件再替换，避免中断时留下半个 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        json.dump(obj, out, ensure_ascii=False, indent=1, default=_jsonable)
        out.write("\n")
    os.chmod(tmp, 0o644)                    # mkstemp 默认 0600；产物需要可被其他读取方打开
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def slug(text: str, limit: int = 40) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return cleaned[:limit].strip("_") or "source"


def mmss(seconds) -> str:
    """MM:SS.mmm；按毫秒四舍五入后再拆分，避免出现 00:60.000。"""
    millis = int(round(float(seconds) * 1000))
    return f"{millis // 60000:02d}:{millis % 60000 // 1000:02d}.{millis % 1000:03d}"


def overlap(a, b) -> bool:
    """源区间均左闭右开。"""
    return max(a[0], b[0]) < min(a[1], b[1])


def merge_spans(spans, lo: float, hi: float) -> list[list[float]]:
    """裁剪到 [lo, hi] 并合并相交或相接的区间。"""
    clipped = sorted([max(lo, float(s)), min(hi, float(e))] for s, e in spans)
    merged: list[list[float]] = []
    for start, end in clipped:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    # 写成毫秒；上界向下取整，保证不超过目标时长
    return [[round(s, 3), min(round(e, 3), math.floor(hi * 1000) / 1000)] for s, e in merged]
