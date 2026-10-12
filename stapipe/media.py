"""固定的解码、分析与导出函数（FFmpeg）。

本模块不解释任何模型输出；所有命令行都由程序拼接。时间一律来自解码得到的
PTS（presentation timestamp，显示时间戳），用整数刻度与有理数时间基表示。
"""
from __future__ import annotations

import bisect
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

from .util import Blocked, sha256_file

FPS = 24
AUDIO_SR = 32000
AUDIO_CH = 2
FEATURE_W, FEATURE_H = 160, 90


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def ffmpeg_version() -> str:
    out = subprocess.run([ffmpeg_exe(), "-version"], capture_output=True, text=True).stdout
    return out.splitlines()[0] if out else "unknown"


def _run(cmd: list[str], binary: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=not binary)


class SourceMedia:
    """全量解码后的源媒体索引。帧号从 0 开始，区间左闭右开。"""

    def __init__(self, path, data: dict):
        self.path = Path(path)
        self.data = data
        self.sha256: str = data["sha256"]
        self.width: int = data["width"]
        self.height: int = data["height"]
        self.tb = Fraction(data["time_base"][0], data["time_base"][1])
        self.pts: list[int] = data["pts"]
        self.durations: list[int] = data["durations"]
        self.n = len(self.pts)
        self.has_audio: bool = data["has_audio_stream"]
        self.audio: dict = data.get("audio") or {}

    @property
    def last_end_known(self) -> bool:
        return self.durations[-1] > 0

    def end_pts(self, index: int) -> int:
        """帧 index 的起始刻度；index == n 时返回最后一帧的显示终点。"""
        if index < self.n:
            return self.pts[index]
        if not self.last_end_known:
            raise Blocked("last_frame_end_unknown")
        return self.pts[-1] + self.durations[-1]

    def t(self, index: int) -> Fraction:
        return self.end_pts(index) * self.tb

    def frame_at(self, seconds) -> int:
        """给定源时间处正在显示的帧号。"""
        tick = Fraction(seconds) / self.tb
        return max(0, min(self.n - 1, bisect.bisect_right(self.pts, tick) - 1))

    def nearest_boundary(self, seconds) -> int:
        """离给定时间最近的帧边界（0..n）。"""
        tick = Fraction(seconds) / self.tb
        i = bisect.bisect_left(self.pts, tick)
        candidates = [c for c in (i - 1, i) if 0 <= c < self.n]
        if self.last_end_known:
            candidates.append(self.n)
        return min(candidates, key=lambda c: abs(self.end_pts(c) - tick))

    def audio_interval(self) -> tuple[float, float] | None:
        if not self.has_audio or not self.audio.get("decoded_samples"):
            return None
        start = self.audio["start_offset_seconds"]
        return start, start + self.audio["decoded_samples"] / self.audio["sample_rate"]


def probe_and_decode(path) -> SourceMedia:
    """全量解码视频与音轨；不能仅凭容器头参数或扩展名通过。"""
    path = Path(path)
    if not path.is_file():
        raise Blocked(f"input_missing:{path}")
    video = _run([ffmpeg_exe(), "-hide_banner", "-nostats", "-i", str(path), "-map", "0:v:0",
                  "-vf", "showinfo", "-an", "-f", "null", "-"])
    log = video.stderr
    frames = re.findall(r"n:\s*(\d+) pts:\s*(-?\d+) pts_time:\S+(?:\s+duration:\s*(-?\d+))?[^\n]*? s:(\d+)x(\d+)", log)
    tb = re.search(r"config in time_base: (\d+)/(\d+)", log)
    if video.returncode != 0 or not frames or not tb:
        raise Blocked(f"video_decode_failed:{log[-400:]}")
    pts = [int(f[1]) for f in frames]
    durations = [int(f[2] or 0) for f in frames]
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise Blocked("non_monotonic_video_pts")
    errors = [line.strip() for line in log.splitlines()
              if re.search(r"error|corrupt|invalid data", line, re.I) and "showinfo" not in line]
    header = [line.strip() for line in log.splitlines() if "Stream #" in line or "Duration:" in line
              or line.startswith("Input #")]
    intervals = np.diff(pts) if len(pts) > 1 else np.array([durations[0] or 1])
    time_base = (int(tb[1]), int(tb[2]))
    data = {
        "path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
        "container_header": header,
        "is_mp4_container": any("mp4" in line for line in header if line.startswith("Input #")),
        "width": int(frames[0][3]), "height": int(frames[0][4]),
        "time_base": list(time_base), "pts": pts, "durations": durations,
        "decoded_frames": len(pts),
        "constant_frame_rate": bool(intervals.max() == intervals.min()),
        "frame_interval_ticks": {"min": int(intervals.min()), "max": int(intervals.max()),
                                 "median": float(np.median(intervals))},
        "last_frame_end_source": "decoded_frame_duration" if durations[-1] > 0 else "unknown",
        "decode_errors": errors,
        "has_audio_stream": False, "audio": None,
    }
    stream = re.search(r"Stream #\S+ Audio: (\w+).*?, (\d+) Hz, ([\w.()]+)", log)
    if stream:
        audio = _run([ffmpeg_exe(), "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0",
                      "-af", "ashowinfo", "-vn", "-f", "null", "-"])
        packets = re.findall(r"pts_time:(-?[\d.]+)[^\n]*?rate:(\d+) nb_samples:(\d+)", audio.stderr)
        if audio.returncode != 0:
            raise Blocked(f"audio_decode_failed:{audio.stderr[-400:]}")
        data["has_audio_stream"] = True
        data["audio"] = {
            "codec": stream[1], "sample_rate": int(stream[2]), "layout": stream[3].rstrip(","),
            "channels": 1 if "mono" in stream[3] else 2,
            "start_offset_seconds": float(packets[0][0]) if packets else 0.0,
            "decoded_samples": sum(int(p[2]) for p in packets),
        }
    return SourceMedia(path, data)


# ---------------------------------------------------------------- 帧特征

def frame_features(path, prefilter: str = "") -> dict:
    """逐帧低分辨率特征：切镜分数、亮度、清晰度、差值哈希和极小灰度图。

    prefilter 是程序拼接的前置滤镜链（例如限定源区间并套用同一缩放补边），用于与导出结果逐帧比对。
    """
    w, h = FEATURE_W, FEATURE_H
    proc = subprocess.Popen([ffmpeg_exe(), "-hide_banner", "-v", "error", "-i", str(path), "-map", "0:v:0",
                             "-vf", f"{prefilter}scale={w}:{h}", "-fps_mode", "passthrough", "-pix_fmt", "rgb24",
                             "-f", "rawvideo", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    size = w * h * 3
    luma, sharp, tiny, dhash, cut = [], [], [], [], []
    prev = prev_hist = None
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3).astype(np.float32)
        gray = frame.mean(axis=2)
        luma.append(float(gray.mean()))
        lap = (gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:])
        sharp.append(float(lap.var()))
        small = gray.reshape(9, h // 9, 16, w // 16).mean(axis=(1, 3))
        tiny.append(small.astype(np.uint8))
        grid = np.array([[gray[int(r * h / 8):int((r + 1) * h / 8), int(c * w / 9):int((c + 1) * w / 9)].mean()
                          for c in range(9)] for r in range(8)])
        dhash.append(np.packbits((grid[:, 1:] > grid[:, :-1]).ravel()))
        hist = np.concatenate([np.histogram(frame[..., c], bins=32, range=(0, 256))[0] for c in range(3)])
        hist = hist / (hist.sum() + 1e-9)
        if prev is None:
            cut.append(0.0)
        else:
            cut.append(0.5 * float(np.abs(frame - prev).mean()) + 50.0 * float(np.abs(hist - prev_hist).sum()) * 0.5)
        prev, prev_hist = frame, hist
    err = proc.stderr.read().decode()
    if proc.wait() != 0:
        raise Blocked(f"frame_feature_decode_failed:{err[-300:]}")
    return {"luma": np.array(luma), "sharpness": np.array(sharp), "tiny": np.array(tiny),
            "dhash": np.array(dhash), "cut_score": np.array(cut)}


def detect_cuts(scores: np.ndarray, thresh: float = 8.0, min_gap: int = 4) -> list[int]:
    """索引 i 处的切镜表示新镜头从第 i 帧开始。绝对阈值 + 局部比值，只作初筛。"""
    cuts: list[int] = []
    n = len(scores)
    for i in range(1, n):
        s = scores[i]
        lo, hi = max(1, i - 6), min(n, i + 7)
        neigh = np.concatenate([scores[lo:i], scores[i + 1:hi]])
        local = float(np.median(neigh)) if len(neigh) else 0.0
        if s >= thresh and s >= 3.0 * local + 3:
            if cuts and i - cuts[-1] < min_gap:
                if s > scores[cuts[-1]]:
                    cuts[-1] = i
                continue
            cuts.append(i)
    return cuts


def tiny_diff(a: np.ndarray, b: np.ndarray) -> float:
    """两张 16×9 灰度图的平均绝对差（0–255）。"""
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


# ---------------------------------------------------------------- 抽帧与音频解码

def extract_frames(path, indices: list[int], width: int | None = None, fmt: str = "jpeg") -> dict[int, bytes]:
    """按零起始帧号提取真实解码帧，返回 {帧号: 图片字节}。"""
    wanted = sorted(set(indices))
    if not wanted:
        return {}
    out: dict[int, bytes] = {}
    for start in range(0, len(wanted), 200):          # 避免 select 表达式过长
        chunk = wanted[start:start + 200]
        vf = "select='" + "+".join(f"eq(n\\,{i})" for i in chunk) + "'"
        if width:
            vf += f",scale={width}:-2"
        codec = ["-c:v", "mjpeg", "-q:v", "3"] if fmt == "jpeg" else ["-c:v", "png"]
        p = _run([ffmpeg_exe(), "-hide_banner", "-v", "error", "-i", str(path), "-map", "0:v:0", "-vf", vf,
                  "-fps_mode", "passthrough", "-f", "image2pipe", *codec, "-"], binary=True)
        if p.returncode != 0:
            raise Blocked(f"frame_extract_failed:{p.stderr.decode()[-300:]}")
        sig = b"\xff\xd8\xff" if fmt == "jpeg" else b"\x89PNG\r\n\x1a\n"
        starts = [m.start() for m in re.finditer(re.escape(sig), p.stdout)]
        if len(starts) != len(chunk):
            raise Blocked(f"frame_extract_count_mismatch:{len(starts)}!={len(chunk)}")
        for k, s in enumerate(starts):
            out[chunk[k]] = p.stdout[s:starts[k + 1] if k + 1 < len(starts) else len(p.stdout)]
    return out


def decode_audio(path, sr: int, channels: int, start: float | None = None, end: float | None = None) -> np.ndarray:
    """解码为 float32 [samples, channels]；可先在源时间上限定范围。"""
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostats", "-v", "error", "-i", str(path), "-map", "0:a:0"]
    if start is not None or end is not None:
        parts = ([f"start={start:.9f}"] if start is not None else []) + ([f"end={end:.9f}"] if end is not None else [])
        cmd += ["-af", "atrim=" + ":".join(parts) + ",asetpts=PTS-STARTPTS"]
    cmd += ["-ar", str(sr), "-ac", str(channels), "-f", "f32le", "-c:a", "pcm_f32le", "-"]
    p = _run(cmd, binary=True)
    if p.returncode != 0:
        raise Blocked(f"audio_decode_failed:{p.stderr.decode()[-300:]}")
    return np.frombuffer(p.stdout, dtype=np.float32).reshape(-1, channels)


def rms_db(audio: np.ndarray, sr: int, hop_seconds: float = 0.05) -> np.ndarray:
    mono = audio.mean(axis=1)
    hop = max(1, int(sr * hop_seconds))
    n = len(mono) // hop
    if n == 0:
        return np.array([-90.0])
    frames = mono[:n * hop].reshape(n, hop)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)


# ---------------------------------------------------------------- 导出

class Exporter:
    """记录每条实际执行的 FFmpeg 命令；失败属于技术阻塞。"""

    def __init__(self, source: SourceMedia):
        self.m = source
        self.log: list[dict] = []

    def _ff(self, args: list[str], step: str) -> None:
        cmd = [ffmpeg_exe(), "-hide_banner", "-v", "error", "-y", *args]
        self.log.append({"step": step, "cmd": cmd})
        p = _run(cmd)
        if p.returncode != 0:
            raise Blocked(f"export_failed:{step}:{p.stderr[-400:]}")

    def wav(self, a: int, b: int, n_samples: int, out: Path) -> None:
        """从源区间 [a, b) 直接提取并重采样；轨道沿用同一源时间原点，只在末尾补静音。"""
        t0, t1 = float(self.m.t(a)), float(self.m.t(b))
        chain = (f"atrim=start={t0:.9f}:end={t1:.9f},asetpts=PTS-{t0:.9f}/TB,"
                 f"aresample={AUDIO_SR}:first_pts=0:filter_size=64:phase_shift=10:cutoff=0.97,"
                 f"aformat=sample_fmts=s16:channel_layouts=stereo,apad=whole_len={n_samples},"
                 f"atrim=end_sample={n_samples}")
        self._ff(["-i", str(self.m.path), "-map", "0:a:0", "-af", chain, "-ac", str(AUDIO_CH),
                  "-c:a", "pcm_s16le", str(out)], f"wav {out.name}")

    def mp4(self, a: int, b: int, n: int, geo: dict, out: Path, wav: Path | None) -> None:
        """精确裁切先于帧率变换、缩放与补边；末尾只重复最后一个合法源帧。"""
        chain = (f"trim=start_pts={self.m.end_pts(a)}:end_pts={self.m.end_pts(b)},setpts=PTS-STARTPTS,"
                 f"fps={FPS}:round=near,scale={geo['scaled_w']}:{geo['scaled_h']}:flags=lanczos,setsar=1,"
                 f"pad={geo['canvas_w']}:{geo['canvas_h']}:{geo['pad_x']}:{geo['pad_y']}:black,"
                 f"tpad=stop_mode=clone:stop={n},trim=end_frame={n},format=yuv420p")
        args = ["-i", str(self.m.path)]
        if wav is not None:
            args += ["-i", str(wav)]
        args += ["-filter_complex", f"[0:v:0]{chain}[v]", "-map", "[v]"]
        if wav is not None:
            args += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-ar", str(AUDIO_SR), "-ac", str(AUDIO_CH)]
        else:
            args += ["-an"]
        args += ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-r", str(FPS), "-fps_mode", "cfr",
                 "-frames:v", str(n), "-movflags", "+faststart", str(out)]
        self._ff(args, f"mp4 {out.name}")

    def image(self, frame: int, out: Path) -> None:
        self._ff(["-i", str(self.m.path), "-map", "0:v:0", "-vf", f"select='eq(n\\,{frame})'",
                  "-fps_mode", "passthrough", "-frames:v", "1", str(out)], f"image {out.name}")

    def origin(self, out: Path, work: Path) -> dict:
        """MP4 原样复制并核对哈希；其他容器保留原件，真实转封装（失败则转码）。"""
        if self.m.data["is_mp4_container"] and self.m.path.suffix.lower() in (".mp4", ".m4v"):
            shutil.copyfile(self.m.path, out)
            return {"method": "byte_copy", "source_sha256": self.m.sha256, "oringin_sha256": sha256_file(out)}
        keep = work / ("source_original" + self.m.path.suffix.lower())
        shutil.copyfile(self.m.path, keep)
        try:
            self._ff(["-i", str(self.m.path), "-map", "0:v:0", "-map", "0:a:0", "-c", "copy",
                      "-movflags", "+faststart", str(out)], "origin remux")
            method = "remux_stream_copy"
        except Blocked:
            self._ff(["-i", str(self.m.path), "-map", "0:v:0", "-map", "0:a:0", "-c:v", "libx264", "-crf", "16",
                      "-fps_mode", "passthrough", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                      str(out)], "origin transcode")
            method = "transcode_h264_aac"
        return {"method": method, "source_sha256": self.m.sha256, "oringin_sha256": sha256_file(out),
                "original_kept_at": f"work/{keep.name}"}


def expected_frame_map(m: SourceMedia, a: int, b: int, n_out: int) -> list[dict]:
    """导出帧 → 源帧/补齐帧。fps 滤镜按最近刻度取帧；超出最后刻度的帧是末帧重复。"""
    t0 = m.t(a)
    ticks = [round((m.t(i) - t0) * FPS) for i in range(a, b)]
    real_frames = (m.t(b) - t0) * FPS
    mapping, j = [], 0
    for k in range(n_out):
        while j + 1 < len(ticks) and ticks[j + 1] <= k:
            j += 1
        mapping.append({"target_frame": k, "source_frame": a + j, "source_seconds": float(m.t(a + j)),
                        "kind": "source" if k < real_frames else "tail_padding_clone"})
    return mapping


def source_clip_prefilter(m: SourceMedia, a: int, b: int, geo: dict) -> str:
    """把源区间 [a, b) 的原始帧套用与导出相同的缩放补边（不做帧率变换）。"""
    return (f"trim=start_pts={m.end_pts(a)}:end_pts={m.end_pts(b)},setpts=PTS-STARTPTS,"
            f"scale={geo['scaled_w']}:{geo['scaled_h']}:flags=lanczos,setsar=1,"
            f"pad={geo['canvas_w']}:{geo['canvas_h']}:{geo['pad_x']}:{geo['pad_y']}:black,")


def best_lag(x: np.ndarray, y: np.ndarray, max_lag: int, step: int = 4) -> tuple[int, float]:
    """在 ±max_lag 采样内找互相关最大的延迟，返回 (延迟, 该延迟下的相关系数)。"""
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    def pair(lag):
        return x[max(0, lag):n + min(0, lag)], y[max(0, -lag):n - max(0, lag)]

    lag = max(range(-max_lag, max_lag + 1, step), key=lambda k: float(np.dot(*pair(k))))
    a, b = pair(lag)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return lag, float("nan")
    return lag, float(np.corrcoef(a, b)[0, 1])
