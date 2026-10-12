"""合法帧数规划与画布计算（reference/media-profile.md 的本地配置）。"""
from __future__ import annotations

from fractions import Fraction

from .media import AUDIO_SR, FPS

AREA_BUDGET = 768 * 1344
SHORT_SIDE = 768


def legal_frames(n: int) -> bool:
    return n >= 5 and (n - 5) % 17 == 0


def plan_legal_video(duration: Fraction, minimum, maximum, search) -> dict | None:
    """为已确认的源区间选合法帧数 N = 17k + 5。

    只接受 N/24 >= duration（不截尾），且尾部补齐同时不超过绝对值与比例上限；
    没有可用方案时返回 None，由调用方尝试其他自然边界。
    """
    plans = []
    for n in range(5, int(FPS * maximum) + 1, 17):
        length = Fraction(n, FPS)
        padding = length - duration
        if not minimum <= length <= maximum or padding < 0:
            continue
        if padding > Fraction(str(search["max_tail_padding_seconds"])):
            continue
        if padding > duration * Fraction(str(search["max_tail_padding_ratio"])):
            continue
        plans.append({"num_frames": n, "duration_seconds": float(length), "padding_seconds": float(padding),
                      "padding_ratio": float(padding / duration), "_key": (padding, n)})
    if not plans:
        return None
    best = min(plans, key=lambda p: p["_key"])
    best.pop("_key")
    return best


def wav_samples(num_frames: int) -> int:
    """S = round(N × 32000 / 24)，用有理数计算；恰好 .5 时向偶数舍入（Python round）。"""
    return round(Fraction(num_frames * AUDIO_SR, FPS))


def canvas_geometry(width: int, height: int) -> dict:
    """短边 768、面积预算 768×1344、宽高为 32 的倍数；等比缩放后补边，不裁切。"""
    landscape = width >= height
    aspect = width / height if landscape else height / width
    long_max = (AREA_BUDGET // SHORT_SIDE) // 32 * 32
    long_side = max(SHORT_SIDE, min(int(round(SHORT_SIDE * aspect / 32)) * 32, long_max))
    cw, ch = (long_side, SHORT_SIDE) if landscape else (SHORT_SIDE, long_side)
    scale = min(cw / width, ch / height)
    sw, sh = int(round(width * scale / 2)) * 2, int(round(height * scale / 2)) * 2
    sw, sh = min(sw, cw), min(sh, ch)
    return {"canvas_w": cw, "canvas_h": ch, "scaled_w": sw, "scaled_h": sh,
            "pad_x": (cw - sw) // 2, "pad_y": (ch - sh) // 2}
