"""Build a narrated Chinese explainer of the implemented stapipe-v1 pipeline.

Unlike make_walkthrough_video.py (a design walkthrough made before the code
existed), this one follows nodes 00-10 as implemented and uses one real
committed run under processed/ as the running example. Numbers shown on screen
are asserted against that run's JSON, so a different run needs new narration.

Run with the repo's .venv (Pillow, numpy, imageio-ffmpeg) on macOS:
    .venv/bin/python video/make_pipeline_video.py
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from make_walkthrough_video import FF, font, run, stamp, wrap, write_wav

HERE = Path(__file__).resolve().parent
STAPIPE = HERE.parent
RUN = STAPIPE / "processed/beauty_the_eye_can_t_see_74d82a19_1b9e123c"
SAMPLE = RUN / "beauty_the_eye_can_t_see_74d82a19_1d4aa5ac_001"
BUILD = HERE / "build/pipeline"
DOCS = HERE / "pipeline"
OUT = STAPIPE / "pipeline_explainer.mp4"
W, H, FPS, SR = 1920, 1080, 24, 48000
BG, CARD, EDGE = "#0B1421", "#142437", "#29445E"
DIM_CARD, DIM_EDGE, DIM_FG = "#0E1929", "#1A2C40", "#51667B"
FG, MUTED, BLUE, AMBER, GREEN, RED = "#F1F5F9", "#A5B8CC", "#70D9ED", "#FFC578", "#91DFC0", "#FF9B9B"
VOICE, RATE = "Tingting", "205"
SRC_FPS, SRC_FRAMES = 25, 750
MOVIE_BOX = (984, 300, 1848, 786)
LAYOUT_WARNINGS = []
DIGITS = "零一二三四五六七八九"


def scene(title, kicker, key, nodes, sentences, movie=None):
    return dict(title=title, kicker=kicker, key=key, nodes=nodes, sentences=sentences, movie=movie)


SCENES = [
    scene("一个视频进来，一条训练样本出去", "Ref2VA 静态数据构造流水线 · stapipe-v1", "intro", [], [
        "这段视频讲解一条静态流水线，它把一个真实视频，加工成参考生成任务的训练样本。",
        "输入是一个带声音的视频。输出是一组参考素材、一段英文提示词，以及从原片里切出来的目标音视频。",
        "参考和提示词是给模型的条件，目标片段是要学会生成的答案，两者必须来自原片里互不重叠的部分。",
        "全程用一次真实运行做例子：右边这条 30 秒的时尚短片，最后产出了一条待复核的样本。",
    ], "raw"),
    scene("程序决定下一步，模型只回答问题", "“静态”指的是什么", "static", [], [
        "所谓静态，是指节点、分支、候选预算、重试次数和退出条件，都在运行之前写死。",
        "模型仍然负责看画面、判断关系、写描述，但每次只回答当前节点提出的一个问题。",
        "模型的回答必须符合固定的数据模式，只能给出内容、等级和证据编号。",
        "路径、标签、时间戳、导出参数和通过状态，都由程序生成。没有任何一步，是让模型决定接下来做什么。",
    ]),
    scene("11 个节点，4 个阶段", "固定工作流全景", "overview", list(range(11)), [
        "整条流水线有 11 个节点，编号从 00 到 10，可以分成 4 个阶段。",
        "第一阶段做准备：冻结配置，完整解码原片。第二阶段理解内容，选出目标和参考。",
        "第三阶段精确导出媒体，在导出结果上重新标注，再写成提示词。",
        "第四阶段做校验和审查，最多修复一次文字，最后择优发布。",
        "中间这一段会对每个候选配对循环执行，但循环次数有预算上限，不是模型发起的回溯。",
    ]),
    scene("00–01  冻结配置，完整解码", "config.py · context.py · models.py · media.py", "decode", [0, 1], [
        "节点 00 读取配置和模型锁文件，记录每个模型节点使用的模型版本和模板哈希，算出运行指纹。",
        "节点 01 把原片从头到尾解码一遍。本例是 750 帧，1280×720，每秒 25 帧，带立体声音轨。",
        "程序还会用一张测试图片，实测模型能不能看图。本例视觉可用，但没有配置能听声音的模型。",
        "分支在这里就定下了：没有音轨直接跳过，没有视觉能力就阻塞。同一个作品的片段共用一个来源分组。",
    ], "raw"),
    scene("02  建立全片事实时间线", "source.py · 固定抽样 + 视觉模型 + 声音证据", "timeline", [2], [
        "节点 02 先建立一张可以查询的内容地图。画面逐帧比较，找出切镜候选。",
        "全片每秒取一帧，切镜两侧再各取一帧，按 12 秒窗口、重叠 2 秒，分三次交给视觉模型。",
        "模型确认了 7 个切镜，把全片分成 8 个镜头，并登记了 10 个主体和 10 个事件。",
        "声音这一路，用语音识别、声音标签和能量曲线做证据。它们只是线索，不会被直接当成事实。",
    ]),
    scene("03  枚举并排序目标候选", "targets.py · 先有目标短名单，再谈参考", "targets", [3], [
        "节点 03 选目标。边界来自片头片尾、确认的切镜、语音停顿，再加上 0.25 秒的网格，一共 129 个。",
        "程序枚举 5 到 15 秒的区间，排不出合法帧数的直接淘汰，剩下 746 个可行区间。",
        "这次用的是小预算配置，粗排之后只有 10 个交给模型评估：内容是否完整，两端是否干净，画面是否清楚。",
        "完整性是硬条件。8 个通过，按固定公式打分排序，前 3 个进入短名单。",
    ]),
    scene("04  在目标之外找参考", "references.py · 召回 → 查重 → 关系判定 → 束搜索", "references", [4], [
        "节点 04 为每个目标，在它的区间之外找参考图片和参考视频。区间左闭右开，跨进目标的直接排除。",
        "先做近重复检查：如果参考只是把目标镜头又放了一遍，就等于泄漏答案，必须拒绝。",
        "然后模型对照真实画面，逐条说明这份参考提供什么。比如同一位女士的脸和发型，或者同一簇白花。",
        "程序用束搜索组合参考，要求每份参考都有用途。配对分数等于六成目标分，加四成参考分。",
        "本例找到 6 个配对，最高 0.928 分：目标是片头的标题镜头，参考是片尾 3 秒多的特写。",
    ]),
    scene("05  合法帧数与精确导出", "planning.py · media.py · build.py", "export", [5], [
        "节点 05 把配对变成媒体文件。输出固定为每秒 24 帧，总帧数必须是 17 的倍数加 5。",
        "目标原长 5.12 秒，规划成 124 帧，只在末尾补 47 毫秒，不到百分之一。",
        "参考原长 3.68 秒，规划成 90 帧，补 70 毫秒，刚好在百分之二的上限之内。",
        "补齐只能重复最后一帧、声音补静音，不能读区间以外的画面。画面等比缩放，再补边到 1344×768。",
        "目标声音直接从原片提取，转成 32000 赫兹立体声，每声道 165333 个采样点。",
    ]),
    scene("06  在导出媒体上重新标注", "build.py · 时间从目标的 0 秒算起", "facts", [6], [
        "节点 06 不再看原片，而是在刚导出的文件上重新标注，时间从目标的零秒算起。",
        "视觉模型给出镜头、主体、可见文字和动作，并在导出画面上重新核对每条参考关系，不成立就淘汰这个配对。",
        "对白来自语音识别。一句话里只要有听不清的词，整句就不写进提示词，只记录它的时间区间。",
        "因为没有模型真正听过声音，整段 5.17 秒都被标成待听辨，这个不确定性会一路带到最后。",
    ], "target"),
    scene("07  稳定编号，渲染六段式提示词", "prompt.py · 模型写内容，程序写结构", "prompt", [7], [
        "节点 07 先由程序分配编号：参考文件是视频一，主体按目标里首次出现的顺序排，说话人另有一套编号。",
        "写作模型只产出六段的内容对象，每句话都要关联事实编号，而且不许自己写标签格式和时间戳。",
        "渲染器按固定顺序拼出六段：主体定义、摘要、保留分析、详细描述、整体声景和配乐。",
        "镜头标记、切镜时间和对白标签都由程序插入，所以提示词里的时间，一定和导出的帧对得上。",
    ]),
    scene("08–09  校验、审查，最多修复一次", "verify.py · build.py · 11 项必需检查", "verify", [8, 9], [
        "节点 08 有 11 项必需检查，开跑前就全部登记。没做的只能写未检查，不会从报告里消失。",
        "其中 9 项是程序类检查：字段和路径、完整解码、帧数和采样数、音画对齐、来源不重叠、提示词结构等等。",
        "另有一个独立的审查模型，拿着真实导出媒体和提示词逐项核对，它看不到写作模型的推理过程。",
        "节点 09 只允许修一次文字：删掉没有证据的修饰，或者用已确认的事实替换错误。修完必须全部重查。",
        "如果问题出在媒体、边界或者参考关系上，就淘汰这个配对，不能靠改文字掩盖。",
    ]),
    scene("10  择优、原子发布与四种状态", "publish.py · 状态如实反映做过什么", "status", [10], [
        "节点 10 在预算内的结果里择优：先选全部通过的，再按分数排。样本先写进暂存目录，再一次性提交。",
        "最终状态只有四种：完成、待复核、跳过和阻塞。内容不合适算跳过，技术故障算阻塞，两者不混淆。",
        "本例 9 项程序检查全部通过，两项语义检查是未检查，所以发布为待复核，并附带 19 条复核事项。",
        "预算内的第二个配对，在审查时遇到服务限流，被如实记成节点错误，没有被悄悄忽略。",
    ]),
    scene("产物：pure 给训练，work 留证据", "输出目录与数据记录", "outputs", [10], [
        "每个样本有两个目录。第一个放数据：一行数据记录、原片副本、参考素材，以及目标视频和波形。",
        "第二个放证据：选择理由、导出计划、事实表、编号表、提示词、校验报告和全部模型调用。",
        "训练时，条件一侧只读提示词和记录里列出的参考。目标只做监督，原片副本只用来溯源。",
        "按照设计，严格的数据入口默认只收完成状态的样本，所以这条待复核样本现在不会进入训练。",
    ]),
    scene("可复现、怎么运行、目前的限制", "record / replay / refresh", "replay", [], [
        "每次模型调用都记入账本。缓存键包含输入内容、模板、模型版本和参数的哈希。",
        "记录模式命中缓存就复用。重放模式只读历史账本，缺了就阻塞。刷新模式全部重新调用，但保留旧版本。",
        "运行只要一条命令，给出视频路径。发布了样本退出码是 0，跳过是 2，阻塞是 1。",
        "第一版的限制也要说清楚：没有听声音的模型，样本最高只到待复核。也没有做模型推理、训练和人工复核。",
        "一句话总结：模型判断内容，程序控制流程，每个结论都能追溯到证据。",
    ]),
]


def spoken(sentence):
    """Subtitle text → text for the speech engine (digit strings it would misread)."""
    s = re.sub(r"节点 (\d)(\d)", lambda m: f"节点{DIGITS[int(m[1])]}{DIGITS[int(m[2])]}", sentence)
    s = s.replace("从 00 到 10", "从零零到一零").replace("1280×720", "一二八零乘七二零").replace("1344×768", "一三四四乘七六八")
    return s.replace("退出码是 0", "退出码是零")


def check_run_facts():
    """The narration quotes this run; fail loudly instead of narrating stale numbers."""
    J = lambda p: json.loads(Path(p).read_text())
    man, tl, cand = J(RUN / "run/source_manifest.json"), J(RUN / "run/timeline.json"), J(RUN / "run/candidates.json")
    pairs, index, plan = J(RUN / "run/pairs.json"), J(RUN / "run/index.json"), J(SAMPLE / "work/export_plan.json")
    ver = J(SAMPLE / "work/verification.json")
    statuses = [c["status"] for c in ver["checks"]]
    facts = [
        (man["decoded_frames"], SRC_FRAMES), ((man["width"], man["height"]), (1280, 720)),
        (len(tl["sampling"]["windows"]), 3), (sum(c["verdict"] == "confirmed" for c in tl["cuts"]), 7),
        (len(tl["shots"]), 8), (len(tl["subjects"]), 10), (len(tl["events"]), 10),
        (len(cand["boundaries"]) + cand["grid_boundary_count"], 129), (cand["enumerated_feasible"], 746),
        (len(cand["evaluated"]), 10), (len(cand["ranked_complete_ids"]), 8), (cand["shortlist_limit"], 3),
        (len(pairs["pair_order"]), 6), (pairs["pair_order"][0]["score"], 0.928),
        ((plan["target"]["start_frame"], plan["target"]["end_frame_excl"], plan["target"]["num_frames"]), (41, 169, 124)),
        ((plan["videos"][0]["start_frame"], plan["videos"][0]["end_frame_excl"], plan["videos"][0]["num_frames"]), (658, 750, 90)),
        (plan["target"]["wav_samples_per_channel"], 165333), (len(statuses), 11), (statuses.count("passed"), 9),
        (statuses.count("not_checked"), 2), (ver["status"], "needs_review"), (len(ver["review_items"]), 19),
        (index["attempts"][0]["outcome"], "node_error"),
    ]
    wrong = [f for f in facts if f[0] != f[1]]
    if wrong:
        raise ValueError(f"Run data no longer matches the narration: {wrong}")
    return tl, ver


def text(d, xy, content, size=34, color=FG, width=None, gap=12, mono=False):
    f = font(size, mono)
    lines = content.split("\n") if width is None else wrap(d, content, f, width)
    x, y = xy
    for line in lines:
        d.text((x, y), line, font=f, fill=color)
        if x + d.textlength(line, font=f) > W - 35:
            LAYOUT_WARNINGS.append((content, "right edge"))
        y += size + gap
    return y


def card(d, box, headline, body=None, accent=BLUE, hs=36, bs=29, state="on"):
    """state: 'off' not reached yet, 'on' already explained, 'now' being explained."""
    off = state == "off"
    d.rounded_rectangle(box, radius=22, fill=DIM_CARD if off else CARD,
                        outline=DIM_EDGE if off else accent if state == "now" else EDGE, width=4 if state == "now" else 2)
    x0, y0, x1, y1 = box
    y = text(d, (x0+28, y0+22), headline, hs, DIM_FG if off else accent, x1-x0-56)
    if body:
        y = text(d, (x0+28, y+10), body, bs, DIM_FG if off else FG, x1-x0-56)
    if y > y1 - 8:
        LAYOUT_WARNINGS.append((headline, f"vertical {y}>{y1}"))


def arrow(d, a, b, color=BLUE):
    d.line([a, b], fill=color, width=5)
    angle = math.atan2(b[1]-a[1], b[0]-a[0])
    d.polygon([b] + [(b[0]-20*math.cos(angle+o), b[1]-20*math.sin(angle+o)) for o in (-0.5, 0.5)], fill=color)


def paste_fit(im, pic, box):
    x0, y0, x1, y1 = box
    ratio = min((x1-x0)/pic.width, (y1-y0)/pic.height)
    pic = pic.resize((round(pic.width*ratio), round(pic.height*ratio)), Image.Resampling.LANCZOS)
    im.paste(pic, (round(x0+(x1-x0-pic.width)/2), round(y0+(y1-y0-pic.height)/2)))


def source_frame(index):
    path = BUILD / f"src_{index:06}.png"
    if not path.exists():
        run([FF, "-v", "error", "-y", "-ss", f"{(index+0.5)/SRC_FPS:.3f}", "-i", str(SAMPLE/"pure/oringin.mp4"),
             "-frames:v", "1", str(path)])
    return Image.open(path).convert("RGB")


def movie_box(im, d, label, poster):
    text(d, (MOVIE_BOX[0], 252), label, 28, BLUE)
    d.rectangle(MOVIE_BOX, fill="#000000")   # the real clip is overlaid on exactly this rectangle
    paste_fit(im, poster, MOVIE_BOX)
    text(d, (MOVIE_BOX[0], 806), "真实运行的素材 · 静音画面配合本片旁白", 24, MUTED)


def frame_x(frame, x0=100, x1=1820):
    return x0 + frame / SRC_FRAMES * (x1 - x0)


def render_scene(s, index, cue, run_data):
    tl, ver = run_data
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    st = lambda k: "now" if cue == k else "on" if cue > k else "off"   # element tied to sentence k
    d.rectangle((0, 0, W, 8), fill=BLUE)
    text(d, (72, 36), "STAPIPE  /  PIPELINE", 24, BLUE, mono=True)
    text(d, (1716, 36), f"{index+1:02d} / {len(SCENES):02d}", 26, MUTED, mono=True)
    for n in range(11):   # persistent node strip: where this chapter sits in the workflow
        x = 569 + n*72
        active = n in s["nodes"]
        d.rounded_rectangle((x, 30, x+62, 68), radius=10, fill="#1B5566" if active else CARD, outline=BLUE if active else EDGE, width=2)
        text(d, (x+16, 37), f"{n:02d}", 23, FG if active else DIM_FG, mono=True)
    text(d, (72, 100), s["title"], 54, FG)
    text(d, (74, 180), s["kicker"], 28, AMBER)
    key = s["key"]
    if key == "intro":
        card(d, (72, 262, 908, 420), "输入", "一个带音轨的真实视频", hs=30, bs=36, state=st(1))
        card(d, (72, 446, 908, 660), "条件 x", "参考图片 / 参考视频\n+ 六段式英文提示词", accent=BLUE, hs=30, bs=36, state=st(2))
        card(d, (72, 686, 908, 866), "监督 y", "原片中的目标视频 + 同期波形", accent=GREEN, hs=30, bs=36, state=st(2))
        movie_box(im, d, "本例原片：30 秒时尚短片", source_frame(124))
    elif key == "static":
        card(d, (72, 262, 936, 560), "模型负责：观察与判断", "看画面 / 判断参考关系 / 写描述\n每次只回答一个节点的问题\n输出受数据模式约束", accent=BLUE, hs=38, bs=31, state=st(1))
        card(d, (984, 262, 1848, 560), "程序负责：流程与结论", "节点 / 分支 / 预算 / 重试 / 退出\n路径、标签、时间戳、导出参数\n通过状态", accent=AMBER, hs=38, bs=31, state="now" if cue in (0, 3) else "on")
        text(d, (78, 596), "固定的模型节点（models.lock.json 锁定模型版本与模板哈希）", 27, MUTED)
        nodes = ["source_visual_facts", "target_completeness", "reference_relations", "duplicate_review",
                 "exported_visual_facts", "prompt_writer", "semantic_auditor", "prompt_repair"]
        for i, name in enumerate(nodes):
            x, y = 72 + (i % 4)*448, 650 + (i//4)*108
            d.rounded_rectangle((x, y, x+432, y+86), radius=16, fill=CARD, outline=EDGE, width=2)
            text(d, (x+22, y+27), name, 25, BLUE, mono=True)
    elif key == "overview":
        stages = [("准备", BLUE, [("00", "冻结配置、能力预检"), ("01", "全量解码、来源分组")]),
                  ("理解与选材", GREEN, [("02", "全片事实时间线"), ("03", "目标候选排序"), ("04", "目标外找参考")]),
                  ("导出与标注", AMBER, [("05", "合法帧数、精确导出"), ("06", "导出媒体上重新标注"), ("07", "六段式提示词")]),
                  ("审查与发布", RED, [("08", "程序校验 + 语义审查"), ("09", "最多修复一次"), ("10", "择优、原子发布")])]
        for i, (name, color, items) in enumerate(stages):
            x = 72 + i*452
            state = "on" if cue in (0, 4) else st(1 if i < 2 else i)
            if cue == 1 and i < 2:
                state = "now"
            off = state == "off"
            d.rounded_rectangle((x, 262, x+420, 740), radius=22, fill=DIM_CARD if off else CARD,
                                outline=DIM_EDGE if off else color if state == "now" else EDGE, width=4 if state == "now" else 2)
            text(d, (x+28, 284), name, 38, DIM_FG if off else color)
            for j, (num, label) in enumerate(items):
                y = 372 + j*118
                d.rounded_rectangle((x+24, y, x+396, y+96), radius=14, fill=BG, outline=DIM_EDGE if off else EDGE, width=2)
                text(d, (x+42, y+12), num, 26, DIM_FG if off else color, mono=True)
                text(d, (x+42, y+50), label, 27, DIM_FG if off else FG)
            if i < 3:
                arrow(d, (x+424, 500), (x+448, 500), MUTED)
        loop = AMBER if cue == 4 else EDGE
        d.line((996, 778, 996, 800, 1780, 800, 1780, 778), fill=loop, width=4)
        text(d, (1010, 818), "05 – 09 对每个候选配对执行，次数受 export_pair_budget 限制", 27, AMBER if cue == 4 else MUTED)
        text(d, (78, 818), "所有失败都进入预先定义的分类", 27, MUTED)
    elif key == "decode":
        card(d, (72, 262, 908, 418), "00  配置快照 + 运行指纹", "models.lock.json：模型版本、模板哈希", hs=33, bs=28, state=st(0))
        card(d, (72, 438, 908, 594), "01  750 帧 · 1280×720 · 25 FPS", "逐帧 PTS / AAC 48000 Hz 立体声 / SHA-256", accent=GREEN, hs=33, bs=28, state=st(1))
        card(d, (72, 614, 908, 730), "能力预检：视觉可用，声音不可用", accent=AMBER, hs=33, state=st(2))
        card(d, (72, 750, 908, 882), "无音轨 → skipped   无视觉 → blocked", "split_group = sha256:74d82a1950e45761", accent=RED, hs=30, bs=25, state=st(3))
        movie_box(im, d, "原片：完整解码，不只看容器头", source_frame(124))
    elif key == "timeline":
        shots = [(sh["start_frame"], sh["end_frame_excl"]) for sh in tl["shots"]]
        for i, (a, b) in enumerate(shots):
            x = 102 + i*216
            im.paste(source_frame((a+b)//2).resize((204, 115), Image.Resampling.LANCZOS), (x, 258))
            text(d, (x, 380), f"s{i+1:03d}  {a/SRC_FPS:5.2f}s", 21, BLUE if cue >= 2 else MUTED, mono=True)
        for i, (a, b) in enumerate(shots):
            d.rectangle((frame_x(a), 436, frame_x(b)-3, 476), fill="#176476" if i % 2 == 0 else "#22495F")
        for c in tl["cuts"]:
            ok = c["verdict"] == "confirmed"
            d.line((frame_x(c["frame"]), 424, frame_x(c["frame"]), 488), fill=(AMBER if ok else MUTED) if cue >= 2 else MUTED, width=4)
        for t in range(0, 31, 5):
            text(d, (frame_x(t*SRC_FPS)-12, 494), f"{t}s", 22, MUTED, mono=True)
        for i, w in enumerate(tl["sampling"]["windows"]):
            y = 540 + i*22
            d.rounded_rectangle((frame_x(w["start_s"]*SRC_FPS), y, frame_x(w["end_s"]*SRC_FPS), y+14), radius=7,
                                fill=GREEN if cue == 1 else EDGE)
        text(d, (frame_x(0), 612), "3 个理解窗口：12 秒，重叠 2 秒", 25, GREEN if cue == 1 else MUTED)
        card(d, (72, 668, 632, 884), "程序规定抽样", "每秒 1 帧 + 切镜两侧\n模型只看给定的帧", hs=32, bs=27, state=st(1))
        card(d, (680, 668, 1240, 884), "模型输出事实", "7 个确认切镜 → 8 个镜头\n10 个主体 / 10 个事件", accent=GREEN, hs=32, bs=27, state=st(2))
        card(d, (1288, 668, 1848, 884), "声音只是证据", "语音识别 / 声音标签 / 能量\n不等于模型听过", accent=AMBER, hs=32, bs=27, state=st(3))
    elif key == "targets":
        funnel = [("129", "边界", "片头尾 / 切镜 / 停顿 / 0.25 s 网格", 0), ("746", "可行区间", "5–15 秒，且能排出合法帧数", 1),
                  ("10", "模型评估", "target_evaluation_budget（小预算）", 2), ("8", "完整", "两端干净、动作与对白未被截断", 3),
                  ("3", "短名单", "按 target_score 稳定排序", 3)]
        for i, (num, name, note, k) in enumerate(funnel):
            y, inset = 262 + i*92, i*46
            state = st(k)
            off = state == "off"
            d.rounded_rectangle((72+inset, y, 1010-inset, y+78), radius=16, fill=DIM_CARD if off else CARD,
                                outline=DIM_EDGE if off else BLUE if state == "now" else EDGE, width=4 if state == "now" else 2)
            text(d, (100+inset, y+17), num, 38, DIM_FG if off else AMBER, mono=True)
            text(d, (220+inset, y+19), name, 32, DIM_FG if off else FG)
            text(d, (1040, y+24), note, 26, DIM_FG if off else MUTED)
        d.rounded_rectangle((72, 742, 1848, 884), radius=22, fill=CARD, outline=AMBER if cue == 3 else EDGE, width=4 if cue == 3 else 2)
        text(d, (104, 762), "target_score = 0.40 × 语义完整性 + 0.30 × 画面质量 + 0.20 × 声音可标注性 + 0.10 × 信息量", 31, FG)
        text(d, (104, 822), "模型只给离散等级和证据帧；分数与排序由程序计算", 27, MUTED)
    elif key == "references":
        d.line((frame_x(0), 300, frame_x(SRC_FRAMES), 300), fill=EDGE, width=6)
        for t in range(0, 31, 5):
            text(d, (frame_x(t*SRC_FPS)-12, 326), f"{t}s", 22, MUTED, mono=True)
        d.rounded_rectangle((frame_x(41), 268, frame_x(169), 316), radius=8, fill="#926335")
        d.rounded_rectangle((frame_x(658), 268, frame_x(750), 316), radius=8, fill="#176476")
        text(d, (frame_x(41)+8, 232), "目标 [41, 169)", 25, AMBER)
        text(d, (frame_x(658)-60, 232), "参考 [658, 750)", 25, BLUE)
        for x, frame, label, color in [(72, 150, "目标：片头标题镜头（源帧 150）", AMBER), (612, 713, "参考视频：片尾特写（源帧 713）", BLUE)]:
            im.paste(source_frame(frame).resize((508, 286), Image.Resampling.LANCZOS), (x, 380))
            d.rectangle((x, 380, x+508, 666), outline=color, width=3)
            text(d, (x, 678), label, 24, color)
        card(d, (72, 730, 580, 884), "近重复检查：clear", "最小帧差 24.6，重复覆盖 0%", accent=GREEN, hs=30, bs=25, state=st(1))
        card(d, (612, 730, 1120, 884), "pair_score = 0.928", "0.6 × 目标分 + 0.4 × 参考集合分", accent=AMBER, hs=30, bs=25, state=st(3))
        card(d, (1160, 380, 1848, 884), "模型判定的关系", "woman_slicked_bun\n身份与外观：脸 / 发型 / 肤色\n→ fully_preserved\n\nwhite_flower_cloud\n道具：白色星形小花、肩部花簇\n→ fully_preserved", accent=BLUE, hs=32, bs=27, state=st(2))
    elif key == "export":
        text(d, (78, 258), "N = 17 × k + 5     |     24 FPS", 42, BLUE if cue == 0 else MUTED, mono=True)
        card(d, (72, 336, 936, 560), "目标：5.12 秒 → 124 帧", "源帧 [41, 169)，25 FPS\n5.1667 秒，末尾补 0.0467 秒（0.91%）", accent=AMBER, hs=38, bs=30, state=st(1))
        card(d, (984, 336, 1848, 560), "参考：3.68 秒 → 90 帧", "源帧 [658, 750)\n3.75 秒，末尾补 0.07 秒（1.90% ≤ 2%）", accent=BLUE, hs=38, bs=30, state=st(2))
        card(d, (72, 590, 936, 884), "画面：等比缩放 + 补边", "1280×720 → 1344×756\n上下各补 6 像素 → 1344×768\n补齐只重复末帧，不读区间外画面", hs=34, bs=29, state=st(3))
        card(d, (984, 590, 1848, 884), "波形：直接取自原片", "32000 Hz / 双声道 / 16-bit PCM\nS = round(124 × 32000 / 24) = 165333\nMP4 音轨由同一 WAV 编码", accent=GREEN, hs=34, bs=29, state=st(4))
    elif key == "facts":
        card(d, (72, 262, 908, 420), "坐标系换成导出媒体", "目标时间从 0 秒开始，切镜定位到帧", hs=33, bs=28, state=st(0))
        card(d, (72, 440, 908, 634), "exported_visual_facts", "镜头 / 主体 / 可见文字 / 动作\n参考关系不成立 → RejectPair", accent=GREEN, hs=31, bs=28, state=st(1))
        card(d, (72, 654, 908, 768), "听不清的对白：整句不写入提示词", accent=AMBER, hs=31, state=st(2))
        card(d, (72, 788, 908, 884), "待听辨区间：[0.0, 5.166] 秒", accent=RED, hs=31, state=st(3))
        movie_box(im, d, "导出的 target.mp4：124 帧，1344×768", Image.open(BUILD/"poster_target.png").convert("RGB"))
    elif key == "prompt":
        d.rounded_rectangle((72, 262, 760, 884), radius=22, fill=CARD, outline=BLUE if cue == 2 else EDGE, width=4 if cue == 2 else 2)
        for i, h in enumerate(["subject_definitions", "summary", "retention_analysis", "detailed_description",
                               "overall_soundscape", "non_diegetic_music"]):
            text(d, (104, 300+i*94), f"{i+1}  {h}:", 31, BLUE if cue >= 2 else MUTED, mono=True)
        card(d, (800, 262, 1848, 446), "程序分配的编号", "<Video 1>   <Subject 1> 白花   <Subject 2> 女士   (S1) 未知说话人", accent=AMBER, hs=31, bs=26, state=st(0))
        card(d, (800, 466, 1848, 650), "模型只写内容对象", "{ text, fact_ids }  不得出现 [Shot、时间戳、<d> 等格式", accent=BLUE, hs=31, bs=26, state=st(1))
        card(d, (800, 670, 1848, 884), "真实输出（节选）", "<Subject 2> (appears in [Shot 1]): fully_preserved - keeps her face, slicked-back dark hair and olive skin tone ...", accent=GREEN, hs=31, bs=25, state=st(3))
    elif key == "verify":
        d.rounded_rectangle((72, 258, 1010, 888), radius=22, fill=CARD, outline=EDGE, width=2)
        for i, c in enumerate(ver["checks"]):
            y = 276 + i*55
            program = c["method"] != "model"
            lit = cue == 0 or (cue == 1 and program) or (cue == 2 and not program) or cue > 2
            color = GREEN if c["status"] == "passed" else AMBER
            text(d, (100, y), c["check"], 22, FG if lit else DIM_FG, mono=True)
            text(d, (704, y), c["method"], 23, MUTED if lit else DIM_FG, mono=True)
            text(d, (832, y), c["status"], 23, color if lit else DIM_FG, mono=True)
        card(d, (1050, 258, 1848, 440), "独立审查模型", "输入：真实导出媒体 + 提示词草稿\n看不到写作节点的推理", accent=BLUE, hs=32, bs=27, state=st(2))
        card(d, (1050, 460, 1848, 664), "可修复（仅一次）", "删去无证据修饰 / 纠正格式\n删去未确认的对白\n修复后全部重查", accent=GREEN, hs=32, bs=27, state=st(3))
        card(d, (1050, 684, 1848, 888), "直接淘汰配对", "参考重复了目标镜头\n目标截断关键动作 / 媒体越界\n核心参考关系不成立", accent=RED, hs=32, bs=27, state=st(4))
    elif key == "status":
        for i, (name, body, color) in enumerate([("completed", "检查通过\n无待复核项", GREEN), ("needs_review", "样本合法\n有明确不确定性", AMBER),
                                                 ("skipped", "无音轨 / 预算内\n无有效配对", MUTED), ("blocked", "输入损坏 / 能力缺失\n服务或程序错误", RED)]):
            x = 72 + i*452
            card(d, (x, 262, x+420, 480), name, body, accent=color, hs=36, bs=27, state="now" if cue == 1 else "on")
        card(d, (72, 510, 936, 884), "本例：needs_review", "9 项程序类检查 passed\n2 项语义检查 not_checked\n19 条复核事项（14 视觉 + 5 声音）\n暂存目录 → 无覆盖提交", accent=AMBER, hs=36, bs=29, state="now" if cue in (0, 2) else "on")
        card(d, (984, 510, 1848, 884), "另一次尝试：node_error", "pair p_000044_000207\nsemantic_auditor 返回 429 限流\n重试耗尽 → 写入 attempts 与\ntechnical_errors，证据归档", accent=RED, hs=36, bs=29, state=st(3))
    elif key == "outputs":
        for x, name, color, lines, k in [
            (72, "pure/", GREEN, ["dataset.jsonl", "oringin.mp4", "inputs/video_1.mp4", "targets/target.mp4", "targets/target_audio.wav"], 0),
            (682, "work/", AMBER, ["selection.json", "export_plan.json  export_log.json", "facts.json  labels.json", "prompt_draft.json  prompt.md",
                                   "verification.json", "provenance.json  model_calls/"], 1)]:
            state = st(k)
            off = state == "off"
            d.rounded_rectangle((x, 262, x+578, 640), radius=22, fill=DIM_CARD if off else CARD,
                                outline=DIM_EDGE if off else color if state == "now" else EDGE, width=4 if state == "now" else 2)
            text(d, (x+28, 282), name, 36, DIM_FG if off else color, mono=True)
            for j, line in enumerate(lines):
                text(d, (x+28, 344+j*46), line, 24, DIM_FG if off else FG, mono=True)
        card(d, (1292, 262, 1848, 640), "run/", "配置快照 / 运行指纹\ntimeline / candidates / pairs\n模型调用账本\n未发布尝试的证据\nindex.json（已提交索引）", accent=BLUE, hs=36, bs=26, state="on")
        card(d, (72, 670, 936, 884), "条件分支只读", "prompt + references\n不扫描 pure/，不读 target 与原片", accent=BLUE, hs=33, bs=28, state=st(2))
        card(d, (984, 670, 1848, 884), "严格入口默认只收 completed", "needs_audio_review: true\nuncertain_audio_spans_seconds: [[0.0, 5.166]]", accent=RED, hs=33, bs=26, state=st(3))
    elif key == "replay":
        for i, (name, body, color) in enumerate([("record", "同指纹复用缓存\n未命中才调用模型", BLUE), ("replay", "只读指定历史账本\n缺失即 blocked", GREEN),
                                                 ("refresh", "全部重新调用\n生成新版本，保留旧的", AMBER)]):
            x = 72 + i*610
            card(d, (x, 262, x+556, 470), name, body, accent=color, hs=38, bs=28, state="now" if cue == 1 else "on")
        d.rounded_rectangle((72, 500, 1848, 660), radius=22, fill=CARD, outline=BLUE if cue == 2 else EDGE, width=4 if cue == 2 else 2)
        text(d, (104, 522), "./ref2va video.mp4 [out_dir] [--config config.yaml] [--set mode=replay]", 30, BLUE, mono=True)
        text(d, (104, 588), "退出码：0 发布了样本    2 skipped    1 blocked", 29, FG)
        card(d, (72, 690, 936, 884), "v1 的限制", "无听音模型 → 最高 needs_review\n未做 H3 推理 / 训练 / 人工复核", accent=RED, hs=33, bs=28, state=st(3))
        card(d, (984, 690, 1848, 884), "模型判断内容，程序控制流程", "每个结论都能追溯到证据", accent=GREEN, hs=35, bs=30, state=st(4))
    return im


def caption_frame(base, sentence, elapsed_ratio, cue_index, cue_count):
    im = base.copy()
    d = ImageDraw.Draw(im)
    d.rectangle((0, 921, W, H), fill="#060D16")
    lines = wrap(d, sentence, font(39), W-160)
    if len(lines) > 2:
        raise ValueError("Subtitle needs more than two lines: " + sentence)
    if len(lines) == 2:   # balance the two lines, breaking after punctuation when possible
        width = lambda t: d.textlength(t, font=font(39))
        cuts = [i for i in range(2, len(sentence)-1) if sentence[i] not in "，。：、）) " and max(width(sentence[:i]), width(sentence[i:])) <= W-160]
        cuts = [i for i in cuts if sentence[i-1] in "，。：、"] or cuts
        best = min(cuts, key=lambda i: abs(width(sentence[:i])-width(sentence[i:])))
        lines = [sentence[:best].strip(), sentence[best:].strip()]
    y = 941 if len(lines) == 2 else 962
    for line in lines:
        d.text(((W-d.textlength(line, font=font(39)))/2, y), line, font=font(39), fill=FG)
        y += 52
    d.rectangle((0, 1072, int(W*elapsed_ratio), 1079), fill=BLUE)
    for i in range(cue_count):
        d.ellipse((1730+i*24, 191, 1740+i*24, 201), fill=AMBER if i == cue_index else EDGE)
    return im


def speech(sentence):
    words = spoken(sentence)
    cache_key = hashlib.sha256(f"{VOICE}|{RATE}|{words}".encode()).hexdigest()[:16]
    aiff, wav = BUILD/f"voice_{cache_key}.aiff", BUILD/f"voice_{cache_key}.wav"
    if not wav.exists():
        run(["say", "-v", VOICE, "-r", RATE, "-o", str(aiff), words])
        run([FF, "-v", "error", "-y", "-i", str(aiff), "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", str(wav)])
    with wave.open(str(wav), "rb") as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()
    active = np.flatnonzero(np.abs(audio.astype(np.int32)) > 90)
    if not len(active):
        raise ValueError("Empty speech: " + sentence)
    return audio[max(0, active[0]-int(.06*SR)):min(len(audio), active[-1]+int(.13*SR))]


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    (DOCS/"previews").mkdir(parents=True, exist_ok=True)
    run_data = check_run_facts()
    movies = {"raw": SAMPLE/"pure/oringin.mp4", "target": SAMPLE/"pure/targets/target.mp4"}
    run([FF, "-v", "error", "-y", "-ss", "3.5", "-i", str(movies["target"]), "-frames:v", "1", str(BUILD/"poster_target.png")])
    all_audio = []
    for i, s in enumerate(SCENES):
        phrases = []
        for j, sentence in enumerate(s["sentences"]):
            a = speech(sentence)
            gap = .85 if j == len(s["sentences"])-1 else .32
            frames = math.ceil((len(a)/SR+gap)*FPS)
            padded = np.zeros(frames*(SR//FPS), dtype=np.int16)
            padded[:len(a)] = a
            phrases.append((sentence, padded, frames, len(a)/SR))
        all_audio.append(phrases)
        print(f"Speech {i+1:02}/{len(SCENES)}: {s['title']}", flush=True)
    total_frames = sum(q[2] for a in all_audio for q in a)
    global_frames = 0
    subtitles, chapters, segments = [], [], []
    narr_md = ["# 流水线讲解视频：旁白与章节\n", f"中文配音由本机 macOS {VOICE} 合成；示例数据来自 `{RUN.relative_to(STAPIPE)}`。\n"]
    for i, (s, phrases) in enumerate(zip(SCENES, all_audio)):
        concat, aud, scene_frames = [], [], 0
        start = global_frames/FPS
        narr_md.append(f"## {stamp(start)}  {i+1:02} · {s['title']}\n")
        for j, (sentence, a, frames, voice_dur) in enumerate(phrases):
            base = render_scene(s, i, j, run_data)
            if j == len(phrases)-1:
                base.save(DOCS/"previews"/f"{i+1:02}_{s['key']}.png")
            png = BUILD/f"scene_{i:02}_cue_{j:02}.png"
            caption_frame(base, sentence, (global_frames+frames)/total_frames, j, len(phrases)).save(png)
            concat.append(f"file '{png.as_posix()}'\noption framerate {FPS}\nduration {frames/FPS:.9f}\n")
            subtitles.append((global_frames/FPS, global_frames/FPS+voice_dur, sentence))
            narr_md.append(sentence+"\n")
            aud.append(a)
            scene_frames += frames
            global_frames += frames
        concat.append(f"file '{png.as_posix()}'\noption framerate {FPS}\n")
        listing = BUILD/f"scene_{i:02}.ffconcat"
        listing.write_text("ffconcat version 1.0\n"+"".join(concat))
        wav = BUILD/f"scene_{i:02}.wav"
        write_wav(wav, np.concatenate(aud))
        seg = BUILD/f"scene_{i:02}.mp4"
        args = [FF, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(wav)]
        if s["movie"]:
            x0, y0, x1, y1 = MOVIE_BOX
            bw, bh = x1-x0, y1-y0
            args += ["-stream_loop", "-1", "-i", str(movies[s["movie"]])]
            graph = (f"[0:v]fps={FPS},format=yuv420p[base];[2:v]setpts=PTS-STARTPTS,fps={FPS},"
                     f"scale={bw}:{bh}:force_original_aspect_ratio=decrease,pad={bw}:{bh}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[clip];"
                     f"[base][clip]overlay={x0}:{y0}:shortest=1[v]")
        else:
            graph = f"[0:v]fps={FPS},format=yuv420p[v]"
        args += ["-filter_complex", graph, "-map", "[v]", "-map", "1:a:0", "-frames:v", str(scene_frames), "-t", f"{scene_frames/FPS:.9f}",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS),
                 "-video_track_timescale", "24000", "-c:a", "aac", "-b:a", "160k", "-ar", str(SR), "-ac", "2", str(seg)]
        print(f"Render {i+1:02}/{len(SCENES)}: {scene_frames/FPS:.1f} s", flush=True)
        run(args)
        chapters.append(dict(index=i+1, title=s["title"], start_seconds=start, end_seconds=global_frames/FPS, frames=scene_frames, movie=s["movie"]))
        segments.append(seg)
    if LAYOUT_WARNINGS:
        raise ValueError(f"Layout overflow: {sorted(set(map(str, LAYOUT_WARNINGS)))}")
    final_list = BUILD/"segments.txt"
    final_list.write_text("".join(f"file '{p.as_posix()}'\n" for p in segments))
    metadata = [";FFMETADATA1", "title=stapipe-v1 Pipeline Explainer",
                "comment=Chinese synthetic narration; footage and numbers from one real stapipe run; no H3 inference or training."]
    for ch in chapters:
        metadata += ["[CHAPTER]", "TIMEBASE=1/24000", f"START={round(ch['start_seconds']*24000)}", f"END={round(ch['end_seconds']*24000)}",
                     f"TITLE={ch['index']:02} {ch['title']}"]
    meta = BUILD/"chapters.ffmeta"
    meta.write_text("\n".join(metadata)+"\n")
    # One sample-aligned narration track; copying per-scene AAC would add priming gaps.
    whole_wav = BUILD/"narration.wav"
    write_wav(whole_wav, np.concatenate([q[1] for a in all_audio for q in a]))
    run([FF, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(final_list), "-i", str(whole_wav), "-i", str(meta),
         "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "2", "-map_chapters", "2", "-c:v", "copy", "-af", "loudnorm=I=-18:TP=-2:LRA=9",
         "-c:a", "aac", "-b:a", "160k", "-ar", str(SR), "-ac", "2", "-t", f"{total_frames/FPS:.9f}", "-movflags", "+faststart", str(OUT)])
    (DOCS/"pipeline_explainer.srt").write_text("\n".join(f"{i}\n{stamp(a, True)} --> {stamp(b, True)}\n{t}\n" for i, (a, b, t) in enumerate(subtitles, 1)))
    (DOCS/"narration.md").write_text("\n".join(narr_md))
    (DOCS/"chapters.json").write_text(json.dumps(chapters, ensure_ascii=False, indent=2))
    manifest = {"output": str(OUT.relative_to(STAPIPE)), "duration_seconds": total_frames/FPS, "frames": total_frames, "resolution": [W, H],
                "fps": FPS, "voice": VOICE, "speech_rate": RATE, "chapters": len(chapters), "subtitle_cues": len(subtitles),
                "example_run": str(RUN.relative_to(STAPIPE)), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}
    (DOCS/"build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
