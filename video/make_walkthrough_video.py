"""Build a narrated Chinese walkthrough from the real repository example.

Run with a Python environment containing Pillow, numpy and imageio-ffmpeg.
macOS `say -v Tingting` supplies local speech. The output is a teaching video,
not a run of the proposed Ref2VA pipeline. Original media remain read-only.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
STAPIPE = HERE.parent
ROOT = STAPIPE.parent
EX = ROOT / "ref2va-workspace/examples"
BUILD = HERE / "build"
OUT = STAPIPE / "walkthrough.mp4"
FF = imageio_ffmpeg.get_ffmpeg_exe()
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"
W, H, FPS, SR = 1920, 1080, 24, 48000
BG, CARD, EDGE = "#0B1421", "#142437", "#29445E"
FG, MUTED, BLUE, AMBER, GREEN, RED = "#F1F5F9", "#A5B8CC", "#70D9ED", "#FFC578", "#91DFC0", "#FF9B9B"
VOICE, RATE = "Tingting", "200"
LAYOUT_WARNINGS = []


def scene(title, kicker, key, sentences, movie=None):
    return dict(title=title, kicker=kicker, key=key, sentences=sentences, movie=movie)


SCENES = [
    scene("从 raw_video 到训练样本", "完整走读 · 一条真实广告", "intro", [
        "这次我们用仓库里的一条三十秒广告，完整走一遍从原始视频到音视频训练样本的过程。",
        "画面里的素材都来自已有样例。我们要得到的目标，是搬沙发和车库卧推这一段，大约七点二九秒。",
        "我会讲清楚每一步的输入、输出和选择依据，也会展示为什么这条样例目前还需要声音复核。",
        "这是对设计的教学走读，不是一次已经执行完成的新流水线。"
    ], "target"),
    scene("静态的是流程，模型仍然可以判断", "先建立整体认识", "static", [
        "这里说的静态，指的是步骤、分支、重试次数和退出条件，都由程序预先规定。",
        "模型仍然可以理解画面、识别声音、判断参考关系，并编写提示词。",
        "区别是模型只回答当前节点的问题，不能自己决定调用什么工具，或者临时改掉质量门槛。",
        "你可以把它理解成一条固定生产线，每个模型是线上一个职责明确的工位。"
    ]),
    scene("先分清：条件 x 和监督 y", "训练样本到底是什么", "xy", [
        "一条训练样本可以理解为一组条件，加上一份真实答案。",
        "条件包含两张人物和产品参考图片，一段办公室参考视频，以及描述目标的六段式提示词。",
        "监督目标则是原片里真实存在的搬沙发和车库画面，以及同一个时间区间的声音。",
        "标注模型可以看答案来写描述，但训练的条件分支不能读入目标媒体或者完整原片。"
    ]),
    scene("00—01  固定规则，完整读取原片", "输入检查与来源分组", "source", [
        "程序首先锁定配置、模型版本和媒体规格，再完整解码原片。",
        "这个例子有七百二十帧，画面是一千二百八十乘七百二十，帧率约为每秒二十三点九七六帧。",
        "原片有真实的双声道音轨。程序还记录文件哈希和每一帧的实际显示时间。",
        "同一作品的片段共用一个来源分组，避免一部分进训练集，另一部分混进验证集。"
    ], "raw"),
    scene("02  先建立全片事实索引", "模型负责观察，程序规定采样", "overview", [
        "先看整部广告。主角、橙色手指、零食包装和戴墨镜的人物，在不同场景反复出现。",
        "程序按固定窗口提供画面和声音，并在切镜附近增加抽帧。模型输出镜头、人物、动作和声音事实。",
        "这里还没有决定最终目标，只是在建立一张可以查询的内容地图。",
        "看见有人卧推，并不等于已经听见喘气或金属碰撞。每个声音描述都需要自己的证据。"
    ]),
    scene("03  先选目标候选", "源帧区间 [200, 374)", "target", [
        "本例沿用已有样例的目标候选，从源视频第两百帧开始，到第三百七十四帧之前结束。",
        "它对应原片八点三四秒到十五点六零秒，包括搬沙发和车库卧推两个小情节。",
        "完整动作、可描述内容，以及目标之外还有可用参考，让这段值得进入下一步。",
        "真实流水线会比较多个候选并保存评分。这里没有重新运行选择器，也不宣称它一定排名第一。"
    ], "target"),
    scene("04  目标之外找参考", "先检索，再核实每份参考的用途", "references", [
        "第一张图片来自原片第409帧，提供主角身份、橙色手指和零食包装。",
        "第二张图片来自第558帧，提供另一个人物的脸、发型和墨镜。",
        "办公室参考视频提供求助、反应、拒绝和人物突然出现的叙事模式。启用的音轨只提供声音风格。",
        "程序必须确认每份参考都有实际用途。两图一视频只是这个例子的组合，不是所有视频都要照搬。"
    ]),
    scene("参考与目标：时间不重叠，内容也要查", "区间检查 + 近重复检查 + 用途确认", "timeline", [
        "把来源放到一条时间线上。办公室参考在前，目标居中，两张图片都来自目标之后。",
        "参考视频恰好在目标开始的边界结束。因为区间右端不包含，所以它们没有交叠。",
        "但时间不同还不够。如果后面只是重播了目标镜头，仍然需要拒绝。",
        "同一个人物的新场景可以作为身份参考，重复播放答案则不行。音色或者音乐相似，也不能写成波形复制。"
    ]),
    scene("05  为什么目标导出为 175 帧", "先保护完整内容，再满足媒体约束", "frames", [
        "目标源区间长七点二五七二五秒，包含一百七十四个源帧。",
        "本地输出规则要求每秒二十四帧，而且总帧数等于十七乘整数，再加五。",
        "一百五十八帧太短，会切掉内容。一百七十五帧对应七点二九一六七秒，只需补约三十四毫秒。",
        "程序只重复最后一个合法源帧，并为声音末尾补静音，不能读入下一个镜头来凑长度。"
    ]),
    scene("真实例子暴露了一个门槛问题", "参考补齐：2.072% > 默认 2%", "padding", [
        "参考视频也要对齐。它从五点七五五七五秒，补到一百四十一帧，也就是五点八七五秒。",
        "需要补零点一一九二五秒，占原片段的百分之二点零七二。",
        "它虽然满足零点一二五秒的绝对上限，却超过设计默认的百分之二比例上限。固定程序应该拒绝这个方案。",
        "为了继续讲解现有素材，文档明确使用百分之二点五的教学配置。门槛在运行前修改，不能让模型临时放宽。"
    ]),
    scene("画面和声音要一起对齐", "等比缩放、补边、源音频直接重采样", "av", [
        "画面等比放大到一千三百四十四乘七百五十六，再上下各补六个像素，得到目标画布。",
        "目标波形直接从原片区间提取，转成三万两千赫兹、双声道、十六位的波形文件。",
        "一百七十五帧对应每声道二十三万三千三百三十三个采样点，时长误差小于一个采样点。",
        "目标视频的声音由同一波形编码。容器尾部的编码填充，要和真正的音画错位分开检查。"
    ]),
    scene("06  导出后重新建立时间线", "提示词时间从目标的 0 秒开始", "shots", [
        "媒体导出之后，必须在导出结果上重新确认镜头和声音事件。",
        "这条目标有六个镜头。第二镜头从目标第29帧开始，换算成提示词时间，就是一点二零八秒。",
        "第五和第六镜头也有自己的目标时间，不能直接抄原片八秒、十五秒这样的绝对位置。",
        "屏幕上的切镜时间来自已有样例标注。实际新运行还需要逐帧测量和核验。"
    ]),
    scene("07  先分配标签，再编写提示词", "素材、可复用主体、发声者是不同编号", "labels", [
        "图片标签和视频标签对应实际输入文件。声音标签一，对应办公室视频中明确启用的同步音轨。",
        "主体标签用于可复用的人物或属性。按照新设计的首次出现顺序，主角是主体一，包装是主体二，墨镜人物是主体三。",
        "说话人又是独立的一套编号。搬运工可以是说话人一，即使他没有专门的参考图片。",
        "历史样例的主体二和主体三顺序不同。重编号时必须更新全文，不能只改一处。"
    ]),
    scene("事实 → 六段式提示词 → 数据记录", "模型写内容，程序控制结构与时间", "prompt", [
        "六段依次是主体定义、任务摘要、保留关系、详细镜头描述、整体声景，以及非画内配乐。",
        "模型基于事实表写英文内容，程序插入固定段名、素材标签和已经确认的镜头时间。",
        "第二镜头的搬运工短句目前听不清，就使用不清楚的标记，并记录目标一点二秒到二点六秒需要声音复核。",
        "最后序列化成一条数据记录，引用输入媒体和实测目标参数，同时保存一份可读提示词。"
    ]),
    scene("08—09  校验，并且最多修复一次", "结构、媒体、事实，分开检查", "verify", [
        "程序先检查文件、帧数、采样数、时长、来源区间和标签。审查模型再对照真实素材核对语义。",
        "如果只是把衣着写错，而事实表里有正确答案，可以修正文案一次，然后重新审查。",
        "如果参考其实重复了目标镜头，或者目标截断了关键动作，就要淘汰配对，不能靠修改文字掩盖。",
        "没有执行过的检查必须写成未检查，不能因为文件已经生成，就自动宣布完成。"
    ]),
    scene("10  本例为什么停在 needs_review", "产物可保存，严格训练入口仍然拦截", "status", [
        "假设其他必需检查全部通过，这条样例仍然有一句未知对白，所以状态应当是需要复核。",
        "媒体和记录可以完整保存，过程目录同时记录来源、提示词、检查结果和待复核区间。",
        "严格训练入口只接收已经完成全部必要核验的样本，因此现在不会把它送入训练。",
        "需要先解决声音问题并复查，或者选另一段可靠的目标。直接把复核标记改成假，并不能解决问题。"
    ]),
    scene("复核通过以后，训练器会读什么", "质量门槛 → 格式适配 → 条件与监督", "train", [
        "当样本未来复核通过，训练打包器才把文本和参考送进条件分支，把真实画面和波形送进监督分支。",
        "完整原片不能进入条件。来源分组也必须保留，用于整组划分训练集和验证集。",
        "还要确认训练配置接收一百七十五帧。如果只接收一百二十四帧，第133帧才开始的第六镜头就会被截掉。",
        "所以不能先截视频，再继续沿用原来的提示词。需要匹配训练配置，或者重新选段和标注。"
    ]),
    scene("可复现、可改进，具体落在哪", "保存输入、模型响应、配置与检查证据", "replay", [
        "记录模式保存模型请求、响应和中间产物。重放模式只读取指定历史记录，不重新请求模型。",
        "重新调用模式则保留旧版本，再调用固定节点，用来观察新的结果是否波动。",
        "想改善声音，就修改声音节点后重跑相关下游。想改善文字，就固定媒体和事实，修改写作模板。",
        "这就是整条流程：原片建立事实，选目标和参考，精确导出，写提示词，通过检查后才进入训练。模型判断内容，程序控制流程。"
    ]),
]


def run(args):
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"Command failed: {args[:5]}\n{p.stderr[-5000:]}")
    return p


def font(size, mono=False):
    return ImageFont.truetype(MONO if mono else FONT, size)


def wrap(d, text, f, width):
    lines, current = [], ""
    for char in text:
        if char == "\n":
            lines.append(current)
            current = ""
        elif current and d.textlength(current + char, font=f) > width:
            lines.append(current)
            current = char
        else:
            current += char
    if current:
        lines.append(current)
    return lines


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


def panel(d, box, fill=CARD, outline=EDGE):
    d.rounded_rectangle(box, radius=22, fill=fill, outline=outline, width=2)


def card(d, box, headline, body=None, accent=BLUE, hs=38, bs=30):
    panel(d, box)
    x0, y0, x1, y1 = box
    y = text(d, (x0+28, y0+23), headline, hs, accent, x1-x0-56)
    if body:
        y = text(d, (x0+28, y+13), body, bs, FG, x1-x0-56)
    if y > y1 - 12:
        LAYOUT_WARNINGS.append((headline, f"vertical {y}>{y1}"))


def arrow(d, a, b, color=BLUE):
    d.line([a,b], fill=color, width=5)
    angle=math.atan2(b[1]-a[1],b[0]-a[0])
    points=[b]
    for offset in [-0.5,0.5]:
        points.append((b[0]-22*math.cos(angle+offset),b[1]-22*math.sin(angle+offset)))
    d.polygon(points, fill=color)


def fit_picture(im, path, box):
    pic = Image.open(path).convert("RGB")
    x0,y0,x1,y1=box
    ratio=min((x1-x0)/pic.width,(y1-y0)/pic.height)
    pic=pic.resize((round(pic.width*ratio),round(pic.height*ratio)),Image.Resampling.LANCZOS)
    im.paste(pic,(round(x0+(x1-x0-pic.width)/2),round(y0+(y1-y0-pic.height)/2)))


def movie_box(im,d,label):
    text(d,(984,260),label,29,BLUE)
    # Actual video overlay occupies this exact 864 x 486 rectangle.
    d.rectangle((984,308,1848,794),fill="#000000")
    fit_picture(im,BUILD/"poster.png",(984,308,1848,794))
    text(d,(984,815),"已有真实素材 · 静音画面配合本片旁白",25,MUTED)


def render_scene(s, index):
    im=Image.new("RGB",(W,H),BG)
    d=ImageDraw.Draw(im)
    d.rectangle((0,0,W,8),fill=BLUE)
    text(d,(72,42),"STAPIPE  /  WALKTHROUGH",25,BLUE,mono=True)
    text(d,(1500,42),f"{index+1:02d} / {len(SCENES):02d}",27,MUTED,mono=True)
    text(d,(72,102),s['title'],54,FG)
    text(d,(74,180),s['kicker'],29,AMBER)
    key=s['key']
    if key=='intro':
        card(d,(72,285,908,470),"RAW → FACTS → SAMPLE","原片 → 事实 → 配对数据",hs=42,bs=39)
        card(d,(72,500,908,710),"30 秒原片 → 7.292 秒目标","两张图片 + 一段参考视频 + 提示词",hs=39,bs=31)
        text(d,(80,765),"中文旁白 / 同步字幕 / 真实素材",29,MUTED)
        movie_box(im,d,"目标：搬沙发 + 车库卧推")
    elif key=='static':
        names=["理解媒体","目标候选","参考配对","媒体导出","提示词","质量检查"]
        for i,name in enumerate(names):
            x=72+i*300
            card(d,(x,320,x+268,464),f"{i+1:02d}",name,hs=31,bs=32)
            if i<5: arrow(d,(x+271,392),(x+294,392))
        card(d,(72,540,936,794),"模型负责判断内容","画面 / 声音 / 关系 / 描述\n输出结构化事实与证据",accent=BLUE)
        card(d,(972,540,1848,794),"程序负责控制流程","固定分支 / 有限重试 / 状态规则\n模型不能临时放宽标准",accent=AMBER)
    elif key=='xy':
        card(d,(72,272,900,806),"条件 x",accent=BLUE,hs=46)
        text(d,(106,365),"1  主角与包装参考图片\n2  戴墨镜人物参考图片\n3  办公室参考视频 + 启用音轨\n4  六段式目标提示词",35,FG,gap=34)
        card(d,(984,272,1848,650),"监督 y",accent=GREEN,hs=46)
        text(d,(1020,376),"target.mp4  真实目标画面\ntarget_audio.wav  同期波形",33,FG,gap=30)
        card(d,(984,683,1848,806),"oringin.mp4 只供溯源",accent=AMBER,hs=35)
    elif key=='source':
        card(d,(72,280,904,470),"720 帧 · 1280 × 720","24000/1001 FPS ≈ 23.976",hs=44,bs=31)
        card(d,(72,500,904,690),"真实音轨：44100 Hz 双声道","完整解码 / 每帧时间 / 文件哈希",hs=34,bs=31)
        text(d,(78,745),"同一作品 → 同一个 split_group",33,AMBER)
        movie_box(im,d,"raw_video：完整原片的真实画面")
    elif key=='overview':
        fit_picture(im,EX/"evidence/overview.jpg",(72,270,1848,735))
        for x,head,body in [(72,"固定采样","概览 + 切镜附近 + 视听窗口"),(682,"结构化事实","人物 / 动作 / 场景 / 声音"),(1292,"证据约束","看见动作 ≠ 已经听见声音")]:
            card(d,(x,760,x+556,887),head,body,hs=30,bs=23)
    elif key=='target':
        card(d,(72,280,908,452),"[200, 374)","包含第 200～373 帧，共 174 个源帧",hs=48,bs=29)
        card(d,(72,478,908,665),"8.341667 → 15.598917 秒","源区间时长：7.25725 秒",hs=35,bs=31)
        text(d,(78,711),"完整动作 → 候选合格\n找到有效参考 → 才能成为配对",33,GREEN,gap=20)
        movie_box(im,d,"已有样例的目标候选；不是新排名结果")
    elif key=='references':
        entries=[(72,'picture_1.png','Picture 1','主角 / 手指 / 产品','源帧 409 · 17.059 秒'),(682,'picture_2.png','Picture 2','脸 / 发型 / 墨镜','源帧 558 · 23.273 秒')]
        for x,fn,label,role,pos in entries:
            panel(d,(x,277,x+556,830))
            text(d,(x+25,300),label,35,BLUE)
            fit_picture(im,EX/'pure/inputs'/fn,(x+20,367,x+536,660))
            text(d,(x+25,697),role,31,FG)
            text(d,(x+25,756),pos,25,MUTED)
        panel(d,(1292,277,1848,830))
        text(d,(1317,300),"Video 1 + Audio 1",32,BLUE)
        fit_picture(im,BUILD/'reference.png',(1312,367,1828,660))
        text(d,(1317,697),"叙事结构 / 声音风格",30,FG)
        text(d,(1317,756),"源帧 [62, 200)",25,MUTED)
    elif key=='timeline':
        x=lambda t:100+t/30*1710
        d.line((x(0),461,x(30),461),fill=EDGE,width=5)
        for t in [0,5,10,15,20,25,30]:
            d.line((x(t),450,x(t),478),fill=MUTED,width=2)
            text(d,(x(t)-10,492),str(t),25,MUTED)
        d.rounded_rectangle((x(2.585917),340,x(8.341667),439),radius=10,fill="#176476")
        d.rounded_rectangle((x(8.341667),340,x(15.598917),439),radius=10,fill="#926335")
        text(d,(x(2.585917)+16,365),"参考视频",30,FG)
        text(d,(x(8.341667)+35,365),"目标视频",30,FG)
        for t,label in [(17.058708,'Picture 1'),(23.27325,'Picture 2')]:
            d.ellipse((x(t)-10,450,x(t)+10,470),fill=GREEN)
            d.line((x(t),439,x(t),351),fill=GREEN,width=3)
            text(d,(x(t)-80,303),label,28,GREEN)
        card(d,(72,585,900,820),"时间层面：左闭右开","[62, 200) 与 [200, 374) 不交叠\n参考图片允许来自目标之后",hs=36,bs=30)
        card(d,(972,585,1848,820),"内容层面：还要排除回放","同一主体的新动作可以参考\n重复播放目标镜头仍然拒绝",accent=AMBER,hs=36,bs=30)
    elif key=='frames':
        text(d,(78,266),"N = 17 × k + 5     |     24 FPS",46,BLUE,mono=True)
        for x,n,dur,desc,col in [(72,'158 帧','6.583333 秒','太短：会截掉内容',RED),(682,'175 帧','7.291667 秒','补 0.034417 秒',GREEN),(1292,'192 帧','8 秒','补 0.74275 秒，过长',RED)]:
            card(d,(x,395,x+556,679),n,dur+'\n'+desc,accent=col,hs=49,bs=31)
        text(d,(84,751),"补齐 0.474%  /  只重复末帧 + 末尾补静音",37,AMBER)
        text(d,(84,817),"不能读取目标区间外的新镜头",32,MUTED)
    elif key=='padding':
        card(d,(72,290,918,546),"参考：5.75575 → 5.875 秒","141 帧 / 补 0.11925 秒\n补齐比例 ≈ 2.072%",accent=BLUE,hs=40,bs=35)
        card(d,(970,290,1848,546),"默认配置：拒绝该参考","0.11925 ≤ 0.125 秒：通过\n2.072% > 2%：不通过",accent=RED,hs=40,bs=35)
        card(d,(72,602,1848,833),"教学配置：运行前显式设为 2.5%","为贯穿已有样例而演示；不是已校准的最佳门槛。\n默认规则不变，其他参考或图片组合仍可继续搜索。",accent=AMBER,hs=36,bs=31)
    elif key=='av':
        card(d,(72,284,920,533),"画面：保持比例","1280 × 720 → 1344 × 756\n上下各补 6 像素 → 1344 × 768",hs=42,bs=33)
        card(d,(970,284,1848,533),"波形：直接来自原片","32000 Hz / 双声道 / 16-bit PCM\n每声道 233333 个采样点",accent=GREEN,hs=42,bs=33)
        panel(d,(72,582,1848,833))
        text(d,(105,615),"S = round(175 / 24 × 32000) = 233333",40,BLUE,mono=True)
        text(d,(105,696),"视频与波形时长误差 ≈ 0.000010417 秒",35,FG)
        text(d,(105,757),"小于一个采样点；仍须核对实际音画内容",31,MUTED)
    elif key=='shots':
        sheet=Image.open(EX/'evidence/target_storyboard.jpg').convert('RGB')
        times=['0.000','1.208','2.125','3.667','4.667','5.542']
        for i,t in enumerate(times):
            col,row=i%3,i//3
            x,y=72+col*604,280+row*285
            tile=sheet.crop((round(col*sheet.width/3),round(row*sheet.height/2),round((col+1)*sheet.width/3),round((row+1)*sheet.height/2)))
            tile.thumbnail((550,212),Image.Resampling.LANCZOS)
            im.paste(tile,(x+(556-tile.width)//2,y))
            text(d,(x+14,y+215),f'Shot {i+1}  /  {t} s',29,BLUE,mono=True)
        text(d,(80,861),"历史切镜标注；实际新运行需在导出媒体上重新确认",24,AMBER)
    elif key=='labels':
        card(d,(72,280,632,845),"素材编号","Picture 1 / Picture 2\nVideo 1\nAudio 1 = 视频同步音轨",hs=39,bs=29)
        card(d,(680,280,1240,845),"主体编号","Subject 1：主角\nSubject 2：零食袋\nSubject 3：墨镜人物",accent=GREEN,hs=39,bs=29)
        card(d,(1288,280,1848,845),"发声编号","S1：搬运工\nS2：墨镜人物\n按首次实际发声排序",accent=AMBER,hs=39,bs=29)
        text(d,(708,676),"新规则的教学编号\n历史编号须全文同步更新",25,MUTED,gap=17)
    elif key=='prompt':
        panel(d,(72,277,952,859))
        headings=['subject_definitions','summary','retention_analysis','detailed_description','overall_soundscape','non_diegetic_music']
        for i,h in enumerate(headings):
            text(d,(104,312+i*78),f'{i+1}  {h}',29,BLUE if i!=3 else AMBER,mono=True)
        card(d,(1000,277,1848,526),"镜头时间由程序写入","[Shot 2] At 00:01.208, ...\n时间来自导出目标第 29 帧",hs=36,bs=30)
        card(d,(1000,566,1848,859),"未知对白保留未知","<d>[English] [unclear]</d>\nneeds_audio_review: true\n复核区间：[1.2, 2.6] 秒",accent=AMBER,hs=36,bs=29)
    elif key=='verify':
        for x,h,b,c in [(72,'程序检查','路径 / 标签 / 帧数\n采样数 / 时长 / 来源',BLUE),(682,'语义审查','真实画面与声音\n人物 / 动作 / 参考关系',GREEN),(1292,'文本修复 ≤ 1 次','已有事实支持才修复\n修复后重新审查',AMBER)]:
            card(d,(x,300,x+556,638),h,b,accent=c,hs=37,bs=32)
        text(d,(87,731),"可修：衣着文案、标签错误     拒绝：重复目标、关键动作截断",34,FG)
        text(d,(87,806),"没有做过的检查 → not_checked",34,RED)
    elif key=='status':
        card(d,(72,289,918,538),"needs_review","素材与记录已经形成\n仍有明确的声音不确定性",accent=AMBER,hs=49,bs=34)
        card(d,(970,289,1848,538),"completed","必需检查完成\n没有已知待复核事项",accent=GREEN,hs=49,bs=34)
        arrow(d,(922,414),(959,414),MUTED)
        card(d,(72,601,1848,843),"本例当前：不能进入严格训练集","解决未知对白 → 更新提示词和质量标记 → 重跑受影响检查\n本视频未新增人工听辨，也未运行 H3 推理或训练。",accent=RED,hs=38,bs=31)
    elif key=='train':
        card(d,(72,285,920,574),"条件分支","完整提示词 + 显式 references\n两张图 / 141 帧参考 / 启用音轨\n禁止扫描原片和目标作为条件",hs=39,bs=31)
        card(d,(970,285,1848,574),"监督分支","目标画面：[175, 3, 768, 1344]\n目标波形：[2, 233333]\n维度是解码层面的教学示意",accent=GREEN,hs=39,bs=31)
        card(d,(72,630,1848,843),"训练配置只收 124 帧？不能直接截短","Shot 6 从第 133 帧开始，会被截掉。\n匹配训练规格，或重新选段、导出和标注。",accent=AMBER,hs=38,bs=30)
    elif key=='replay':
        for x,h,b,c in [(72,'record','保存模型请求与响应\n记录配置、事实和检查',BLUE),(682,'replay','只读指定历史账本\n不重新请求模型',GREEN),(1292,'refresh','重新调用固定节点\n保留旧版，比较差异',AMBER)]:
            card(d,(x,290,x+556,591),h,b,accent=c,hs=44,bs=31)
        text(d,(82,649),"事实 → 目标与参考 → 精确导出 → 提示词 → 校验 → 训练入口",35,FG)
        text(d,(82,727),"模型判断内容，程序控制流程。",48,BLUE)
        text(d,(82,816),"配套文档：stapipe/WALKTHROUGH.md",28,MUTED)
    return im


def caption_frame(base, sentence, elapsed_ratio, cue_index, cue_count):
    im=base.copy()
    d=ImageDraw.Draw(im)
    d.rectangle((0,921,W,H),fill="#060D16")
    lines=wrap(d,sentence,font(39),W-160)
    if len(lines)>2:
        raise ValueError('Subtitle needs more than two lines: '+sentence)
    y=941 if len(lines)==2 else 962
    for line in lines:
        x=(W-d.textlength(line,font=font(39)))/2
        d.text((x,y),line,font=font(39),fill=FG)
        y+=52
    d.rectangle((0,1072,int(W*elapsed_ratio),1079),fill=BLUE)
    for i in range(cue_count):
        d.ellipse((1730+i*24,181,1740+i*24,191),fill=AMBER if i==cue_index else EDGE)
    return im


def speech(sentence, index):
    cache_key=hashlib.sha256(f'{VOICE}|{RATE}|{sentence}'.encode()).hexdigest()[:16]
    aiff=BUILD/f'voice_{cache_key}.aiff'
    wav=BUILD/f'voice_{cache_key}.wav'
    if not wav.exists():
        # All content is passed as argv, never interpolated into a shell.
        run(['say','-v',VOICE,'-r',RATE,'-o',str(aiff),sentence])
        run([FF,'-v','error','-y','-i',str(aiff),'-ar',str(SR),'-ac','1','-c:a','pcm_s16le',str(wav)])
    with wave.open(str(wav),'rb') as w:
        assert w.getframerate()==SR and w.getnchannels()==1 and w.getsampwidth()==2
        audio=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').copy()
    # Remove excessive system-voice trailing silence; preserve a small onset/offset margin.
    active=np.flatnonzero(np.abs(audio.astype(np.int32))>90)
    if not len(active):
        raise ValueError('Empty speech: '+sentence)
    audio=audio[max(0,active[0]-int(.06*SR)):min(len(audio),active[-1]+int(.13*SR))]
    return audio


def write_wav(path,audio):
    with wave.open(str(path),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(SR)
        w.writeframes(audio.astype('<i2').tobytes())


def stamp(seconds, srt=False):
    ms=round(seconds*1000)
    hh,ms=divmod(ms,3600000);mm,ms=divmod(ms,60000);ss,ms=divmod(ms,1000)
    return f'{hh:02}:{mm:02}:{ss:02}{"," if srt else "."}{ms:03}'


def extract_poster(source,path,at):
    run([FF,'-v','error','-y','-ss',str(at),'-i',str(source),'-frames:v','1',str(path)])


def main():
    BUILD.mkdir(parents=True,exist_ok=True)
    (HERE/'previews').mkdir(exist_ok=True)
    extract_poster(EX/'pure/targets/target.mp4',BUILD/'poster.png',2.5)
    extract_poster(EX/'pure/inputs/video_1.mp4',BUILD/'reference.png',3.0)
    movies={'target':EX/'pure/targets/target.mp4','raw':EX/'input.mp4'}
    global_frames=0
    subtitles=[]; chapters=[]; segments=[]; narr_md=['# 视频旁白与章节\n','本片为静态流水线设计的教学走读。中文配音由本机 macOS Tingting 合成。\n']
    # First generate all speech so global progress, chapters and captions align exactly.
    all_audio=[]
    for i,s in enumerate(SCENES):
        phrases=[]
        for j,sentence in enumerate(s['sentences']):
            a=speech(sentence,(i,j))
            gap=.85 if j==len(s['sentences'])-1 else .32
            frames=math.ceil((len(a)/SR+gap)*FPS)
            padded=np.zeros(frames*(SR//FPS),dtype=np.int16)
            padded[:len(a)]=a
            phrases.append((sentence,padded,frames,len(a)/SR))
        all_audio.append(phrases)
        print(f'Speech {i+1:02}/{len(SCENES)}: {s["title"]}',flush=True)
    total_frames=sum(q[2] for a in all_audio for q in a)
    for i,(s,phrases) in enumerate(zip(SCENES,all_audio)):
        base=render_scene(s,i)
        base.save(HERE/'previews'/f'{i+1:02}_{s["key"]}.png')
        concat=[]; aud=[]; scene_frames=0
        start=global_frames/FPS
        narr_md.append(f'## {stamp(start)}  {i+1:02} · {s["title"]}\n')
        for j,(sentence,a,frames,voice_dur) in enumerate(phrases):
            png=BUILD/f'scene_{i:02}_cue_{j:02}.png'
            caption_frame(base,sentence,(global_frames+frames)/total_frames,j,len(phrases)).save(png)
            concat.append(f"file '{png.as_posix()}'\noption framerate {FPS}\nduration {frames/FPS:.9f}\n")
            subtitles.append((global_frames/FPS,global_frames/FPS+voice_dur,sentence))
            narr_md.append(sentence+'\n')
            aud.append(a);scene_frames+=frames;global_frames+=frames
        concat.append(f"file '{png.as_posix()}'\noption framerate {FPS}\n")
        listing=BUILD/f'scene_{i:02}.ffconcat';listing.write_text('ffconcat version 1.0\n'+''.join(concat))
        wav=BUILD/f'scene_{i:02}.wav';write_wav(wav,np.concatenate(aud))
        seg=BUILD/f'scene_{i:02}.mp4'
        args=[FF,'-v','error','-y','-f','concat','-safe','0','-i',str(listing),'-i',str(wav)]
        if s['movie']:
            args+=['-stream_loop','-1','-i',str(movies[s['movie']])]
            graph=f'[0:v]fps={FPS},format=yuv420p[base];[2:v]setpts=PTS-STARTPTS,fps={FPS},scale=864:486:force_original_aspect_ratio=decrease,pad=864:486:(ow-iw)/2:(oh-ih)/2:black,setsar=1[clip];[base][clip]overlay=984:308:shortest=1[v]'
        else:
            graph=f'[0:v]fps={FPS},format=yuv420p[v]'
        args+=['-filter_complex',graph,'-map','[v]','-map','1:a:0','-frames:v',str(scene_frames),'-t',f'{scene_frames/FPS:.9f}',
               '-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-r',str(FPS),
               '-video_track_timescale','24000','-c:a','aac','-b:a','160k','-ar',str(SR),'-ac','2','-movflags','+faststart',str(seg)]
        print(f'Render {i+1:02}/{len(SCENES)}: {scene_frames/FPS:.1f} s',flush=True)
        run(args)
        chapters.append(dict(index=i+1,title=s['title'],start_seconds=start,end_seconds=global_frames/FPS,frames=scene_frames,movie=s['movie']))
        segments.append(seg)
    if LAYOUT_WARNINGS:
        (HERE/'layout_warnings.json').write_text(json.dumps(LAYOUT_WARNINGS,ensure_ascii=False,indent=2))
        raise ValueError(f'Layout overflow: {LAYOUT_WARNINGS}')
    final_list=BUILD/'segments.txt';final_list.write_text(''.join(f"file '{p.as_posix()}'\n" for p in segments))
    metadata=[';FFMETADATA1','title=Static Ref2VA Walkthrough','comment=Chinese synthetic narration; existing media examples; pipeline design walkthrough, not a new pipeline run.']
    for ch in chapters:
        metadata+=['[CHAPTER]','TIMEBASE=1/24000',f'START={round(ch["start_seconds"]*24000)}',f'END={round(ch["end_seconds"]*24000)}',f'TITLE={ch["index"]:02} {ch["title"]}']
    meta=BUILD/'chapters.ffmeta';meta.write_text('\n'.join(metadata)+'\n')
    # Keep a single sample-aligned narration track; copying segmented AAC could add gaps.
    whole_wav=BUILD/'narration.wav'
    write_wav(whole_wav,np.concatenate([q[1] for a in all_audio for q in a]))
    run([FF,'-v','error','-y','-f','concat','-safe','0','-i',str(final_list),'-i',str(whole_wav),'-i',str(meta),
         '-map','0:v:0','-map','1:a:0','-map_metadata','2','-map_chapters','2','-c:v','copy','-af','loudnorm=I=-18:TP=-2:LRA=9',
         '-c:a','aac','-b:a','160k','-ar',str(SR),'-ac','2','-t',f'{total_frames/FPS:.9f}','-movflags','+faststart',str(OUT)])
    (HERE/'walkthrough.srt').write_text('\n'.join(f'{i}\n{stamp(a,True)} --> {stamp(b,True)}\n{t}\n' for i,(a,b,t) in enumerate(subtitles,1)))
    (HERE/'narration.md').write_text('\n'.join(narr_md))
    (HERE/'chapters.json').write_text(json.dumps(chapters,ensure_ascii=False,indent=2))
    manifest={'output':str(OUT.relative_to(STAPIPE)),'duration_seconds':total_frames/FPS,'frames':total_frames,'resolution':[W,H],
              'fps':FPS,'voice':VOICE,'speech_rate':RATE,'chapters':len(chapters),'subtitle_cues':len(subtitles),'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),
              'source_sha256':hashlib.sha256((EX/'input.mp4').read_bytes()).hexdigest(),'ffmpeg_version':run([FF,'-version']).stdout.splitlines()[0],
              'scope':'Teaching walkthrough; original example audio uncertainty remains unresolved; no pipeline, model training or H3 inference performed.'}
    (HERE/'build_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
