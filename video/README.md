# Walkthrough 讲解视频

成片位于 [walkthrough.mp4](../walkthrough.mp4)。它按照 [WALKTHROUGH.md](../WALKTHROUGH.md) 讲解完整流程，采用 1920 × 1080、24 FPS（frames per second，每秒帧数）、H.264 视频和 AAC（Advanced Audio Coding）双声道音频。

视频包含 18 个章节、中文合成旁白、烧录字幕和真实原片/目标片段画面。原广告声音不混入旁白；声音复核问题仍按文档标记为未解决。成片是设计走读，不代表已经执行新流水线、完成新的人工听辨或运行训练。

## 配套文件

- [旁白与章节时间](narration.md)
- [独立字幕](walkthrough.srt)，SRT（SubRip Subtitle）格式
- [章节数据](chapters.json)，也已写入 MP4 容器
- [生成清单](build_manifest.json)：时长、帧数、配音参数和媒体哈希
- [验证结果](verification.json)：实际成片解码、媒体参数、字幕范围与声音信号检查
- [制作脚本](make_walkthrough_video.py)
- `previews/`：讲解画面与用于检查的预览；`build/`：语音缓存和编码中间文件

## 重新生成

脚本采用 macOS 的本地 `say -v Tingting` 配音，依赖系统中文字体和 Python 的 Pillow、numpy、imageio-ffmpeg。首次制作使用 Pillow 12.3.0、numpy 2.5.3、imageio-ffmpeg 0.6.0；FFmpeg 版本记录在生成清单中。

在仓库根目录使用当前已有环境运行：

```bash
ref2va-workspace/.venv/bin/python stapipe/video/make_walkthrough_video.py
```

也可以使用安装上述依赖的独立 Python 环境。脚本通过自身位置定位素材，不依赖启动目录；视频素材来自 `ref2va-workspace/examples/`，只读使用。语音按旁白、声音和语速内容哈希缓存。

修改脚本中的 `SCENES` 可以调整章节标题、旁白和画面类型，`render_scene()` 控制布局。重新运行会覆盖本教程的成片和制作产物，不会改动原片、样例数据或流水线设计。跨系统版本重新合成声音或编码时，不保证视频文件逐字节一致。

本片将默认 2% 与教学 2.5% 补齐限制分别说明，并保留样例的 `needs_review` 状态；不要把教学配置当成已完成校准的新默认值。
