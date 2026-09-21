# photo-poem-blank 照片填空诗

把任意一张照片做成「填空小诗」竖版视频：先读懂照片里的具体细节和情绪，写一首 3–4 行的小诗，再把 5–7 个局部从照片里抠出来，飞进诗句的空格里。切片原本的位置留一个半透明取景框，照片始终完整可见。

输出 1080×1440 竖版视频（含配乐与音效）、封面图、成品静帧。

## 目录结构

```
SKILL.md                  技能说明与工作流（先读这个）
agents/openai.yaml        技能元信息
references/poem-writing.md  读照片与写诗的规则
references/visual-spec.md   版式、动效参数与设计理由
references/audio-spec.md    配乐与音效结构
scripts/poem_video.py       主流程：渲染视频与封面
scripts/inspect_photo.py    取景辅助：坐标网格、候选切片对比图
scripts/score.py            配乐/音效合成与混音
scripts/selftest.py         端到端自检
scripts/example_config.json 配置模板
```

## 依赖

`python3`（Pillow、numpy、scipy）与 `ffmpeg`。字体自动在系统里找中文字体，找不到就在配置里指定 `font.path`。

读图会自动处理 EXIF 旋转、透明通道，并在 Pillow 读不了时回退到 `sips`（macOS）/`ffmpeg` 转码，因此 iPhone 的 HEIC、AVIF 等也能直接喂进来。

## 快速开始

```bash
# 0. 环境自检
python3 scripts/selftest.py

# 1. 读照片，挑出 6–8 个候选细节
python3 scripts/inspect_photo.py photo.jpg --grid
python3 scripts/inspect_photo.py photo.jpg --boxes "花:690,302,92,60;牌:100,62,150,70"

# 2. 复制 scripts/example_config.json 改成本片的配置（图片、lines、slices）
python3 scripts/poem_video.py build config.json
python3 scripts/poem_video.py stills config.json --times 2.6,9.6
```

`build` 会打印版式体检（每行宽度、每块切片尺寸、原位取景框坐标、总时长），有问题的项以 `!` 报出。渲染不落盘，每帧合成后直接走管道喂给 ffmpeg。

完整工作流、硬规则与不可改的默认效果见 [SKILL.md](SKILL.md)。
