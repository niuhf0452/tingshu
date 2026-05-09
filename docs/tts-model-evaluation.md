# TTS 模型选型结论

## 架构决策路径

原始方案：**iOS 端本地实时合成**。
中止原因：**所有跑得起来的本地 TTS 模型音质都不达标**——发音机械、韵律平淡，不能满足"听有声书"的体验门槛。
这不是工程问题，是 iPhone 平台硬限制（无独立 GPU、可用 RAM 4-6 GB、Neural Engine 11-17 TOPS）下能跑的最大模型 ~80M 参数所决定的天花板。

转向方案：**Mac Mini M4 离线预合成 + iOS 端读取播放**。
本文档记录两个阶段的选型结论，重点在**每个模型为什么不行**或**为什么入选**。

| 阶段 | 部署位置 | 选定模型 |
|------|---------|---------|
| 已放弃 | iOS（实时合成） | — |
| 当前 | Mac Mini M4（离线预合成） | **Qwen3-TTS-12Hz-0.6B-Base-4bit** + **Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit** |

---

## 第一阶段：iOS 端本地合成（已放弃）

### 共有结论

**所有能在 iOS 上跑起来的中文 TTS 模型，音质都达不到有声书要求。**

- iPhone 12+ 可用 RAM 4-6 GB，Neural Engine 算力 11-17 TOPS。
- 这个算力预算下能加载的最大模型在 ~80M 参数量级。
- 80M 量级的中文 TTS 自然度在"能听懂"和"机械感明显"之间，无法接近真人朗读。
- 调参（noiseScale / noiseScaleW / lengthScale）改善有限，是模型容量上限的问题，不是参数问题。

→ 这一发现直接驱动了"放弃 iOS 端实时合成、改走服务端预合成"的架构转向。

### 实测模型记录

#### vits-zh-hf-fanchen-C

- 116 MB onnx + 187 个音色（数字 ID `0-186`，**无性别/年龄元数据**）。
- 训练数据：AISHELL-3 扩展集。
- **问题**：发音机械感最强，韵律平淡，句号/逗号停顿生硬。
- **问题**：187 个音色全是数字 ID，必须人工逐个试听才能选用。

#### kokoro-int8-multi-lang-v1_1（INT8 量化版）

- 109 MB onnx + 51 MB voices.bin，103 个音色（其中 100 个中文，`zf_*` 女声 / `zm_*` 男声）。
- **阻断 bug**：iPhone 17 模拟器上 ONNX Runtime 推理 INT8 模型时输出 PCM samples **全部为 NaN**：
  ```
  TTS: got 93600 samples, sampleRate=24000, range=[nan, nan]
  ```
  真机是否正常未验证，但模拟器调试不可用本身就是阻塞性问题。
- 结论：iOS 上 INT8 量化模型不可用，必须用全精度。

#### kokoro-multi-lang-v1_1（全精度版）

- **311 MB** onnx + 51 MB voices.bin + 60 MB espeak-ng-data，82M 参数，24 kHz 采样率。
- 跑得起来，**iOS 本地能用的最好模型**。
- 音色有性别标注（`zf_*` / `zm_*`），可按角色性别分配。
- **问题**：音质仍偏机械，对话情绪表达平淡。模拟器上 RTF ~1.5（一句 4 秒音频要 ~6 秒合成），勉强能用但不流畅。
- 即使把 iOS 本地方案推到当前技术上限，也只能做到这个水平 → 触发架构转向。

### CosyVoice 300M（属性级阻断，未实测）

| 维度 | 数值 |
|------|------|
| 磁盘大小 | **1.74 GB**（llm.pt 1.24 GB + flow.pt 420 MB + hift.pt 82 MB） |
| 推理 RAM | ~3-5 GB（三组件同时驻留） |
| iPhone 12+ 可用 RAM | 4-6 GB |
| iOS 集成路径 | **无**——无完整 ONNX、Sherpa-ONNX 不支持、CoreML 无方案 |

任一维度都阻断 iOS 部署。CosyVoice 在产品里的位置只能是服务端，但服务端我们也没选它（见下一阶段）。

---

## 第二阶段：服务端选型（Mac Mini M4）

### 硬指标

1. **本地推理**——不依赖云 API（成本、隐私、限流、离线场景都是阻断条件）。
2. **Apple Silicon 友好**——CUDA-only 模型出局；FP16 必须真正在 Metal 上生效。
3. **零样本克隆**——60+ 角色音色不可能逐个训练，必须支持"5–15 s 参考音频 + 转写"克隆。
4. **RTF ≤ 1**（理想 ≤ 0.6）——批量预生成长篇小说必须在合理时间内完成。
5. **中文质量**——朗读小说，机械感强不可接受。
6. **情绪可控**——至少要能用自然语言指令影响合成情绪。

### ⛔ CosyVoice 300M（实测过，性能不达标）

**为什么看好**：阿里达摩院出品，中文质量业界第一梯队，支持零样本克隆。
**首版方案就是它**——写了完整 production 代码（`server/app/services/tts_cosyvoice.py` + `tts_cosyvoice_test.py`，已随选型清理），通过 `scripts/annotate_voices.py --preset sft-all` 跑通：加载 + 7 条 SFT 音色合成 + Gemini 标注。

**阻断性问题**：

1. **FP16 推理只对 CUDA 生效**：在 Mac 上加载时 `fp16=True` flag **静默回落为 FP32**，loader 不报错，模型直接吃 ~2× 显存。源码里写死了 `if torch.cuda.is_available() is False and fp16 is True: fp16 = False`。
2. **CPU/MPS 推理 RTF ≈ 2.6**：M4 实测单句合成 25-30 s（含 Gemini 往返），离生产要求 RTF ≤ 0.6 差**一个数量级**。
3. **官方无现成单文件 FP16/INT8 包**：ModelScope 上的 FP16 是按子模块拆分的 zip，需自己组装且未经官方支持。

→ 在 Apple Silicon 上要么改源码、要么吃 FP32 显存且性能不达标，工程税不值得为单点质量优势付。

### ⛔ CosyVoice2 / Fun-CosyVoice3-0.5B（mlx-audio 分支冲突）

CosyVoice2/3 在 HF 上没有预转换的 MLX 权重；要走 [mlx-audio-plus](https://github.com/Trans-N-ai/swama) 分支才有 `cosyvoice3` 模型类。

**致命问题**：mlx-audio-plus 与上游 `Blaizzy/mlx-audio` **共用 `mlx_audio` import path 互相覆盖**。要测 CosyVoice3 必须 `pip uninstall mlx-audio && pip install mlx-audio-plus`，会污染已经走通的 Qwen3-TTS 路径。每次模型升级还要重做权重转换。

→ 与 Qwen3-TTS 路径互斥，维护成本高于潜在收益。

### ⛔ Voxtral TTS（Mistral AI）

方向是低延迟对话，公开 demo 主要是英文短句；没有面向中文小说朗读的稳定性 / 韵律实测。
社区量化版本几乎没有，mlx-audio 集成路径不成熟。

→ 朗读长文本场景的可信度不足。

### ⛔ VoxCPM2-4bit（实测过，集成路径不通）

OpenBMB 出品，0.5B 量级，社区有 `mlx-community/VoxCPM2-4bit` 等 MLX 包。
下载 ~1.5 GB + 写 `scripts/test_voxcpm_bleed.py` 做零样本克隆 bleed 复现测试。

**阻断性问题**：

1. **mlx-audio 不识别 voxcpm2**：v0.4.2/0.4.3 的 `MODEL_REMAPPING` 只认 `voxcpm` / `voxcpm1.5`。Monkey-patch 让 v2 走 v1 loader 也加载失败。
2. **改装 OpenBMB 官方 voxcpm2 模块后**，参数命名跟 HuggingFace README 对不上 —— OpenBMB 包是 `prompt_wav_path` + `generate_audio()`，mlx-audio 包是 `prompt_audio` + `generate()`；按多种组合调参输出始终不正确。
3. **集成路径不稳定**：要稳定上生产必须自己 fork loader 或贴身跟 OpenBMB 上游同步，工程成本远高于"一个备选模型"应有的投入。

→ 不是音质问题，是上下游集成成本高于回报。

> **VoxCPM 1.5-fp16**：VoxCPM2 集成受阻时建议过同体量的 1.5 版作为替代，被否决（不降级到老版本）。下了 ~90 MB 即清理。

### ⛔ VibeVoice（同 ICL 范式，未实测）

微软出品，主打"表现力 / 情绪"，社区有 `mlx-community/VibeVoice-Realtime-0.5B-4bit` 等量化包。
架构层调研发现走 in-context learning（同 ref+target 拼接），与 Qwen3-TTS 同范式 → bleed bug 风险同样存在；中文长文本稳定性也无可信 benchmark。

→ 同范式不解决 bleed，且文档与社区生态不足以承担生产部署。

### ⛔ Qwen3-TTS-12Hz-0.6B-CustomVoice（架构层不匹配，未实测）

Qwen 团队的"自定义音色"变体。从模型卡 + 官方示例可以看出：**API 不接受 `ref_audio`**，只能用 checkpoint 内训好的预设音色 + `instruct` 参数调情绪。

**架构层判断**：
- 优点：从根上避免 bleed bug（无 ref/target 拼接）。
- 阻断性缺点：只有 ~9 个中文 speaker，**撑不住 30+ 角色的中型小说**，听感会重复。

→ 单旁白朗读场景适合，但本项目的多角色定位下损失音色多样性的代价更高。

### ⛔ 同 ICL 范式批量调研：F5-TTS / ChatterBox / Spark-TTS / Bark / XTTS

调试 Qwen3-TTS bleed bug 时（详见 §"沉淀知识"），系统调研了所有走"参考文本 + 目标文本拼接"的 in-context-learning 类 TTS。结论是 bleed 是**整个 ICL 范式的固有风险**，并不是 Qwen3-TTS 独有：

| 模型 | bleed 风险 | 备注 |
|------|----------|------|
| CosyVoice 系列 | ★★★★★ | 同 ref+target 拼接，训练量大可能更稳 |
| F5-TTS / ChatterBox / Spark-TTS | ★★★ | 同类 |
| VoxCPM 系列 | ★★★ | 同类，diffusion 解耦稍好 |
| VibeVoice | ★★★ | 同 ICL 家族 |
| Bark | ★★ | 训练范式略不同 |
| XTTS-v2 (single-speaker) | – | speaker-embedding 类，无 bleed，但克隆保真度通常差一档 |

**真正消除 bleed 只有放弃 ICL**（CustomVoice 预设音色 / 训练 LoRA / 厂商云 TTS），代价是失去音色多样性。

→ 不切换。换到同范式的其他模型不解决问题，只是换一个 bleed 概率分布；通过 ref_text 工程化（短/平/软尾）已能压住。

### ✅ Qwen3-TTS-12Hz-0.6B-Base-4bit ⭐ 选定（运行时引擎）

| 项目 | 值 |
|------|---|
| 参数量 | 0.6B |
| 磁盘大小 | ~1.6 GB（MLX 4bit DWQ） |
| RTF（M4 实测） | **~0.6** |
| API | `model.generate(text, ref_audio, ref_text, instruct=自然语言情绪)` |

**关键实测发现**：

- 64 条参考音频逐条克隆，**音色保真度高**，长辈/少年/女声/男声区分明显。
- `instruct` 实测有效：传入"愤怒、激烈"等中文指令，输出有可识别的情绪倾向。
- 触发了 **bleed bug**（详见 §"沉淀知识"），不是模型缺陷而是 ref_text 设计问题，可工程规避。

→ 硬指标全部满足，是唯一实测胜出者。

### ✅ Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit ⭐ 选定（音色库生成工具）

根据自然语言描述直接合成虚拟音色，不需要参考音频。

| 项目 | 值 |
|------|---|
| 参数量 | 1.7B |
| 磁盘大小 | ~1.7 GB（MLX 8bit；目前没有 0.6B 也没有 4bit 构建） |
| API | `model.generate_voice_design(text, instruct=自然语言描述)` |
| 用途 | **离线一次性生成参考音频**，不参与运行时 |

**关键实测发现**：

- 用 64 条覆盖 (gender × age × personality) 矩阵的中文描述，一次性生成 64 条 ~10 s 的 wav。
- **输出非确定性**：同一描述跑两次"虚拟音色"略有差异 → 必须冻结到磁盘，否则同一角色每次播放音色都漂移。
- prompt 设计模式：负向对比 framing + 克制类形容词 + 具体意象；不要堆叠"表现力"词语。

→ 解决了零样本克隆"必须先有参考音频"的供应链瓶颈，让我们能用一段中文描述凭空生成参考音色。
开发期跑一次（几分钟），上线后扩充音色库时再跑。

### 服务端最终架构

```
┌──────────── 开发期 / 扩充音色库时 ────────────┐
│  Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit          │
│       ↓ generate_voice_design                  │
│  data/voices/prompts/vd_<tag>.wav + .txt × 64  │
└────────────────────────────────────────────────┘
                  ↓ 持久化、冻结
┌──────────── 运行时（每次 /api/tts） ───────────┐
│  Qwen3-TTS-12Hz-0.6B-Base-4bit                 │
│       ↑ ref_audio = vd_<tag>.wav               │
│       ↑ ref_text  = vd_<tag>.txt               │
│       ↑ instruct  = 句子 tone 对应的中文指令    │
│  → 24 kHz mono WAV 写入 .tsb                   │
└────────────────────────────────────────────────┘
```

**为什么是这个分工**：VoiceDesign 输出非确定性，作为运行时引擎会音色漂移；
Base 模型小、快，零样本克隆已胜任运行时。

---

## 沉淀知识

### Reference text bleed bug（适用于所有 Qwen3-TTS 类零样本克隆）

**症状**：克隆输出尾部带上 ref_text 末尾的几个字。
例如目标句"今天天气真好。"，输出变成"今天天气真好。**只有强者，和弱者**"。

**机制**：零样本克隆在 token 层把 `[ref_text_tokens] + [target_text_tokens]` 拼接，**没有强分隔符**。
当 ref_text 较长 / 戏剧化 / 用硬结尾标点（`。` `！`）时，模型把"参考朗读还没结束"作为先验信号，
合成 target 后**继续朗读 ref_text 的尾巴**。本质类比 LLM 的 prompt injection / stop sequence 缺失。

**触发概率（标点排序）**：
1. **U+2026 `…`**（水平省略号）—— 最高
2. `。` `！` `？` —— 中等
3. `，` `、` —— 几乎不触发

**修复策略**：

1. ref_text 改写为短、平、温和结尾的句子（避免戏剧化长句）。
2. ref_text 末尾用"……"软结尾代替"。"硬结束。
3. 不让多条音色共用同一段 ref_text。
4. **不在目标文本动手**（保持原文朗读自然），只在 ref_text 上做软化。

### `instruct` 参数 vs 文本前缀

把"【愤怒】"等语气描述拼到 `text` 字段前缀，模型会**当成需要朗读的文字直接念出来**。
正确做法：用独立的 `instruct` 参数传入自然语言情绪指令。

### `linear="no"` spine items / EPUB 目录页

EPUB 解析阶段曾把"目录"页当成普通章节，导致播放第 1 章是一长段章节标题列表的朗读。
不是 TTS 模型问题，但表现为 TTS 出错。修复见 `server/app/core/parsers/epub.py` 的 `_is_toc_page` 启发式。

### iOS Sherpa-ONNX 集成（即使本地合成方案已弃用，工程结论仍有价值）

如果未来在 iOS 端追加任何本地 TTS（如离线降级模式）时复用：

- **采样率必须动态匹配** AVAudioEngine 节点连接：fanchen-C 16 kHz、Kokoro 24 kHz；硬编码会导致播放速度异常或杂音。
- **音量幅值因模型而异**：fanchen-C 输出偏小要 `volume ≈ 4.0`，Kokoro 输出正常 `volume = 1.0` 即可。不要硬编码增益。
- **Lexicon `Unknown token` 警告可忽略**：例如 Kokoro `lexicon-zh.txt` 中"呣"标了 `❓` 但 `tokens.txt` 没定义。
- **输入文本必须 sanitize**：模型对 emoji / 特殊 Unicode 无映射，合成前要过滤。
- **INT8 模型在 iOS 模拟器上不可用**：ONNX Runtime + 模拟器组合下 INT8 推理输出 NaN，必须用全精度。

---

## 不会再考虑的选项

- **iOS 端任何 ≥ 0.5B 参数的本地模型**——磁盘 + RAM + 算力三重阻断，平台限制非工程问题。
- **任何 CUDA-only 或显存 ≥ 16 GB 的服务端模型**——Mac Mini M4 unified memory 必须给 LLM 调用 / OS / 其他 daemon 留余量。
- **训练 SFT 音色**——每条音色至少需要 10-30 分钟干净录音，60+ 角色无法承担。
- **Cloud TTS API**（豆包 / 腾讯云 / Azure / MiniMax 等）——per-character 计费随业务量爆炸，
  且失去离线 / 隐私 / 角色定制核心卖点。**唯一例外**：未来若引入"高音质付费可选模式"，候选首选腾讯云（800 万字免费额度）或火山引擎（音质最好）。

---

## 何时重新评估

### 服务端

- 出现明显比 Qwen3-TTS 更稳定的中文 zero-shot 克隆开源模型，**且有可用的 MLX 量化包**。
- VoiceDesign 出 0.6B / 4bit 量化版本（让运行时也能用 VoiceDesign，"按描述创建角色"）。
- Qwen3-TTS 出原生 streaming 输出 API（降低长句首字节延迟）。

### iOS 端实时合成（现在已放弃）

只有以下任一条件成立才值得回头再看：

- Apple 出 ANE 优化的、支持任意音色的 TTS 模型（类似 Speech.framework 但开放音色）。
- 出现真正适合 iOS 的 0.5B+ 中文克隆模型，**且实测 RAM ≤ 1 GB / 真机 RTF ≤ 1 / 音质能听**。
- iPhone 硬件迭代到可用 RAM ≥ 8 GB 同时 Neural Engine 算力翻倍。

否则继续走"Mac M4 预合成 + iOS 端读取"路线。
