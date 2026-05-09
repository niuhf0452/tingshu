# 听书 App 技术方案

## Context

开发一款 iOS 听书 App，用户导入 TXT/EPUB 文件，通过云端 AI TTS 多角色朗读。架构为 C/S 模式：Mac Mini M4 作为服务端，负责预处理和 TTS 合成；iOS App 作为客户端，负责书架管理、正文阅读和音频播放。

## 系统架构

```
┌─────────────── Mac Mini M4（服务端）──────────────────┐
│                                                       │
│  API 服务                                             │
│  ├── 书籍管理（导入、列表、下载、删除）                 │
│  ├── 预处理（渐进式按需处理）                           │
│  │   ├── 书籍解析（TXT/EPUB → 章节纯文本）             │
│  │   └── 元数据生成（断句 + 归因 + 语气 + 角色画像，   │
│  │       一次 LLM 调用产出两段 NDJSON）                │
│  └── TTS 合成（Qwen3-TTS 0.6B MLX，零样本克隆）       │
│                                                       │
│  LLM：DeepSeek V4 flash（云，OpenAI 兼容协议）         │
│  TTS：Qwen3-TTS 0.6B MLX 4bit（本地）                 │
└───────────────────────┬───────────────────────────────┘
                        │ HTTP API
┌───────────────────────▼───────────────────────────────┐
│              iOS App（客户端）                          │
│                                                       │
│  书架管理 → 导入书籍 → 上传服务端                       │
│  正文阅读 → 句级高亮 → 跟读/自由浏览                   │
│  音频播放 → 调用服务端 TTS API → 预加载 + 播放          │
│                                                       │
│  技术栈：SwiftUI + AVAudioEngine                      │
└───────────────────────────────────────────────────────┘
```

---

## 一、预处理文件格式

每本书预处理后对应一个目录：

```
book_<id>/
├── meta.json                     ← 书籍元数据（导入后一次性生成，永不变更）
├── characters.json               ← 全书累积角色表（仅服务端内部，不对外）
├── chapters/
│   ├── 0001.txt                  ← 第 1 章正文（导入后一次性生成）
│   ├── 0001.json                 ← 第 1 章元数据（按需 lazy 生成；
│   │                                含句子 + 本章角色快照）
│   ├── 0002.txt
│   ├── 0002.json
│   └── ...
```

**数据同步边界**：

- **服务端持有全量**：`meta.json` + `characters.json`（内部）+ 所有
  `chapters/*.txt` + 已生成的 `chapters/*.json`
- **下载到 App 端**：`meta.json` + 所有 `chapters/*.txt`（不含章节元
  数据，**不含 characters.json**）
- **章节元数据**：App 端按需 `GET /api/books/{id}/chapters/{ch}/meta`，
  返回的 JSON 同时包含本章句子 + 本章涉及角色的画像快照
- **App 端不需要 characters.json** —— 它读章节 meta 内嵌的快照就够
  了；characters.json 是服务端用来在 LLM prompt 里给 hint + 维护跨
  章节 id 一致性的内部数据

### meta.json（书籍元数据，导入后不变）

```json
{
  "version": 1,
  "book_id": "a1b2c3d4",
  "title": "斗破苍穹",
  "author": "天蚕土豆",
  "cover": "cover.jpg",
  "summary": "一个天才少年的修炼之路...",
  "chapters": [
    {"id": 1, "title": "第一章 陨落的天才", "text_file": "chapters/0001.txt", "meta_file": "chapters/0001.json"},
    {"id": 2, "title": "第二章 斗之气三段", "text_file": "chapters/0002.txt", "meta_file": "chapters/0002.json"}
  ],
  "status": "ready",
  "source_filename": "斗破苍穹.txt"
}
```

`meta.json` 在导入完成时一次性写入，**之后永不变更**（除了 status 从
processing 一次性翻到 ready）。App 端下载一次即终局。

### characters.json（服务端内部，不暴露）

全书累积的角色表，每章分析后增量更新：

```json
[
  {"id": 0, "name": "旁白", "identity": "", "gender": "neutral", "age": "adult", "personality": ["calm"]},
  {"id": 1, "name": "萧炎", "identity": "少年弟子，主角，手持玄重尺", "gender": "male", "age": "teen", "personality": ["determined", "brave"]},
  {"id": 2, "name": "萧薰儿", "identity": "萧家千金，温柔聪慧", "gender": "female", "age": "teen", "personality": ["gentle", "kind"]},
  {"id": 3, "name": "药老", "identity": "萧炎的师父，魂体形态，精通炼药", "gender": "male", "age": "elder", "personality": ["calm", "wise"]}
]
```

用途与生命周期：
- 章节分析时作为 prompt hint 喂给 LLM（让它复用已知名字、做别名推理、
  判断已知角色是否需要更新画像 —— 详见 §2.3）
- 维护跨章节角色 id 一致性
- LLM 输出的画像更新（新增角色或已知角色演变）合并回这个表
- **不通过 API 暴露**，不进 download zip
- App 端没有"全书角色表"概念 —— 章节 meta 里嵌入的快照就是 App 看到
  的全部角色信息

### 性别（gender）预定义取值

| 值 | 说明 |
|----|------|
| male | 男 |
| female | 女 |
| neutral | 中性（旁白可用，或性别不明的角色） |

### 年龄（age）预定义取值

| 值 | 说明 | 典型范围 |
|----|------|---------|
| child | 儿童 | ~12 岁以下 |
| teen | 少年 | ~13-19 岁 |
| youth | 青年 | ~20-35 岁 |
| adult | 中年 | ~36-55 岁 |
| elder | 老年 | ~56 岁以上 |

### 性格（personality）预定义取值

性格为标签数组，可同时标记多个特征。预定义标签库：

| 值 | 说明 |
|----|------|
| calm | 沉稳 |
| gentle | 温柔 |
| cheerful | 开朗 |
| serious | 严肃 |
| cold | 冷淡 |
| fierce | 凶悍 |
| determined | 坚定 |
| timid | 怯懦 |
| playful | 活泼 |
| mature | 成熟 |
| naive | 天真 |
| wise | 智慧 |
| arrogant | 傲慢 |
| kind | 善良 |
| cunning | 狡黠 |
| brave | 勇敢 |
| melancholy | 忧郁 |
| passionate | 热情 |

### chapters/0001.json（章节元数据）

包含两部分：本章句子（不重复正文，仅记录位置 + 归因 + 语气）和**本
章涉及角色的画像快照**（带画像数据，App 端音色匹配用）。

```json
{
  "sentences": [
    {"start_line": 1, "start_col": 0, "end_line": 1, "end_col": 8, "character_id": 0, "tone": "neutral"},
    {"start_line": 1, "start_col": 8, "end_line": 1, "end_col": 24, "character_id": 0, "tone": "neutral"},
    {"start_line": 3, "start_col": 0, "end_line": 3, "end_col": 12, "character_id": 1, "tone": "angry"},
    {"start_line": 5, "start_col": 0, "end_line": 5, "end_col": 10, "character_id": 2, "tone": "gentle"}
  ],
  "characters": [
    {"id": 0, "name": "旁白", "identity": "", "gender": "neutral", "age": "adult", "personality": ["calm"]},
    {"id": 1, "name": "萧炎", "identity": "少年弟子，主角，手持玄重尺", "gender": "male", "age": "teen", "personality": ["determined", "brave"]},
    {"id": 2, "name": "萧薰儿", "identity": "萧家千金，温柔聪慧", "gender": "female", "age": "teen", "personality": ["gentle", "kind"]}
  ]
}
```

`characters[]` 是**本章分析时的画像快照**（值，不是引用）：只包含本章
出现的角色，画像在写入时就被冻结。即便后续章节里 LLM 把萧炎升格为
youth，重新读取本章 meta 拿到的还是 teen 的快照 —— 章节内一致 +
跨章自然过渡。

### 语气预定义列表

程序预定义语气列表，不绑定具体模型，能够映射到 TTS 模型参数：

| 语气 ID | 名称 | 描述 | 适用场景 |
|---------|------|------|---------|
| neutral | 平静 | 默认语气 | 旁白、陈述 |
| happy | 高兴 | 愉悦、开心 | 喜悦场景 |
| sad | 悲伤 | 低沉、忧郁 | 悲伤场景 |
| angry | 愤怒 | 激烈、强硬 | 争吵、愤怒 |
| fearful | 恐惧 | 紧张、害怕 | 惊恐场景 |
| surprised | 惊讶 | 惊叹、意外 | 意外发现 |
| gentle | 温柔 | 轻柔、关切 | 亲密对话 |
| serious | 严肃 | 郑重、庄严 | 重要场合 |
| playful | 戏谑 | 调侃、轻松 | 打趣场景 |
| whisper | 低语 | 压低声音 | 密谈、自语 |

---

## 二、服务端

### 2.1 技术栈

- **语言**：Python
- **API 框架**：FastAPI
- **LLM**：DeepSeek V4 flash（云端 OpenAI 兼容 API，`thinking: disabled`）
  - 一章 ≈ $0.0016（3K 输入 + 5K 输出 token），整本 600 章长篇 ≈ $1
  - prompt + NDJSON parser 在 `app/services/llm_prompts.py`，
    backend 在 `app/services/llm_deepseek.py`
- **TTS**：Qwen3-TTS 0.6B（Apple MLX 4bit 量化，本地推理，RTF ~0.6）
- **硬件**：Mac Mini M4, 16GB（LLM 走云，本地只跑 TTS，~2 GB 内存）
- **存储**：**基于文件系统管理书籍数据，不使用数据库**

**为何不本地 LLM**：早期实测过 7B 以下的本地中文模型对小说"断句 +
多角色对话归因 + 跨章节别名识别"能力不足，且 verbatim 位置数字
不可靠（locate_sentences 命中率低）。把 LLM 路径固化在云端，本地
只留 TTS。如果将来开源中文模型质量追上、或要自托管，再加 backend；
现在不为离线做过早抽象。

### 2.1.1 书籍 ID 与存储布局

- 每本书在导入时由服务端分配唯一 `book_id`（UUID 或短 hash，永久不变）
- 服务端按 `book_id` 组织文件目录，App 端也以 `book_id` 作为书籍主键
- 所有关于书籍的持久化（播放进度、下载状态等）均以 `book_id` 为键，避免书名冲突

**服务端目录布局**：

```
data/
└── books/
    └── <book_id>/
        ├── meta.json                 ← 书籍元数据 + 导入状态
        ├── cover.jpg                 ← 封面（可选）
        └── chapters/
            ├── 0001.txt
            ├── 0001.json             ← lazy 生成
            ├── 0002.txt
            └── ...
```

### 2.1.2 数据边界

| 数据 | 存储位置 | 说明 |
|------|---------|------|
| 书籍正文 + 元数据 | 服务端（全量）+ App（已下载） | 按 `book_id` 组织 |
| 章节句子元数据 | 服务端（lazy 生成）| App 不持久化，按需请求 |
| 播放进度 | **仅 App 端**（SwiftData） | 按 `book_id` 为主键 |
| 用户偏好（字号/音量/语速/左右手/暗色模式等） | **仅 App 端**（SwiftData/UserDefaults） | 不上传服务端 |
| 音色库 | 服务端 | 开发阶段生成 |

服务端不存储任何用户个性化配置和播放状态；App 端配置不同步到服务端。

### 2.2 API 设计

| API | 方法 | 说明 |
|-----|------|------|
| `/api/books/upload` | POST | 上传书籍（TXT/EPUB） |
| `/api/books` | GET | 获取书籍列表（含状态：处理中/已完成） |
| `/api/books/{id}` | DELETE | 删除书籍 —— 服务端清理导入产生的全部文件 |
| `/api/books/{id}/download` | GET | 下载已处理书籍（仅含 `meta.json` + 章节正文 `chapters/*.txt`，不含章节元数据） |
| `/api/books/{id}/chapters/{ch}/meta` | GET | 获取章节元数据（触发渐进式处理） |
| `/api/tts` | POST | TTS 合成（传入文本 + 角色特征 + tone，返回音频） |

#### 2.2.1 DELETE 语义

- **清理范围**：递归删除 `data/books/<book_id>/` 目录，即 `meta.json` +
  服务端内部 `characters.json` + 全部章节正文 `chapters/*.txt` + 懒
  生成的章节元数据 `chapters/*.json` + 封面（若存在）。该目录是书籍
  导入产生文件的唯一归宿，删除后无残留。
- **进程内状态**：无 —— 章节分析是同步的，没有跨请求的后台任务队列
  需要清理（早期的异步 profile worker 已废弃）。
- **TTS 缓存不清理**：`data/tts_cache/` 是按 `sha1(text||speaker||tone||speed)`
  全局键控、非书籍维度，跨书命中常见。删除书籍不触发缓存清理；旧条目
  自然老化即可。
- **状态码**：成功 `204 No Content`；书籍不存在 `404 Not Found`。
  删除不幂等（第二次调用返回 404），和 `/download` 等端点保持一致。
- **客户端协同**：App 端收到 204 后立即删除本地副本（`<Documents>/library/<id>/`）
  并从书架移除。服务端 DELETE 失败时 App 不动本地副本 —— 下次刷新
  仍能看到、用户可重试。

### 2.3 预处理逻辑

采用渐进式按需处理策略：

```
客户端请求章节 N 的元数据
    ↓
服务端检查 chapters/N.json 是否已存在
    ├── 已存在 → 直接返回
    └── 不存在 → 开始处理
         ├── 处理章节 N（同步，完成后立即返回）
         └── 后台处理章节 N+1（异步，不阻塞响应）
```

#### 章节分析的一体化调用

单次 LLM 调用同时产出**句子 + 角色画像更新**。章节正文 + 已知角色表
（带 identity / gender / age / personality）作为输入；输出是**两段
NDJSON**用 `---` 分隔。

```
章节正文 + characters.json（全书已知角色，含完整画像）
    ↓
LLM 单次调用：断句 + 对话归因 + 语气分类 + 角色画像更新
    ↓
两段 NDJSON 输出：
    {"t":"verbatim","s":"speaker","o":"tone"}      ← 第 1 段：句子
    {"t":"...","s":"...","o":"..."}
    ---
    {"c":"name","g":"...","a":"...","p":[...],"i":"..."}  ← 第 2 段：画像
    ↓
Python reconcile：
    · 句子的 speaker 字符串 → 全局 character_id
    · 第 2 段画像合并到 characters.json：新角色追加，已知角色覆盖
    ↓
Python locate_sentences：把 verbatim 句子文本定位回原文，
                        计算 (start_line, start_col, end_line, end_col)
    ↓
写入 chapters/N.json，包含：
    · sentences[]：每句的位置 + character_id + tone
    · characters[]：本章涉及的角色快照（含完整画像）
```

**第 2 段的输出策略**（LLM 自行判断）：
- **新角色**（不在已知列表）—— 必须输出完整画像
- **已知角色发生重要演变** —— 输出更新覆盖：年龄段跨越（少年→青年→
  老年）、身份重大变化（弟子→长老；活人→魂体）、性格显著转变、
  identity 描述需要修订
- **已知且未演变的角色** —— 不输出，避免无谓更新

支持小说中跨度数十年的角色"成长"：萧炎在第 5 章是 teen，到第 50 章
LLM 判断他已成长为 youth → 输出新画像 → 后续章节用青年音色。
**章节内一致 + 章节边界自然过渡**。

**为何一体化调用 LLM**：

- **断句与归因耦合**：判断一段文字是一个句子还是两个，往往取决于对话结构。
  例如 "你快点！他喊道。"，若将 "他喊道" 识别为引述标签，应拆为两句并分别归因给说话人和旁白；
  若单纯按标点断，会把归因线索切碎。三者同时决策 → 结果更自洽。
- **LLM 上下文利用率**：一次调用中 LLM 同时看到全章和角色表，可横向比较所有候选句子；
  拆成三个 prompt 时每步都要重新构造上下文，整体 token 反而更多。
- **工程简化**：只维护一个 LLM 调用和一个输出 schema，减少接口面。

**位置计算分离**：LLM 仅输出句子正文（verbatim 片段）和属性。Python 在原文中做
子串顺序匹配（cursor 单调前进），精确产出 (line, col) 坐标。
这样 LLM 专注语义分析，位置计算不受 LLM 幻觉影响。

#### 角色数据的存储与暴露

**两层存储 + 严格暴露规则**：

| 文件 | 内容 | 生命周期 | 是否对外 |
|---|---|---|---|
| `meta.json` | 书籍信息（title/author/chapters）| **导入时一次性生成，永不变更** | 是（整书 zip 下载） |
| `characters.json` | 全书累积角色表（name → id + 当前画像）| 章节分析时持续增长 | **否，仅服务端内部** |
| `chapters/N.json` | 第 N 章 sentences + **本章涉及角色的快照**（含画像）| 章节分析时一次性写入 | 是（按需 GET）|

**App 端**只读 `meta.json`（导入后一次性下载）+ `chapters/N.json`
（按需），从 chapter meta 的 `characters` 字段拿当前章节要用的音色
信息。**不读、也读不到 `characters.json`** —— 它纯属服务端用来维护
跨章节角色 id 一致性 + 给 LLM 提供已知角色 hint 的内部数据结构。

**章节快照是值不是引用**（关键设计）：每章的 `characters[]` 是
**该章节分析时的画像快照**。即便 50 章后某角色画像被 LLM 更新，
重新阅读第 5 章拿到的还是当时的快照（少年）—— 跨章节自然过渡，
章节内一致。

**取消的旧机制**（拥抱新模型，不再保留）：
- ~~`X-Characters-Version` 响应头~~：不需要 —— 没有跨请求的角色版本同步
- ~~异步 profile worker + ThreadPoolExecutor~~：不需要 —— 角色画像
  和句子分析在同一次 LLM 调用产出
- ~~"首次音色偏差"窗口期~~：消失 —— chapter meta 到达的瞬间，
  里面的角色都是终态
- ~~`BookMeta.characters` / `characters_version`~~：从 meta.json 移除

**仍然接受的边界 case**：
- 别名识别靠 LLM：偶尔把「萧家少主」当成新角色 → 创建重复条目；
  我们**不做事后 merge**
- 非顺序阅读：首次进入中间章时已知角色表只有此前看过的章节累积，
  LLM 可能把老角色当新角色；接受
- 画像 thrash：极少情况下 LLM 在两章之间反复改判 → 音色抖；
  实测罕见，不优化

**为何不预先做全书角色识别**：
- 章节分析本身就在做"谁在说话"的推理，独立的人名识别冗余
- 全书扫描会成为导入延迟瓶颈；按章摊薄成本，首章能听的时延显著下降
- LLM 在章节语境下的判断往往比无语境的人名识别更准

### 2.4 书籍导入流程

```
接收上传文件（TXT/EPUB）
    ↓
解析书籍基本信息（标题、作者、封面、简介）
    ↓
章节识别 + 拆分（§2.4.1）→ 每章写入独立 txt 文件
    ↓
写入 meta.json（title / author / chapters 列表 / status）
写入 characters.json：[{id:0, name:"旁白", ...}]（仅旁白，服务端内部）
    ↓
标记状态为"已完成"
```

**导入阶段不做全书角色识别和画像**。`meta.json` 写入后**永不变更**；
角色表（`characters.json`，仅服务端内部）由章节元数据生成过程逐步
增长（见 §2.3）。每章的角色快照随 chapters/N.json 一同写入，App 端
按章读取，无需额外的同步协议。

### 2.4.1 章节识别

章节识别策略按文件格式分两条路径，因为 EPUB 自带结构化目录而 TXT
是裸文本，两者的最优方案完全不同。

#### EPUB：直接读取目录，不做识别

EPUB 规范在包内自带目录文件：EPUB 2.0 用 `toc.ncx`，EPUB 3.0 用
`nav.xhtml`；每个条目显式指向一个 HTML 文件或锚点。读取步骤：

1. 解析 `META-INF/container.xml` → 拿到 `.opf` 文件路径
2. 解析 `.opf` 的 `<manifest>` + `<spine>` → 拿到章节 HTML 文件的顺序列表
3. 解析 `toc.ncx` / `nav.xhtml` → 拿到章节标题与文件的映射
4. 按 spine 顺序读取 HTML，剥离标签得到纯文本章节内容

**不调用 LLM**。目录是书籍元数据的一部分，权威且稳定。

**实现细节**（[app/core/parsers/epub.py](server/app/core/parsers/epub.py)，纯 stdlib，无第三方依赖）：

- **路径解析**：manifest / ncx / nav 里的 `href` 都相对于所在目录；解析器处理 `../` 与任意嵌套的子目录（例如 `OEBPS/package/content.opf` 指向 `../text/ch1.xhtml`）。
- **TOC 优先级**：先找 manifest 中 `properties="nav"` 的 EPUB 3 导航文件；其次按 spine 的 `toc` 属性找 NCX；最后兜底扫 manifest 里 `media-type=application/x-dtbncx+xml` 的条目。TOC 缺失时，用章节开头第一行作标题，再退到 "第 N 章"。
- **spine 过滤**：跳过 `linear="no"` 的辅助页（脚注/索引），跳过 nav 文档自身（否则会产出一个包含章节标题列表的伪章节）。
- **HTML → 纯文本**：stdlib `HTMLParser`，为块级标签（`p/div/h1-6/li/section/...`）插入换行；跳过 `<head>/<title>/<script>/<style>` 里的内容；自动解码实体（`&amp;`、`&#x4e2d;`）；空白压缩。
- **失败分层降级**：损坏 ZIP、缺 `container.xml`、损坏 XML → `EpubParseError`（API 层映射为 400）。全章节为空 → 兜底生成一个用 fallback_title 命名的单章。

#### TXT：LLM 驱动的章节识别

TXT 没有结构化目录，各种书源写作习惯差异极大：`第X章`、`楔子`、
`序章`、`上篇 风起`、`一、初遇`、`Chapter 1`、`番外：旧事` 等等。
硬编码正则无法覆盖这些变体，因此**完全放弃硬编码正则路径**，改由
LLM 做语义识别：LLM 只产出结构化的标题列表或"LLM 归纳出的"正则
表达式，定位仍由 Python 以 O(n) 完成。

**核心思路**：小说 TXT 要么开头有目录块（作者手工贴或导出工具生成），
要么每一章标题本身具有稳定的重复模式。两种场景下 LLM 输出都是
结构化、可验证的 —— 章节标题列表（场景 1）或一条章节标题正则
（场景 2）—— 后续大规模正文扫描仍走确定性代码。

```
LLM 读取 TXT 开头 200 行
    ↓
LLM 判断：是否包含目录块？
    ├── 有目录 → 抽取当前批次内的章节标题
    │             │ toc_complete=false：外层代码再喂下一个 200 行批次，
    │             │                      带上已抽到的标题作为 prompt hint
    │             │ toc_complete=true ：拿到完整标题列表
    │             ↓
    │           Python: full_text.find(title, cursor) 顺序定位
    │
    └── 无目录 → LLM 返回首章标题 + 描述标题模式的正则
                   ↓
                 Python: 用该正则在全文 scan 标题，锚定每章位置
```

**LLM Prompt 输出形式**（JSON schema 约束）：

```json
// 场景 1：检测到目录（可能分多次调用，每次返回本批次新抽到的标题 + 是否处理完）
{
  "has_toc": true,
  "chapter_titles": ["楔子", "第一章 初见", "第二章 对决", "…"],
  "toc_complete": false       // 本批次结尾还没进入正文时为 false，提示调用方继续喂下一批
}

// 场景 2：未检测到目录，返回首章标题 + LLM 从样本中归纳出的正则
{
  "has_toc": false,
  "preface_titles": ["序章"],                          // 楔子/序章等非正文章节，可为空
  "first_chapter_title": "第一章 初见",
  "chapter_pattern": "^第[一二三四五六七八九十百千0-9]+章.*$"   // LLM 基于实际首章写出，Python 编译后全文 scan
}
```

**有目录时的分页读取循环**（外层 Python，单轮 LLM 调用仍是单 prompt）：

1. 喂入第 1-200 行 → LLM 返回本批次抽到的标题 + `toc_complete`
2. `toc_complete=false` → 喂入第 201-400 行，prompt 里附上已抽到的标题
   作为 hint，LLM 继续追加；如此循环
3. 直到 `toc_complete=true`，或达到上限（默认 10 批 = 2000 行）
4. 若 10 批后仍未完成 → 视为识别失败，降级为单章兜底

这要求 prompt 里让 LLM 显式判断"本批次末尾是否已经离开目录进入正文"，
而不是简单数标题数。小说 TXT 的目录通常连续出现，少有穿插正文的情况，
这个信号足够稳定。

**位置映射规则（Python 侧）**：
- **有目录**：对每个标题做 `full_text.find(title, cursor)` 顺序扫描，
  cursor 单调前进 —— 防止同名/重复前缀标题错配（如"第一章"既在目录
  出现也在正文出现）。
- **无目录**：`re.compile(chapter_pattern)` 后全文 scan。首章位置先用
  `find(first_chapter_title)` 锚定，防止 LLM 给出的正则过宽、
  把作者话或书评首行误匹配进来。

**标题归一化**（目录文字与正文章节首行偶尔存在差异时的兜底）：
- 中文数字 ↔ 阿拉伯数字互转（"第一章" ↔ "第1章"）
- 去除所有空白
- 忽略大小写

**失败降级**：
- LLM 响应非法 JSON / schema 不符 / `chapter_pattern` 编译失败 → 重试一次
- 重试后仍失败，或扫出的章节数 < 2 → 全文作为单章，标题 = 文件名

**为何不全程由 LLM 做章节切分**：
- 章节数可达几百至上千，让 LLM 逐章扫成本过高（token + 时间）
- 章节定位是纯字符串操作，LLM 反而不擅长（容易算错 offset，类似
  `locate_sentences` 里 Python 侧负责位置映射的思路）
- LLM 只做"语义层判断"（目录/非目录、首章标题、模式归纳），其余交给
  确定性代码

**LLM 调用策略：每批单 prompt，不使用 tools**

章节识别里 LLM 每次只看一个固定大小的文本批次（200 行 / ≤5K tokens），
直接把这段作为 prompt 一次性喂进去。**不走 tool-calling 回路** ——
目录分页是外层 Python 控制的普通循环，每次循环仍是独立的一次单
prompt 调用。

整个项目的所有 LLM 调用都是单次 prompt（输入规模由代码切块控制），
没有 tool-calling 循环：

| 任务 | 输入规模 | 调用形式 |
|---|---|---|
| §2.4.1 章节识别 | 每批 200 行（≤5K tokens） | 单次 prompt，必要时外层循环分页 |
| §2.3 章节元数据（句子 + 角色画像更新） | 单章正文 + 已知角色表（数千字） | 单次 prompt 全量喂入，输出两段 NDJSON |

规则：**输入规模可控 → 单次 prompt**。tool-calling 引入多轮 LLM
调用，总耗时近似线性增加轮数；当前所有任务的输入都能一次塞下，
没有必要引入。

> **历史变化**：早期版本曾打算给"全书角色画像"用 tool-calling
> （让 LLM 用 `find_character` / `read_lines` 工具按需读取整书）。
> 这套方案在改用按章一体化分析后被废弃 —— 见 §2.4.2。

### 2.4.2 角色画像更新（合并到章节分析）

> **历史变化**：早期版本通过独立的 tool-calling 异步 worker 单独
> 分析每个新角色（让 LLM 用 `find_character` / `read_lines` 工具查
> 全书）。这套机制已被废弃 —— 见 §2.3 的"章节分析的一体化调用"。
>
> 现在，角色画像和句子断句 / 归因 / 语气在 `analyze_chapter` 的同一
> 次 LLM 调用里产出（两段 NDJSON 中的第二段）。`AnalyzedChapter`
> 同时携带 sentences 和 character_updates。
>
> 优势：
> - **首次音色偏差消失**：画像和 sentence 同步到达，无窗口期
> - **无异步同步协议**：删掉了 `X-Characters-Version` / 后台
>   ThreadPoolExecutor / `analyze_character_profile` 工具调用循环
> - **支持角色"成长"**：LLM 在每章基于本章上下文判断已知角色是否
>   需要更新画像（年龄段跨越 / 身份变化 / 性格转变），写回会覆盖
>   `characters.json` 里的旧画像；后续章节用新画像生成的章节快照
>   自然过渡到新音色
>
> 详见 §2.3 第 2 段输出策略。

### 2.5 音色管理与匹配

**设计决策：采用零样本克隆路线（zero-shot voice cloning），不使用 TTS 模型自带的 SFT 预置音色。**

理由：
- **音色数量自由**：SFT 预置音色数量由 TTS 模型发布方固化，中文高质量音色通常只有个位数（CosyVoice-300M-SFT 仅"中文男"、"中文女"两个可用于 Mandarin），无法满足多角色小说的覆盖需求。
- **扩展成本低**：新增一条音色 = 增加一组 (wav, txt) 文件 + 一条 JSON 元数据，无需模型训练或指令微调。
- **引擎可替换**：零样本克隆是现代 TTS 引擎的通用能力，相同的 (ref_wav, ref_text) 接口可在 Qwen3-TTS / CosyVoice / F5-TTS 等引擎间平移，不绑定特定厂商。
- **语气独立于音色**：通过 Qwen3-TTS 的 `instruct` 参数（自然语言指令）控制语气，不需要为每种语气各训练一个音色。

---

音色库设计如下：

- **音色载体**：每个音色由一段 3-10 秒的参考音频（`data/voices/prompts/*.wav`）+ 配套转写文本（`*.txt`）组成
- **音色库元数据**：[data/voices/speakers.json](server/data/voices/speakers.json) 记录每条音色的属性（性别、年龄、性格标签），使用与角色相同的预定义枚举值
- **运行时合成**：TTS 引擎（Qwen3-TTS）做零样本克隆，每次请求把参考音频 + 目标文本传给模型，输出克隆音色朗读目标文本的音频
- **匹配策略**：性别和年龄为硬约束，性格标签做相似度匹配（交集越大优先）
- **引擎无关**：speakers.json 与 TTS 引擎解耦，`zs:<prompt_id>` 类型的音色可在 Qwen3-TTS / CosyVoice / F5-TTS 等支持零样本克隆的引擎间平移
- **扩充便捷**：新增一条音色 = 增加一组 (wav, txt) 文件 + 一条 JSON 元数据，无需模型训练

### 2.5.1 参考音频生成与音色库构建

开发阶段一次性工作：用 **Qwen3-TTS VoiceDesign** 模型在本地按自然语言描述生成 64 条覆盖全属性组合的中文参考音频，手工建立 tag → {gender, age, personality} 映射。

**双模型分工**（重要）：

VoiceDesign 只用于**参考音频生成阶段**，运行时 `/api/tts` 服务仍走 Base + 零样本克隆，不变。两个模型差异：

| 用途 | 模型变体 | 调用方式 |
|---|---|---|
| **生成 prompts/*.wav** | `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` | `model.generate_voice_design(text=..., instruct=<自然语言描述>)` |
| **运行时合成 /api/tts** | `Qwen3-TTS-12Hz-0.6B-Base-4bit` | `model.generate(text=..., ref_audio=<wav>, ref_text=<txt>, instruct=<tone>)` |

为何不直接用 VoiceDesign 跑运行时：VoiceDesign 输出非确定（temperature 0.9），同一描述每次音色略有差异；Base + 固定 ref_audio 完全确定，听感一致。把 VoiceDesign 的随机性"冻结"到磁盘 wav 上，运行时拿这个 wav 做克隆，得到稳定可复现的音色。

**整体流程**：

```
人工写 VOICE_CATALOG（45+ 条）：tag + 自然语言音色描述
    ↓
对每条调 VoiceDesign（约 10s/条）→ 24kHz mono 16-bit WAV
    ↓
落盘 data/voices/prompts/vd_<tag>.wav + 同名 .txt（转写）
    ↓
人工 TAG_TO_ATTRS 映射（generate_voicedesign_voices.py 内）→ {gender, age, personality}
    ↓
写入 speakers.json：speaker_id=zs:vd_<tag>
```

执行命令在 [server/README.md](server/README.md) "Seed the voice library" 章节。

**自然语言描述模式**（VoiceDesign 对描述敏感，区分两类音色）：

| 类别 | 示例 tag | 描述策略 |
|---|---|---|
| **旁白类**（讲述，runtime tone 多为 NEUTRAL） | `narrator_male_mature` / `narrator_female_adult` | 强调讲述自然、克制；用反例对比"像朋友讲故事**而不是**舞台朗诵"；不堆叠 "洪亮 / 铿锵 / 富感染力" 等戏剧性词 |
| **角色类**（对白，runtime tone 多变） | `arrogant_male_adult` / `cold_female_youth` 等 | 戏剧化 + persona-anchored：用影视/小说角色锚（"宫斗剧妃嫔"/"武侠侠客"）、明确演绎指令（重音/语调习惯）、反例对比"日常对话的平淡"；character 类比 narrator 类需要更多表演感、避免被 runtime tone 反向稀释 |

**标注台词分桶**（参考音频的内容也影响克隆出来的演绎风格 —— 平铺的叙述会让克隆的角色音色显得"念稿"，戏剧化的对白才能让音色具备演绎潜力）：

| 类别 | 适用 tag | 文本 |
|---|---|---|
| 旁白叙述 | 2 条 narrator | "春天的午后，阳光透过窗户洒在桌上。他放下书本，缓缓抬起头，问道：「你真的决定要走了吗？」..." |
| 角色对白 (adult) | youth / adult / elder 角色 | "「我等了十年，就是为了今天这一刻。」他冷冷地笑了，「你以为躲得掉吗？这世上从来就没有公平可言..." |
| 角色对白 (teen) | teen 角色 | "「不可能！」桌上的书被一把推开，「你说的根本不是真的！如果真是这样，他怎么会一直瞒着我？」" |
| 角色对白 (child) | child 角色 | "「我看见啦！我看见啦！」蹦着拍手叫起来，「那只小狗从那边跑过去了！它的尾巴一摇一摇的，我们快去追！」" |

派发逻辑在 `scripts/generate_voicedesign_voices.py:sample_text_for(tag)`：narrator → 叙述；其余按 tag 的 age 落到 child / teen / adult+ 三档。

**当前音色库（64 条）属性分布**：

| 性别 \ 年龄 | child | teen | youth | adult | elder |
|---|---|---|---|---|---|
| male | - | 2 | 9 | 13 | 4 |
| female | - | 3 | 10 | 13 | 4 |
| neutral | 6 | - | - | - | - |

- **旁白用音色**：男 / 女各 1 条（`narrator_male_mature` / `narrator_female_adult`），不做更多 narrator 变体
- **童声**：6 条全部 `Gender.NEUTRAL`，按性格区分（lively / gentle / timid / cheerful / clever / melancholy）—— 童年期男女声物理特征重叠，性别区分价值低于性格区分
- **老年**：男女各 4 条（wise / kind / fierce / cunning male；wise / kind / fierce / melancholy female），覆盖武侠老前辈、慈祥外公外婆、宫斗反派、沧桑老妇等常见角色

**性格标签覆盖**（18 个预定义性格全部覆盖）：

```
gentle      █████████████████████████ 25    cold        ██████ 6
mature      ████████████ 12                  wise        █████ 5
calm        ████████████ 12                  passionate  █████ 5
cheerful    ██████████ 10                    brave       █████ 5
kind        █████████ 9                      melancholy  █████ 5
naive       ████████ 8                       arrogant    ████ 4
playful     ██████ 6                         cunning     ████ 4
serious     ████ 4                           timid       ███ 3
fierce      ██ 2                             determined  ██ 2
```

`gentle` 高频是结构性的：旁白和大量"温柔"型角色都用它做基底；niche 性格 fierce / determined 各 2 条够用（出现频率本就低）。

**(gender × age × personality) 唯一性约束**：matcher 在 cell 内按 personality 交集排序、最后按 speaker_id 字典序 tie-break；如果两条音色的 (gender, age, personality) 完全相同，字典序后的那条永远轮不到。所有 64 条经过去重审计，每个 (gender, age, personality) 三元组唯一，64 条都可达。

**参考音频来源可扩展**：

| 来源 | 说明 | 标注方式 |
|------|------|---------|
| Qwen3-TTS VoiceDesign | 当前主路径，本地生成 64 条 | 自然语言描述 + 手工 tag→attrs 映射 |
| 商业 TTS API（Volcengine / Azure / ElevenLabs / Minimax 等） | 可选扩充：目录里的预设音色 | 目录标签 → 手工映射 |
| 影视 / 播客剪辑 | 从公开音视频提取干净的单人 3-10 秒片段 | 多模态 LLM 标注 |
| 真人录音 | 最终落地方案，可录制专业声优 | 录音时直接标注 |

---

## 三、iOS App

### 3.1 技术栈

- **UI**：SwiftUI（iOS 16+）
- **音频**：AVAudioEngine + AVAudioSession
- **存储**：SwiftData（播放进度等持久化）
- **网络**：URLSession

### 3.2 页面结构

```
书架页（首页）
    │
    ├── 导入书籍（文件选择器，支持 TXT/EPUB）
    ├── 下拉刷新
    ├── 设置入口
    │
    └── 点击书籍 → 播放页
                    │
                    ├── 正文区域（上下滚动，左右滑动切换章节）
                    ├── 顶栏：返回按钮 │ 书名 │ 目录按钮
                    ├── 底栏：播放/暂停 │ 跟读 │ 字号- │ 字号+ │ 章节N/总数
                    │
                    └── 目录按钮 → 目录页
                                    │
                                    └── 点击条目 → 返回播放页，定位到所选章节
```

### 3.2.1 设置页

| 设置项 | 类型 | 范围/可选值 | 默认值 | 说明 |
|--------|------|------------|--------|------|
| 音量增益 | Slider | -12 dB ~ +20 dB，步进 1 dB | 0 dB | 客户端增益，主要用于嘈杂环境放大。详见 3.8 |
| 语速 | Slider | 0.5x - 2.0x | 1.0x | 传递给 TTS API |
| 字号 | Stepper | 12 - 30，步进 2 | 18 | 播放页正文字号，底栏字号按钮也可调 |
| 暗色模式 | Toggle | 开/关/跟随系统 | 跟随系统 | 影响全局外观 |
| 左右手模式 | Toggle | 左手 / 右手 | 右手 | 控制栏布局镜像，适配单手操作 |
| 服务端地址 | TextField | `host:port` | 局域网 IP + 端口 | App 连接的 Mac Mini M4 地址 |
| 音频缓存上限 | Stepper | 100 MB - 2 GB，步进 100 MB | 500 MB | TTS 音频缓存容量上限，详见 3.6.1 |
| 版本号 | Label | - | - | 只读显示 |

所有设置项仅存储在 App 端（SwiftData / UserDefaults），不同步到服务端。

### 3.3 书架管理

书籍显示三种状态：

| 状态 | 说明 | 交互 |
|------|------|------|
| 导入中（uploading/processing） | 已上传，服务端正在预处理 | 点击提示"处理中，请稍后" |
| 下载中（downloading） | 服务端已处理完，App 正在下载 | 点击提示"下载中，请稍后"，显示进度 |
| 已导入（ready） | 下载完成，可正常播放 | 点击进入播放页 |

- 导入书籍后自动上传服务端进行预处理；上传过程显示上传进度条，可取消
- 预处理完成后自动触发下载，下载过程显示进度条
- 删除书籍：App 端删除本地缓存；是否同时删除服务端数据为可选行为（通过 API）

**刷新触发条件**：

| 触发事件 | 说明 |
|---------|------|
| 进入书架页面 | 启动 / 从其它页面返回书架时，由 `.task` 触发一次 |
| App 从后台切到前台**且**当前在书架页面 | 监听 `scenePhase` → `.active` 转换，路由路径为空时刷新 |
| 用户下拉 | SwiftUI `.refreshable` |
| 上传完成 | `uploadBook` 返回时 `await refresh()` 兜底，让占位条目和服务端记录对齐 |
| **状态轮询**（仅当书架里有 uploading / processing 书） | 在上一次响应返回后等 1 秒（固定，非可配置），再发下一次 `GET /api/books`；接口响应时间不计入这 1 秒。所有书进入终态（ready / failed）后链条自动停止 |

**稳态下（全部 ready / failed）不主动轮询**；只有正在导入的书才会触发周期性 API
调用——没有书在动时书架完全安静。进入播放页等其它页面同样不刷新。

**刷新动作的语义**：

1. 调用 `GET /api/books`：
   - **成功（HTTP 2xx）**：以服务端列表为准。本地存在但服务端未列出的书视为已被
     删除（用户在服务端 `DELETE /api/books/{id}`、或直接 `rm` 缓存目录），
     **同时清掉本地 `Documents/library/<book_id>/` 和保存的播放进度**。
   - **失败 / 非 2xx / 网络错误**：保持现有 `books` 数组不动；**绝不**触发本地清理。
     这样断网 / 服务端临时挂掉时书架不会被错误清空。
2. 服务端列出但本地没有的书：照常下载到本地。

**进行中状态的指示**：

- **不在 nav bar / toolbar 上放全局 spinner** —— 这种"无差别忙碌指示器"对用户没有信息量，
  反而会被误当成系统级阻塞、并且 iOS 26 的 toolbar 胶囊合并机制会让它和实际按钮在视觉上
  纠缠。
- **静默后台同步**：自动刷新（进入页面、前台恢复、上传后兜底刷新）不显示任何加载指示。
  下拉刷新本身有 SwiftUI `.refreshable` 自带的转圈，那是用户主动触发的反馈、保留。
- **正在导入的书籍 = 行内显示 spinner**：书架的每条书籍 Row 在 uploading / processing /
  downloading 三种状态下，状态徽章右侧紧跟一个 `ProgressView()` + 进度提示文案
  （"正在上传并识别章节…" / "服务端处理中" / "正在下载到本机"）。

### 3.3.1 App 端书籍存储布局

```
<Documents>/
├── library/
│   └── <book_id>/              ← 与服务端一致的 book_id
│       ├── meta.json
│       └── chapters/
│           ├── 0001.txt
│           └── ...             ← 仅正文，章节元数据按需请求不持久化
├── cache/
│   └── tts/                    ← TTS 音频预加载临时文件（可清理）
└── <SwiftData>                 ← 播放进度、用户偏好
```

- **持久化存储（library/）**：与服务端已导入书籍同步
- **缓存（cache/）**：可随时清理不影响功能
- **SwiftData**：以 `book_id` 为主键的播放进度表，及用户偏好

### 3.4 播放位置与浏览位置

分离两个独立概念：

- **播放位置**：当前 TTS 朗读到的句子。每本书有且仅有一个，持久化存储。新书首次打开默认定位到第一章第一句。
- **浏览位置**：用户当前查看正文的位置。

**播放位置的表示与持久化**：

- 不引入句子 ID
- 位置由 `(chapter_id, start_line, start_col)` 定位，与章节元数据中句子的起始位置对齐
- 通过 SwiftData 持久化，以 `book_id` 为主键：每本书对应一条记录 `{book_id, chapter_id, start_line, start_col}`
- 切换播放位置时根据章节元数据查找对应句子，找不到则取最接近的句子

**跟读模式**（默认开启）：浏览位置自动跟随播放位置 —— 当播放推进到新句子时，浏览侧自动跳到该句子所在的页（同章则切页，跨章则换章并定位到首页）。

退出跟读的触发条件：
- 手动左右翻页（包括跨章自动连续翻页中的用户手动滑动）
- 从目录选择章节

恢复跟读：点击底栏跟读按钮 → 正文自动跳转到播放位置所在的页，恢复跟读状态。

**翻页交互（重要决策，2026-04-26）**：

播放页正文区域采用**统一的水平翻页手势**，没有独立的"翻章"手势：

- **页 = 一屏正文**（不是章节）。左右滑动 = 翻一页。
- **跨章连续翻页**：当前章末页右滑 → 直接进入下一章首页；某章首页左滑 → 直接进入上一章末页。翻页与翻章使用同一手势、同一动画，对用户透明。
- **不再支持上下滚动**。早期版本采用"章节级 TabView + 章内 ScrollView"：横向滑动翻章、纵向滚动读章节正文。该方案存在两个核心问题：
  1. 章内 ScrollView 的拖动手势会与外层 TabView 的横向翻章手势竞争识别，导致左右翻章频繁失灵；
  2. 跟读模式下需要把"当前句滚动到屏幕中央"，跨章节时滚动位置与章节切换的时序难以协调。
- **底栏序号仍只显示章节号**（`N / 总章数`），不显示页号 —— 这是有意的简化决策：显示全书页号需要先把所有章节都分页一遍，对一个 600 章的书是数秒级开销；而显示章号只需当前章的分页结果，几乎免费。

**分页机制**：

- iOS 端使用 TextKit 1（`NSLayoutManager` + `NSTextContainer`）按当前字号、视图尺寸把章节正文切成页，输出每页的 UTF-16 偏移区间 + 行起始偏移表。一章 3-15ms，缓存在 `PaginationStore` 中。
- 缓存以 `(fontSize, viewWidth, viewHeight)` 为签名 —— 字号变化或屏幕尺寸变化（旋转、分屏）时整体失效并重排。
- **惰性预分页**：进入某章时，分页当前章 + 相邻 ±1 章，确保跨章翻页时下一章首页已就绪。
- 句子位置 `(start_line, start_col)`（UTF-16 列偏移）→ 通过行起始偏移表换算成章内 UTF-16 偏移 → 在分页结果中二分定位所在页。这一映射使得"跟读模式跳页"和"判断高亮区间是否在当前页"都是 O(log pages) 的查表。

**双击翻页正文**：识别本页内首个句子（按起始偏移）→ 跳转播放到该句子。本页若无句子起始（如跨页延续段），双击不响应。

### 3.5 播放逻辑

**播放状态**：

| 事件 | 效果 |
|------|------|
| 进入播放页 | 进入播放状态（即使暂停） |
| 点击返回按钮退出播放页 | 退出播放状态；若正在播放则自动暂停 |
| App 切换到后台 | 保持播放状态，继续后台播放 |
| App 被 kill | 下次启动恢复为非播放状态 |

播放状态控制：
- 后台音频播放（AVAudioSession `.playback` 模式）
- 锁屏媒体信息显示（MPNowPlayingInfoCenter）
- 锁屏播放/暂停控制（MPRemoteCommandCenter）

**暂停行为**：点击暂停按钮后立即停止播放，不等待当前句子播放完成。

**逐句播放流程**：

```
进入播放页 → 获取当前章节元数据（调用服务端 API）
    ↓
点击播放 → 从播放位置开始逐句播放
    ↓
当前句 → 调用服务端 TTS API → 获取音频 → 播放 → 高亮当前句
    ↓                                         ↓
播放完成 → 播放位置移动到下一句            预加载下 N 句音频
    ↓
章节末尾 → 自动切换到下一章节
```

### 3.6 音频预加载

为减少 TTS 等待延迟，采用预加载策略：

- 播放当前句时，提前调用 TTS API 预生成后续若干句子的音频
- 采用**单 worker 串行预取**（详见 §3.6.2），按 anchor → anchor+1 →
  ... 的优先级顺序合成，跨章续航
- 完成后落盘到 TTS 缓存（M4A，§3.6.1），后续命中可秒回

### 3.6.1 音频缓存管理

TTS 音频持久化到本地 `cache/tts/` 目录。文件名是 `SHA256(bookId + chapterId
+ text + speaker-attrs + tone + speed).m4a`，保证相同参数的音频一一对应、
可复用。同目录下 `index.json` 存 `hash → (bookId, chapterId, sentenceIndex)`
映射，用于淘汰策略（见下）。

**音频格式：M4A（AAC 48 kbps mono @ 24 kHz）**

服务端生产环境用 Qwen3-TTS 出原始 PCM，写成 16-bit mono 24kHz WAV 后用
macOS 自带的 `afconvert -f m4af -d aac -b 48000 -c 1` 编码为 M4A 容器
中的 AAC 比特流，再返回给 App。决策对比（实测）：

| 方案 | 单句大小 | 一本书缓存 | 编码开销 | 主观质量 |
|---|---|---|---|---|
| 原始 WAV (24kHz/16-bit) | 250-700 KB | ~3.5 GB | 0 | 基线 |
| **M4A AAC 48 kbps**（采用）| **30-100 KB**（**~13%**） | **~470 MB** | **~30 ms/句** | 听不出差别 |

理由：
- WAV 累积量 ~3.5GB 远超默认 `audioCacheLimitMB`（500MB），实战中缓存
  会频繁淘汰、命中率低；切到 AAC 后整本书都能装进默认缓存
- 编码开销（~30ms）相比 TTS 合成本身（~3s/句）< 1%，完全可忽略
- iOS `AVAudioFile` 原生支持 M4A，解码无需额外依赖
- 测试用的 stub TTS 仍输出 WAV；endpoint 通过 sniff 字节头自动设
  `Content-Type: audio/mp4` 或 `audio/wav`，两条路径都能跑

为何不用 Opus（更省）：iOS 解码需要第三方库，48kbps AAC 已经远低于
缓存压力点，再省没有实际收益。

### 3.6.2 顺序预读（Serial Prefetch）

**设计决策**：App 端的 TTS 预取使用**单 worker 串行**模型，而不是 N
个并行任务一槽一个 Task。

**反例（为什么并行不合适）**：早期版本对每个窗口槽位都 spawn 一个
detached Task：

```
windowAhead = 20 → 21 个并行 fetchAudio Task → 21 个并发 HTTP 请求
```

服务端 Qwen3-TTS 用 `gpu_guard` 在 Metal 上串行（一次只能跑一个 TTS），
所以 21 个请求实际是在服务端的 asyncio 调度池里排队。**问题**：
asyncio 选下一个请求的顺序不可控（不一定是 FIFO，受协程调度策略影响），
所以 anchor 句子的音频未必先生成。表现为：进度条上能看到很多浅色块
亮起（散落在窗口里随机的句子），但 anchor 槽位仍然空着 → 播放卡顿。

**正确做法（当前设计）**：App 端串行，服务端继续串行：

```
                    单 worker 按优先级顺序循环
position=N
   ↓
window = [N, N+1, ..., N+windowAhead]    (priority-ordered)
   ↓
worker iterates:
   1. fetchAudio(N)     ← anchor 永远第一个完成
   2. fetchAudio(N+1)   ← anchor+1 第二
   3. ...
   每完成一个 → 重新评估 ordered window（响应 anchor 变化）
```

**前提条件**：TTS 合成时间（~3s/句）短于该句的播放时间（5-15s）。
所以即使串行，下一句的合成在当前句播放完之前就准备好了 —— 串行预取
**总能跑赢**播放线索，不会变成阻塞瓶颈。

**收益**：

- **anchor 永远最先完成** —— 播放线索基本不需要等待
- **简单**：一个 Task 而不是 N 个 → 取消、推进、跳转的状态机都简化
- **可观测**：进度条上看到的浅色块按顺序从左向右填充，与播放游标
  之间始终保持一段窗口，符合直觉
- **降低服务端无效合成**：跳转时取消单个 in-flight 任务而非 N 个

**如果预取赶不上播放呢？** 极端情况下（TTS 突然变慢，比如 GPU 降频或
服务端拥塞）会出现"预取追不上播放"的情况。这时播放线索会停在 anchor
槽位上等，进度条游标和最后一个浅色块距离会缩短到 0。这是预期内的
回退行为 —— 总比并行情况下"很多句子缓存了但 anchor 没缓存"更好处理。

**实现要点**：

| 元素 | 类型 | 含义 |
|---|---|---|
| `prefetchedURLs` | `[SentenceAddress: Result<URL, Error>]` | 已完成的窗口槽位（成功或失败）|
| `prefetchTask` | `Task<Void, Never>?` | 单 worker 句柄；nil 表示空闲 |
| `prefetchSubscribers` | `[SentenceAddress: [CheckedContinuation]]` | player loop 等待的连续点 |
| `runPrefetchWorker()` | `async` loop | 主循环：`orderedWindowSlots` → 找第一个未完成的 → fetch → 通知订阅者 |
| `awaitPrefetched(slot:)` | `async` | player loop 接口：缓存命中即返回，否则订阅等待 |

`reconcileWindow` 在 `position` 变化时被调用：仅做"修剪窗口外的已完成
URL + 取消窗口外的等待者 + 唤起 worker"，**不直接发起 fetch**。worker
独立调度 fetch 顺序。

**当前参数**：`windowAhead = 20`（窗口 21 句）。够覆盖跨章过渡，又不至于
浪费带宽（如用户立刻跳走，最多浪费一个 in-flight fetch）。

**缓存 ≠ 预下载窗口**（见 §3.9）：
- 窗口是"当前要合成几句"，决定主动预取负载
- 缓存是"已经合成过的结果保留多少"，决定跳转时的命中率
- 缓存容量应**远大于**窗口尺寸，这样用户在播放位置附近做小幅度前后跳转
  能充分复用

**缓存淘汰策略：按距离当前播放位置远近淘汰（不是 FIFO）**

FIFO 在"顺序播放 + 偶尔跳转"的使用模式下反直觉：

> 举例：缓存了第 50-100 句刚好满，播放到第 90 句。用户跳回第 40 句，
> 缓存 miss 要下载。按 FIFO 会淘汰掉 50，但顺序播放 40→41→...→50
> 时又得重新下载 50。按距离淘汰则先踢掉离 40 最远的 100，50 保留
> 命中率。

所以：**淘汰时计算每个缓存条目到当前播放位置的"句子距离"，最远的先
淘汰**。距离定义：

- 同本书、同章节：`|Δsentence|`
- 同本书、跨章节：`|Δchapter| * 1000 + |Δsentence|`（1000 ≈ 一章的句子数
  上限，跨一章的代价 ≈ 一章内所有句）
- 不同书：`Int.max`（视为永远不复用，优先淘汰）
- 无索引条目（旧文件 / 索引丢失）：`Int.max - 1`

**实现细节**：

- **上限容量**：可在设置页配置（默认 500 MB，范围 100 MB - 2 GB）
- **清理时机**：每次 `store()` 后检查；超过上限才淘汰（避免每句都扫盘）
- **Anchor 提供方**：`PlaybackService.reconcileWindow` 在每次 `position`
  变化时调用 `cache.setAnchor(coord)`，把当前播放位置同步给缓存
- **Anchor 缺失时**：退回 FIFO（按文件 creation time），用于 app 冷启动
  还没开始播放、或 `PlaybackService.stop()` 之后的场景
- **跨 session 持久性**：`index.json` 每次变更后写盘，app 重启后距离信息
  还在

**缓存命中**：

- 同一句子（同文本、同角色、同语气、同语速）直接读本地文件，不走 TTS API
- 用户跳回已听过段落 → 命中
- 不同语速/语气/角色产生不同缓存条目，互不覆盖

### 3.7 网络与错误处理

所有对服务端的 HTTP 请求采用统一策略：

| 场景 | 策略 |
|------|------|
| TTS API 请求失败 | 自动重试 3 次（指数退避），仍失败则跳过该句并在 UI 提示 |
| 章节元数据请求超时 | 超时 30 秒，失败时提示用户并允许手动重试 |
| 上传中断 | 不自动续传，允许用户重新上传 |
| 下载中断 | 支持断点续传（HTTP Range） |
| 预处理状态轮询间隔 | 固定 1 秒（详见 §3.3「刷新触发条件」），不暴露给用户调整 |
| 离线状态 | 已下载书籍的正文和播放进度可正常使用；TTS 合成需联网，离线时 UI 明确提示 |

服务端地址（Mac Mini M4）需在 App 端可配置：设置页提供"服务端地址"项（默认局域网 IP + 端口）。

### 3.8 音量增益与限幅保护

音量控制完全在 App 端实现，不经过服务端 TTS API（避免破坏缓存命中、节省算力）。

**目的**：在嘈杂环境下放大语音，改善可懂度。

**范围**：-12 dB ~ +20 dB，步进 1 dB，默认 0 dB（对应倍数 0.25x ~ 10x）。

**实现方式（AVAudioEngine 音频链）**：

```
AVAudioPlayerNode
    ↓
AVAudioUnitEQ（globalGain = 用户设置 dB）
    ↓
AVAudioUnitEffect(AUPeakLimiter)   ← 限幅保护，阈值 -0.3 dBFS
    ↓
mainMixerNode → output
```

**限幅器参数**：

- 使用 iOS 原生 `kAudioUnitSubType_PeakLimiter`（lookahead 峰值限幅器）
- **阈值**：-0.3 dBFS（保留少量 headroom 避免浮点数字削波）
- **Attack**：3 ms（快速响应语音瞬态）
- **Release**：50-100 ms（避免语音段落间出现 pump 效应）

**效果保证**：

- 即便增益 +20 dB，超过阈值的峰值会被 look-ahead 限幅器平滑压缩，不产生削波失真
- 语音的清晰度保留，听感自然
- 衰减区间（< 0 dB）无任何音质损失

### 3.9 UI 响应性与后台任务协同（全局强约束）

本节约束**整个 iOS App**，不限于播放页。任何调用网络的 View / Service /
状态机都必须遵循这些原则。

---

#### 3.9.1 核心原则 1：UI 更新不得阻塞等待任何网络/远程请求

本地已有的数据必须立即上屏；依赖网络的部分异步到达、按需补齐。
**任何 user-visible 的状态变化，从用户动作到发布第一条 `@Published` 更新
之间都不得出现 `await api.xxx(...)`。**

具体体现：

| 动作 | UI 立刻响应（同步 / 本地） | 后台异步（不阻塞 UI） |
|---|---|---|
| App 冷启动打开书架 | 立即读 `<Documents>/library/*/meta.json`，渲染已下载书籍列表 | 异步 `GET /api/books` 同步服务端状态，成功后 reconcile overlay，失败则留 banner 提示离线 |
| 点击书籍进入播放页 | 立即读本地 `meta.json` 构造 `LocalBook`，切换到 PlayerView | 章节元数据按需拉取（仅播放需要） |
| 点击下一章 / 上一章 / 目录跳转 | 读 `chapters/<n>.txt`、切换标题、切换正文 | 按需拉取该章句子元数据 |
| 点击播放 | 进入 playing 状态、显示暂停按钮 | 保证句子元数据已加载 → TTS 合成 → 解码入队 |
| 点击暂停 | 立刻停止音频 | （无） |
| 双击句子跳转 | 更新 `position` 发布给 UI | 从新位置继续 TTS 合成 |
| 退出播放页 | **必须**立刻停止音频 + 取消所有后台任务 | （无） |
| 选择文件导入 | 书架顶部立即插入占位行（status=uploading、spinner） | 异步读文件字节 + POST `/api/books/upload` |
| 滑动删除书籍 | 弹二次确认对话框（本地操作） | 点击确认后 `DELETE /api/books/<id>` → 成功再 `library.delete(id)` + 从列表移除 |
| 下拉刷新 | 保留当前列表（避免闪烁） | 异步 `GET /api/books` reconcile |
| 打开设置页 | 立即显示 UserDefaults 当前值 | （设置页无网络依赖） |
| 调节字号 / 增益 / 语速 | 立即生效：@Published 字段更新触发 View 重渲染 | 下一句 TTS 请求带新参数 |

**等价反向约束**：任何 `@MainActor` 标注的方法不得在 MainActor 上做可能
长时阻塞的磁盘 / 解压 / 大 JSON 编解码等 IO —— 这些操作要走
`Task.detached` 或专门的 actor。MainActor 只用来发布 `@Published` 更新和
响应用户事件。

**执行原则**（规避"本地先发布"被代码结构绕过的陷阱）：

1. `await network()` 前必须已经 publish 了本地快照（如果该 View 有本地
   数据源）。
2. `catch` 分支里"兜底 publish 本地"是**反例**，不是解决方案 —— 它
   意味着只有网络失败才展示本地，失败前的几秒用户盯着空页面。
3. 一个 async 方法里出现多次 `state.xx = ...`，**第一次**必须发生在
   **第一个 await 之前**。
4. await 后的 publish 要用"generation token"或"比对 current key"的方式
   二次确认（e.g. `guard state.currentChapterId == chapterId else { return }`），
   防止迟到的响应覆盖新状态。

**反例清单（全局）**：
- `await api.xx(); state.xx = ...; catch { state.xx = localFallback() }`
  —— 顺序搞反，冷启动时用户看不到任何东西直到网络失败。
- `if loadingX { ProgressView() } else { localContent }`
  —— 本地数据就绪时不应整屏 spinner。小 banner 可以，整屏不行。
- 在 View 的 body / computed property 里做同步磁盘 IO —— 每次 SwiftUI
  re-render 都会重跑。
- 在 `.navigationDestination` / `.sheet` 的 content closure 里阻塞式
  读文件解析 JSON —— 用 async `.task` + `@State` 做异步装配。

---

#### 3.9.2 核心原则 2：播放由两条合作的后台线索构成 —— 网络下载线索 + 音频播放线索 —— 共享一个按句子索引的滑动窗口

两条线索分工：

| 线索 | 职责 | 读写 |
|---|---|---|
| **网络下载线索** | 调用 `/chapters/{n}/meta` 拉章节元数据；调用 `/api/tts` 合成句子音频；把音频落盘到 `cache/tts/` | 读 `position`（决定窗口锚点）；写 `ttsCache`、`metaByChapter` |
| **音频播放线索** | 从缓存读取当前句子的音频文件，用 AVAudioEngine 串行播放；推进 `position`；更新高亮 | 读 `ttsCache`、`metaByChapter`；写 `position`、UI 高亮 |

**滑动窗口**：以"句子"为单位。窗口锚定在 `position` 上，覆盖
`[position, position + windowAhead]` 一共 `windowAhead + 1` 个句位
（当前 `windowAhead = 20` → 窗口 21 句，跨章续航）。窗口包含**当前正在
播放**的句 + **前方待播**的若干句。

- 网络线索：单 worker 串行（详见 §3.6.2），按 anchor → anchor+1 → ...
  的优先级顺序填空。一次只在飞一个 TTS 请求，不并行。
- 播放线索按序推进：`await awaitPrefetched(anchor)` —— 命中即秒回；
  未命中则订阅 worker 完成事件等待，播放，`position += 1`。
- 播放线索推进 `position` → 网络线索的窗口锚点随之滑动 → 旧槽位
  （已经在播放线索背后的句子）不再属于窗口；新的前沿槽位进入窗口并
  在下次 worker 迭代时被处理。

**窗口大小 ≠ 缓存大小**（非常重要）：

- **窗口**是"活跃预下载区间"。只有窗口内的句位是 worker 的工作目标；
  超出窗口的已完成 URL 会被丢弃以节约内存（音频本身仍在 TTSCache 里）。
  窗口越大，越能吸收网络抖动；但因为是串行预取，主要影响是"提前到
  多远以避免播放追赶"。当前选 20。
- **缓存**是已落盘的历史合成结果，容量远大于窗口（默认 500 MB，
  按距离当前位置淘汰，见 §3.6.1）。缓存的键是
  `SHA1(text | speaker | tone | speed)`，**不是**按位置索引 ——
  这意味着用户跳回已听过的段落、或跳到几秒前的上一句，只要 TTS 参数
  相同就秒播，不走网络。
- 换句话说：窗口决定"主动预取多少"，缓存决定"被动复用多少"。前者锁定
  GPU/网络负载上限，后者最大化跳转时的命中率。

**跳转（jump）= 移动窗口锚点**：

- 双击句子 / 上下章按钮 / 目录跳转 / 暂停后继续到新位置，统一建模为
  "写入新的 `position`"。
- 播放线索：`.cancel()` 当前 `playerTask`（停 `AVAudioPlayerNode`、
  停等待音频的 await、停序列推进），用新 anchor 重启。
- 网络线索：重算窗口，把**不在新窗口内**的正在飞 TTS 任务全部
  `.cancel()`（URLSession 层取消请求）；窗口内已完成的缓存条目自然保留；
  窗口内未开始的槽位触发新 TTS 请求。
- **逻辑取消兜底**：若请求已经发出但还没返回、技术上来不及 cancel 也
  没关系 —— 任务一旦被 `.cancel()`，其 `.value` 在播放线索侧永远不会被
  consume（播放线索已经在别的 Task 里了）；即便响应抵达，网络线索把
  它 `store(key)` 到 `ttsCache` 也只是多一条缓存条目，未来可能还会用到，
  不会污染 UI。

**章节元数据的协同**：

- 元数据同样按"章节"维度缓存在内存 `metaByChapter: [Int: [Sentence]]`。
- 窗口推进到当前章节末尾附近时，网络线索**提前**为 `currentChapterId + 1`
  拉元数据（`ensureMeta(nextChapterId)`），避免跨章时的空白。
- 章节元数据拉取失败不影响本章已有的播放；只会阻塞"窗口向下章扩张"。

**实现映射（`PlaybackService`）**：

| 概念 | 代码对应 |
|---|---|
| 播放位置 | `state.position: PlaybackPosition?` |
| 播放线索 | `playerTask: Task<Void, Never>?` |
| 已完成的预取 | `prefetchedURLs: [SentenceAddress: Result<URL, Error>]` |
| 单 worker 句柄 | `prefetchTask: Task<Void, Never>?` |
| player loop 等待者 | `prefetchSubscribers: [SentenceAddress: [CheckedContinuation]]` |
| 章节元数据缓存 | `metaByChapter: [Int: ChapterMeta]` + `metaInFlight: [Int: Task<Void, Never>]` |
| TTS 音频缓存 | `TTSCache`（`cache/tts/<sha1>.m4a`，AAC 48 kbps mono @ 24 kHz；§3.6.1） |
| 窗口滑动 | `reconcileWindow()` —— 每次 `position` 变化调用，唤起 worker |
| 串行 worker | `runPrefetchWorker()` —— 详见 §3.6.2 |
| player loop 等音频 | `awaitPrefetched(slot:)` —— 缓存命中即返回，否则订阅 worker 完成 |
| 跳转 | `jumpPlay(chapterId:sentenceIndex:)` → `pause()` → set position → `reconcileWindow` → `play()` |

**播放动作 × 取消清单**（补充 §3.9.1 总表里的"后台异步"列，详列每个动作
要取消的既有后台工作）：

| 动作 | 要取消的既有后台工作 |
|---|---|
| 点击下一章 / 上一章 / 目录跳转 | playerTask 中止；窗口锚点变 → reconcile 修剪窗口外 URL + 等待者；prefetch worker 在当前 fetch 完成后续按新窗口工作 |
| 点击播放 | （无，首次启动）|
| 点击暂停 | 当前 playerTask（其内部正在 await 的 TTS 不取消，让 worker 继续填缓存）|
| 双击句子跳转 | playerTask 中止；窗口锚点变 → reconcile 修剪 + 重启 worker（新 anchor 优先）|
| 自动切到下一章 | 无需特殊处理 —— 窗口本就跨章续航，worker 沿同一队列继续 |
| 退出播放页 | playerTask + prefetchTask + 所有 metaInFlight + `AVAudioEngine.stop()` + 清空 `prefetchedURLs` 与 `prefetchSubscribers` |

**分层职责**（`PlaybackService` + `PlaybackState`）：

- **UI state（本地、同步可得）**：`currentChapterId` / `currentChapterText`。
  `switchToChapter` 只摸本地磁盘，发布完这两个字段就返回。章节标题
  和正文在用户点击后的几毫秒内必然上屏。
- **播放 state（可能等待网络）**：`currentChapterSentences`。由
  `ensureSentencesLoaded` 懒加载，仅在真正需要播放（或高亮跟读）时
  才触发 `GET /chapters/{n}/meta`。即便这个请求耗时 30 秒，UI 完全
  不会 block —— 用户可以继续翻章节、阅读正文。
- **`fetchingSentences` 标志**只控制正文上方一小条 "正在加载句子信息…"
  提示条，**绝不**遮挡正文或禁用翻页按钮。

**播放页退出清理**（`PlayerView.onDisappear → playback.stop()`）：

调用 `PlaybackService.stop()` 会串行执行：

1. `pause()` → `playerNode.stop()` + 取消 `playbackTask`（不等当前句放完）
2. `state.position = nil`
3. 取消 `metaFetchTask`（章节元数据请求）
4. 取消并清空 `preloadTasks`（提前合成的下 N 句任务）
5. `engine.stop()` 卸掉 AVAudioEngine

**等价的反向约束**：`PlaybackService` 的任何 public 方法都不得在
MainActor 上做可能阻塞的磁盘/网络 IO。磁盘 IO（读 chapter 正文）走
`BookStore.chapterText` 的 `Task.detached`；网络走 async APIClient
actor。MainActor 只用来发布 `@Published` 更新。

**反例（不要再写）**：
- 在 `loadChapter` 里 `await api.chapterMeta(...)` **之后**才发布
  `state.currentChapterId` —— 章节切换被网络往返绑架。
- 在 ChapterTextArea 里用 `if loadingChapter { ProgressView() }` 全屏
  遮挡 —— 正文数据就在本地，没理由让用户盯着转圈。
- 在 `jumpTo` 里 `loadChapter` 失败后仍旧写 `state.position` ——
  chapterId 与 position.chapterId 会错位。
- 在 `fetchAudio` 的 retry 循环里 `try?` 吞掉取消信号、让被取消的
  playbackTask 继续把剩下 2 次 retry 跑完 —— 已取消的那一句必须立刻
  停止，不给浪费 GPU/网络的机会。
- 在 `kickOffPreload` 里只把旧 key 从 dict 里 `filter` 掉却不 `.cancel()`
  背后的 Task —— 跳过的句子仍在后台做 TTS 合成。preload 的活跃窗口是
  `[current, current+preloadAhead]`，窗口外的 Task 全部 cancel。
- 切章时不清 preloadTasks —— 老章的预加载会和新章的预加载抢 GPU。
- 在 async 的 `switchToChapter` / `ensureSentencesLoaded` 里 await 完网络
  或磁盘后无脑 publish 到 `state` —— 必须在 await 后重新用
  `state.currentChapterId == chapterId` 二次确认，老请求的迟到响应不得
  污染新章的 UI。

---

## 四、开发阶段计划

### Phase 1：服务端基础
1. 项目初始化（FastAPI 项目结构）
2. Qwen3-TTS 0.6B MLX 部署与 TTS API（4bit 量化，零样本克隆）
3. 书籍上传 + TXT/EPUB 解析 + 章节拆分
4. 书籍列表 + 下载 + 删除 API
5. 参考音频生成（Qwen3-TTS VoiceDesign 本地合成 64 条）+ 音色库标注

### Phase 2：服务端预处理
6. LLM backend：DeepSeek V4 flash（OpenAI 兼容协议，no thinking）
7. 章节识别（§2.4.1，TXT 走 LLM 分页，EPUB 走 TOC）
8. 章节元数据生成（§2.3，一次 LLM 调用产出两段 NDJSON：
   句子 + 角色画像更新，Python 做位置映射）
9. 渐进式章节元数据处理 API + lookahead

### Phase 3：iOS App MVP
10. 书架页面（书籍列表、状态显示）
11. 书籍导入（文件选择 + 上传服务端）
12. 播放页面（正文渲染 + 上下滚动 + 左右切换章节）
13. 目录页面
14. 基础播放控制（播放/暂停 + 逐句高亮）
15. 调用服务端 TTS API + 音频预加载

### Phase 4：iOS 播放体验
16. 播放位置与浏览位置分离 + 跟读模式
17. 双击句子切换播放位置
18. 播放位置持久化（SwiftData）
19. 后台播放 + 锁屏控制
20. 书架下拉刷新 + 导入中状态轮询

### Phase 5：进阶功能
21. 语速调节
22. 定时停止（睡眠模式）
23. 服务端导入书籍支持
24. 字号调节

---

## 五、技术选型汇总

| 组件 | 技术选择 | 说明 |
|------|---------|------|
| iOS UI | SwiftUI | iOS 16+ |
| 音频播放 | AVAudioEngine | 后台播放 + 锁屏 |
| iOS 存储 | SwiftData | 播放进度持久化 |
| 服务端框架 | FastAPI | Python，异步支持 |
| TTS 引擎 | Qwen3-TTS 0.6B (MLX 4bit) | 本地推理，零样本克隆，RTF ~0.6 |
| 音色库 | Qwen3-TTS VoiceDesign 1.7B 本地合成的 64 条参考音频 | 手工 tag→属性映射；运行时仍走 Base 0.6B 零样本克隆 |
| LLM | DeepSeek V4 flash | 云端 OpenAI 兼容 API，no thinking |
| 预处理硬件 | Mac Mini M4, 16GB | 服务端 + TTS 模型推理（LLM 走云，本地只跑 TTS ~2 GB 内存）|

---

## 六、验证方式

1. 准备一本中文 TXT 小说（10 万字级别），手动走通服务端导入 + 预处理全流程
2. 验证 TTS API 合成质量：多角色音色、语气表现
3. 在 iOS 真机上测试：书籍导入 → 播放 → 句级高亮 → 章节跳转
4. 验证播放体验：预加载延迟、跟读模式、后台播放、锁屏控制
5. 验证渐进式处理：首次打开章节的响应速度
