# 抖音收藏知识库

把个人抖音收藏夹变成一个**可搜索、可问答的本地知识库**：爬取收藏元数据入库 → 按需语音转写 → AI 生成概要 → 全文检索 + 对 AI 提问。全程本地运行，数据不出本机（仅转写/概要/问答调用 LLM API）。

## 功能一览

- **采集**：扫码登录抖音（登录态落盘），自动滚动收藏夹采集「标题 / 标签 / 作者 / 链接 / 视频直链」入库，支持增量同步
- **搜索**：SQLite FTS5 trigram 全文索引（标题、标签、作者、转写文本），jieba 分词 + bm25 排序，忘记关键词也能用整句召回；≤2 字短词自动降级 LIKE
- **转写**：yt-dlp 即时下载音频（直链有时效，不提前囤）→ faster-whisper（CPU int8，本地推理）→ OpenCC 繁转简后入库
- **AI 概要 / 问答**：基于已转写内容生成概要；提问时用本地 bge 句向量做**语义 + 关键词混合检索**（RRF 融合，向量在本地 ONNX 推理、不联网、零 API 成本），把命中的**相关片段/概要**而不是固定开头喂给 LLM，回答末尾标注来源视频
- **自动分类**：LLM 先抽样浏览收藏内容、自动设计一套贴合的类别（6~12 个 + 兜底「其他」，存库复用），再批量归类；换台机器、换个收藏夹，类别会跟着内容变，不写死
- **Web 界面**：单页应用，搜索高亮、分类筛选、勾选按需转写、任务进度条 + 取消 + 历史记录、一键转写/概要/分类

## 项目结构

```
├── main.py              # CLI 入口（login/crawl/search/transcribe/summarize/ask/web/stats）
├── app/
│   ├── crawler.py       # Playwright 登录 + 旁听接口采集（不逆向签名）
│   ├── db.py            # SQLite + FTS5 层：建表/迁移/搜索/去重入库
│   ├── transcribe.py    # yt-dlp 下载音频 + faster-whisper 转写 + 繁转简
│   ├── llm.py           # OpenAI 兼容客户端（DeepSeek 等），概要与问答
│   ├── embedder.py      # 本地句向量 bge-small-zh-v1.5 ONNX（onnxruntime + tokenizers）
│   ├── indexer.py       # 转写/概要/标题 → 分块 → 向量索引（内容签名增量更新）
│   ├── retriever.py     # 语义 + 关键词混合召回（RRF 融合）→ 问答上下文
│   └── web.py           # FastAPI 后端：搜索/问答/任务（进度/取消/历史）
├── static/index.html    # 单页前端（原生 JS，无框架）
├── data/                # favorites.db（SQLite 数据库）
├── browser_data/        # Playwright 登录态
├── models/              # 本地模型：faster-whisper-small + bge-small-zh-v1.5
├── audio_cache/         # 转写用音频临时目录
└── .env                 # LLM 密钥配置（不入库）
```

## 快速开始

### 方式一：一键启动（Windows 推荐）

1. 安装 [Python 3.10+](https://www.python.org/downloads/)（安装时勾选 **Add python.exe to PATH**）
2. 把整个项目文件夹放到任意位置，**双击 `start.bat`**，脚本会按步骤执行并显示进度：
   - `[1/5]` 创建虚拟环境
   - `[2/5]` 安装依赖（清华 pip 源 + npmmirror Chromium 镜像）
   - `[3/5]` 下载 Whisper 语音模型（**约 461MB，仅首次**，来自 hf-mirror.com 镜像，存到 `models/`，来源/去向/用途都会打印在屏幕上）
   - `[4/5]` 下载 BGE 句向量模型（**约 120MB，仅首次**，问 AI 的语义检索用；缺了不致命，问答会自动退化为纯关键词检索）
   - `[5/5]` 启动服务并自动打开浏览器
3. 首次运行会提示：打开 `.env` 填入 `LLM_API_KEY`（[DeepSeek](https://platform.deepseek.com) 等任何 OpenAI 兼容服务的 key），保存后再双击一次 `start.bat`
4. 页面里点「扫码登录」→「同步收藏夹」→「一键分类全部」，之后就能搜索和转写了

### 方式二：手动安装（跨平台 / 想了解细节）

```bash
# 建议在项目目录内建虚拟环境
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 Chromium（国内镜像，官方 CDN 可能极慢）
$env:PLAYWRIGHT_DOWNLOAD_HOST="https://registry.npmmirror.com/-/binary/playwright"
playwright install chromium
```

### 配置 LLM（概要 / 问答需要）

复制 `.env.example` 为 `.env`，填入你的 key：

```ini
LLM_API_KEY=sk-xxxx          # 任何 OpenAI 兼容服务的 key
LLM_BASE_URL=https://api.deepseek.com/v1/
LLM_MODEL=deepseek-chat
```

### 首次使用

```bash
python main.py login     # 弹出浏览器扫码登录，登录态自动落盘
python main.py crawl     # 滚动采集收藏夹入库
python main.py web       # 打开 http://127.0.0.1:8642
```

之后所有操作（搜索、转写、概要、分类、同步）都在 Web 页面上完成；CLI 同样可用：

```bash
python main.py search 熵增
python main.py transcribe 10          # 转写 10 条；--all 转写全部；--model base 换模型
python main.py summarize 10
python main.py index                  # 预先构建语义索引；不跑的话 ask 时也会自动增量构建
python main.py ask "RAG 面试会问哪些问题？"
python main.py stats
```

### 分享给别人

把项目文件夹打包发出去即可，但**排除这些本机数据目录**（接收方会自动生成自己的）：

```
排除：.venv/  data/  browser_data/  models/  audio_cache/  __pycache__/  .env
保留：start.bat  main.py  app/  static/  requirements.txt  .env.example  README.md
```

接收方只需装 Python，双击 `start.bat`，填一次自己的 API key 就能用。

## 技术要点

- **采集不逆向**：Playwright 打开真实浏览器，旁听 `aweme/favorite` 与 `listcollection` 接口响应，签名参数由页面自己生成
- **FTS5 trigram**：中文按 3 字滑窗建索引，支持任意子串匹配；`search()` 内 FTS 与 LIKE 双路径自动切换
- **混合检索（问 AI）**：`app/retriever.py` 把本地 bge 语义向量（float32 存进 SQLite BLOB，numpy 暴力余弦；语料千级下矩阵乘微秒级，故不引入 sqlite-vec）与 FTS 关键词两路结果用 **RRF** 融合。关键词路只搜「有效转写」、全词 AND 优先、bm25 列权重（标题 8 / 标签 3 / 作者 2 / 转写 1）；上下文取**命中片段**（向量命中的转写块，或问题词附近的窗口），不再固定取转写开头
- **索引自愈**：每个视频存内容签名（sig），转写/概要更新只重嵌入该视频；`ask` 前自动检查补建，`meta.embed_model` 变化（换模型）触发全量重建；向量模型缺失时向量路静默跳过，问答退化为关键词检索，不会报错卡住
- **即时下载转写**：视频直链有时效，转写时才用 yt-dlp + cookies 拉音频，失败重试 3 次（随机退避），仍失败标记「音频不可用」不阻塞队列
- **协作式取消**：后台任务在条目间检查取消标志，已抓到的数据照常入库
- **容错优先**：接口解析层全 `.get()` 容错，抖音字段改版只影响解析不影响整体

## 常见问题

| 现象 | 处理 |
|---|---|
| 搜索命中为 0 | 换更短的关键词（≤2 字走 LIKE 路径）；或先转写更多视频扩大语料 |
| 问答答非所问 / 检索不到相关视频 | 页面点「重建语义索引」；语义检索需要 `models/bge-small-zh-v1.5`（start.bat 会自动下载，也可手动 `python scripts/download_model.py bge`），缺模型时会退化为纯关键词检索 |
| 转写报 "Fresh cookies needed" | 登录过期，页面上点「扫码登录」重新登录后同步 |
| 转写慢 | 默认 small 模型 CPU 推理，可 `--model base` 或 `tiny` 换速度 |
| LLM 报 401/余额 | 检查 `.env` 的 key；key 泄露过请及时在服务商处轮换 |
| 想重置 | 关服务后删除 `data/`（数据库）、`browser_data/`（登录态）即可 |

## 隐私

收藏数据、登录态、转写文本全部存在本机；`.env`、`browser_data/`、`data/` 均已在 `.gitignore` 中，不会被提交。
