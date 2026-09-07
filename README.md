# 抖音收藏知识库

把个人抖音收藏夹变成一个**可搜索、可问答的本地知识库**：爬取收藏元数据入库 → 按需语音转写 → AI 生成概要 → 全文检索 + 对 AI 提问。全程本地运行，数据不出本机（仅转写/概要/问答调用 LLM API）。

## 功能一览

- **采集**：扫码登录抖音（登录态落盘），自动滚动收藏夹采集「标题 / 标签 / 作者 / 链接 / 视频直链」入库，支持增量同步
- **搜索**：SQLite FTS5 trigram 全文索引（标题、标签、作者、转写文本），jieba 分词 + bm25 排序，忘记关键词也能用整句召回；≤2 字短词自动降级 LIKE
- **转写**：yt-dlp 即时下载音频（直链有时效，不提前囤）→ faster-whisper（CPU int8，本地推理）→ OpenCC 繁转简后入库
- **AI 概要 / 问答**：基于已转写内容生成概要；提问时检索相关转写片段喂给 LLM，回答末尾标注来源视频
- **自动分类**：8 个固定类别批量归类（AI 技术 / 编程开发 / 知识学习 / 心理成长 / 生活 / 小说写作 / 游戏娱乐 / 软件工具）
- **Web 界面**：单页应用，搜索高亮、分类筛选、勾选按需转写、任务进度条 + 取消 + 历史记录、一键转写/概要/分类

## 项目结构

```
├── main.py              # CLI 入口（login/crawl/search/transcribe/summarize/ask/web/stats）
├── app/
│   ├── crawler.py       # Playwright 登录 + 旁听接口采集（不逆向签名）
│   ├── db.py            # SQLite + FTS5 层：建表/迁移/搜索/去重入库
│   ├── transcribe.py    # yt-dlp 下载音频 + faster-whisper 转写 + 繁转简
│   ├── llm.py           # OpenAI 兼容客户端（DeepSeek 等），概要与问答
│   └── web.py           # FastAPI 后端：搜索/问答/任务（进度/取消/历史）
├── static/index.html    # 单页前端（原生 JS，无框架）
├── data/                # favorites.db（SQLite 数据库）
├── browser_data/        # Playwright 登录态
├── models/              # 本地 whisper 模型（faster-whisper-small）
├── audio_cache/         # 转写用音频临时目录
└── .env                 # LLM 密钥配置（不入库）
```

## 快速开始

### 1. 安装依赖

```bash
# 建议在项目目录内建虚拟环境
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 Chromium（国内镜像，官方 CDN 可能极慢）
$env:PLAYWRIGHT_DOWNLOAD_HOST="https://registry.npmmirror.com/-/binary/playwright"
playwright install chromium
```

### 2. 配置 LLM（概要 / 问答需要）

在项目根目录建 `.env`：

```ini
LLM_API_KEY=sk-xxxx          # 任何 OpenAI 兼容服务的 key
LLM_BASE_URL=https://api.deepseek.com/v1/
LLM_MODEL=deepseek-chat
```

### 3. 首次使用

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
python main.py ask "RAG 面试会问哪些问题？"
python main.py stats
```

## 技术要点

- **采集不逆向**：Playwright 打开真实浏览器，旁听 `aweme/favorite` 与 `listcollection` 接口响应，签名参数由页面自己生成
- **FTS5 trigram**：中文按 3 字滑窗建索引，支持任意子串匹配；`search()` 内 FTS 与 LIKE 双路径自动切换
- **即时下载转写**：视频直链有时效，转写时才用 yt-dlp + cookies 拉音频，失败重试 3 次（随机退避），仍失败标记「音频不可用」不阻塞队列
- **协作式取消**：后台任务在条目间检查取消标志，已抓到的数据照常入库
- **容错优先**：接口解析层全 `.get()` 容错，抖音字段改版只影响解析不影响整体

## 常见问题

| 现象 | 处理 |
|---|---|
| 搜索命中为 0 | 换更短的关键词（≤2 字走 LIKE 路径）；或先转写更多视频扩大语料 |
| 转写报 "Fresh cookies needed" | 登录过期，页面上点「扫码登录」重新登录后同步 |
| 转写慢 | 默认 small 模型 CPU 推理，可 `--model base` 或 `tiny` 换速度 |
| LLM 报 401/余额 | 检查 `.env` 的 key；key 泄露过请及时在服务商处轮换 |
| 想重置 | 关服务后删除 `data/`（数据库）、`browser_data/`（登录态）即可 |

## 隐私

收藏数据、登录态、转写文本全部存在本机；`.env`、`browser_data/`、`data/` 均已在 `.gitignore` 中，不会被提交。
