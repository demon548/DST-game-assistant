# 🎮 饥荒联机版 智能游戏助手 — 基于 RAG 的 AI 问答系统

> NLP 期末大作业 · 2026 年夏季学期  
> 技术栈：Python + LangChain + ChromaDB + BGE Embedding + DeepSeek API + Streamlit

---

## 📋 项目概述

一个面向《饥荒联机版》(Don't Starve Together) 的领域 AI 助手，基于 **RAG（检索增强生成）** 技术构建。系统从灰机 Wiki 等来源收集游戏攻略知识，通过文本切分、向量化存入 ChromaDB，用户提问时自动检索最相关的内容并交由 DeepSeek 大模型生成回答。

### 核心能力

| 功能 | 说明 |
|------|------|
| 🔍 语义检索 | BGE 中文 Embedding 模型，理解游戏术语 |
| 🧠 知识增强生成 | DeepSeek API + 检索到的知识上下文 |
| 📎 来源引用 | 回答中标注 `[1]` `[2]` 角标，注明信息来源 |
| 🚫 边界拒答 | 知识库未覆盖时明确告知，不编造内容 |
| 💬 对话界面 | Streamlit 聊天式 UI，支持连续追问 |

---

## 📁 项目结构

```
game_assistant/
│
├── app.py                  # 🖥️ Streamlit 网页界面（对话交互）
├── rag_pipeline.py         # ⚙️ RAG 核心引擎（文档加载/切分/向量化/检索/生成）
├── build_kb.py             # 🔨 预构建向量库脚本（运行一次，后续秒开）
├── scrape_huijiwiki.py     # 🕷️ 灰机Wiki 爬虫（知识库采集工具）
│
├── .env                    # 🔑 API Key 配置（不提交 Git）
├── env.example             # 📄 环境变量模板（提交 Git，不含真实 Key）
├── requirements.txt        # 📦 依赖清单（待生成）
│
├── data/
│   ├── raw/                # 📚 原始知识库（45 个 txt，329,000 字，219 页）
│   │   ├── 角色背景故事/   # 温蒂、威尔逊、麦斯威尔等全部角色 lore
│   │   ├── 食物与烹饪/     # 80 道料理配方 + 系数系统
│   │   ├── 生物与Boss/     # 10 个 Boss 详细打法 + 中立/敌对生物
│   │   ├── 装备与合成/     # 武器/护甲/工具/魔法/建筑全套
│   │   ├── 四季生存与新手/ # 秋冬春夏 + 开荒建家攻略
│   │   ├── 游戏机制/       # 三维/温度/湿度/战斗/复活/光源
│   │   ├── 种田攻略/       # 14 种作物 + 巨大作物 + 黄金配比
│   │   ├── 洞穴与月岛/     # 洞穴探索 + 天体线完整流程
│   │   ├── 2026版本更新/   # 咒终对决·上部 + WX-78 重做
│   │   └── 灰机Wiki/       # 从灰机Wiki爬取的36个补充页面
│   │
│   └── chroma_db/          # 🗄️ ChromaDB 向量数据库（8.9MB）
│
├── test_cases/             # 🧪 测试用例（待填充）
│
└── NLP期末大作业_2026...pdf # 📄 作业要求文档
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# Python 3.10+ 推荐
pip install langchain langchain-openai langchain-community langchain-huggingface \
            langchain-chroma langchain-text-splitters python-dotenv sentence-transformers \
            chromadb tiktoken cloudscraper beautifulsoup4 lxml streamlit
```

### 2. 配置 API Key

复制 `env.example` 为 `.env`，填入 DeepSeek API Key：

```bash
cp env.example .env
# 编辑 .env 文件，将 DEEPSEEK_API_KEY 改为你的真实 Key
```

> 🔑 在 [platform.deepseek.com](https://platform.deepseek.com) 获取 API Key（新用户有免费额度）。

### 3. 构建向量知识库（首次必须）

```bash
python build_kb.py
```

首次运行会自动：
1. 加载 `data/raw/` 下所有 45 个 txt 文件
2. 下载 BGE Embedding 模型（~92MB，仅首次，缓存在 `~/.cache/huggingface/`）
3. 切分成 ~1000 个 500 字文本块
4. 向量化并存入 `data/chroma_db/`

> ⏱️ **耗时约 2-5 分钟（CPU）**。之后启动无需重建。

### 4. 启动 Web 界面

```bash
streamlit run app.py --server.port 8501
```

---

## 📊 知识库规模

| 指标 | 数值 |
|------|------|
| 总字符数 | **329,708 字** |
| 约合页数 | **219 页**（按 1500 字/页） |
| 知识文件数 | **45 个 txt** |
| 主题模块数 | **18 个类别** |
| 向量块数 | ~1000 chunks |
| 向量库大小 | 8.9 MB |

### 知识来源

| 来源 | 内容 | 比例 |
|------|------|------|
| WebSearch 编译 | 核心攻略（角色/Boss/食物/装备/四季/机制） | ~13% |
| 灰机Wiki 爬取 | 36 个页面（机制细节/角色专题/物品/更新） | ~87% |
| 2026 最新更新 | 咒终对决·上部、WX-78 重做、春节活动 | ✅ 已包含 |

---

## 🔧 如何扩展知识库

知识库设计为**模块化、可增量更新**：

1. **添加新知识**：在 `data/raw/` 下创建新的 `.txt` 文件（按分类组织到子文件夹）。
2. **重新构建**：运行 `python build_kb.py`，新内容自动纳入向量库。
3. **验证效果**：在 Streamlit 中提问新知识相关问题，确认能正确检索。

> 💡 `scrape_huijiwiki.py` 中的 `PAGES` 列表可以直接添加新的 Wiki 页面 URL，扩展爬取范围。

---

## 🧪 当前进度

### ✅ 已完成

| 阶段 | 内容 | 状态 |
|------|------|------|
| Step 1 | 数据收集 + 知识库构建（329,000 字/219 页） | ✅ |
| Step 2 | 文本切分 + BGE 向量化 + ChromaDB 存储 | ✅ |
| Step 3 | RAG 核心问答链路（检索→增强→生成） | ✅ |
| Step 4 | Streamlit 聊天界面（含角标来源引用） | ✅ |
| Step 5 | 设计 20 个覆盖不同模块的测试用例 | ✅ |
| Step 6 | 查询改写：口语化问题→检索友好查询 | ✅ |
| Step 7 | 问题分解：复杂问题拆成 2-3 个子问题 | ✅ |
| Step 8 | 边界拒答：超出知识范围时明确拒绝 | ✅ |
| Step 6 | 查询改写：口语化问题→检索友好查询 | ✅ |

### ⏳ 待完成

| 阶段 | 内容 | 预估工作量 |
|------|------|-----------|
| Step 5 | **测试与对比实验** | 2-3 天 |
| | — LLM-only vs RAG 回答质量对比 | |
| | — 2 个失败案例分析（何时 RAG 回答不好？） | |
| | — 准确率/召回率评估 | |
| Step 6 | **Agent 增强（加分项）** | 3-5 天 |
| | — 查询路由：判断问题类型 → 选择检索策略 | |
| | — 回答自检：生成后检查是否有知识支撑 | |
| 文档 | 项目报告 + 演示视频 | 2-3 天 |

---

## 📄 各文件详细说明

### `rag_pipeline.py` — RAG 核心引擎（255 行）

完整的 RAG 流水线实现，包含 7 个核心函数：

| 函数 | 功能 | 关键参数 |
|------|------|----------|
| `load_documents()` | 加载 `data/raw/` 下所有 txt | DirectoryLoader, glob 匹配 |
| `split_documents()` | 切分长文本为小块 | chunk_size=500, overlap=50 |
| `create_vector_store()` | BGE 向量化 + 存入 Chroma | persist 到 data/chroma_db/ |
| `load_vector_store()` | 加载已有向量库 | 检测目录是否已存在 |
| `retrieve()` | 语义相似度检索 | k=4, 返回最相关 chunk |
| `generate_answer()` | 基于检索结果生成回答 | DeepSeek API + Prompt 模板 |
| `RAGAssistant` | 封装所有功能的类 | `ask(question)` 一行调用 |

**技术决策**：
- LLM：DeepSeek API（deepseek-chat），兼容 OpenAI 接口，成本低
- Embedding：BAAI/bge-small-zh-v1.5（本地运行，免费，中文优化）
- 向量库：ChromaDB（轻量，本地持久化，无需额外服务）
- `HF_HUB_OFFLINE=1`：强制离线模式，跳过 HuggingFace 联网验证

### `app.py` — Streamlit 网页界面（105 行）

- 聊天式 UI，输入框自然沉底
- `@st.cache_resource` 缓存 RAGAssistant 实例
- 首次使用显示欢迎消息 + 功能模块介绍
- 侧边栏展示技术栈、知识库覆盖范围
- 支持清空对话按钮

### `build_kb.py` — 预构建脚本（15 行）

独立于 Streamlit 运行，提前完成耗时的向量化工作。知识库更新后运行一次即可。

### `scrape_huijiwiki.py` — Wiki 爬虫（234 行）

- 使用 MediaWiki API 获取页面内容（比 HTML 爬取更稳定）
- 支持 URL 编码页面名 + 中文搜索两种方式
- 礼貌爬取：每次请求间隔 1-1.5 秒
- 自动清洗 Wiki 导航栏、引用标记等噪声

---

## 🛠️ 技术架构

```
用户浏览器 (Streamlit)
        │
        ▼
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│   app.py     │────▶│ rag_pipeline  │────▶│  DeepSeek API│
│  Web 界面    │     │  RAG 核心引擎  │     │  生成回答     │
└──────────────┘     └───────┬───────┘     └──────────────┘
                             │
                    ┌────────▼────────┐
                    │    ChromaDB     │
                    │   向量数据库     │
                    │  data/chroma_db │
                    └────────▲────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼────────┐          ┌────────▼────────┐
     │  BGE Embedding  │          │  data/raw/      │
     │  文本 → 向量     │          │  45 个 txt      │
     │  (本地 CPU)      │          │  329,000 字     │
     └─────────────────┘          └─────────────────┘
```

---

## ⚠️ 注意事项

1. **`.env` 文件不要提交到 Git** — 包含真实 API Key。`.gitignore` 应包含 `.env`。
2. **`data/chroma_db/` 无需提交** — 每个协作者在自己的机器上运行 `python build_kb.py` 构建即可。
3. **BGE 模型首次下载** — 如果网络不通，可设置 `HF_ENDPOINT=https://hf-mirror.com` 使用国内镜像。
4. **DeepSeek API 费用** — deepseek-chat 模型价格为 ¥1/百万 tokens，本项目的问答成本极低。

---

## 👥 协作说明

1. 各自配置自己的 `.env` 文件（不共享 API Key）
2. 各自运行 `python build_kb.py` 构建本地向量库
3. 知识库更新：往 `data/raw/` 添加新文件 → 提交 → 协作者 pull 后重新 `build_kb.py`
4. 代码修改：`rag_pipeline.py` 是核心，`app.py` 是 UI，分工明确

---

## 📝 下一步要改进的地方：
①	安全对齐	🔴 高	Prompt 加"一旦判定违规，立即停止，不提供任何细节"

③	幻觉编造	🔴 高	Prompt 加强"不要编造知识库中没有的机制"

④	空/乱码输入	🟡 中	代码层加输入验证前置

②	检索召回	🟡 中	增大 k 值或对多 chunk 做重排序

⑤	回答缺细节	🟢 低	Prompt 加"一次性给出完整回答，不要反问"

---
> 📅 最后更新：2026-07-07
