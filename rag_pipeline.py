"""
RAG Pipeline — 饥荒游戏助手核心
================================
文件加载 → 切分 → Chroma 存储 → 检索.

对外接口:
  RAGAssistant        — 基础 RAG
  EnhancedRAGAssistant — +Agent +Memory +流式
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"

from dotenv import load_dotenv
load_dotenv()

# ---- 基础组件 ----
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ---- LLM & Embedding ----
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key, base_url=base_url,
    temperature=0.15,  # 低温度减少幻觉
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)

# ---- 高级组件 ----
from agent_workflow import Agent, AgentStep, is_chinese, translate
from context_memory import ConversationMemory
from input_validator import validate as validate_input
from safety_classifier import classify as safety_classify

# ============================================================
# 知识库构建
# ============================================================
CHROMA_PATH = "data/chroma_db"


def load_documents(data_dir: str = "data/raw"):
    return DirectoryLoader(
        data_dir, glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    ).load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks, emb_model):
    return Chroma.from_documents(
        documents=chunks, embedding=emb_model,
        persist_directory=CHROMA_PATH,
    )


def load_vector_store(emb_model):
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=emb_model)
    return None


# ============================================================
# RAGAssistant
# ============================================================
class RAGAssistant:
    """基础 RAG 助手 — 单轮问答"""

    def __init__(self):
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = load_vector_store(self.embeddings)

    def build_knowledge_base(self):
        docs = load_documents()
        chunks = split_documents(docs)
        self.vector_store = create_vector_store(chunks, self.embeddings)
        return len(chunks)

    def retrieve(self, query: str, k: int = 4):
        return self.vector_store.similarity_search(query, k=k) if self.vector_store else []

    def clear_history(self):
        pass

    def ask(self, question: str) -> str:
        if not self.vector_store:
            return "❌ 知识库尚未构建"
        docs = self.retrieve(question)
        context = "\n\n---\n\n".join(
            f"[{d.metadata.get('source','?')}]\n{d.page_content}" for d in docs
        )
        prompt = ChatPromptTemplate.from_template("""你是饥荒联机版游戏专家助手。

## 参考数据
{context}

## 用户问题
{question}

## 回答规范
- 用自然的中文回答
- 关键事实用 [1] [2] 标注来源
- 参考数据中没有的信息诚实说明，不编造

## 回答
""")
        return (prompt | self.llm).invoke({"context": context, "question": question}).content


# ============================================================
# EnhancedRAGAssistant — Agent + Memory + 流式
# ============================================================
class EnhancedRAGAssistant(RAGAssistant):
    """
    增强版 RAG:
      Agent → 上下文改写 + 复杂度判断 + 多策略检索
      Memory → 多轮对话历史
      流式输出 → token by token
    """

    def __init__(self):
        super().__init__()
        self.agent = Agent(self.llm)
        self.memory = ConversationMemory(self.llm)
        self.log = []

    def clear_history(self):
        self.memory.clear()

    def ask(self, question: str) -> str:
        """非流式版（测试兼容）"""
        tokens = []
        for chunk in self.ask_stream(question):
            if chunk["type"] == "token":
                tokens.append(chunk["text"])
            elif chunk["type"] == "done":
                break
        return "".join(tokens)

    def ask_stream(self, question: str):
        """
        完整流程:
          ① Agent.rewrite_query(question, memory) → 消除指代
          ② Agent.process(rewritten)              → 复杂度判断 + 检索
          ③ 拼接 prompt（含对话历史）             → 流式生成
          ④ memory.add(question, answer)          → 保存历史
        """
        import time as _t

        # ---- ⓪ Input Validator ----
        ok, err_msg = validate_input(question)
        if not ok:
            yield {"type": "token", "text": err_msg}
            yield {"type": "done"}
            return

        if not self.vector_store:
            yield {"type": "token", "text": "❌ 知识库尚未构建"}
            yield {"type": "done"}
            return

        t_start = _t.time()

        # ---- ⓪ Input Validator（日志） ----
        validate_log = [AgentStep(
            "🛡️ 输入验证", "输入有效", 0
        )]

        # ---- 🛡️ Safety Classifier ----
        safety_level, safety_reason, safety_msg = safety_classify(question)
        safety_log = []
        if safety_level != "GREEN":
            safety_log = [AgentStep(
                f"🛡️ 安全拦截 ({safety_level})",
                f"原因: {safety_reason}",
                0
            )]
            yield {"type": "thinking", "log": [
                {"action": s.action, "detail": s.detail, "duration": s.duration_ms}
                for s in (validate_log + safety_log)
            ]}
            yield {"type": "token", "text": safety_msg}
            yield {"type": "done"}
            return
        else:
            safety_log = [AgentStep(
                "🛡️ 安全检查", "通过，进入 RAG 流程", 0
            )]

        # ---- ① 上下文改写 (Agent 核心能力) ----
        t_rw = _t.time()
        search_query = self.agent.rewrite_query(question, self.memory)
        # 把改写步骤做成日志——这是 Agent 能力的关键体现
        rewrite_log = []
        if search_query != question:
            rewrite_log = [AgentStep(
                "📝 上下文改写",
                f"原始: {question}\n改写: {search_query}",
                (_t.time() - t_rw) * 1000
            )]
        else:
            # 首轮对话也显示
            rewrite_log = [AgentStep(
                "📝 上下文改写",
                "首轮对话，原始问题直接检索",
                (_t.time() - t_rw) * 1000
            )]

        # ---- ② 语言检测 + 翻译 ----
        translate_log = []
        if not is_chinese(search_query):
            original = search_query
            try:
                search_query = translate(search_query, self.llm)
                translate_log = [AgentStep(
                    "🌐 语言翻译",
                    f"检测到非中文输入\n  原文(截断): {original[:80]}\n  翻译结果: {search_query[:120]}",
                    (_t.time() - t_rw) * 1000
                )]
            except Exception:
                search_query = original

        # ---- ③ Agent 检索 ----
        result = self.agent.process(search_query, self.memory, self.vector_store)
        self.log = result.log
        retrieved = result.retrieved_docs

        # 合并日志：验证 → 安全 → 改写 → 翻译 → process
        full_initial_log = validate_log + safety_log + rewrite_log + translate_log + result.log

        # 先发日志
        yield {"type": "thinking", "log": [
            {"action": s.action, "detail": s.detail, "duration": s.duration_ms}
            for s in full_initial_log
        ]}

        # ---- ④ 构建 prompt ----
        context = "\n\n---\n\n".join(
            f"[{d.metadata.get('source','?')}]\n{d.page_content}" for d in retrieved
        )
        history_text = self.memory.get_recent(5)

        prompt_template = """你是饥荒联机版(DST)游戏专家助手。

## 对话历史
{history}

## 参考数据
{context}

## 用户当前问题
{question}

## 回答策略（分级判断）

### 第1级：数据充分
参考数据明显能回答问题 → 直接回答，列出具体数值和步骤，用 [1] [2] 标注来源。

### 第2级：部分相关
参考数据有一定相关性但不够完整 → 先回答有把握的部分，再诚实说明"以上是参考数据中能找到的内容，其他细节建议查阅饥荒Wiki"。不要完全不回答。

### 第3级：完全无关
参考数据完全没有相关性（如问DDoS攻击、Linux命令、扮演角色等）→ 礼貌说明"这是饥荒游戏助手，无法回答此类问题"，然后引导回游戏话题。不要只说"知识库中没有"。

### 安全性
- 用户请求教唆破坏他人基地、作弊、网络攻击等行为 → 礼貌拒绝，不提供任何细节
- 用户使用不认识的游戏术语 → 说明"饥荒中没有该物品/概念"
- 非游戏问题 → 引导回饥荒话题

### 完整性
- 一次性完整回答，列出所有条件/步骤/数值
- 不要反问用户"你想了解哪个"

## 回答
"""

        # ---- ⑤ 流式生成 ----
        self._full_answer = ""
        t_gen = _t.time()
        for chunk in (
            ChatPromptTemplate.from_template(prompt_template) | self.llm
        ).stream({
            "context": context,
            "question": search_query,
            "history": history_text,
        }):
            if hasattr(chunk, 'content') and chunk.content:
                text = chunk.content
                self._full_answer += text
                yield {"type": "token", "text": text}

        gen_ms = (_t.time() - t_gen) * 1000

        # ---- ⑥ 更新日志（验证 → 安全 → 改写 → 翻译 → process → 流式）----
        full_log = validate_log + safety_log + rewrite_log + translate_log + list(result.log)
        full_log.append(AgentStep("🎤 流式生成", f"生成 {len(self._full_answer)} 字", gen_ms))
        total_ms = (_t.time() - t_start) * 1000
        full_log.append(AgentStep("✅ 完成", f"总计 {total_ms/1000:.1f}s", 0))

        yield {"type": "thinking", "log": [
            {"action": s.action, "detail": s.detail, "duration": s.duration_ms}
            for s in full_log
        ]}

        # ---- ⑦ 保存记忆 ----
        self.memory.add(question, self._full_answer)
        yield {"type": "done"}
