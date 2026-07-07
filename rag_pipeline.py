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
llm = ChatOpenAI(model="deepseek-chat", api_key=api_key, base_url=base_url)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)

# ---- 高级组件 ----
from agent_workflow import Agent, AgentStep, is_chinese, translate
from context_memory import ConversationMemory

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

        if not self.vector_store:
            yield {"type": "token", "text": "❌ 知识库尚未构建"}
            yield {"type": "done"}
            return

        t_start = _t.time()

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

        # ---- ② 语言检测 ----
        if not is_chinese(search_query):
            try:
                search_query = translate(search_query, self.llm)
            except Exception:
                pass

        # ---- ③ Agent 检索 ----
        result = self.agent.process(search_query, self.memory, self.vector_store)
        self.log = result.log
        retrieved = result.retrieved_docs

        # 合并日志：改写在前，process 日志在后
        full_initial_log = rewrite_log + result.log

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

## 回答规范
- 参考对话历史理解上下文。如上一轮讨论了"巨鹿"，本轮"它"就是指巨鹿
- 用自然的中文回答，跟老玩家交流一样
- 关键事实用 [1] [2] 标注来源
- 用户使用不认识的术语时，直接说明游戏中没有，不要猜测
- 参考数据中没有的信息诚实说明，不编造

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

        # ---- ⑥ 更新日志（保留改写步骤！）----
        full_log = rewrite_log + list(result.log)
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
