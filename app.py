"""
饥荒游戏助手 — Streamlit 界面
启动: streamlit run app.py
"""
import re
import streamlit as st
from rag_pipeline import EnhancedRAGAssistant
from agent_workflow import AgentStep

st.set_page_config(page_title="饥荒联机版 游戏助手", page_icon="🎮", layout="wide")
st.title("🎮 饥荒联机版 游戏助手")
st.caption("基于 RAG + Agent + Memory 的智能问答系统 — 329,000 字知识库")

# ---- 初始化（用 session_state 替代 cache_resource，保证代码更新生效） ----
if "assistant" not in st.session_state:
    st.session_state.assistant = EnhancedRAGAssistant()
    if st.session_state.assistant.vector_store is None:
        with st.spinner("首次运行，构建知识库..."):
            n = st.session_state.assistant.build_knowledge_base()
            st.success(f"完成！{n} 个文本块")

assistant = st.session_state.assistant

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---- 引用渲染 ----
def render_citations(text: str, log: list) -> str:
    sources = []
    if log:
        for s in log:
            if "检索" in s.action and "命中" in s.detail:
                for line in s.detail.split("\n"):
                    if "查询:" in line:
                        sources.append(line.strip())
    def repl(m):
        n = int(m.group(1))
        tip = sources[n-1].replace('"', '&quot;') if n <= len(sources) else f"引用 #{n}"
        return f'<sup title="{tip}" style="cursor:help;color:#1a73e8;border-bottom:1px dotted #1a73e8">[{n}]</sup>'
    return re.sub(r'\[(\d+)\]', repl, text)


# ---- 渲染历史消息 ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        if msg.get("log"):
            content = render_citations(content, msg["log"])
        st.markdown(content, unsafe_allow_html=True)
        if msg.get("log"):
            log = msg["log"]
            total_ms = sum(s.duration_ms for s in log)
            total_s = f"{total_ms/1000:.1f}s" if total_ms >= 1000 else f"{int(total_ms)}ms"
            with st.expander(f"🧠 思考过程 · {total_s}", expanded=False):
                for s in log:
                    st.caption(f"{s.action} — `{s.duration_str}`")
                    if s.detail:
                        st.text(s.detail)


# ---- 欢迎消息 ----
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""你好！我是饥荒联机版游戏助手 🎮

| 模块 | 内容 |
|------|------|
| 🧑 角色 | 全部角色背景故事、技能数据 |
| 🍲 烹饪 | 80道料理配方、食材系数、保鲜 |
| ⚔️ Boss | 10个Boss打法、走位、掉落 |
| 🌾 种田 | 14种作物、巨大作物、黄金配比 |
| 🏠 装备 | 武器/护甲/魔法/建筑全套 |
| 🌤️ 四季 | 季节应对、温度湿度管理 |
| 🕳️ 洞穴 | 远古遗迹、中庭、影织者 |
| 🌙 月岛 | 天体线、玻璃科技、月石获取 |

**支持多轮对话** — 你可以追问"它怎么打？""那掉什么？"。
试试问我一些问题吧！""")

# ---- 输入 ----
if prompt := st.chat_input("输入你的饥荒问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "log": None})

    with st.chat_message("assistant"):
        with st.spinner("🧠 分析中..."):
            stream = assistant.ask_stream(prompt)
            first_chunk = next(stream)

        saved_log = []

        if first_chunk.get("type") == "thinking" and first_chunk.get("log"):
            raw = first_chunk["log"]
            saved_log = [AgentStep(s["action"], s["detail"], s["duration"]) for s in raw]

        tokens = []
        if first_chunk.get("type") == "token":
            tokens.append(first_chunk["text"])

        answer_placeholder = st.empty()
        for chunk in stream:
            if chunk["type"] == "token":
                tokens.append(chunk["text"])
                if len(tokens) % 5 == 0:
                    answer_placeholder.markdown("".join(tokens) + "▌", unsafe_allow_html=True)
            elif chunk["type"] == "thinking":
                raw = chunk["log"]
                saved_log = [AgentStep(s["action"], s["detail"], s["duration"]) for s in raw]
            elif chunk["type"] == "done":
                break

        answer = "".join(tokens)
        rendered = render_citations(answer, saved_log)
        if saved_log:
            total_ms = sum(s.duration_ms for s in saved_log)
            total_s = f"{total_ms/1000:.1f}s" if total_ms >= 1000 else f"{int(total_ms)}ms"
            steps = "<br>".join(
                f"<b>{s.action}</b> <code>{s.duration_str}</code>"
                + (f" — {s.detail.replace(chr(10), '<br>')}" if s.detail else "")
                for s in saved_log
            )
            rendered += (
                f'\n\n<details style="font-size:0.85em;color:#888;margin-top:1em">'
                f'<summary>🧠 思考过程 · {total_s}</summary><p>{steps}</p></details>'
            )
        answer_placeholder.markdown(rendered, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "log": saved_log,
    })
    st.rerun()

# ---- 侧边栏 ----
with st.sidebar:
    st.header("📋 关于")
    st.markdown("""
**饥荒联机版 游戏助手**

**技术栈:**
- DeepSeek API (对话)
- BGE Embedding (向量化)
- ChromaDB (向量库)
- LangChain (框架)
- Streamlit (界面)

**Agent 能力:**
- 📝 上下文改写 — 消除"它""这个"等指代
- ✂️ 问题分解 — 复杂问题拆子问题
- 🔎 多策略检索 — 语义+关键词兜底
- 🎤 流式输出 — token by token

**知识库:** 329,000 字 / 219 页
""")
    st.divider()
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        assistant.clear_history()
        st.rerun()
    st.caption("NLP 期末大作业 · 2026")
