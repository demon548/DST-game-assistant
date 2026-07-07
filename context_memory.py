"""
上下文记忆模块 — 多轮对话核心
==============================
提供三个能力:
  1. 对话历史存储
  2. 上下文拼接
  3. 问题改写（消除指代，生成独立可检索的完整问题）

使用方式:
  from context_memory import ConversationMemory
  memory = ConversationMemory(llm)
  rewritten = memory.rewrite("它怎么打？")  # → "巨鹿怎么打？"
  memory.add("它怎么打？", assistant_answer)
"""
from langchain_core.prompts import ChatPromptTemplate


class ConversationMemory:
    """对话记忆 — 存储历史 + 问题改写"""

    def __init__(self, llm):
        self.llm = llm
        self.history: list[dict] = []  # [{"user": ..., "assistant": ...}, ...]

    # ---- Step 1: 存储 ----
    def add(self, user_question: str, assistant_answer: str):
        """追加一轮对话"""
        self.history.append({
            "user": user_question,
            "assistant": assistant_answer
        })

    def clear(self):
        self.history = []

    # ---- Step 2: 拼接上下文 ----
    def build_context(self, current_query: str = "") -> str:
        """将历史对话拼成自然语言"""
        if not self.history:
            return "（无历史对话）"

        parts = []
        for i, h in enumerate(self.history[-5:]):  # 最近5轮
            parts.append(f"第{i+1}轮 — 用户: {h['user']}\n助手: {h['assistant'][:400]}")
        return "\n\n".join(parts)

    # ---- Step 3: 问题改写（核心） ----
    def rewrite(self, current_query: str) -> str:
        """
        输入: 可能含指代的当前问题（如"那它掉什么？"）
        输出: 独立完整的问题（如"巨鹿的掉落物品有哪些"）

        用 LLM 根据历史对话补全指代。
        """
        # 无历史，无需改写
        if not self.history:
            return current_query

        # 无指代词，也无需改写
        referents = ['它', '他', '她', '这个', '那个', '这种', '那种',
                      '这', '那', '这个怪', '那个BOSS', '这些', '那些']
        has_referent = any(r in current_query for r in referents)
        is_short = len(current_query) <= 15
        if not has_referent and not is_short:
            return current_query

        # 构建改写 prompt
        history_text = self.build_context()

        prompt = ChatPromptTemplate.from_template("""你是饥荒联机版专家。将用户的当前问题改写为一个完整、独立的问题。

规则:
- 根据对话历史，把"它""这个""那个"等指代词替换为具体实体名
- 把省略的主语/宾语补全
- 改写后的问题应该可以脱离历史独立理解
- 改写后的问题用中文，适合做知识库检索
- 直接输出改写后的问题，不要解释

对话历史:
{history}

用户当前问题: {query}

改写后的问题:""")

        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "history": history_text,
                "query": current_query
            })
            rewritten = response.content.strip()
            # 清理
            rewritten = rewritten.strip('"''').strip()
            if len(rewritten) >= 3:
                return rewritten
        except Exception:
            pass

        return current_query

    # ---- Step 4: 完整流程 ----
    def augment_query(self, current_query: str) -> str:
        """
        一步完成: 拼接上下文 + LLM 改写 + 返回可检索问题
        """
        return self.rewrite(current_query)

    def get_recent(self, n: int = 4) -> str:
        """获取最近 n 轮对话的文本，供 prompt 使用"""
        if not self.history:
            return "（这是第一轮对话，无历史）"

        parts = []
        for h in self.history[-n:]:
            parts.append(f"用户: {h['user']}\n助手: {h['assistant'][:300]}")
        return "\n\n".join(parts)
