"""
Agent 工作器 — 上下文改写 + 问题分解 + 多策略检索
=================================================
Agent 三个核心能力:
  1. 📝 上下文改写 — 检测指代，调用 LLM 补全为独立问题
  2. ✂️ 问题分解 — 复杂问题拆为子问题
  3. 🔎 多策略检索 — 向量搜索 + 关键词兜底

每一步都有日志，供 UI 展示。
"""
import re
import time as _time


class AgentStep:
    """单步日志"""
    def __init__(self, action: str, detail: str, duration_ms: float = 0):
        self.action = action
        self.detail = detail
        self.duration_ms = duration_ms

    @property
    def duration_str(self) -> str:
        ms = self.duration_ms
        return f"{ms/1000:.1f}s" if ms >= 1000 else f"{int(ms)}ms"


class AgentResult:
    """Agent 执行结果"""
    def __init__(self):
        self.log: list[AgentStep] = []
        self.search_query: str = ""       # 最终用于检索的查询
        self.sub_queries: list[str] = []  # 如果被分解，子问题列表
        self.retrieved_docs: list = []    # 检索到的文档


class Agent:
    """
    Agent 工作器。

    使用:
        agent = Agent(llm)
        result = agent.process(question, memory, vector_store)
        # result.search_query → 改写后的查询
        # result.retrieved_docs → 检索到的文档
        # result.log → 每一步日志
    """

    def __init__(self, llm):
        self.llm = llm

    # ================================================================
    # 能力 1: 上下文改写
    # ================================================================
    def needs_rewrite(self, query: str, history_len: int) -> bool:
        """检查是否需要改写"""
        if history_len == 0:
            return False
        referents = ['它', '他', '她', '这个', '那个', '这种', '那种',
                      '这', '那', '这个怪', '那个BOSS', '这些', '那些', '他的', '它的']
        return any(r in query for r in referents) or len(query) <= 15

    def rewrite_query(self, query: str, memory) -> str:
        """
        Agent 调用 LLM，把含指代的问题改写为独立完整问题。
        这是 Agent 的核心能力之一。
        """
        if not self.needs_rewrite(query, len(memory.history)):
            return query

        history_text = memory.get_recent(5)

        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template("""根据对话历史，将用户当前问题改写为一个独立完整的问题。

对话历史:
{history}

用户当前问题: {query}

改写规则:
- "它""他""她""这个""那个"等指代词 → 替换为历史中的具体实体名
- 省略的主语/宾语 → 补全
- 输出一个脱离历史也能理解的完整中文问题
- 直接输出改写后的问题，不要任何解释

改写后的问题:""")

        try:
            chain = prompt | self.llm
            response = chain.invoke({"history": history_text, "query": query})
            rewritten = response.content.strip().strip('"''')
            if len(rewritten) >= 3:
                return rewritten
        except Exception:
            pass
        return query

    # ================================================================
    # 能力 2: 复杂度判断 + 问题分解
    # ================================================================
    def is_complex(self, question: str) -> bool:
        if question.count('？') + question.count('?') >= 2:
            return True
        markers = ['并且', '同时', '还有', '另外', '以及',
                    '第一步', '第二步', '首先', '然后', '最后',
                    '怎么.*怎么', '什么.*什么', '和.*区别', '对比']
        for m in markers:
            if re.search(m, question):
                return True
        return False

    def decompose(self, question: str) -> list[str]:
        """LLM 拆复杂问题"""
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template("""将下面的复杂问题拆成2-4个独立子问题。
每行一个，以 "- " 开头，用中文。

复杂问题: {question}

子问题:""")
        try:
            chain = prompt | self.llm
            r = chain.invoke({"question": question})
            subs = []
            for line in r.content.strip().split('\n'):
                line = line.strip().lstrip('- ').strip()
                if line and len(line) > 3:
                    subs.append(line)
            return subs[:4] if subs else [question]
        except Exception:
            return [question]

    # ================================================================
    # 能力 3: 关键词 + 检索
    # ================================================================
    def extract_keywords(self, text: str) -> list[str]:
        noise = ['是啥', '咋搞', '咋办', '咋整', '怎么搞', '怎么弄', '好难打',
                 '怎么', '什么', '如何', '为什么', '怎样', '什么样', '好难',
                 '是什么', '怎么做', '怎么办', '搞的', '那个', '这个', '有啥',
                 '可以', '需要', '一个', '哪些', '一下', '是不是', '有什么办法',
                 '怎么处理', '怎么解决', '啊', '呀', '呢', '吧', '嘛', '哦',
                 '哈', '啦', '喔', '诶', '吗', '的']
        c = text
        for n in noise:
            c = c.replace(n, '')
        c = re.sub(r'\s+', '', c)
        kws = re.findall(r'[\d]+|[A-Za-z]+|[一-鿿]{2,}', c)
        result = []
        for k in kws:
            if len(k) > 4 and re.match(r'^[一-鿿]+$', k):
                for i in range(0, len(k) - 1):
                    if k[i:i+2] not in result:
                        result.append(k[i:i+2])
            else:
                if k not in result:
                    result.append(k)
        return result

    RETRIEVAL_K = 8  # 扩大召回范围（原 4）

    def search(self, query: str, vector_store, k: int = None) -> list:
        if k is None:
            k = self.RETRIEVAL_K
        return vector_store.similarity_search(query, k=k)

    def rerank_by_keyword(self, query: str, docs: list, top_k: int = 4) -> list:
        """
        轻量 rerank：按关键词命中密度重新排序 top-K chunks。
        不需要额外模型，零依赖。
        """
        kws = self.extract_keywords(query)
        if not kws or len(docs) <= top_k:
            return docs[:top_k]

        scored = []
        for d in docs:
            text = d.page_content[:500]
            score = sum(1 for kw in kws if kw in text)
            # 奖励短文本密度
            density = score / max(len(text), 1)
            scored.append((d, density))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored[:top_k]]

    def search_with_fallback(self, query: str, vector_store, k: int = None) -> list:
        """检索(k=8) + 关键词兜底 + rerank → 返回 top-4"""
        if k is None:
            k = self.RETRIEVAL_K

        docs = self.search(query, vector_store, k)
        all_text = ' '.join(d.page_content[:200] for d in docs)
        kws = self.extract_keywords(query)
        matched = [kw for kw in kws if kw in all_text]
        if not matched and len(kws) >= 2:
            docs = self.search(' '.join(kws[:5]), vector_store, k)

        # rerank：从 top-8 中选出关键词最密集的 top-4
        return self.rerank_by_keyword(query, docs, top_k=4)

    # ================================================================
    # 主流程
    # ================================================================
    def process(self, question: str, memory, vector_store) -> AgentResult:
        """
        Agent 完整流程:
          1. 上下文改写（消除指代）
          2. 复杂度判断 → 简单/复杂
          3. 检索（复杂则分解后多轮检索）
          4. 返回结果（含日志 + 检索到的文档）
        """
        result = AgentResult()
        t0 = _time.time()

        # ---- Step 1: 上下文改写 ----
        t1 = _time.time()
        rewritten = self.rewrite_query(question, memory)
        result.search_query = rewritten
        if rewritten != question:
            result.log.append(AgentStep(
                "📝 上下文改写",
                f"原始: {question}\n改写: {rewritten}",
                (_time.time() - t1) * 1000
            ))

        # ---- Step 2: 复杂度判断 ----
        t2 = _time.time()
        if self.is_complex(rewritten):
            result.log.append(AgentStep(
                "🔍 复杂度判断",
                "检测到复杂问题",
                (_time.time() - t2) * 1000
            ))

            # 分解
            t3 = _time.time()
            subs = self.decompose(rewritten)
            result.sub_queries = subs
            result.log.append(AgentStep(
                "✂️ 问题分解",
                f"拆为 {len(subs)} 个子问题:\n" + '\n'.join(f"  • {q}" for q in subs),
                (_time.time() - t3) * 1000
            ))

            # 每个子问题检索
            all_docs = []
            for i, sq in enumerate(subs):
                t_s = _time.time()
                docs = self.search_with_fallback(sq, vector_store, k=self.RETRIEVAL_K)
                all_docs.extend(docs)
                result.log.append(AgentStep(
                    f"🔎 检索子问题{i+1}",
                    f"查询: {sq} | 命中 {len(docs)} 个chunk",
                    (_time.time() - t_s) * 1000
                ))
            result.retrieved_docs = all_docs[:10]
        else:
            result.log.append(AgentStep(
                "🔍 复杂度判断",
                "简单问题，直接检索",
                (_time.time() - t2) * 1000
            ))

            t3 = _time.time()
            docs = self.search_with_fallback(rewritten, vector_store, k=4)
            result.retrieved_docs = docs
            result.log.append(AgentStep(
                "🔎 语义检索",
                f"查询: {rewritten[:60]} | 命中 {len(docs)} 个chunk",
                (_time.time() - t3) * 1000
            ))

        result.log.append(AgentStep(
            "✅ 分析完成",
            f"总耗时 {(_time.time()-t0)*1000:.0f}ms",
            (_time.time() - t0) * 1000
        ))
        return result


# ---- 语言检测 ----
def is_chinese(text: str) -> bool:
    return bool(re.search(r'[一-鿿]', text))


# ---- 多语言→中文游戏术语映射 ----
MULTILINGUAL_MAP = {
    # 英文→中文
    'bearger': '熊獾',
    'deerclops': '巨鹿',
    'dragonfly': '龙蝇',
    'bee queen': '蜂后',
    'klaus': '克劳斯',
    'fuelweaver': '影织者',
    'shadow sword': '暗影剑',
    'dark sword': '暗影剑',
    'ham bat': '火腿棒',
    'pierogi': '波兰水饺',
    'meatballs': '肉丸',
    'dragonpie': '火龙果派',
    'cave': '洞穴',
    'ruins': '远古遗迹',
    'atrium': '中庭',
    'lunar island': '月岛',
    'celestial champion': '天体英雄',
    'crock pot': '烹饪锅',
    'science machine': '科学机器',
    'alchemy engine': '炼金引擎',
    'sanity': '理智',
    'health': '生命值',
    'hunger': '饱食度',
    'thermal stone': '暖石',
    'eyebrella': '眼球伞',
    'beefalo hat': '牛毛帽',
    'winter': '冬季',
    'summer': '夏季',
    'spring': '春季',
    'autumn': '秋季',
    'farming': '种田',
    'giant crop': '巨大作物',
    'combat': '战斗攻略',
    'beginner': '新手',
    'survival guide': '生存指南',
    'walkthrough': '攻略',
    'crafting': '合成',
    'recipe': '食谱',
    'boss': 'Boss攻略',
}


def translate_and_rewrite(text: str, llm) -> tuple[str, str, str]:
    """
    多语言处理完整链路:
      ① 多语言词表直接映射
      ② LLM 翻译 + 改写为饥荒领域检索表达
    返回: (translation, rewrite, debug_info)
    """
    debug_parts = [f"原文: {text[:100]}"]
    lower = text.lower()

    # ① 先检查词表映射
    direct_matches = []
    for en, zh in MULTILINGUAL_MAP.items():
        if en in lower:
            direct_matches.append(zh)
    if direct_matches:
        direct_str = ' '.join(direct_matches)
        debug_parts.append(f"词表匹配: {direct_str}")

    # ② LLM 翻译 + 领域改写
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(
        """将下面的问题翻译为中文，并改写为适合饥荒Wiki检索的查询。

规则:
- 翻译成中文
- 补充饥荒游戏关键词（如"新手"→"新手开荒攻略"、"怎么开始"→"新手入门指南"）
- 输出精简的检索关键词，而非完整句子
- 不添加不存在的信息

输入: {text}

中文检索关键词:"""
    )
    translated = (prompt | llm).invoke({"text": text}).content.strip()
    debug_parts.append(f"翻译+改写: {translated}")

    # ③ 如果词表有匹配，拼到翻译结果后面
    if direct_matches:
        translated = translated + ' ' + ' '.join(direct_matches)

    debug_info = '\n'.join(debug_parts)
    return translated, debug_info


# 保持旧接口兼容
def translate(text: str, llm) -> str:
    result, _ = translate_and_rewrite(text, llm)
    return result
