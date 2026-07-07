"""
Reasoning Agent — 检索前语义分析
=================================
在 RAG 检索前分析问题。不做检索、不生成答案，只负责:
  1. 实体提取 — 识别游戏实体（生物/物品/作物/地点/机制）
  2. 前提检查 — 检测错误假设（如"非洲之星是可种植作物"）
  3. Query规划 — 给出修正查询 + 回答策略

输出 JSON:
  {"intent":"", "entities":[], "premise_check":{"has_error":false, "reason":""},
   "corrected_query":"", "answer_strategy":"normal_rag"|"correct_user_then_answer"}

使用:
  from reasoning_agent import ReasoningAgent
  agent = ReasoningAgent(llm)
  plan = agent.analyze("春天怎么种植非洲之星")
"""
import json
import re
from langchain_core.prompts import ChatPromptTemplate


# 饥荒游戏真实实体词表（用于 LLM 提示）
_DST_ENTITIES = {
    "生物": ["巨鹿", "Deerclops", "熊獾", "Bearger", "龙蝇", "Dragonfly", "蜂后", "Bee Queen",
             "克劳斯", "Klaus", "影织者", "Fuelweaver", "织影者", "远古守护者", "远古犀牛",
             "蚁狮", "Antlion", "猪人", "Pig", "兔人", "Bunnyman", "蜘蛛", "高脚鸟", "触手",
             "猎犬", "Hound", "齿轮怪", "Clockwork", "企鹅", "牛", "Beefalo",
             "阿比盖尔", "Abigail", "影怪", "暗影生物", "坎普斯"],
    "角色": ["温蒂", "Wendy", "威尔逊", "Wilson", "沃尔夫冈", "Wolfgang", "WX-78",
             "薇克伯顿", "Wickerbottom", "麦斯威尔", "Maxwell", "韦伯", "Webber",
             "韦斯", "Wes", "薇洛", "Willow", "沃利", "Warly", "沃拓克斯", "Wortox",
             "沃姆伍德", "Wormwood", "沃特", "Wurt", "旺达", "Wanda", "薇诺娜", "Winona"],
    "物品": ["暗影剑", "影刀", "影刃", "暗影甲", "影甲", "火腿棒", "铥矿棒", "铥矿甲",
             "玻璃刀", "晨星", "眼球伞", "暖石", "步行手杖", "矿工帽", "灯笼",
             "触手尖刺", "触手钉锤", "长矛", "木甲", "草甲", "大理石甲", "足球头盔",
             "猪皮帽", "牛毛帽", "冬帽", "熊皮背心", "雨伞", "雨衣", "背包"],
    "作物": ["土豆", "番茄", "火龙果", "大蒜", "辣椒", "洋葱", "南瓜", "玉米", "芦笋",
             "茄子", "榴莲", "石榴", "西瓜", "胡萝卜", "巨大作物"],
    "食物": ["火龙果派", "波兰水饺", "肉丸", "炖肉汤", "培根煎蛋", "蜜汁火腿",
             "冰淇淋", "太妃糖", "蔬菜鸡尾酒", "奶油土豆泥", "香蕉冰"],
    "地点": ["中庭", "遗迹", "远古遗迹", "洞穴", "月岛", "沼泽", "沙漠", "草原", "桦树林",
             "稀树草原", "岩石地", "荧光果平原", "远古档案区"],
    "机制": ["理智", "san", "饱食度", "生命", "保暖", "温度", "湿度", "季节",
             "种田", "烹饪", "合成", "战斗", "走位"],
}

# 统一扁平列表
_ALL_VALID = []
for lst in _DST_ENTITIES.values():
    _ALL_VALID.extend(lst)


class ReasoningAgent:
    """推理 Agent — 检索前语义分析"""

    def __init__(self, llm):
        self.llm = llm

    def analyze(self, query: str) -> dict:
        """
        分析用户问题，输出规划 JSON。
        """
        if not query.strip():
            return {
                "intent": "unknown",
                "entities": [],
                "premise_check": {"has_error": False, "reason": ""},
                "corrected_query": query,
                "answer_strategy": "normal_rag",
            }

        # 快速预检 — 没有出现任何游戏关键词也可能跳过 LLM 调用
        has_game_term = any(
            k.lower() in query.lower() for k in _ALL_VALID if len(k) >= 2
        )
        # 如果问题很短且无游戏术语，直接 normal_rag
        if not has_game_term and len(query) <= 6:
            return {
                "intent": "unknown",
                "entities": [],
                "premise_check": {"has_error": False, "reason": ""},
                "corrected_query": query,
                "answer_strategy": "normal_rag",
            }

        known_entities_text = "\n".join(
            f"  {cat}: {', '.join(ents[:15])}"
            for cat, ents in _DST_ENTITIES.items()
        )

        prompt = ChatPromptTemplate.from_template("""你是饥荒联机版专家。分析以下用户问题。

## 已知游戏实体
{known_entities}

## 用户问题
{query}

## 分析要求
1. **intent**: 用户意图（攻略/数据查询/合成配方/Boss打法/种田/机制/闲聊/其他）
2. **entities**: 问题中提到的游戏实体列表（生物/角色/物品/作物/地点/机制）
3. **premise_check**: 检查用户是否有错误假设
   - 例如"非洲之星"不在已知作物中 → has_error=true
   - 例如"如何用小推车带猪人"中"小推车"不存在 → has_error=true
   - 如果没错误 → has_error=false
4. **corrected_query**:
   - 如果 has_error=true → 给出修正后的检索查询（去除错误前提，保留真实意图）
   - 如果 has_error=false → 保持原 query
5. **answer_strategy**:
   - "normal_rag" → 正常RAG回答
   - "correct_user_then_answer" → 先纠正错误前提，再回答真正的问题

输出严格 JSON（不要多余文字）:
{{"intent":"","entities":[],"premise_check":{{"has_error":false,"reason":""}},"corrected_query":"","answer_strategy":""}}""")

        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "query": query,
                "known_entities": known_entities_text,
            })
            text = response.content.strip()

            # 提取 JSON
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # 确保必要字段存在
                result.setdefault("intent", "unknown")
                result.setdefault("entities", [])
                result.setdefault("premise_check", {"has_error": False, "reason": ""})
                result.setdefault("corrected_query", query)
                result.setdefault("answer_strategy", "normal_rag")
                return result
        except Exception:
            pass

        return {
            "intent": "unknown",
            "entities": [],
            "premise_check": {"has_error": False, "reason": ""},
            "corrected_query": query,
            "answer_strategy": "normal_rag",
        }
