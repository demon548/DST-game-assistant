"""
Query Expansion — 游戏同义词扩展
================================
检索前自动将玩家俗称扩展为游戏标准术语，提升召回率。

使用:
  from query_expansion import expand
  expanded = expand("影刀怎么制作")
  # → "影刀 暗影剑 怎么制作"
"""

import re

# 游戏同义词映射表 (玩家俗称 → 标准术语)
SYNONYMS = {
    # 武器
    "影刀": "暗影剑",
    "暗影刀": "暗影剑",
    "火腿球棒": "火腿棒",
    "触手棒": "触手尖刺",
    "眼球刀": "玻璃刀",
    # Boss
    "独眼巨鹿": "巨鹿 Deerclops",
    "龙蝇": "Dragonfly",
    "蜂后": "Bee Queen",
    "熊大": "熊獾 Bearger",
    "克劳斯": "Klaus",
    "织影者": "影织者 Fuelweaver",
    "远古犀牛": "远古守护者",
    # 生物
    "猪哥": "猪人",
    "兔哥": "兔人",
    "齿轮马": "机械马",
    "齿轮主教": "机械主教",
    # 装备
    "猪皮帽": "足球头盔 橄榄球头盔",
    "牛帽": "牛毛帽",
    "火腿棍": "火腿棒",
    "眼伞": "眼球伞",
    "步行杖": "步行手杖",
    # 机制
    "san": "理智",
    "San": "理智",
    "san值": "理智值",
    "精神值": "理智值",
    "饱食": "饱食度",
    # 物品
    "齿轮": "齿轮 Gears",
    "活木": "活木 Living Log",
    "噩梦燃料": "噩梦燃料 Nightmare Fuel",
    # 食物
    "水饺": "波兰水饺 pierogi",
    "肉丸": "肉丸 meatballs",
    "火龙果派": "火龙果派 dragonpie",
    "蜜汁火腿": "蜜汁火腿 honey ham",
    "培根蛋": "培根煎蛋",
    # 季节/活动
    "春": "春季 Spring",
    "夏": "夏季 Summer",
    "秋": "秋季 Autumn",
    "冬": "冬季 Winter",
    # 通用
    "怎么做": "制作 配方 材料",
    "怎么搞": "制作 获得 获取",
    "咋整": "方法 攻略",
    "咋搞": "方法 攻略",
}


def expand(query: str) -> str:
    """
    扩展查询：在不改变原义的前提下，追加标准术语。
    返回: 扩展后的查询字符串
    """
    parts = [query]

    for slang, formal in SYNONYMS.items():
        if slang in query and formal not in query:
            parts.append(formal)

    # 去重
    seen = set()
    result_parts = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            result_parts.append(p)

    return ' '.join(result_parts)
