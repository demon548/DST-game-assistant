"""
Safety Classifier — 安全分类器
===============================
在 RAG 流程之前拦截恶意请求。纯规则匹配，不依赖 LLM。

规则层级:
  RED    → 坚决拒绝（DDoS、扮演、越狱）
  YELLOW → 警告 + 拒绝（恶意破坏、作弊、泄漏）
  GREEN  → 放行

使用:
  from safety_classifier import classify
  level, reason = classify("如何破坏别人的基地？")
  if level == "RED" or level == "YELLOW":
      return build_refusal(level, reason)
"""

import re

# RED 规则：直接命中 → 必须拒绝，不解释原因
RED_RULES = [
    # DDoS / 网络攻击
    (r'ddos|denial.of.service|洪水攻击|ddos.*attack|attack.*server', "网络攻击"),
    # 角色扮演绕过
    (r'act\s+as\s+(a\s+)?linux\s+terminal|扮演.*终端|pretend.*terminal'
     r'|act\s+like\s+a\s+(linux|terminal|bash|shell)', "角色扮演绕过"),
    (r'developer\s+mode|dev\s+mode|developer.*output|系统提示.*输出', "Prompt泄露"),
    # 越狱
    (r'ignore\s+(all\s+)?(previous|prior)\s+instructions|DAN\s|jailbreak', "越狱指令"),
    # API Key 泄露
    (r'output\s+your\s+(system\s+)?prompt|输出.*系统.*提示|泄露.*配置'
     r'|输出.*API.*密钥|show.*api.*key|print.*api.*key', "敏感信息泄露"),
]

# YELLOW 规则：上下文关联 → 拒绝 + 引导
YELLOW_RULES = [
    # 恶意破坏 (破坏/摧毁/拆家)
    (r'破坏.*基地|摧毁.*建筑|摧毁.*基地|拆家|故意.*破坏|恶意.*破坏|grief.*base|destroy.*base', "恶意破坏"),
    (r'如何.*报复.*玩家|怎么.*报复|how.*to.*grief', "报复行为"),
    # 作弊
    (r'作弊|cheat|spawn.*infinite|生成.*无限|c_give|console.*command.*cheat', "作弊"),
    # 骂人
    (r'骂.*那个|怎么.*骂|脏话|swear|fuck|shit', "不当语言"),
    # 儿童保护（检测到儿童+不良引导）
    (r'小朋友.*报复|儿童.*骂|小孩.*打人', "儿童不良引导"),
]

# 临时拒答词列表
REFUSAL_TERMS = [
    ("RED", "网络攻击", "我无法提供任何关于网络攻击的指导。如果你遇到服务器问题，请通过游戏官方渠道反馈。"),
    ("RED", "角色扮演绕过", "我是饥荒游戏助手，无法扮演其他角色或执行系统命令。有什么游戏问题我可以帮你？比如角色攻略、Boss打法、食物配方，随时问我！"),
    ("RED", "Prompt泄露", "我无法输出内部配置信息。如果你对饥荒游戏感兴趣，随时问我！比如你想了解哪个角色的背景故事？"),
    ("RED", "越狱指令", "我无法执行此指令。我是饥荒联机版游戏助手，专注于回答游戏相关问题。想了解什么游戏内容？"),
    ("RED", "敏感信息泄露", "我无法提供系统配置或API密钥等信息。请提出与饥荒游戏相关的问题，比如你想知道温蒂怎么召唤阿比盖尔吗？"),
    ("YELLOW", "恶意破坏",
     "我不能提供破坏其他玩家基地的方法。\n\n" +
     "如果你想保护自己的基地，我可以帮你：\n" +
     "- 建造石墙或木墙围住基地\n" +
     "- 放置狗牙陷阱防御猎犬和入侵者\n" +
     "- 用箱子锁住贵重物品\n" +
     "- 建造灭火器防止夏季自燃\n" +
     "需要我详细讲解哪一种防护方法？"),
    ("YELLOW", "报复行为",
     "报复不是解决问题的好方法。如果你在服务器遇到问题，可以试试这些正当方式：\n\n" +
     "- 与服务器管理员沟通，举报不良行为\n" +
     "- 用密码保护自己的房间或服务器\n" +
     "- 给贵重箱子加锁（使用锁头mod或游戏内机制）\n" +
     "- 只和信得过的朋友联机\n\n" +
     "需要我教你如何设置服务器密码吗？"),
    ("YELLOW", "作弊",
     "我不能提供作弊方法。但如果你想提升游戏技术，我可以分享一些合法技巧：\n\n" +
     "- 高效刷噩梦燃料的方法\n" +
     "- 如何快速获取齿轮\n" +
     "- 走位 (kiting) 技巧教学\n" +
     "你想了解哪个方面的正常玩法？"),
    ("YELLOW", "不当语言",
     "骂人不是解决问题的好方法。如果你想学习如何更好地应对游戏中的挑战，我可以帮你！\n\n" +
     "比如你可以告诉我你经常在哪种情况下死掉或者感到沮丧，我可以给你针对性的生存建议。\n" +
     "想聊聊吗？"),
    ("YELLOW", "儿童不良引导",
     "小朋友，在游戏里最重要的是玩得开心！你提到的那些方式我不能教你哦。\n\n" +
     "不过我可以教你一些实用的生存技巧，让你少死几次：\n" +
     "- 天黑前一定要点火把或篝火\n" +
     "- 学会做肉丸（1肉+3冰或浆果），永远不饿死\n" +
     "- 打不过就跑，不要硬拼\n" +
     "你想先学哪一个？"),
]


def classify(text: str) -> tuple[str, str, str]:
    """
    安全分类。
    返回: (level, reason, refusal_message)
      level: "RED" | "YELLOW" | "GREEN"
      reason: 匹配到的规则原因
      refusal_message: 如果需拒绝，返回的拒答话术
    """
    if not text or not text.strip():
        return "GREEN", "", ""

    lower = text.lower()

    # 先检查 RED（最高优先级）
    for pattern, reason in RED_RULES:
        if re.search(pattern, lower):
            for level, r, msg in REFUSAL_TERMS:
                if r == reason:
                    return "RED", reason, msg
            return "RED", reason, "我无法执行此请求。"

    # 再检查 YELLOW
    for pattern, reason in YELLOW_RULES:
        if re.search(pattern, lower):
            for level, r, msg in REFUSAL_TERMS:
                if r == reason:
                    return "YELLOW", reason, msg
            return "YELLOW", reason, "我不能提供这方面的帮助。"

    return "GREEN", "", ""
