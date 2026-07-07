"""
Input Validator — 输入验证层
=============================
在用户输入进入 RAG 前验证，拦截无效输入。
新增 v4: 有效字符比例 + 游戏领域关键词 + 多语种乱码检测

使用:
  from input_validator import validate
  ok, message = validate(user_input)
  if not ok:
      return message
"""

import re

# 饥荒游戏领域关键词（用于语义检测）
_GAME_KEYWORDS = [
    # 中文关键词
    '饥荒', '联机', '角色', '温蒂', '威尔逊', '阿比盖尔', '巨鹿', '冬季', '夏季',
    '烹饪', '食谱', '装备', '武器', '护甲', 'Boss', '种田', '作物', '巨大',
    '理智', '生命', '饱食', '保暖', '暖石', '眼球伞', '洞穴', '月岛',
    '合成', '材料', '掉落', '攻略', '怎么', '什么', '如何', '打法',
    '新手', '生存', '开荒', '季节', '秋天', '春天', '冬天', '夏天',
    '影刀', '暗影剑', '锤子', '猪人', '兔人', '蜘蛛', '触手', '浣猫',
    'Dragonfly', 'Deerclops', 'Klaus', 'Crock', 'Abigail', 'Wendy',
    'DST', 'RAG', 'Don.t Starve',
]


def validate(text: str) -> tuple[bool, str]:
    """
    验证用户输入。新增 v4 三层检查。
    返回: (是否有效, 错误信息或空字符串)
    """
    # ---- ① 空输入 ----
    if not text or not text.strip():
        return False, "请输入有效问题"

    cleaned = text.strip()

    # ---- ② 过短 ----
    valid_chars = [c for c in cleaned if c.isalnum() or '一' <= c <= '鿿' or c.isascii()]
    if len(valid_chars) < 2:
        return False, "问题过短，请重新输入"

    # ---- ③ 纯符号 ----
    alpha = sum(1 for c in cleaned if c.isalpha() or '一' <= c <= '鿿')
    if alpha == 0:
        return False, "请输入有效问题"

    # ---- ④ 键盘乱敲（无意义英文） ----
    if _is_keyboard_mash(cleaned):
        return False, "检测到无意义输入，请输入关于饥荒游戏的有效问题"

    # ---- ⑤ v4 新增：有效字符比例检测 ----
    # 总字符中，有效文字（中文/英文/数字）占比 < 30% → 拦截
    meaningful = sum(1 for c in cleaned if c.isalnum() or '一' <= c <= '鿿' or c in '?？!！.。，,''- ')
    ratio = meaningful / max(len(cleaned), 1)
    if ratio < 0.30:
        return False, "无法理解您的输入，请用中文或英文描述您的问题"

    # ---- ⑥ v4 新增：游戏领域关键词检测 ----
    # 如果输入全是英文但没有已知游戏词，也没有英文单词特征 → 拦截
    has_chinese = any('一' <= c <= '鿿' for c in cleaned)
    has_game_term = any(k.lower() in cleaned.lower() for k in _GAME_KEYWORDS)

    if not has_chinese and not has_game_term:
        # 纯英文/多语种 → 检查是否有英文单词特征
        letters_only = ''.join(c.lower() for c in cleaned if c.isalpha())
        if letters_only:
            # 有元音交替 = 可能是正常英文 → 放行（后续翻译会处理）
            vowels = sum(1 for c in letters_only if c in 'aeiou')
            if len(letters_only) >= 5 and vowels / len(letters_only) < 0.15:
                return False, "无法理解您的输入，请用中文或英文描述您的问题"
            # 完全没有任何元音的乱码 → 拦截
            if vowels == 0 and len(letters_only) >= 4:
                return False, "无法理解您的输入，请用中文或英文描述您的问题"

    # ---- ⑥ v4：多语种垃圾检测 ----
    # 符号比例 > 20% 且 无中文 且 无游戏关键词 → 垃圾输入
    symbol_count = sum(1 for c in cleaned if not c.isalnum() and c not in ' ?？!！.。，,\'-_')
    if symbol_count / max(len(cleaned), 1) > 0.20 and not has_chinese and not has_game_term:
        return False, "无法理解您的输入，请用中文或英文描述您的问题"

    return True, ""


def _is_keyboard_mash(text: str) -> bool:
    """检测键盘乱敲"""
    letters = ''.join(c for c in text if c.isalpha())

    if letters and all(c.isascii() and c.isalpha() for c in letters):
        vowels = sum(1 for c in letters.lower() if c in 'aeiou')
        if len(letters) >= 5 and vowels / len(letters) < 0.15:
            return True
        keyboard_rows = [
            set('qwertyuiop'), set('asdfghjkl'), set('zxcvbnm')
        ]
        for row in keyboard_rows:
            row_chars = sum(1 for c in letters.lower() if c in row)
            if row_chars / len(letters) > 0.8 and len(letters) >= 4:
                return True
    return False
