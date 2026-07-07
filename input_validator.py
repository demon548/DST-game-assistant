"""
Input Validator — 输入验证层
=============================
在用户输入进入 RAG 前验证，拦截无效输入。

使用:
  from input_validator import validate
  ok, message = validate(user_input)
  if not ok:
      return message  # 拒绝，直接返回提示
  # ok → 继续 RAG 流程
"""


def validate(text: str) -> tuple[bool, str]:
    """
    验证用户输入。
    返回: (是否有效, 错误信息或空字符串)
    """
    if not text or not text.strip():
        return False, "请输入有效问题"

    # 去掉空白后
    cleaned = text.strip()

    # 过短（<2个有效字符）
    valid_chars = [c for c in cleaned if c.isalnum() or '一' <= c <= '鿿']
    if len(valid_chars) < 2:
        return False, "问题过短，请重新输入"

    # 全是符号/标点/空白
    alpha = sum(1 for c in cleaned if c.isalpha() or '一' <= c <= '鿿')
    if alpha == 0:
        return False, "请输入有效问题"

    # 乱码检测：连续相同字母过长（如 "asdfghjkl"）
    # 策略：如果全是ASCII字母且无英文单词特征（无元音交替），判为乱码
    if _is_keyboard_mash(cleaned):
        return False, "检测到无意义输入，请输入关于饥荒游戏的有效问题"

    return True, ""


def _is_keyboard_mash(text: str) -> bool:
    """检测键盘乱敲"""
    # 去掉符号和数字
    letters = ''.join(c for c in text if c.isalpha())

    # 全英文但全是辅音或全是同一行键盘 → 乱码
    if letters and all(c.isascii() and c.isalpha() for c in letters):
        # 检查元音比例（正常英文约30-50%元音）
        vowels = sum(1 for c in letters.lower() if c in 'aeiou')
        if len(letters) >= 5 and vowels / len(letters) < 0.15:
            return True
        # 连续同一行键盘字母（qwerty/asdf/zxcv 模式）
        keyboard_rows = [
            set('qwertyuiop'), set('asdfghjkl'), set('zxcvbnm')
        ]
        for row in keyboard_rows:
            row_chars = sum(1 for c in letters.lower() if c in row)
            if row_chars / len(letters) > 0.8 and len(letters) >= 4:
                return True

    return False
