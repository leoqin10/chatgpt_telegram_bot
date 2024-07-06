import re

def detect_language(text):
    # 中文字符的正则表达式
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    # 英文字符的正则表达式
    english_pattern = re.compile(r'[a-zA-Z]+')

    has_chinese = bool(chinese_pattern.search(text))
    has_english = bool(english_pattern.search(text))

    if has_chinese and has_english:
        return "en"
    elif has_chinese:
        return "ch"
    elif has_english:
        return "en"
    else:
        return "en"