# `.\marker\marker\ocr\utils.py`

```py
# 导入必要的模块和类
from typing import Optional
from nltk import wordpunct_tokenize
from spellchecker import SpellChecker
from marker.settings import settings
import re

# 检测 OCR 文本质量是否差，返回布尔值
def detect_bad_ocr(text, spellchecker: Optional[SpellChecker], misspell_threshold=.7, space_threshold=.6, newline_threshold=.5, alphanum_threshold=.4):
    # 如果文本长度为0，则假定 OCR 失败
    if len(text) == 0:
        return True

    # 使用 wordpunct_tokenize 函数将文本分词
    words = wordpunct_tokenize(text)
    # 过滤掉空白字符
    words = [w for w in words if w.strip()]
    # 提取文本中的字母数字字符
    alpha_words = [word for word in words if word.isalnum()]

    # 如果提供了拼写检查器
    if spellchecker:
        # 检查文本中的拼写错误
        misspelled = spellchecker.unknown(alpha_words)
        # 如果拼写错误数量超过阈值，则返回 True
        if len(misspelled) > len(alpha_words) * misspell_threshold:
            return True

    # 计算文本中空格的数量
    spaces = len(re.findall(r'\s+', text))
    # 计算文本中字母字符的数量
    alpha_chars = len(re.sub(r'\s+', '', text))
    # 如果空格占比超过阈值，则返回 True
    if spaces / (alpha_chars + spaces) > space_threshold:
        return True

    # 计算文本中换行符的数量
    newlines = len(re.findall(r'\n+', text))
    # 计算文本中非换行符的数量
    non_newlines = len(re.sub(r'\n+', '', text))
    # 如果换行符占比超过阈值，则返回 True
    if newlines / (newlines + non_newlines) > newline_threshold:
        return True

    # 如果文本中字母数字字符比例低于阈值，则返回 True
    if alphanum_ratio(text) < alphanum_threshold: # Garbled text
        return True

    # 计算文本中无效字符的数量
    invalid_chars = len([c for c in text if c in settings.INVALID_CHARS])
    # 如果无效字符数量超过阈值，则返回 True
    if invalid_chars > max(3.0, len(text) * .02):
        return True

    # 默认情况下返回 False
    return False

# 将字体标志拆解为可读的形式
def font_flags_decomposer(flags):
    l = []
    # 检查字体标志中是否包含上标
    if flags & 2 ** 0:
        l.append("superscript")
    # 检查字体标志中是否包含斜体
    if flags & 2 ** 1:
        l.append("italic")
    # 检查字体标志中是否包含衬线
    if flags & 2 ** 2:
        l.append("serifed")
    else:
        l.append("sans")
    # 检查字体标志中是否包含等宽字体
    if flags & 2 ** 3:
        l.append("monospaced")
    else:
        l.append("proportional")
    # 检查字体标志中是否包含粗体
    if flags & 2 ** 4:
        l.append("bold")
    # 返回拆解后的字体标志字符串
    return "_".join(l)

# 计算文本中字母数字字符的比例
def alphanum_ratio(text):
    # 去除文本中的空格和换行符
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    # 统计文本中的字母数字字符数量
    alphanumeric_count = sum([1 for c in text if c.isalnum()])

    # 如果文本长度为0，则返回1
    if len(text) == 0:
        return 1

    # 计算字母数字字符比例
    ratio = alphanumeric_count / len(text)
    # 返回变量 ratio 的值
    return ratio
```