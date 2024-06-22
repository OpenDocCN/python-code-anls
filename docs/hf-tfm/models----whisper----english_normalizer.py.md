# `.\transformers\models\whisper\english_normalizer.py`

```py
# 导入所需的模块
import re
import unicodedata
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union

import regex

# 含有额外附加重音符号的非 ASCII 字母（未经 "NFKD" 标准化）
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}

# 移除符号和重音符号
def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space, and drop any diacritics (category 'Mn' and some
    manual mappings)
    """
    def replace_character(char):
        if char in keep:
            return char
        elif char in ADDITIONAL_DIACRITICS:
            return ADDITIONAL_DIACRITICS[char]
        elif unicodedata.category(char) == "Mn":
            return ""
        elif unicodedata.category(char)[0] in "MSP":
            return " "
        return char

    return "".join(replace_character(c) for c in unicodedata.normalize("NFKD", s))

# 移除符号
def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(" " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s))

# 基本文本规范化器类
class BasicTextNormalizer:
    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        self.split_letters = split_letters

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        if self.split_letters:
            s = " ".join(regex.findall(r"\X", s, regex.U))

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespace characters with a space

        return s

# 英文数字规范化器类
class EnglishNumberNormalizer:
    """
    Convert any spelled-out numbers into arabic numbers, while handling:

    - remove any commas
    - keep the suffixes such as: `1960s`, `274th`, `32nd`, etc.
    - spell out currency symbols after the number. e.g. `$20 million` -> `20000000 dollars`
    - spell out `one` and `ones`
    - interpret successive single-digit numbers as nominal: `one oh one` -> `101`
    """

    # 定义 preprocess 方法，接受字符串参数 s
    def preprocess(self, s: str):
        # 将字符串 s 按照 "and a half" 分割成多个片段
        results = []
        segments = re.split(r"\band\s+a\s+half\b", s)
        # 遍历每个片段
        for i, segment in enumerate(segments):
            # 如果片段去除空格后长度为 0，则继续下一轮循环
            if len(segment.strip()) == 0:
                continue
            # 如果当前片段为最后一个片段，则添加到结果中
            if i == len(segments) - 1:
                results.append(segment)
            else:
                results.append(segment)
                # 获取片段的最后一个单词
                last_word = segment.rsplit(maxsplit=2)[-1]
                # 如果最后一个单词是在 self.decimals 或 self.multipliers 中，则添加 "point five" 到结果中
                if last_word in self.decimals or last_word in self.multipliers:
                    results.append("point five")
                # 否则添加 "and a half" 到结果中
                else:
                    results.append("and a half")

        # 用空格连接结果中的每个片段
        s = " ".join(results)

        # 在数字和字母之间加一个空格
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)

        # 但移除可能是后缀的空格
        s = re.sub(r"([0-9])\s+(st|nd|rd|th|s)\b", r"\1\2", s)

        # 返回预处理后的字符串 s
        return s

    # 定义 postprocess 方法，接受字符串参数 s
    def postprocess(self, s: str):
        # 定义 combine_cents 函数，接受 Match 对象参数 m
        def combine_cents(m: Match):
            try:
                # 获取货币符号、整数部分和分数部分
                currency = m.group(1)
                integer = m.group(2)
                cents = int(m.group(3))
                # 返回组合后的字符串，保留两位小数
                return f"{currency}{integer}.{cents:02d}"
            except ValueError:
                return m.string

        # 定义 extract_cents 函数，接受 Match 对象参数 m
        def extract_cents(m: Match):
            try:
                # 获取分数部分
                return f"¢{int(m.group(1))}"
            except ValueError:
                return m.string

        # 对货币进行后处理，"$2 and ¢7" -> "$2.07"
        s = re.sub(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b", combine_cents, s)
        s = re.sub(r"[€£$]0.([0-9]{1,2})\b", extract_cents, s)

        # 将 "1(s)" 替换为 "one(s)"，仅为了可读性
        s = re.sub(r"\b1(s?)\b", r"one\1", s)

        # 返回后处理后的字符串 s
        return s

    # 定义 __call__ 方法，接受字符串参数 s
    def __call__(self, s: str):
        # 预处理字符串 s
        s = self.preprocess(s)
        # 将分割后的单词传递给 process_words 方法进行处理，然后用空格连接单词
        s = " ".join(word for word in self.process_words(s.split()) if word is not None)
        # 后处理字符串 s
        s = self.postprocess(s)

        # 返回处理后的字符串 s
        return s
class EnglishSpellingNormalizer:
    """
    Applies British-American spelling mappings as listed in [1].

    [1] https://www.tysto.com/uk-us-spelling-list.html
    """

    # 创建英语拼写规范化类，根据提供的英语拼写映射
    def __init__(self, english_spelling_mapping):
        # 初始化拼写映射
        self.mapping = english_spelling_mapping

    # 定义调用该类实例时的行为，将输入的字符串拆分为单词，根据映射进行替换，并返回
    def __call__(self, s: str):
        return " ".join(self.mapping.get(word, word) for word in s.split())


class EnglishTextNormalizer:
    # 创建英语文本规范化类
    def __init__(self, english_spelling_mapping):
        # 定义要忽略的模式
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um)\b"
        
        # 定义替换规则，将匹配的模式替换成对应的规范化形式
        self.replacers = {
            # 常见缩写
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            ...
            r"'m\b": " am",
        }

        # 初始化数字标准化类
        self.standardize_numbers = EnglishNumberNormalizer()
        
        # 初始化英语拼写规范化，传入英语拼写映射
        self.standardize_spellings = EnglishSpellingNormalizer(english_spelling_mapping)
    # 定义一个方法，用于处理输入的字符串
    def __call__(self, s: str):
        # 把输入字符串转换为小写
        s = s.lower()
    
        # 删除字符串中括号内的内容
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
        # 删除字符串括号内的内容
        s = re.sub(r"\(([^)]+?)\)", "", s)
        # 删除字符串中指定的忽略模式
        s = re.sub(self.ignore_patterns, "", s)
        # 标准化字符串中空格和撇号的组合
        s = re.sub(r"\s+'", "'", s)
    
        # 遍历替换模式列表，替换字符串中匹配的内容
        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)
    
        # 删除数字间的逗号
        s = re.sub(r"(\d),(\d)", r"\1\2", s)
        # 删除不跟数字的句点
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)
        # 移除字符串中的符号和变音符号，保留 .%$¢€£
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")
    
        # 标准化字符串中的数字
        s = self.standardize_numbers(s)
        # 标准化字符串中的拼写
        s = self.standardize_spellings(s)
    
        # 删除不跟数字的前缀/后缀符号
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)
    
        # 将连续的空白字符替换为单个空格
        s = re.sub(r"\s+", " ", s)
    
        # 返回处理后的字符串
        return s
```