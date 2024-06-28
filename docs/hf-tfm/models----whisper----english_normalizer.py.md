# `.\models\whisper\english_normalizer.py`

```py
# 版权声明及引用信息
# Copyright 2022 The OpenAI team and The HuggingFace Team. All rights reserved.
# Most of the code is copy pasted from the original whisper repository
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入正则表达式和Unicode相关模块
import re
import unicodedata
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union

# 导入第三方正则表达式模块
import regex

# 非ASCII字母，不通过"NFKD"规范化分隔的其他附加重音符号和特殊字符映射表
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

def remove_symbols_and_diacritics(s: str, keep=""):
    """
    替换文本中的标记、符号和标点为空格，并且移除所有重音符号（类别为'Mn'）和一些手动映射的特殊字符
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

def remove_symbols(s: str):
    """
    替换文本中的标记、符号和标点为空格，保留重音符号
    """
    return "".join(" " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s))

class BasicTextNormalizer:
    """
    文本基础清理类，根据初始化参数移除重音符号和分隔字母
    """
    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        self.split_letters = split_letters

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # 移除括号内的单词
        s = re.sub(r"\(([^)]+?)\)", "", s)  # 移除括号内的内容
        s = self.clean(s).lower()

        if self.split_letters:
            s = " ".join(regex.findall(r"\X", s, regex.U))  # 按字母分隔文本

        s = re.sub(r"\s+", " ", s)  # 将连续的空白字符替换为单个空格

        return s

class EnglishNumberNormalizer:
    """
    英文数字标准化类，将文本中的英文数字转换为阿拉伯数字，并保留后缀，如`1960s`, `274th`, `32nd`等
    """
    def __init__(self):
        pass  # 此类不需要额外的初始化

    def __call__(self, s: str):
        """
        对输入的字符串进行处理，替换文本中的英文数字为阿拉伯数字，并保留后缀
        """
        s = s.lower()
        s = re.sub(r"\s+", " ", s)  # 将连续的空白字符替换为单个空格
        return s
    # spell out currency symbols after the number. e.g. `$20 million` -> `20000000 dollars`
    # spell out `one` and `ones`
    # interpret successive single-digit numbers as nominal: `one oh one` -> `101`
    """
    This class provides methods for preprocessing and postprocessing text transformations related to numbers and currencies.
    """

    def preprocess(self, s: str):
        # replace "<number> and a half" with "<number> point five"
        results = []

        # Split the input string by the phrase "and a half"
        segments = re.split(r"\band\s+a\s+half\b", s)
        for i, segment in enumerate(segments):
            if len(segment.strip()) == 0:
                continue
            if i == len(segments) - 1:
                results.append(segment)
            else:
                results.append(segment)
                # Check if the last word in the segment is a decimal or a multiplier
                last_word = segment.rsplit(maxsplit=2)[-1]
                if last_word in self.decimals or last_word in self.multipliers:
                    results.append("point five")
                else:
                    results.append("and a half")

        s = " ".join(results)

        # Put a space at number/letter boundary
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)

        # Remove spaces which could be a suffix like "1st", "2nd", etc.
        s = re.sub(r"([0-9])\s+(st|nd|rd|th|s)\b", r"\1\2", s)

        return s

    def postprocess(self, s: str):
        def combine_cents(m: Match):
            try:
                currency = m.group(1)
                integer = m.group(2)
                cents = int(m.group(3))
                # Combine currency, integer part, and cents with correct formatting
                return f"{currency}{integer}.{cents:02d}"
            except ValueError:
                return m.string

        def extract_cents(m: Match):
            try:
                # Extract cents from the matched pattern and format as cents symbol
                return f"¢{int(m.group(1))}"
            except ValueError:
                return m.string

        # Apply currency postprocessing: "$2 and ¢7" -> "$2.07"
        s = re.sub(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b", combine_cents, s)
        s = re.sub(r"[€£$]0.([0-9]{1,2})\b", extract_cents, s)

        # Replace "1(s)" with "one(s)" for readability
        s = re.sub(r"\b1(s?)\b", r"one\1", s)

        return s

    def __call__(self, s: str):
        # Process input string `s` through preprocessing, word processing, and postprocessing steps
        s = self.preprocess(s)
        s = " ".join(word for word in self.process_words(s.split()) if word is not None)
        s = self.postprocess(s)

        return s
class EnglishSpellingNormalizer:
    """
    Applies British-American spelling mappings as listed in [1].

    [1] https://www.tysto.com/uk-us-spelling-list.html
    """

    def __init__(self, english_spelling_mapping):
        # 初始化时接收一个英语拼写映射的字典
        self.mapping = english_spelling_mapping

    def __call__(self, s: str):
        # 在调用实例时，根据映射替换输入字符串中的单词
        return " ".join(self.mapping.get(word, word) for word in s.split())


class EnglishTextNormalizer:
    def __init__(self, english_spelling_mapping):
        # 忽略的模式，用于匹配需要保留原始形式的词语
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um)\b"
        # 替换规则字典，用于将特定模式替换为标准化的形式
        self.replacers = {
            # 常见缩略语
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "got to",
            r"\bgonna\b": "going to",
            r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have",
            r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have",
            r"\bma'am\b": "madam",
            # 头衔和称谓中的缩略语
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
            # 完成时态的缩略语
            r"'d been\b": " had been",
            r"'s been\b": " has been",
            r"'d gone\b": " had gone",
            r"'s gone\b": " has gone",
            r"'d done\b": " had done",  # "'s done" is ambiguous
            r"'s got\b": " has got",
            # 一般缩略语
            r"n't\b": " not",
            r"'re\b": " are",
            r"'s\b": " is",
            r"'d\b": " would",
            r"'ll\b": " will",
            r"'t\b": " not",
            r"'ve\b": " have",
            r"'m\b": " am",
        }
        # 初始化时创建一个用于标准化数字的对象
        self.standardize_numbers = EnglishNumberNormalizer()
        # 初始化时创建一个用于标准化拼写的对象，传入英语拼写映射
        self.standardize_spellings = EnglishSpellingNormalizer(english_spelling_mapping)
    # 定义一个特殊方法 __call__，接受一个字符串参数 s，并将其转换为小写形式
    def __call__(self, s: str):
        # 将字符串 s 转换为小写形式
        s = s.lower()

        # 使用正则表达式去除尖括号或方括号中的内容
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        
        # 使用正则表达式去除圆括号中的内容
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        
        # 使用预定义的忽略模式列表去除特定模式的内容
        s = re.sub(self.ignore_patterns, "", s)
        
        # 将空格后的撇号标准化为撇号
        s = re.sub(r"\s+'", "'", s)  # standardize when there's a space before an apostrophe

        # 根据预定义的替换规则字典，依次替换字符串 s 中的模式
        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        # 去除数字之间的逗号
        s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas between digits
        
        # 去除句点后非数字字符的句点
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)  # remove periods not followed by numbers
        
        # 调用外部函数 remove_symbols_and_diacritics 去除 s 中的特定符号和变音符号，保留一些特定符号用于数字
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")  # keep some symbols for numerics

        # 使用对象内定义的方法 standardize_numbers 对 s 中的数字进行标准化处理
        s = self.standardize_numbers(s)
        
        # 使用对象内定义的方法 standardize_spellings 对 s 中的拼写进行标准化处理
        s = self.standardize_spellings(s)

        # 去除非数字前后的前缀/后缀符号，如 . $ ¢ € £
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        
        # 去除非数字前的百分号，将其前后加上空格
        s = re.sub(r"([^0-9])%", r"\1 ", s)

        # 将任意连续的空白字符替换为单个空格
        s = re.sub(r"\s+", " ", s)  # replace any successive whitespace characters with a space

        # 返回处理后的字符串 s
        return s
```