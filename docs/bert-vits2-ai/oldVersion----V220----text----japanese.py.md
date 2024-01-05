# `d:/src/tocomm/Bert-VITS2\oldVersion\V220\text\japanese.py`

```
# 导入所需的库
import re  # 导入正则表达式模块
import unicodedata  # 导入Unicode数据模块
from transformers import AutoTokenizer  # 从transformers库中导入AutoTokenizer类
from . import punctuation, symbols  # 从当前包中导入punctuation和symbols模块
from num2words import num2words  # 从num2words库中导入num2words函数
import pyopenjtalk  # 导入pyopenjtalk库
import jaconv  # 导入jaconv库

def kata2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""
    text = text.strip()  # 去除文本两端的空格
    if text == "ー":  # 如果文本为"ー"
        return ["ー"]  # 返回包含"ー"的列表
    elif text.startswith("ー"):
        # 如果文本以 "ー" 开头，则将 "ー" 添加到结果列表中，并递归调用 kata2phoneme 函数处理剩余部分的文本
        return ["ー"] + kata2phoneme(text[1:])
    res = []
    prev = None
    while text:
        # 当文本不为空时循环
        if re.match(_MARKS, text):
            # 如果文本匹配 _MARKS 正则表达式，则将文本添加到结果列表中，并将文本的第一个字符去除
            res.append(text)
            text = text[1:]
            continue
        if text.startswith("ー"):
            # 如果文本以 "ー" 开头
            if prev:
                # 如果前一个字符存在，则将前一个字符的最后一个字符添加到结果列表中
                res.append(prev[-1])
            text = text[1:]
            continue
        # 使用 pyopenjtalk.g2p 函数将文本转换为音素，并将结果转换为小写，将 "cl" 替换为 "q"，然后按空格分割并添加到结果列表中
        res += pyopenjtalk.g2p(text).lower().replace("cl", "q").split(" ")
        break
    # res = _COLON_RX.sub(":", res)
    # 返回结果列表
    return res
        # 如果有假名，则将假名转换为片假名
        res.append(hira2kata(yomi))
        else:
            # 如果没有假名，则保持原样
            res.append(word)
    # 将列表中的片假名或原文拼接成字符串
    return "".join(res)


def replace_punctuation(text: str) -> str:
    # 使用正则表达式替换文本中的标点符号
    return _MARKS.sub("", text)
            # 如果 yomi 符合 _MARKS 的正则表达式
            if re.match(_MARKS, yomi):
                # 如果 word 的长度大于 1
                if len(word) > 1:
                    # 将 word 中的标点符号替换，并转换为列表
                    word = [replace_punctuation(i) for i in list(word)]
                    # 将 yomi 更新为 word
                    yomi = word
                    # 将 yomi 添加到 res 中
                    res += yomi
                    # 将 word 添加到 sep 中
                    sep += word
                    # 继续下一次循环
                    continue
                # 如果 word 不在 rep_map 的键和值中
                elif word not in rep_map.keys() and word not in rep_map.values():
                    # 将 word 更新为逗号
                    word = ","
                # 将 yomi 更新为 word
                yomi = word
            # 将 yomi 添加到 res 中
            res.append(yomi)
        # 如果不符合上述条件
        else:
            # 如果 word 在 _SYMBOL_TOKENS 中
            if word in _SYMBOL_TOKENS:
                # 将 word 添加到 res 中
                res.append(word)
            # 如果 word 是 "っ" 或 "ッ"
            elif word in ("っ", "ッ"):
                # 将 "ッ" 添加到 res 中
                res.append("ッ")
            # 如果 word 在 _NO_YOMI_TOKENS 中
            elif word in _NO_YOMI_TOKENS:
                # 什么也不做
                pass
            # 如果不符合上述条件
            else:
                # 将 word 添加到 res 中
                res.append(word)
    return hira2kata("".join(res))  # 将列表res中的元素连接成一个字符串，然后将其作为参数传递给hira2kata函数，并返回函数的结果


def text2sep_kata(text: str) -> (list, list):  # 定义一个函数text2sep_kata，接受一个字符串参数text，返回一个元组类型的值，包含两个列表
    parsed = pyopenjtalk.run_frontend(text)  # 使用pyopenjtalk模块的run_frontend函数对文本进行解析，并将结果赋值给parsed变量

    res = []  # 创建一个空列表res
    sep = []  # 创建一个空列表sep
    for parts in parsed:  # 遍历parsed中的元素，每个元素赋值给变量parts
        word, yomi = replace_punctuation(parts["string"]), parts["pron"].replace(
            "’", ""
        )  # 从parts字典中获取"string"和"pron"对应的值，分别赋值给word和yomi变量，并对yomi中的"’"进行替换

        if yomi:  # 如果yomi不为空
            if re.match(_MARKS, yomi):  # 如果yomi符合_MARKS的正则表达式
                if len(word) > 1:  # 如果word的长度大于1
                    word = [replace_punctuation(i) for i in list(word)]  # 对word中的每个字符进行标点替换，并将结果重新赋值给word
                    yomi = word  # 将word赋值给yomi
                    res += yomi  # 将yomi中的元素添加到res列表中
                    sep += word  # 将word中的元素添加到sep列表中
                    continue  # 继续下一次循环
                elif word not in rep_map.keys() and word not in rep_map.values():  # 如果单词不在替换映射的键和值中
                    word = ","  # 将单词替换为逗号
                yomi = word  # 将yomi设置为当前单词
            res.append(yomi)  # 将yomi添加到结果列表中
        else:  # 如果不是第一个单词
            if word in _SYMBOL_TOKENS:  # 如果单词是符号
                res.append(word)  # 将单词添加到结果列表中
            elif word in ("っ", "ッ"):  # 如果单词是"っ"或"ッ"
                res.append("ッ")  # 将"ッ"添加到结果列表中
            elif word in _NO_YOMI_TOKENS:  # 如果单词在_NO_YOMI_TOKENS中
                pass  # 什么也不做
            else:  # 如果单词不是特殊符号
                res.append(word)  # 将单词添加到结果列表中
        sep.append(word)  # 将单词添加到分隔列表中
    return sep, [hira2kata(i) for i in res], get_accent(parsed)  # 返回分隔列表、结果列表中的单词的片假名转换为片假名的列表，以及解析后的音调


def get_accent(parsed):  # 定义函数get_accent，接受参数parsed
    labels = pyopenjtalk.make_label(parsed)  # 使用pyopenjtalk.make_label函数对parsed进行标记
    phonemes = []  # 创建一个空列表，用于存储音素
    accents = []   # 创建一个空列表，用于存储音调

    for n, label in enumerate(labels):  # 遍历labels列表中的元素，同时获取索引值
        phoneme = re.search(r"\-([^\+]*)\+", label).group(1)  # 使用正则表达式从label中提取音素信息
        if phoneme not in ["sil", "pau"]:  # 如果提取的音素不是"sil"或"pau"
            phonemes.append(phoneme.replace("cl", "q").lower())  # 将提取的音素添加到phonemes列表中，并将其转换为小写
        else:  # 如果提取的音素是"sil"或"pau"
            continue  # 继续下一次循环

        a1 = int(re.search(r"/A:(\-?[0-9]+)\+", label).group(1))  # 使用正则表达式从label中提取A1的值
        a2 = int(re.search(r"\+(\d+)\+", label).group(1))  # 使用正则表达式从label中提取A2的值

        if re.search(r"\-([^\+]*)\+", labels[n + 1]).group(1) in ["sil", "pau"]:  # 如果下一个音素是"sil"或"pau"
            a2_next = -1  # 将a2_next设为-1
        else:  # 如果下一个音素不是"sil"或"pau"
            a2_next = int(re.search(r"\+(\d+)\+", labels[n + 1]).group(1))  # 使用正则表达式从下一个label中提取A2的值

        # Falling
        if a1 == 0 and a2_next == a2 + 1:  # 如果A1为0且下一个音素的A2等于当前音素的A2加1
            accents.append(-1)  # 将-1添加到accents列表中，表示下降音调
        # Rising
        elif a2 == 1 and a2_next == 2:  # 如果A2为1且下一个音素的A2为2
            accents.append(1)  # 将1添加到accents列表中，表示上升音调
        else:
            accents.append(0)  # 如果没有找到对应的音调，将0添加到音调列表中
    return list(zip(phonemes, accents))  # 将音素列表和音调列表压缩成一个元组列表并返回


_ALPHASYMBOL_YOMI = {
    "#": "シャープ",  # 将#映射为“シャープ”
    "%": "パーセント",  # 将%映射为“パーセント”
    "&": "アンド",  # 将&映射为“アンド”
    "+": "プラス",  # 将+映射为“プラス”
    "-": "マイナス",  # 将-映射为“マイナス”
    ":": "コロン",  # 将:映射为“コロン”
    ";": "セミコロン",  # 将;映射为“セミコロン”
    "<": "小なり",  # 将<映射为“小なり”
    "=": "イコール",  # 将=映射为“イコール”
    ">": "大なり",  # 将>映射为“大なり”
    "@": "アット",  # 将@映射为“アット”
    "a": "エー",  # 将a映射为“エー”
    "b": "ビー",  # 将b映射为“ビー”
    "c": "シー",  # 将c映射为“シー”
    "d": "ディー",  # 将字母"d"对应的日文读音存入字典中
    "e": "イー",   # 将字母"e"对应的日文读音存入字典中
    "f": "エフ",   # 将字母"f"对应的日文读音存入字典中
    "g": "ジー",   # 将字母"g"对应的日文读音存入字典中
    "h": "エイチ",  # 将字母"h"对应的日文读音存入字典中
    "i": "アイ",   # 将字母"i"对应的日文读音存入字典中
    "j": "ジェー",  # 将字母"j"对应的日文读音存入字典中
    "k": "ケー",   # 将字母"k"对应的日文读音存入字典中
    "l": "エル",   # 将字母"l"对应的日文读音存入字典中
    "m": "エム",   # 将字母"m"对应的日文读音存入字典中
    "n": "エヌ",   # 将字母"n"对应的日文读音存入字典中
    "o": "オー",   # 将字母"o"对应的日文读音存入字典中
    "p": "ピー",   # 将字母"p"对应的日文读音存入字典中
    "q": "キュー",  # 将字母"q"对应的日文读音存入字典中
    "r": "アール",  # 将字母"r"对应的日文读音存入字典中
    "s": "エス",   # 将字母"s"对应的日文读音存入字典中
    "t": "ティー",  # 将字母"t"对应的日文读音存入字典中
    "u": "ユー",   # 将字母"u"对应的日文读音存入字典中
    "v": "ブイ",   # 将字母"v"对应的日文读音存入字典中
    "w": "ダブリュー",  # 将字母"w"对应的日文读音存入字典中
    "x": "エックス",  # 将键值对 "x" 和 "エックス" 添加到字典中
    "y": "ワイ",  # 将键值对 "y" 和 "ワイ" 添加到字典中
    "z": "ゼット",  # 将键值对 "z" 和 "ゼット" 添加到字典中
    "α": "アルファ",  # 将键值对 "α" 和 "アルファ" 添加到字典中
    "β": "ベータ",  # 将键值对 "β" 和 "ベータ" 添加到字典中
    "γ": "ガンマ",  # 将键值对 "γ" 和 "ガンマ" 添加到字典中
    "δ": "デルタ",  # 将键值对 "δ" 和 "デルタ" 添加到字典中
    "ε": "イプシロン",  # 将键值对 "ε" 和 "イプシロン" 添加到字典中
    "ζ": "ゼータ",  # 将键值对 "ζ" 和 "ゼータ" 添加到字典中
    "η": "イータ",  # 将键值对 "η" 和 "イータ" 添加到字典中
    "θ": "シータ",  # 将键值对 "θ" 和 "シータ" 添加到字典中
    "ι": "イオタ",  # 将键值对 "ι" 和 "イオタ" 添加到字典中
    "κ": "カッパ",  # 将键值对 "κ" 和 "カッパ" 添加到字典中
    "λ": "ラムダ",  # 将键值对 "λ" 和 "ラムダ" 添加到字典中
    "μ": "ミュー",  # 将键值对 "μ" 和 "ミュー" 添加到字典中
    "ν": "ニュー",  # 将键值对 "ν" 和 "ニュー" 添加到字典中
    "ξ": "クサイ",  # 将键值对 "ξ" 和 "クサイ" 添加到字典中
    "ο": "オミクロン",  # 将键值对 "ο" 和 "オミクロン" 添加到字典中
    "π": "パイ",  # 将键值对 "π" 和 "パイ" 添加到字典中
    "ρ": "ロー",  # 将键值对 "ρ" 和 "ロー" 添加到字典中
    "σ": "シグマ",  # 将希腊字母σ转换为日语假名シグマ
    "τ": "タウ",  # 将希腊字母τ转换为日语假名タウ
    "υ": "ウプシロン",  # 将希腊字母υ转换为日语假名ウプシロン
    "φ": "ファイ",  # 将希腊字母φ转换为日语假名ファイ
    "χ": "カイ",  # 将希腊字母χ转换为日语假名カイ
    "ψ": "プサイ",  # 将希腊字母ψ转换为日语假名プサイ
    "ω": "オメガ",  # 将希腊字母ω转换为日语假名オメガ
}


_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")  # 匹配带有千位分隔符的数字的正则表达式
_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}  # 定义货币符号到日语货币名称的映射
_CURRENCY_RX = re.compile(r"([$¥£€])([0-9.]*[0-9])")  # 匹配货币符号和金额的正则表达式
_NUMBER_RX = re.compile(r"[0-9]+(\.[0-9]+)?")  # 匹配数字的正则表达式


def japanese_convert_numbers_to_words(text: str) -> str:
    res = _NUMBER_WITH_SEPARATOR_RX.sub(lambda m: m[0].replace(",", ""), text)  # 将带有千位分隔符的数字替换为不带分隔符的数字
    res = _CURRENCY_RX.sub(lambda m: m[2] + _CURRENCY_MAP.get(m[1], m[1]), res)  # 将货币符号和金额替换为日语货币名称和金额
    res = _NUMBER_RX.sub(lambda m: num2words(m[0], lang="ja"), res)  # 将数字转换为对应的日语文本表示
    return res
```
这行代码是一个函数的结尾，返回变量 res 的值。

```python
def japanese_convert_alpha_symbols_to_words(text: str) -> str:
    return "".join([_ALPHASYMBOL_YOMI.get(ch, ch) for ch in text.lower()])
```
这是一个函数，将输入的文本中的英文字母和符号转换为对应的日语单词。

```python
def japanese_text_to_phonemes(text: str) -> str:
    """Convert Japanese text to phonemes."""
    res = unicodedata.normalize("NFKC", text)
    res = japanese_convert_numbers_to_words(res)
    # res = japanese_convert_alpha_symbols_to_words(res)
    res = text2kata(res)
    res = kata2phoneme(res)
    return res
```
这是一个函数，将输入的日语文本转换为音素。

```python
def is_japanese_character(char):
    # 定义日语文字系统的 Unicode 范围
    japanese_ranges = [
```
这是一个函数，用于判断输入的字符是否为日语字符。在函数内部定义了日语文字系统的 Unicode 范围。
        (0x3040, 0x309F),  # 定义平假名的 Unicode 范围
        (0x30A0, 0x30FF),  # 定义片假名的 Unicode 范围
        (0x4E00, 0x9FFF),  # 定义汉字 (CJK Unified Ideographs) 的 Unicode 范围
        (0x3400, 0x4DBF),  # 定义汉字扩展 A 的 Unicode 范围
        (0x20000, 0x2A6DF),  # 定义汉字扩展 B 的 Unicode 范围
        # 可以根据需要添加其他汉字扩展范围
    ]

    # 将字符的 Unicode 编码转换为整数
    char_code = ord(char)

    # 检查字符是否在任何一个日语范围内
    for start, end in japanese_ranges:
        if start <= char_code <= end:
            return True  # 如果字符在日语范围内，则返回 True

    return False  # 如果字符不在任何日语范围内，则返回 False


rep_map = {
    "：": ",",  # 将中文冒号替换为英文逗号
    "；": ",",  # 将中文分号替换为英文逗号
    "，": ",",  # 将中文逗号替换为英文逗号
    "。": ".",  # 将中文句号替换为英文句号
    "！": "!",  # 将中文感叹号替换为英文感叹号
    "？": "?",  # 将中文问号替换为英文问号
    "\n": ".",  # 将换行符替换为英文句号
    "．": ".",  # 将中文句号替换为英文句号
    "…": "...",  # 将中文省略号替换为英文省略号
    "···": "...",  # 将中文省略号替换为英文省略号
    "・・・": "...",  # 将中文省略号替换为英文省略号
    "·": ",",  # 将中文间隔号替换为英文逗号
    "・": ",",  # 将中文间隔号替换为英文逗号
    "、": ",",  # 将中文顿号替换为英文逗号
    "$": ".",  # 将美元符号替换为英文句号
    "“": "'",  # 将中文左双引号替换为英文单引号
    "”": "'",  # 将中文右双引号替换为英文单引号
    '"': "'",  # 将双引号替换为单引号
    "‘": "'",  # 将中文左单引号替换为英文单引号
    "’": "'",  # 将中文右单引号替换为英文单引号
    "（": "'",  # 将中文括号（）替换为英文单引号'
    "）": "'",  # 将中文括号（）替换为英文单引号'
    "(": "'",   # 将左括号替换为英文单引号'
    ")": "'",   # 将右括号替换为英文单引号'
    "《": "'",  # 将中文书名号《》替换为英文单引号'
    "》": "'",  # 将中文书名号《》替换为英文单引号'
    "【": "'",  # 将中文方括号【】替换为英文单引号'
    "】": "'",  # 将中文方括号【】替换为英文单引号'
    "[": "'",   # 将左方括号替换为英文单引号'
    "]": "'",   # 将右方括号替换为英文单引号'
    "—": "-",   # 将中文破折号—替换为英文破折号-
    "−": "-",   # 将减号−替换为英文破折号-
    "～": "-",  # 将中文波浪号～替换为英文破折号-
    "~": "-",   # 将波浪号~替换为英文破折号-
    "「": "'",  # 将中文书名号「」替换为英文单引号'
    "」": "'",  # 将中文书名号「」替换为英文单引号'
}

def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))  # 创建一个正则表达式模式，用于匹配替换映射中的键

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)  # 使用正则表达式模式替换文本中匹配的内容

    replaced_text = re.sub(
        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"  # 匹配日文字符范围
        + "".join(punctuation)  # 匹配标点符号
        + r"]+",
        "",
        replaced_text,  # 替换后的文本
    )

    return replaced_text  # 返回替换后的文本


def text_normalize(text):
    res = unicodedata.normalize("NFKC", text)  # 使用 NFKC 规范化 Unicode 文本
    res = japanese_convert_numbers_to_words(res)  # 将文本中的数字转换为对应的日文单词
    # res = "".join([i for i in res if is_japanese_character(i)])  # 将文本中的日文字符连接起来
    res = replace_punctuation(res)  # 替换文本中的标点符号
    res = res.replace("゙", "")  # 用空字符串替换字符串中的特定字符"゙"
    return res  # 返回处理后的字符串


def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word  # 创建一个长度为n_word的列表，每个元素初始化为0
    for task in range(n_phone):  # 循环n_phone次
        min_tasks = min(phones_per_word)  # 找到phones_per_word中的最小值
        min_index = phones_per_word.index(min_tasks)  # 找到最小值的索引
        phones_per_word[min_index] += 1  # 最小值对应的元素加1
    return phones_per_word  # 返回phones_per_word列表


def handle_long(sep_phonemes):
    for i in range(len(sep_phonemes)):  # 遍历sep_phonemes列表
        if sep_phonemes[i][0] == "ー":  # 如果列表中第i个元素的第一个字符是"ー"
            sep_phonemes[i][0] = sep_phonemes[i - 1][-1]  # 将第i个元素的第一个字符替换为前一个元素的最后一个字符
        if "ー" in sep_phonemes[i]:  # 如果列表中第i个元素包含"ー"
            for j in range(len(sep_phonemes[i])):  # 遍历第i个元素
                if sep_phonemes[i][j] == "ー":  # 如果第i个元素的第j个字符是"ー"
                    sep_phonemes[i][j] = sep_phonemes[i][j - 1][-1]
```
这行代码的作用是将sep_phonemes[i][j - 1][-1]的值赋给sep_phonemes[i][j]。

```
    return sep_phonemes
```
这行代码的作用是返回sep_phonemes变量的值。

```
tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese-char-wwm")
```
这行代码的作用是使用预训练模型"./bert/deberta-v2-large-japanese-char-wwm"创建一个tokenizer对象。

```
def align_tones(phones, tones):
    res = []
    for pho in phones:
        temp = [0] * len(pho)
        for idx, p in enumerate(pho):
            if len(tones) == 0:
                break
            if p == tones[0][0]:
                temp[idx] = tones[0][1]
                if idx > 0:
                    temp[idx] += temp[idx - 1]
                tones.pop(0)
        temp = [0] + temp
```
这段代码定义了一个名为align_tones的函数，该函数接受两个参数phones和tones。在函数内部，它使用循环和条件语句对phones和tones进行处理，并将结果存储在res变量中。

希望这些注释能够帮助你理解这些代码的作用。
        temp = temp[:-1]  # 从列表temp中去掉最后一个元素
        if -1 in temp:  # 检查列表temp中是否包含-1
            temp = [i + 1 for i in temp]  # 如果包含-1，则对列表temp中的每个元素加1
        res.append(temp)  # 将处理后的temp添加到列表res中
    res = [i for j in res for i in j]  # 将res列表中的嵌套列表展开成一个一维列表
    assert not any([i < 0 for i in res]) and not any([i > 1 for i in res])  # 断言：res列表中的所有元素都不小于0且不大于1
    return res  # 返回处理后的res列表


def rearrange_tones(tones, phones):
    res = [0] * len(tones)  # 创建一个长度为tones列表长度的全0列表
    for i in range(len(tones)):  # 遍历tones列表的索引
        if i == 0:  # 如果索引为0
            if tones[i] not in punctuation:  # 如果tones列表中的第一个元素不在标点符号列表中
                res[i] = 1  # 则将res列表中对应位置的元素设为1
        elif tones[i] == prev:  # 如果tones列表中的当前元素等于上一个元素
            if phones[i] in punctuation:  # 如果phones列表中的当前元素在标点符号列表中
                res[i] = 0  # 则将res列表中对应位置的元素设为0
            else:
                res[i] = 1  # 否则将res列表中对应位置的元素设为1
        elif tones[i] > prev:
            # 如果当前音调大于前一个音调，将结果数组中对应位置的值设为2
            res[i] = 2
        elif tones[i] < prev:
            # 如果当前音调小于前一个音调，将结果数组中前一个位置的值设为3，当前位置的值设为1
            res[i - 1] = 3
            res[i] = 1
        prev = tones[i]
    return res


def g2p(norm_text):
    # 将规范化后的文本转换为分隔后的文本、分隔后的假名、重音信息
    sep_text, sep_kata, acc = text2sep_kata(norm_text)
    sep_tokenized = []
    for i in sep_text:
        if i not in punctuation:
            # 如果分隔后的文本不是标点符号，则进行分词处理
            sep_tokenized.append(tokenizer.tokenize(i))
        else:
            # 如果分隔后的文本是标点符号，则直接添加到分词结果中
            sep_tokenized.append([i])

    sep_phonemes = handle_long([kata2phoneme(i) for i in sep_kata])
    # 处理长音，将分隔后的假名转换为音素

    # 异常处理，MeCab不认识的词的话会一路传到这里来，然后炸掉。目前来看只有那些超级稀有的生僻词会出现这种情况
    for i in sep_phonemes:  # 遍历分离的音素列表
        for j in i:  # 遍历每个音素
            assert j in symbols, (sep_text, sep_kata, sep_phonemes)  # 检查每个音素是否在符号列表中，如果不在则抛出异常并显示相关信息
    tones = align_tones(sep_phonemes, acc)  # 调用 align_tones 函数对分离的音素列表进行音调对齐

    word2ph = []  # 初始化单词到音素的映射列表
    for token, phoneme in zip(sep_tokenized, sep_phonemes):  # 遍历分离的标记化单词和音素列表
        phone_len = len(phoneme)  # 获取音素列表的长度
        word_len = len(token)  # 获取单词的长度

        aaa = distribute_phone(phone_len, word_len)  # 调用 distribute_phone 函数进行音素分配
        word2ph += aaa  # 将分配的音素添加到单词到音素的映射列表中
    phones = ["_"] + [j for i in sep_phonemes for j in i] + ["_"]  # 构建包含起始和结束符号的音素列表
    tones = [0] + tones + [0]  # 在音调列表的开头和结尾添加零
    word2ph = [1] + word2ph + [1]  # 在单词到音素的映射列表的开头和结尾添加一
    assert len(phones) == len(tones)  # 检查音素列表和音调列表的长度是否相等，如果不相等则抛出异常
    return phones, tones, word2ph  # 返回音素列表、音调列表和单词到音素的映射列表
if __name__ == "__main__":
    # 从预训练模型中加载分词器
    tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese")
    # 待处理的文本
    text = "hello,こんにちは、世界ー！……"
    # 从自定义模块中导入函数
    from text.japanese_bert import get_bert_feature

    # 对文本进行规范化处理
    text = text_normalize(text)
    # 打印处理后的文本
    print(text)

    # 使用自定义模块中的函数将文本转换为音素、音调，并生成单词到音素的映射
    phones, tones, word2ph = g2p(text)
    # 使用自定义模块中的函数获取文本的 BERT 特征
    bert = get_bert_feature(text, word2ph)

    # 打印音素、音调、单词到音素映射以及 BERT 特征的形状
    print(phones, tones, word2ph, bert.shape)
```