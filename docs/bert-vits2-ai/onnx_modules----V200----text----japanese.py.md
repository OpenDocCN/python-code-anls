# `d:/src/tocomm/Bert-VITS2\onnx_modules\V200\text\japanese.py`

```
# 导入所需的模块
import re  # 导入正则表达式模块
import unicodedata  # 导入Unicode数据模块
from transformers import AutoTokenizer  # 从transformers模块中导入AutoTokenizer类
from . import punctuation, symbols  # 从当前包中导入punctuation和symbols模块
from num2words import num2words  # 从num2words模块中导入num2words函数
import pyopenjtalk  # 导入pyopenjtalk模块
import jaconv  # 导入jaconv模块

def kata2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""
    text = text.strip()  # 去除文本两端的空白字符
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
            # 如果文本匹配 _MARKS 正则表达式，则将文本添加到结果列表中，并将文本向后移动一位
            res.append(text)
            text = text[1:]
            continue
        if text.startswith("ー"):
            # 如果文本以 "ー" 开头
            if prev:
                # 如果前一个字符存在，则将前一个字符的最后一个音素添加到结果列表中
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
    # 将文本中的标点符号替换为空格
    return re.sub(_MARKS, " ", text)
# 如果 yomi 符合 _MARKS 的正则表达式
if re.match(_MARKS, yomi):
    # 如果 word 的长度大于 1
    if len(word) > 1:
        # 将 word 中的标点符号替换后存入列表 word
        word = [replace_punctuation(i) for i in list(word)]
        # 将 word 赋值给 yomi
        yomi = word
        # 将 yomi 添加到 res 列表中
        res += yomi
        # 将 word 添加到 sep 列表中
        sep += word
        # 继续下一次循环
        continue
    # 如果 word 不在 rep_map 的键和值中
    elif word not in rep_map.keys() and word not in rep_map.values():
        # 将 word 赋值为逗号
        word = ","
    # 将 word 赋值给 yomi
    yomi = word
# 将 yomi 添加到 res 列表中
res.append(yomi)
# 如果 yomi 不符合 _MARKS 的正则表达式
else:
    # 如果 word 在 _SYMBOL_TOKENS 中
    if word in _SYMBOL_TOKENS:
        # 将 word 添加到 res 列表中
        res.append(word)
    # 如果 word 是 "っ" 或 "ッ"
    elif word in ("っ", "ッ"):
        # 将 "ッ" 添加到 res 列表中
        res.append("ッ")
    # 如果 word 在 _NO_YOMI_TOKENS 中
    elif word in _NO_YOMI_TOKENS:
        # 什么也不做
        pass
    else:
        # 将 word 添加到 res 列表中
        res.append(word)
    return hira2kata("".join(res))  # 将列表res中的假名转换为片假名，然后将其连接成一个字符串并返回

def text2sep_kata(text: str) -> (list, list):
    parsed = pyopenjtalk.run_frontend(text)  # 使用pyopenjtalk库对文本进行解析

    res = []  # 创建一个空列表，用于存储解析后的假名
    sep = []  # 创建一个空列表，用于存储解析后的单词
    for parts in parsed:  # 遍历解析后的结果
        word, yomi = replace_punctuation(parts["string"]), parts["pron"].replace(
            "’", ""
        )  # 获取单词和假名，并去除假名中的特殊字符
        if yomi:  # 如果假名不为空
            if re.match(_MARKS, yomi):  # 如果假名符合特定的模式
                if len(word) > 1:  # 如果单词长度大于1
                    word = [replace_punctuation(i) for i in list(word)]  # 将单词中的标点符号替换为空格
                    yomi = word  # 将单词作为假名
                    res += yomi  # 将假名添加到res列表中
                    sep += word  # 将单词添加到sep列表中
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
        (0x3040, 0x309F),  # 定义平假名范围的 Unicode 编码
        (0x30A0, 0x30FF),  # 定义片假名范围的 Unicode 编码
        (0x4E00, 0x9FFF),  # 定义汉字 (CJK Unified Ideographs) 范围的 Unicode 编码
        (0x3400, 0x4DBF),  # 定义汉字扩展 A 范围的 Unicode 编码
        (0x20000, 0x2A6DF),  # 定义汉字扩展 B 范围的 Unicode 编码
        # 可以根据需要添加其他汉字扩展范围
    ]

    # 将字符的 Unicode 编码转换为整数
    char_code = ord(char)

    # 检查字符是否在任何一个日语范围内
    for start, end in japanese_ranges:
        if start <= char_code <= end:  # 如果字符的 Unicode 编码在范围内
            return True  # 返回 True
    # 如果字符的 Unicode 编码不在任何一个日语范围内
    return False  # 返回 False


rep_map = {  # 定义替换映射的字典
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 使用 open 函数读取文件内容，然后封装成字节流
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    zip.close()  # 关闭 ZIP 对象
    return fdict  # 返回结果字典
    "）": "'",  # 将中文右括号替换为英文单引号
    "(": "'",   # 将中文左括号替换为英文单引号
    ")": "'",   # 将中文右括号替换为英文单引号
    "《": "'",  # 将中文书名号左边的符号替换为英文单引号
    "》": "'",  # 将中文书名号右边的符号替换为英文单引号
    "【": "'",  # 将中文方括号左边的符号替换为英文单引号
    "】": "'",  # 将中文方括号右边的符号替换为英文单引号
    "[": "'",   # 将英文方括号替换为英文单引号
    "]": "'",   # 将英文方括号替换为英文单引号
    "—": "-",   # 将中文破折号替换为英文破折号
    "−": "-",   # 将数学符号减号替换为英文破折号
    "～": "-",  # 将中文波浪号替换为英文破折号
    "~": "-",   # 将英文波浪号替换为英文破折号
    "「": "'",  # 将中文书名号左边的符号替换为英文单引号
    "」": "'",  # 将中文书名号右边的符号替换为英文单引号
}

def replace_punctuation(text):
    # 创建一个正则表达式模式，用于匹配需要替换的标点符号
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    # 使用正则表达式替换文本中匹配的内容
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    # 使用正则表达式替换文本中不属于指定日语字符和标点符号的内容
    replaced_text = re.sub(
        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
        + "".join(punctuation)
        + r"]+",
        "",
        replaced_text,
    )

    # 返回经过处理后的文本
    return replaced_text


def text_normalize(text):
    # 使用 unicodedata 模块对文本进行 NFKC 规范化
    res = unicodedata.normalize("NFKC", text)
    # 将文本中的数字转换为对应的日语单词
    res = japanese_convert_numbers_to_words(res)
    # 替换文本中的标点符号
    res = replace_punctuation(res)
    # 返回处理后的文本
    return res
# 分配电话号码给单词
def distribute_phone(n_phone, n_word):
    # 创建一个长度为 n_word 的列表，每个元素初始化为 0，表示每个单词的电话号码数量
    phones_per_word = [0] * n_word
    # 遍历每个电话号码
    for task in range(n_phone):
        # 找到电话号码数量最少的单词
        min_tasks = min(phones_per_word)
        # 找到该单词在列表中的索引
        min_index = phones_per_word.index(min_tasks)
        # 为该单词分配一个电话号码
        phones_per_word[min_index] += 1
    # 返回每个单词分配的电话号码数量列表
    return phones_per_word


# 处理长音节
def handle_long(sep_phonemes):
    # 遍历分离的音节列表
    for i in range(len(sep_phonemes)):
        # 如果当前音节以 "ー" 开头
        if sep_phonemes[i][0] == "ー":
            # 将当前音节的开头改为前一个音节的结尾
            sep_phonemes[i][0] = sep_phonemes[i - 1][-1]
        # 如果当前音节包含 "ー"
        if "ー" in sep_phonemes[i]:
            # 遍历当前音节中的每个字符
            for j in range(len(sep_phonemes[i])):
                # 如果当前字符是 "ー"
                if sep_phonemes[i][j] == "ー":
                    # 将当前字符改为前一个字符的结尾
                    sep_phonemes[i][j] = sep_phonemes[i][j - 1][-1]
    # 返回处理后的音节列表
    return sep_phonemes
# 使用预训练模型的路径创建一个自动分词器
tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese")

# 定义一个函数，用于将音素和音调对齐
def align_tones(phones, tones):
    res = []  # 创建一个空列表，用于存储结果
    for pho in phones:  # 遍历音素列表
        temp = [0] * len(pho)  # 创建一个与音素列表相同长度的全为0的临时列表
        for idx, p in enumerate(pho):  # 遍历音素列表中的每个音素
            if len(tones) == 0:  # 如果音调列表为空
                break  # 跳出循环
            if p == tones[0][0]:  # 如果当前音素与音调列表中的第一个元素的第一个元素相同
                temp[idx] = tones[0][1]  # 将临时列表中对应位置的值设置为音调列表中第一个元素的第二个元素
                if idx > 0:  # 如果当前位置不是第一个位置
                    temp[idx] += temp[idx - 1]  # 将当前位置的值加上前一个位置的值
                tones.pop(0)  # 移除音调列表中的第一个元素
        temp = [0] + temp  # 在临时列表的开头添加一个0
        temp = temp[:-1]  # 去掉临时列表的最后一个元素
        if -1 in temp:  # 如果临时列表中包含-1
            temp = [i + 1 for i in temp]  # 对列表temp中的每个元素加1，生成新的列表
        res.append(temp)  # 将处理后的temp列表添加到res列表中
    res = [i for j in res for i in j]  # 将res列表中的嵌套列表展开成一个一维列表
    assert not any([i < 0 for i in res]) and not any([i > 1 for i in res])  # 断言，确保res列表中所有元素都大于等于0且小于等于1
    return res  # 返回处理后的res列表


def g2p(norm_text):
    sep_text, sep_kata, acc = text2sep_kata(norm_text)  # 调用text2sep_kata函数，将norm_text分割成sep_text、sep_kata和acc
    sep_tokenized = [tokenizer.tokenize(i) for i in sep_text]  # 对sep_text中的每个元素进行分词处理，生成分词后的列表
    sep_phonemes = handle_long([kata2phoneme(i) for i in sep_kata])  # 对sep_kata中的每个元素进行音素处理，生成音素列表
    # 异常处理，MeCab不认识的词的话会一路传到这里来，然后炸掉。目前来看只有那些超级稀有的生僻词会出现这种情况
    for i in sep_phonemes:  # 遍历sep_phonemes列表
        for j in i:  # 遍历i列表
            assert j in symbols, (sep_text, sep_kata, sep_phonemes)  # 断言，确保sep_phonemes中的每个元素都在symbols中，否则抛出异常
    tones = align_tones(sep_phonemes, acc)  # 调用align_tones函数，对sep_phonemes和acc进行音调对齐处理

    word2ph = []  # 初始化空列表word2ph
    for token, phoneme in zip(sep_tokenized, sep_phonemes):  # 遍历sep_tokenized和sep_phonemes列表
        phone_len = len(phoneme)  # 获取phoneme列表的长度
        word_len = len(token)  # 获取单词的长度

        aaa = distribute_phone(phone_len, word_len)  # 调用 distribute_phone 函数，计算单词长度和电话长度的分布
        word2ph += aaa  # 将计算得到的分布添加到 word2ph 中
    phones = ["_"] + [j for i in sep_phonemes for j in i] + ["_"]  # 生成包含下划线的电话列表
    tones = [0] + tones + [0]  # 在 tones 列表的开头和结尾添加 0
    word2ph = [1] + word2ph + [1]  # 在 word2ph 列表的开头和结尾添加 1
    assert len(phones) == len(tones)  # 断言电话列表和音调列表的长度相等
    return phones, tones, word2ph  # 返回电话列表、音调列表和 word2ph 列表


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese")  # 从预训练模型中加载分词器
    text = "hello,こんにちは、世界ー！……"  # 待处理的文本
    from text.japanese_bert import get_bert_feature  # 从 text.japanese_bert 模块中导入 get_bert_feature 函数

    text = text_normalize(text)  # 对文本进行规范化处理
    print(text)  # 打印处理后的文本

    phones, tones, word2ph = g2p(text)  # 调用 g2p 函数，将文本转换为音素、音调和 word2ph 列表
    # 调用 get_bert_feature 函数，传入文本和 word2ph 参数，将结果赋值给 bert 变量
    bert = get_bert_feature(text, word2ph)

    # 打印 phones、tones、word2ph 和 bert.shape 的值
    print(phones, tones, word2ph, bert.shape)
```