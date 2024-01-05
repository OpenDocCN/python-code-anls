# `d:/src/tocomm/Bert-VITS2\oldVersion\V111\text\fix\japanese.py`

```
# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
# 导入所需的模块
import re  # 导入正则表达式模块
import unicodedata  # 导入Unicode数据模块

from transformers import AutoTokenizer  # 从transformers模块中导入AutoTokenizer类

from .. import punctuation, symbols  # 从当前包的上一级包中导入punctuation和symbols模块

from num2words import num2words  # 从num2words模块中导入num2words函数

import pyopenjtalk  # 导入pyopenjtalk模块
import jaconv  # 导入jaconv模块


def kata2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""
    # 去除文本两端的空格
    text = text.strip()
    # 如果文本为"ー"，则返回包含"ー"的列表
    if text == "ー":
        return ["ー"]
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
def hira2kata(text: str) -> str:
    # 将输入的平假名文本转换为片假名文本
    return jaconv.hira2kata(text)

# 定义一些特殊符号和不需要读音的符号
_SYMBOL_TOKENS = set(list("・、。？！"))
_NO_YOMI_TOKENS = set(list("「」『』―（）［］[]"))
# 定义一个正则表达式，用于匹配需要替换的标点符号
_MARKS = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

def text2kata(text: str) -> str:
    # 使用 pyopenjtalk 库对文本进行解析
    parsed = pyopenjtalk.run_frontend(text)

    res = []
    # 遍历解析后的文本
    for parts in parsed:
        # 获取单词和对应的读音
        word, yomi = replace_punctuation(parts["orig"]), parts["pron"].replace("’", "")
        # 如果有读音
        if yomi:
            # 如果读音中包含需要替换的标点符号
            if re.match(_MARKS, yomi):
                # 如果单词长度大于1
                if len(word) > 1:
                    word = [replace_punctuation(i) for i in list(word)]  # 将单词中的标点符号替换为空格
                    yomi = word  # 将单词赋值给yomi
                    res += yomi  # 将yomi添加到res中
                    sep += word  # 将单词添加到sep中
                    continue  # 继续下一次循环
                elif word not in rep_map.keys() and word not in rep_map.values():  # 如果单词不在rep_map的键和值中
                    word = ","  # 将单词替换为逗号
                yomi = word  # 将单词赋值给yomi
            res.append(yomi)  # 将yomi添加到res中
        else:  # 如果不满足上述条件
            if word in _SYMBOL_TOKENS:  # 如果单词在_SYMBOL_TOKENS中
                res.append(word)  # 将单词添加到res中
            elif word in ("っ", "ッ"):  # 如果单词是"っ"或"ッ"
                res.append("ッ")  # 将"ッ"添加到res中
            elif word in _NO_YOMI_TOKENS:  # 如果单词在_NO_YOMI_TOKENS中
                pass  # 什么也不做
            else:  # 如果不满足上述条件
                res.append(word)  # 将单词添加到res中
    return hira2kata("".join(res))  # 将res中的内容转换为片假名并返回
# 将文本转换为分词和假名列表
def text2sep_kata(text: str) -> (list, list):
    # 使用 pyopenjtalk 运行前端处理文本
    parsed = pyopenjtalk.run_frontend(text)

    # 初始化结果列表和分词列表
    res = []
    sep = []

    # 遍历处理后的文本
    for parts in parsed:
        # 获取单词和假名，替换标点符号
        word, yomi = replace_punctuation(parts["orig"]), parts["pron"].replace("’", "")
        
        # 如果假名不为空
        if yomi:
            # 如果假名匹配标点符号的正则表达式
            if re.match(_MARKS, yomi):
                # 如果单词长度大于1
                if len(word) > 1:
                    # 将单词中的标点符号替换，并将假名列表赋值为单词列表
                    word = [replace_punctuation(i) for i in list(word)]
                    yomi = word
                    # 将假名列表添加到结果列表，将单词列表添加到分词列表
                    res += yomi
                    sep += word
                    continue
                # 如果单词不在替换映射的键和值中
                elif word not in rep_map.keys() and word not in rep_map.values():
                    # 将单词替换为逗号
                    word = ","
                # 将假名赋值为单词
                yomi = word
            # 将假名添加到结果列表
            res.append(yomi)
        else:  # 如果不是以上情况，执行以下操作
            if word in _SYMBOL_TOKENS:  # 如果单词在符号标记中
                res.append(word)  # 将单词添加到结果列表中
            elif word in ("っ", "ッ"):  # 如果单词是"っ"或"ッ"
                res.append("ッ")  # 将"ッ"添加到结果列表中
            elif word in _NO_YOMI_TOKENS:  # 如果单词在无读音标记中
                pass  # 不做任何操作
            else:  # 如果以上条件都不满足
                res.append(word)  # 将单词添加到结果列表中
        sep.append(word)  # 将单词添加到分隔列表中
    return sep, [hira2kata(i) for i in res]  # 返回分隔列表和结果列表中每个单词的片假名转片假名的结果


_ALPHASYMBOL_YOMI = {  # 创建一个字典_ALPHASYMBOL_YOMI
    "#": "シャープ",  # 键为"#"，值为"シャープ"
    "%": "パーセント",  # 键为"%"，值为"パーセント"
    "&": "アンド",  # 键为"&"，值为"アンド"
    "+": "プラス",  # 键为"+"，值为"プラス"
    "-": "マイナス",  # 键为"-"，值为"マイナス"
    ":": "コロン",  # 键为":"，值为"コロン"
    ";": "セミコロン",  # 将分号";"映射为日语中的"セミコロン"
    "<": "小なり",  # 将小于号"<"映射为日语中的"小なり"
    "=": "イコール",  # 将等于号"="映射为日语中的"イコール"
    ">": "大なり",  # 将大于号">"映射为日语中的"大なり"
    "@": "アット",  # 将符号"@"映射为日语中的"アット"
    "a": "エー",  # 将字母"a"映射为日语中的"エー"
    "b": "ビー",  # 将字母"b"映射为日语中的"ビー"
    "c": "シー",  # 将字母"c"映射为日语中的"シー"
    "d": "ディー",  # 将字母"d"映射为日语中的"ディー"
    "e": "イー",  # 将字母"e"映射为日语中的"イー"
    "f": "エフ",  # 将字母"f"映射为日语中的"エフ"
    "g": "ジー",  # 将字母"g"映射为日语中的"ジー"
    "h": "エイチ",  # 将字母"h"映射为日语中的"エイチ"
    "i": "アイ",  # 将字母"i"映射为日语中的"アイ"
    "j": "ジェー",  # 将字母"j"映射为日语中的"ジェー"
    "k": "ケー",  # 将字母"k"映射为日语中的"ケー"
    "l": "エル",  # 将字母"l"映射为日语中的"エル"
    "m": "エム",  # 将字母"m"映射为日语中的"エム"
    "n": "エヌ",  # 将字母"n"映射为日语中的"エヌ"
    "o": "オー",  # 将字母"o"映射为日语中的"オー"
    "p": "ピー",  # 将字母 "p" 映射为日语中的发音 "ピー"
    "q": "キュー",  # 将字母 "q" 映射为日语中的发音 "キュー"
    "r": "アール",  # 将字母 "r" 映射为日语中的发音 "アール"
    "s": "エス",  # 将字母 "s" 映射为日语中的发音 "エス"
    "t": "ティー",  # 将字母 "t" 映射为日语中的发音 "ティー"
    "u": "ユー",  # 将字母 "u" 映射为日语中的发音 "ユー"
    "v": "ブイ",  # 将字母 "v" 映射为日语中的发音 "ブイ"
    "w": "ダブリュー",  # 将字母 "w" 映射为日语中的发音 "ダブリュー"
    "x": "エックス",  # 将字母 "x" 映射为日语中的发音 "エックス"
    "y": "ワイ",  # 将字母 "y" 映射为日语中的发音 "ワイ"
    "z": "ゼット",  # 将字母 "z" 映射为日语中的发音 "ゼット"
    "α": "アルファ",  # 将希腊字母 "α" 映射为日语中的发音 "アルファ"
    "β": "ベータ",  # 将希腊字母 "β" 映射为日语中的发音 "ベータ"
    "γ": "ガンマ",  # 将希腊字母 "γ" 映射为日语中的发音 "ガンマ"
    "δ": "デルタ",  # 将希腊字母 "δ" 映射为日语中的发音 "デルタ"
    "ε": "イプシロン",  # 将希腊字母 "ε" 映射为日语中的发音 "イプシロン"
    "ζ": "ゼータ",  # 将希腊字母 "ζ" 映射为日语中的发音 "ゼータ"
    "η": "イータ",  # 将希腊字母 "η" 映射为日语中的发音 "イータ"
    "θ": "シータ",  # 将希腊字母 "θ" 映射为日语中的发音 "シータ"
    "ι": "イオタ",  # 将希腊字母 "ι" 映射为日语中的发音 "イオタ"
    "κ": "カッパ",  # 将希腊字母 κ 映射为日文字符 カッパ
    "λ": "ラムダ",  # 将希腊字母 λ 映射为日文字符 ラムダ
    "μ": "ミュー",  # 将希腊字母 μ 映射为日文字符 ミュー
    "ν": "ニュー",  # 将希腊字母 ν 映射为日文字符 ニュー
    "ξ": "クサイ",  # 将希腊字母 ξ 映射为日文字符 クサイ
    "ο": "オミクロン",  # 将希腊字母 ο 映射为日文字符 オミクロン
    "π": "パイ",  # 将希腊字母 π 映射为日文字符 パイ
    "ρ": "ロー",  # 将希腊字母 ρ 映射为日文字符 ロー
    "σ": "シグマ",  # 将希腊字母 σ 映射为日文字符 シグマ
    "τ": "タウ",  # 将希腊字母 τ 映射为日文字符 タウ
    "υ": "ウプシロン",  # 将希腊字母 υ 映射为日文字符 ウプシロン
    "φ": "ファイ",  # 将希腊字母 φ 映射为日文字符 ファイ
    "χ": "カイ",  # 将希腊字母 χ 映射为日文字符 カイ
    "ψ": "プサイ",  # 将希腊字母 ψ 映射为日文字符 プサイ
    "ω": "オメガ",  # 将希腊字母 ω 映射为日文字符 オメガ
}

_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")  # 创建正则表达式对象，用于匹配带逗号分隔的数字
_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}  # 创建货币符号到日文名称的映射字典
_CURRENCY_RX = re.compile(r"([$¥£€])([0-9.]*[0-9])")  # 创建一个正则表达式对象，用于匹配货币符号和金额
_NUMBER_RX = re.compile(r"[0-9]+(\.[0-9]+)?")  # 创建一个正则表达式对象，用于匹配数字

def japanese_convert_numbers_to_words(text: str) -> str:
    res = _NUMBER_WITH_SEPARATOR_RX.sub(lambda m: m[0].replace(",", ""), text)  # 使用正则表达式替换文本中的数字和逗号
    res = _CURRENCY_RX.sub(lambda m: m[2] + _CURRENCY_MAP.get(m[1], m[1]), res)  # 使用正则表达式替换文本中的货币符号和金额
    res = _NUMBER_RX.sub(lambda m: num2words(m[0], lang="ja"), res)  # 使用正则表达式替换文本中的数字为对应的日文单词
    return res

def japanese_convert_alpha_symbols_to_words(text: str) -> str:
    return "".join([_ALPHASYMBOL_YOMI.get(ch, ch) for ch in text.lower()])  # 将文本中的字母和符号转换为对应的日文单词

def japanese_text_to_phonemes(text: str) -> str:
    """Convert Japanese text to phonemes."""
    res = unicodedata.normalize("NFKC", text)  # 使用Unicode规范化函数将文本转换为NFKC格式
    res = japanese_convert_numbers_to_words(res)  # 调用上面定义的函数，将文本中的数字转换为对应的日文单词
    # res = japanese_convert_alpha_symbols_to_words(res)  # 可能是注释掉的代码，暂时不起作用
    res = text2kata(res)  # 使用 text2kata 函数将输入的文本转换为片假名
    res = kata2phoneme(res)  # 使用 kata2phoneme 函数将片假名转换为音素
    return res  # 返回转换后的音素文本


def is_japanese_character(char):
    # 定义日语文字系统的 Unicode 范围
    japanese_ranges = [
        (0x3040, 0x309F),  # 平假名
        (0x30A0, 0x30FF),  # 片假名
        (0x4E00, 0x9FFF),  # 汉字 (CJK Unified Ideographs)
        (0x3400, 0x4DBF),  # 汉字扩展 A
        (0x20000, 0x2A6DF),  # 汉字扩展 B
        # 可以根据需要添加其他汉字扩展范围
    ]

    # 将字符的 Unicode 编码转换为整数
    char_code = ord(char)

    # 检查字符是否在任何一个日语范围内
    for start, end in japanese_ranges:
        # 遍历日文字符范围，判断给定字符编码是否在范围内
        if start <= char_code <= end:
            # 如果在范围内，返回 True
            return True
    # 如果不在范围内，返回 False
    return False


rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "...": "…",
    "···": "…",
    "・・・": "…",
    "·": ",",
```
- rep_map 是一个用于替换字符的字典，将字典中的键替换为对应的值。
    "・": ",",  # 将"・"替换为","
    "、": ",",  # 将"、"替换为","
    "$": ".",   # 将"$"替换为"."
    "“": "'",   # 将“”‘’（）()《》【】[]—−～替换为"'"
    "”": "'",   # 将“”‘’（）()《》【】[]—−～替换为"'"
    "‘": "'",   # 将“”‘’（）()《》【】[]—−～替换为"'"
    "’": "'",   # 将“”‘’（）()《》【】[]—−～替换为"'"
    "（": "'",   # 将“”‘’（）()《》【】[]—−～替换为"'"
    "）": "'",   # 将“”‘’（）()《》【】[]—−～替换为"'"
    "(": "'",    # 将“”‘’（）()《》【】[]—−～替换为"'"
    ")": "'",    # 将“”‘’（）()《》【】[]—−～替换为"'"
    "《": "'",   # 将“”‘’（）()《》【】[]—−～替换为"'"
    "》": "'",   # 将“”‘’（）()《》【】[]—−～替换为"'"
    "【": "'",   # 将“”‘’（）()《》【】[]—−～替换为"'"
    "】": "'",   # 将“”‘’（）()《》【】[]—−～替换为"'"
    "[": "'",    # 将“”‘’（）()《》【】[]—−～替换为"'"
    "]": "'",    # 将“”‘’（）()《》【】[]—−～替换为"'"
    "—": "-",    # 将“”‘’（）()《》【】[]—−～替换为"-"
    "−": "-",    # 将“”‘’（）()《》【】[]—−～替换为"-"
    "～": "-",    # 将“”‘’（）()《》【】[]—−～替换为"-"
    "~": "-",  # 将波浪线替换为短横线
    "「": "'",  # 将左双引号替换为单引号
    "」": "'",  # 将右双引号替换为单引号
}

# 定义一个函数，用于替换文本中的标点符号
def replace_punctuation(text):
    # 创建一个正则表达式模式，用于匹配需要替换的标点符号
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    # 使用正则表达式模式替换文本中的标点符号
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    # 使用正则表达式去除文本中除了日文、片假名、中文和标点符号以外的字符
    replaced_text = re.sub(
        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF"
        + "".join(punctuation)
        + r"]+",
        "",
        replaced_text,
    )

    return replaced_text  # 返回替换后的文本
def text_normalize(text):
    # 使用 unicodedata.normalize 函数将文本进行 Unicode 规范化
    res = unicodedata.normalize("NFKC", text)
    # 调用 japanese_convert_numbers_to_words 函数将文本中的数字转换为对应的日文单词
    res = japanese_convert_numbers_to_words(res)
    # 调用 replace_punctuation 函数替换文本中的标点符号
    res = replace_punctuation(res)
    return res


def distribute_phone(n_phone, n_word):
    # 创建一个长度为 n_word 的列表，用于记录每个单词分配到的电话数量
    phones_per_word = [0] * n_word
    # 遍历 n_phone 次，将电话分配给单词
    for task in range(n_phone):
        # 找到 phones_per_word 中值最小的元素
        min_tasks = min(phones_per_word)
        # 找到最小值的索引
        min_index = phones_per_word.index(min_tasks)
        # 将电话分配给对应的单词
        phones_per_word[min_index] += 1
    return phones_per_word


def handle_long(sep_phonemes):
    # 在这里添加对 handle_long 函数的注释
    for i in range(len(sep_phonemes)):  # 遍历分隔后的音素列表
        if sep_phonemes[i][0] == "ー":  # 如果当前音素列表的第一个音素是 "ー"
            sep_phonemes[i][0] = sep_phonemes[i - 1][-1]  # 将当前音素列表的第一个音素替换为前一个音素列表的最后一个音素
        if "ー" in sep_phonemes[i]:  # 如果当前音素列表包含 "ー"
            for j in range(len(sep_phonemes[i])):  # 遍历当前音素列表
                if sep_phonemes[i][j] == "ー":  # 如果当前音素列表中的元素是 "ー"
                    sep_phonemes[i][j] = sep_phonemes[i][j - 1][-1]  # 将当前音素列表中的 "ー" 替换为前一个音素列表的最后一个音素
    return sep_phonemes  # 返回处理后的音素列表

tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")  # 从预训练模型中加载分词器

def g2p(norm_text):  # 定义 g2p 函数，将文本转换为音素
    sep_text, sep_kata = text2sep_kata(norm_text)  # 调用 text2sep_kata 函数将文本分隔为文本和假名
    sep_tokenized = [tokenizer.tokenize(i) for i in sep_text]  # 使用分词器对分隔后的文本进行分词
    sep_phonemes = handle_long([kata2phoneme(i) for i in sep_kata])  # 调用 kata2phoneme 函数将假名转换为音素，并处理长音
    # 异常处理，MeCab不认识的词的话会一路传到这里来，然后炸掉。目前来看只有那些超级稀有的生僻词会出现这种情况
    for i in sep_phonemes:  # 遍历音素列表
        for j in i:  # 遍历音素列表中的元素
    assert j in symbols, (sep_text, sep_kata, sep_phonemes)
    # 检查 j 是否在 symbols 中，如果不在则抛出异常，异常信息为 (sep_text, sep_kata, sep_phonemes)

    word2ph = []
    # 初始化空列表 word2ph
    for token, phoneme in zip(sep_tokenized, sep_phonemes):
        # 遍历 sep_tokenized 和 sep_phonemes 中对应位置的元素
        phone_len = len(phoneme)
        # 获取 phoneme 的长度
        word_len = len(token)
        # 获取 token 的长度

        aaa = distribute_phone(phone_len, word_len)
        # 调用 distribute_phone 函数，传入 phone_len 和 word_len 作为参数，返回结果赋值给 aaa
        word2ph += aaa
        # 将 aaa 中的元素添加到 word2ph 列表中
    phones = ["_"] + [j for i in sep_phonemes for j in i] + ["_"]
    # 生成 phones 列表，包含 "_" 和 sep_phonemes 中的元素
    tones = [0 for i in phones]
    # 生成与 phones 长度相同的全为 0 的列表 tones
    word2ph = [1] + word2ph + [1]
    # 在 word2ph 列表的开头和结尾添加 1
    return phones, tones, word2ph
    # 返回 phones, tones, word2ph 作为结果


if __name__ == "__main__":
    # 如果当前脚本被直接执行
    tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")
    # 从预训练模型 "./bert/bert-base-japanese-v3" 中加载 tokenizer
    text = "hello,こんにちは、世界ー！……"
    # 初始化文本变量 text
    from text.japanese_bert import get_bert_feature
    # 从 text.japanese_bert 模块中导入 get_bert_feature 函数
    # 对文本进行规范化处理
    text = text_normalize(text)
    # 打印处理后的文本
    print(text)

    # 使用文本进行音素到字音的转换，返回音素列表、音调列表和字词到音素的映射
    phones, tones, word2ph = g2p(text)
    # 获取文本的 BERT 特征
    bert = get_bert_feature(text, word2ph)

    # 打印音素列表、音调列表、字词到音素的映射和 BERT 特征的形状
    print(phones, tones, word2ph, bert.shape)
```