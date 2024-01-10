# `Bert-VITS2\oldVersion\V111\text\fix\japanese.py`

```
# 导入所需的模块
import re
import unicodedata
from transformers import AutoTokenizer
from .. import punctuation, symbols
from num2words import num2words
import pyopenjtalk
import jaconv

# 定义函数，将片假名文本转换为音素
def kata2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""
    # 去除文本两端的空格
    text = text.strip()
    # 如果文本为"ー"，返回包含"ー"的列表
    if text == "ー":
        return ["ー"]
    # 如果文本以"ー"开头，返回包含"ー"的列表和剩余部分的转换结果
    elif text.startswith("ー"):
        return ["ー"] + kata2phoneme(text[1:])
    res = []
    prev = None
    # 循环处理文本
    while text:
        # 如果文本匹配标点符号的正则表达式，将其添加到结果列表中
        if re.match(_MARKS, text):
            res.append(text)
            text = text[1:]
            continue
        # 如果文本以"ー"开头，且前一个字符存在，则将前一个字符的最后一个音素添加到结果列表中
        if text.startswith("ー"):
            if prev:
                res.append(prev[-1])
            text = text[1:]
            continue
        # 使用 pyopenjtalk 将文本转换为小写的音素序列，并替换特定字符后拆分为列表，添加到结果列表中
        res += pyopenjtalk.g2p(text).lower().replace("cl", "q").split(" ")
        break
    # 返回结果列表
    return res

# 定义函数，将平假名文本转换为片假名
def hira2kata(text: str) -> str:
    return jaconv.hira2kata(text)

# 定义函数，将文本转换为片假名
def text2kata(text: str) -> str:
    # 使用 pyopenjtalk 的前端处理文本
    parsed = pyopenjtalk.run_frontend(text)
    res = []
    # 遍历解析后的文本
    for parts in parsed:
        # 获取单词和假名，替换标点符号和特殊字符
        word, yomi = replace_punctuation(parts["orig"]), parts["pron"].replace("’", "")
        # 如果假名不为空
        if yomi:
            # 如果假名符合特定模式
            if re.match(_MARKS, yomi):
                # 如果单词长度大于1
                if len(word) > 1:
                    # 替换单词中的标点符号
                    word = [replace_punctuation(i) for i in list(word)]
                    # 将假名设置为单词
                    yomi = word
                    # 将假名添加到结果列表中
                    res += yomi
                    # 将单词添加到分隔符列表中
                    sep += word
                    # 继续下一次循环
                    continue
                # 如果单词不在替换映射中，并且不在值中
                elif word not in rep_map.keys() and word not in rep_map.values():
                    # 将单词设置为逗号
                    word = ","
                # 将假名设置为单词
                yomi = word
            # 将假名添加到结果列表中
            res.append(yomi)
        # 如果假名为空
        else:
            # 如果单词是特殊符号
            if word in _SYMBOL_TOKENS:
                # 将单词添加到结果列表中
                res.append(word)
            # 如果单词是特殊假名
            elif word in ("っ", "ッ"):
                # 将特殊假名添加到结果列表中
                res.append("ッ")
            # 如果单词是无假名的特殊符号
            elif word in _NO_YOMI_TOKENS:
                # 跳过
                pass
            # 如果单词不是特殊符号
            else:
                # 将单词添加到结果列表中
                res.append(word)
    # 将结果列表中的假名转换为片假名
    return hira2kata("".join(res))
# 将文本转换为分隔片假名
def text2sep_kata(text: str) -> (list, list):
    # 使用 pyopenjtalk 库的 run_frontend 方法解析文本
    parsed = pyopenjtalk.run_frontend(text)

    # 初始化结果列表和分隔片假名列表
    res = []
    sep = []

    # 遍历解析结果
    for parts in parsed:
        # 获取单词和假名，替换标点符号和特殊字符
        word, yomi = replace_punctuation(parts["orig"]), parts["pron"].replace("’", "")
        
        # 如果假名不为空
        if yomi:
            # 如果假名匹配特殊标记
            if re.match(_MARKS, yomi):
                # 如果单词长度大于1，将单词拆分为字符列表
                if len(word) > 1:
                    word = [replace_punctuation(i) for i in list(word)]
                    yomi = word
                    res += yomi
                    sep += word
                    continue
                # 如果单词不在替换映射中，将单词替换为逗号
                elif word not in rep_map.keys() and word not in rep_map.values():
                    word = ","
                yomi = word
            res.append(yomi)
        else:
            # 如果单词是特殊符号
            if word in _SYMBOL_TOKENS:
                res.append(word)
            # 如果单词是 "っ" 或 "ッ"
            elif word in ("っ", "ッ"):
                res.append("ッ")
            # 如果单词是无假名的特殊符号
            elif word in _NO_YOMI_TOKENS:
                pass
            else:
                res.append(word)
        sep.append(word)
    # 返回分隔片假名列表和转换为片假名的结果列表
    return sep, [hira2kata(i) for i in res]

# 片假名到假名的映射
_ALPHASYMBOL_YOMI = {
    "#": "シャープ",
    "%": "パーセント",
    "&": "アンド",
    "+": "プラス",
    "-": "マイナス",
    ":": "コロン",
    ";": "セミコロン",
    "<": "小なり",
    "=": "イコール",
    ">": "大なり",
    "@": "アット",
    "a": "エー",
    "b": "ビー",
    # ... 其他映射
}
    # 键为"ψ"，值为"プサイ"
    "ψ": "プサイ",
    # 键为"ω"，值为"オメガ"
    "ω": "オメガ",
# 定义一个正则表达式，用于匹配带有逗号分隔符的数字
_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")
# 定义一个货币符号到对应日语名称的映射字典
_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
# 定义一个正则表达式，用于匹配货币符号和对应的数字
_CURRENCY_RX = re.compile(r"([$¥£€])([0-9.]*[0-9])")
# 定义一个正则表达式，用于匹配数字
_NUMBER_RX = re.compile(r"[0-9]+(\.[0-9]+)?")

# 将输入的文本中的数字转换为对应的日语单词
def japanese_convert_numbers_to_words(text: str) -> str:
    res = _NUMBER_WITH_SEPARATOR_RX.sub(lambda m: m[0].replace(",", ""), text)
    res = _CURRENCY_RX.sub(lambda m: m[2] + _CURRENCY_MAP.get(m[1], m[1]), res)
    res = _NUMBER_RX.sub(lambda m: num2words(m[0], lang="ja"), res)
    return res

# 将输入的文本中的字母和符号转换为对应的日语单词
def japanese_convert_alpha_symbols_to_words(text: str) -> str:
    return "".join([_ALPHASYMBOL_YOMI.get(ch, ch) for ch in text.lower()])

# 将输入的日语文本转换为音素
def japanese_text_to_phonemes(text: str) -> str:
    """Convert Japanese text to phonemes."""
    res = unicodedata.normalize("NFKC", text)
    res = japanese_convert_numbers_to_words(res)
    # res = japanese_convert_alpha_symbols_to_words(res)
    res = text2kata(res)
    res = kata2phoneme(res)
    return res

# 检查字符是否为日语字符
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
        if start <= char_code <= end:
            return True

    return False

# 定义一些字符替换规则的映射字典
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
    "・": ",",
    "、": ",",
    "$": ".",
    "“": "'",
    "”": "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "−": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    # 将中文标点符号「」替换为英文单引号'
    "」": "'",
# 替换文本中的标点符号
def replace_punctuation(text):
    # 创建正则表达式模式，用于匹配需要替换的标点符号
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    # 使用正则表达式模式替换文本中的标点符号
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    # 移除文本中除了日语、中文和特定标点符号之外的所有字符
    replaced_text = re.sub(
        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF"
        + "".join(punctuation)
        + r"]+",
        "",
        replaced_text,
    )
    # 返回替换后的文本
    return replaced_text


# 对文本进行规范化处理
def text_normalize(text):
    # 使用 NFKC 规范化文本
    res = unicodedata.normalize("NFKC", text)
    # 将文本中的数字转换为对应的日文单词
    res = japanese_convert_numbers_to_words(res)
    # 替换文本中的标点符号
    res = replace_punctuation(res)
    # 返回规范化后的文本
    return res


# 分配电话
def distribute_phone(n_phone, n_word):
    # 初始化每个单词的电话数列表
    phones_per_word = [0] * n_word
    # 遍历每个电话任务
    for task in range(n_phone):
        # 找到电话数最少的单词
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        # 为该单词分配一个电话任务
        phones_per_word[min_index] += 1
    # 返回每个单词分配的电话数列表
    return phones_per_word


# 处理长音
def handle_long(sep_phonemes):
    # 遍历分隔后的音素列表
    for i in range(len(sep_phonemes)):
        # 如果当前音素是长音符号
        if sep_phonemes[i][0] == "ー":
            # 将长音符号替换为前一个音素的最后一个音节
            sep_phonemes[i][0] = sep_phonemes[i - 1][-1]
        # 如果当前音素包含长音符号
        if "ー" in sep_phonemes[i]:
            # 遍历当前音素中的每个字符
            for j in range(len(sep_phonemes[i])):
                # 如果当前字符是长音符号
                if sep_phonemes[i][j] == "ー":
                    # 将长音符号替换为当前音素中前一个音素的最后一个音节
                    sep_phonemes[i][j] = sep_phonemes[i][j - 1][-1]
    # 返回处理后的音素列表
    return sep_phonemes


# 从预训练模型中加载分词器
tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")


# 文本转换为音素
def g2p(norm_text):
    # 将文本分隔为假名和片假名
    sep_text, sep_kata = text2sep_kata(norm_text)
    # 使用分词器对文本进行分词
    sep_tokenized = [tokenizer.tokenize(i) for i in sep_text]
    # 将片假名转换为音素，并处理长音
    sep_phonemes = handle_long([kata2phoneme(i) for i in sep_kata])
    # 异常处理，检查是否有未知的音素
    for i in sep_phonemes:
        for j in i:
            assert j in symbols, (sep_text, sep_kata, sep_phonemes)
    # 分配电话任务给每个单词
    word2ph = []
    for token, phoneme in zip(sep_tokenized, sep_phonemes):
        phone_len = len(phoneme)
        word_len = len(token)
        # 分配电话任务
        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa
    # 创建一个包含下划线的列表，然后将sep_phonemes中的所有元素展开到列表中，最后再添加一个下划线
    phones = ["_"] + [j for i in sep_phonemes for j in i] + ["_"]
    # 创建一个与phones列表相同长度的全零列表
    tones = [0 for i in phones]
    # 在word2ph列表的开头和结尾添加1，然后返回结果
    word2ph = [1] + word2ph + [1]
    # 返回phones, tones, word2ph三个列表
    return phones, tones, word2ph
# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 从预训练模型中加载自动分词器
    tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")
    # 定义一个文本字符串
    text = "hello,こんにちは、世界ー！……"
    # 从文本.japanese_bert模块中导入get_bert_feature函数
    from text.japanese_bert import get_bert_feature

    # 对文本进行规范化处理
    text = text_normalize(text)
    # 打印处理后的文本
    print(text)

    # 使用g2p函数将文本转换为音素、音调和单词到音素的映射关系
    phones, tones, word2ph = g2p(text)
    # 使用get_bert_feature函数获取文本的BERT特征
    bert = get_bert_feature(text, word2ph)

    # 打印音素、音调、单词到音素的映射关系以及BERT特征的形状
    print(phones, tones, word2ph, bert.shape)
```