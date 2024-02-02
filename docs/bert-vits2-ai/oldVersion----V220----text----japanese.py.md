# `Bert-VITS2\oldVersion\V220\text\japanese.py`

```py
# 导入所需的库和模块
# 从正则表达式模块导入 re
# 从 unicodedata 模块导入 unicodedata
# 从 transformers 模块导入 AutoTokenizer
# 从当前目录下的 __init__.py 文件中导入 punctuation 和 symbols
# 从 num2words 模块导入 num2words
# 导入 pyopenjtalk 模块
# 从 jaconv 模块导入 hira2kata
import re
import unicodedata
from transformers import AutoTokenizer
from . import punctuation, symbols
from num2words import num2words
import pyopenjtalk
import jaconv

# 定义函数 kata2phoneme，接受一个字符串参数并返回一个字符串
def kata2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""
    # 去除字符串两端的空白字符
    text = text.strip()
    # 如果字符串为 "ー"，返回包含 "ー" 的列表
    if text == "ー":
        return ["ー"]
    # 如果字符串以 "ー" 开头，返回包含 "ー" 的列表和去除开头字符后的结果
    elif text.startswith("ー"):
        return ["ー"] + kata2phoneme(text[1:])
    # 初始化结果列表
    res = []
    prev = None
    # 循环处理字符串
    while text:
        # 如果字符串匹配 _MARKS 中定义的正则表达式，将其添加到结果列表中并去除该字符
        if re.match(_MARKS, text):
            res.append(text)
            text = text[1:]
            continue
        # 如果字符串以 "ー" 开头且前一个字符存在，将前一个字符的最后一个字符添加到结果列表中并去除 "ー"
        if text.startswith("ー"):
            if prev:
                res.append(prev[-1])
            text = text[1:]
            continue
        # 使用 pyopenjtalk.g2p 将文本转换为音素，转换为小写并替换 "cl" 为 "q"，然后按空格分割并添加到结果列表中
        res += pyopenjtalk.g2p(text).lower().replace("cl", "q").split(" ")
        break
    # 返回结果列表
    return res

# 定义函数 hira2kata，接受一个字符串参数并返回一个字符串
def hira2kata(text: str) -> str:
    # 使用 jaconv.hira2kata 将平假名文本转换为片假名文本并返回结果
    return jaconv.hira2kata(text)

# 初始化一个包含特殊符号的集合
_SYMBOL_TOKENS = set(list("・、。？！"))
# 初始化一个包含不需要读音的特殊符号的集合
_NO_YOMI_TOKENS = set(list("「」『』―（）［］[]"))
# 使用正则表达式模块创建一个用于匹配非日语字符的正则表达式对象
_MARKS = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# 定义函数 text2kata，接受一个字符串参数并返回一���字符串
def text2kata(text: str) -> str:
    # 使用 pyopenjtalk.run_frontend 处理文本并将结果赋值给 parsed
    parsed = pyopenjtalk.run_frontend(text)
    # 初始化结果列表
    res = []
    # 遍历解析后的文本列表
    for parts in parsed:
        # 获取单词和假名，替换标点符号和特殊字符
        word, yomi = replace_punctuation(parts["string"]), parts["pron"].replace(
            "’", ""
        )
        # 如果假名存在
        if yomi:
            # 如果假名符合特定模式
            if re.match(_MARKS, yomi):
                # 如果单词长度大于1
                if len(word) > 1:
                    # 替换单词中的标点符号
                    word = [replace_punctuation(i) for i in list(word)]
                    # 将单词作为假名
                    yomi = word
                    # 将假名添加到结果列表
                    res += yomi
                    # 将单词添加到分隔符列表
                    sep += word
                    # 继续下一次循环
                    continue
                # 如果单词不在替换映射的键和值中
                elif word not in rep_map.keys() and word not in rep_map.values():
                    # 将单词替换为逗号
                    word = ","
                # 将单词作为假名
                yomi = word
            # 将假名添加到结果列表
            res.append(yomi)
        # 如果假名不存在
        else:
            # 如果单词是特定符号
            if word in _SYMBOL_TOKENS:
                # 将单词添加到结果列表
                res.append(word)
            # 如果单词是特定假名
            elif word in ("っ", "ッ"):
                # 将特定假名添加到结果列表
                res.append("ッ")
            # 如果单词是特定无假名符号
            elif word in _NO_YOMI_TOKENS:
                # 跳过
                pass
            # 如果单词不是特定无假名符号
            else:
                # 将单词添加到结果列表
                res.append(word)
    # 将结果列表中的平假名转换为片假名，然后连接成字符串返回
    return hira2kata("".join(res))
# 将文本转换为分词和假名的列表
def text2sep_kata(text: str) -> (list, list):
    # 使用 pyopenjtalk 运行前端处理文本，返回解析结果
    parsed = pyopenjtalk.run_frontend(text)

    # 初始化结果列表和分词列表
    res = []
    sep = []

    # 遍历解析结果
    for parts in parsed:
        # 替换标点符号，获取单词和假名
        word, yomi = replace_punctuation(parts["string"]), parts["pron"].replace(
            "’", ""
        )
        # 如果有假名
        if yomi:
            # 如果假名匹配标点符号
            if re.match(_MARKS, yomi):
                # 如果单词长度大于1，将单词拆分为字符列表，更新假名和结果列表
                if len(word) > 1:
                    word = [replace_punctuation(i) for i in list(word)]
                    yomi = word
                    res += yomi
                    sep += word
                    continue
                # 如果单词不在替换映射中，也不在值中，将单词替换为逗号
                elif word not in rep_map.keys() and word not in rep_map.values():
                    word = ","
                yomi = word
            res.append(yomi)
        # 如果没有假名
        else:
            # 如果单词是符号标记，将单词添加到结果列表
            if word in _SYMBOL_TOKENS:
                res.append(word)
            # 如果单词是特殊符号，将单词添加到结果列表
            elif word in ("っ", "ッ"):
                res.append("ッ")
            # 如果单词是无假名的特殊符号，跳过
            elif word in _NO_YOMI_TOKENS:
                pass
            # 否则将单词添加到结果列表
            else:
                res.append(word)
        # 将单词添加到分词列表
        sep.append(word)
    # 返回分词列表、假名转换为片假名后的结果列表和重音信息
    return sep, [hira2kata(i) for i in res], get_accent(parsed)


# 获取重音信息
def get_accent(parsed):
    # 使用 pyopenjtalk 生成标签
    labels = pyopenjtalk.make_label(parsed)

    # 初始化音素列表和重音列表
    phonemes = []
    accents = []

    # 遍历标签
    for n, label in enumerate(labels):
        # 获取音素
        phoneme = re.search(r"\-([^\+]*)\+", label).group(1)
        # 如果音素不是 "sil" 或 "pau"，将音素添加到音素列表
        if phoneme not in ["sil", "pau"]:
            phonemes.append(phoneme.replace("cl", "q").lower())
        else:
            continue
        # 获取重音位置信息
        a1 = int(re.search(r"/A:(\-?[0-9]+)\+", label).group(1))
        a2 = int(re.search(r"\+(\d+)\+", label).group(1))
        # 获取下一个音素的重音位置信息
        if re.search(r"\-([^\+]*)\+", labels[n + 1]).group(1) in ["sil", "pau"]:
            a2_next = -1
        else:
            a2_next = int(re.search(r"\+(\d+)\+", labels[n + 1]).group(1)
        # 如果重音是下降的，将-1添加到重音列表
        if a1 == 0 and a2_next == a2 + 1:
            accents.append(-1)
        # 如果重音是上升的，将1添加到重音列表
        elif a2 == 1 and a2_next == 2:
            accents.append(1)
        # 否则将0添加到重音列表
        else:
            accents.append(0)
    # 返回音素和重音的元组列表
    return list(zip(phonemes, accents))
# 将特殊符号映射为对应的日语读音
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
    "c": "シー",
    ...
    # 其他字母和特殊符号的映射
}

# 匹配带有分隔符的数字的正则表达式
_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")
# 货币符号映射为对应的日语读音
_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
# 匹配货币的正则表达式
_CURRENCY_RX = re.compile(r"([$¥£€])([0-9.]*[0-9])")
# 匹配数字的正则表达式
_NUMBER_RX = re.compile(r"[0-9]+(\.[0-9]+)?")

# 将文本中的数字转换为对应的日语读音
def japanese_convert_numbers_to_words(text: str) -> str:
    res = _NUMBER_WITH_SEPARATOR_RX.sub(lambda m: m[0].replace(",", ""), text)
    res = _CURRENCY_RX.sub(lambda m: m[2] + _CURRENCY_MAP.get(m[1], m[1]), res)
    res = _NUMBER_RX.sub(lambda m: num2words(m[0], lang="ja"), res)
    return res

# 将文本中的字母和特殊符号转换为对应的日语读音
def japanese_convert_alpha_symbols_to_words(text: str) -> str:
    return "".join([_ALPHASYMBOL_YOMI.get(ch, ch) for ch in text.lower()])

# 将日语文本转换为音素
def japanese_text_to_phonemes(text: str) -> str:
    """Convert Japanese text to phonemes."""
    res = unicodedata.normalize("NFKC", text)
    res = japanese_convert_numbers_to_words(res)
    # res = japanese_convert_alpha_symbols_to_words(res)
    res = text2kata(res)
    res = kata2phoneme(res)
    return res
# 判断字符是否为日语字符
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


# 定义替换映射表
rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "…": "...",
    "···": "...",
    "・・・": "...",
    "·": ",",
    "・": ",",
    "、": ",",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
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
    "」": "'",
}


# 替换文本中的标点符号
def replace_punctuation(text):
    # 使用正则表达式构建替换模式
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    # 使用替换映射表替换文本中的标点符号
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    # 移除非日语字符和标点符号之外的所有字符
    replaced_text = re.sub(
        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
        + "".join(punctuation)
        + r"]+",
        "",
        replaced_text,
    )

    return replaced_text


# 文本规范化
def text_normalize(text):
    # 使用 NFKC 规范化 Unicode 文本
    res = unicodedata.normalize("NFKC", text)
    res = japanese_convert_numbers_to_words(res)  # 未提供此函数的实现，需要补充
    # res = "".join([i for i in res if is_japanese_character(i)])  # 注释掉的代码，需要确认是否需要保留
    res = replace_punctuation(res)  # 替换文本中的标点符号
    res = res.replace("゙", "")  # 移除特定字符
    return res


# 分配电话号码
def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word
# 处理长音符号，将其转换为对应的音节
def handle_long(sep_phonemes):
    # 遍历分离的音节列表
    for i in range(len(sep_phonemes)):
        # 如果当前音节的第一个字符是长音符号
        if sep_phonemes[i][0] == "ー":
            # 将当前音节的第一个字符替换为前一个音节的最后一个字符
            sep_phonemes[i][0] = sep_phonemes[i - 1][-1]
        # 如果当前音节包含长音符号
        if "ー" in sep_phonemes[i]:
            # 遍历当前音节的每个字符
            for j in range(len(sep_phonemes[i])):
                # 如果当前字符是长音符号
                if sep_phonemes[i][j] == "ー":
                    # 将当前字符替换为前一个字符的最后一个字符
                    sep_phonemes[i][j] = sep_phonemes[i][j - 1][-1]
    # 返回处理后的音节列表
    return sep_phonemes


# 创建一个来自预训练模型的分词器
tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese-char-wwm")


# 将音节与音调对齐
def align_tones(phones, tones):
    res = []
    # 遍历音节列表
    for pho in phones:
        temp = [0] * len(pho)
        # 遍历音节中的每个字符
        for idx, p in enumerate(pho):
            # 如果音调列表为空，则跳出循环
            if len(tones) == 0:
                break
            # 如果当前字符与音调列表的第一个元素相同
            if p == tones[0][0]:
                # 将临时列表中对应位置的值设为音调值
                temp[idx] = tones[0][1]
                # 如果当前位置大于0，则加上前一个位置的音调值
                if idx > 0:
                    temp[idx] += temp[idx - 1]
                # 移除已经匹配的音调
                tones.pop(0)
        # 在临时列表前后各添加一个0
        temp = [0] + temp
        temp = temp[:-1]
        # 如果临时列表中存在-1，则将其替换为递增的值
        if -1 in temp:
            temp = [i + 1 for i in temp]
        # 将临时列表添加到结果列表中
        res.append(temp)
    # 将结果列表扁平化
    res = [i for j in res for i in j]
    # 断言结果列表中不包含负数和大于1的数值
    assert not any([i < 0 for i in res]) and not any([i > 1 for i in res])
    # 返回结果列表
    return res


# 重新排列音调
def rearrange_tones(tones, phones):
    res = [0] * len(tones)
    prev = 0
    # 遍历音调列表
    for i in range(len(tones)):
        # 如果是第一个音调
        if i == 0:
            # 如果音调不在标点符号中，则设为1
            if tones[i] not in punctuation:
                res[i] = 1
        # 如果当前音调与前一个音调相同
        elif tones[i] == prev:
            # 如果当前音节在标点符号中，则设为0，否则设为1
            if phones[i] in punctuation:
                res[i] = 0
            else:
                res[i] = 1
        # 如果当前音调大于前一个音调
        elif tones[i] > prev:
            res[i] = 2
        # 如果当前音调小于前一个音调
        elif tones[i] < prev:
            res[i - 1] = 3
            res[i] = 1
        # 更新前一个音调
        prev = tones[i]
    # 返回重新排列后的音调列表
    return res


# 将文本转换为音素
def g2p(norm_text):
    # 将文本分离成音节、片假名和重音符号
    sep_text, sep_kata, acc = text2sep_kata(norm_text)
    sep_tokenized = []
    # 遍历分离的音节列表
    for i in sep_text:
        # 如果当前音节不在标点符号中，则进行分词处理
        if i not in punctuation:
            sep_tokenized.append(tokenizer.tokenize(i))
        else:
            sep_tokenized.append([i])

    # 处理长音符号，将其转换为对应的音节
    sep_phonemes = handle_long([kata2phoneme(i) for i in sep_kata])
    # 异常处理，MeCab不认识的词的话会一路传到这里来，然后炸掉。目前来看只有那些超级稀有的生僻词会出现这种情况
    # 检查分离的音素是否都在符号集合中，如果不在则抛出异常
    for i in sep_phonemes:
        for j in i:
            assert j in symbols, (sep_text, sep_kata, sep_phonemes)
    # 调用 align_tones 函数对音素进行音调对齐
    tones = align_tones(sep_phonemes, acc)

    # 初始化 word2ph 列表
    word2ph = []
    # 遍历分离的分词和音素
    for token, phoneme in zip(sep_tokenized, sep_phonemes):
        # 获取音素和分词的长度
        phone_len = len(phoneme)
        word_len = len(token)
        # 调用 distribute_phone 函数对音素进行分配
        aaa = distribute_phone(phone_len, word_len)
        # 将分配的音素添加到 word2ph 列表中
        word2ph += aaa
    # 初始化 phones 列表，包含下划线和所有音素
    phones = ["_"] + [j for i in sep_phonemes for j in i] + ["_"]
    # 初始化 tones 列表，包含起始和结束的音调
    tones = [0] + tones + [0]
    # 在 word2ph 列表的开头和结尾添加标记
    word2ph = [1] + word2ph + [1]
    # 检查 phones 和 tones 的长度是否相等
    assert len(phones) == len(tones)
    # 返回 phones, tones, word2ph
    return phones, tones, word2ph
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 从预训练模型中加载自动分词器
    tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese")
    # 定义文本内容
    text = "hello,こんにちは、世界ー！……"
    # 从文本.japanese_bert模块中导入get_bert_feature函数
    from text.japanese_bert import get_bert_feature

    # 对文本进行规范化处理
    text = text_normalize(text)
    # 打印处理后的文本
    print(text)

    # 使用g2p函数获取文本的音素、音调和单词到音素的映射关系
    phones, tones, word2ph = g2p(text)
    # 使用get_bert_feature函数获取文本的BERT特征
    bert = get_bert_feature(text, word2ph)

    # 打印音素、音调、单词到音素的映射关系和BERT特征的形状
    print(phones, tones, word2ph, bert.shape)
```