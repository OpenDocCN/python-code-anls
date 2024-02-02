# `Bert-VITS2\text\english.py`

```py
# 导入 pickle 模块
import pickle
# 导入 os 模块
import os
# 导入 re 模块
import re
# 从 g2p_en 模块中导入 G2p 类
from g2p_en import G2p
# 从 transformers 模块中导入 DebertaV2Tokenizer 类
from transformers import DebertaV2Tokenizer
# 从 text 模块中导入 symbols 和 punctuation
from text import symbols
from text.symbols import punctuation

# 获取当前文件的路径
current_file_path = os.path.dirname(__file__)
# 拼接路径，得到 CMU_DICT_PATH
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
# 拼接路径，得到 CACHE_PATH
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")
# 创建 G2p 对象
_g2p = G2p()
# 设置 LOCAL_PATH
LOCAL_PATH = "./bert/deberta-v3-large"
# 从 LOCAL_PATH 加载 DebertaV2Tokenizer 对象
tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)

# 定义 arpa 集合
arpa = {
    "AH0",
    "S",
    # ... 其他元素
    "(": "'",
}

# 定义函数 post_replace_ph
def post_replace_ph(ph):
    # 定义替换映射表
    rep_map = {
        "：": ",",
        "；": ",",
        # ... 其他映射
        "(": "'",
    }
    # 如果 ph 在 rep_map 的键中
    if ph in rep_map.keys():
        # 将 ph 替换为 rep_map 中对应的值
        ph = rep_map[ph]
    # 如果 ph 在 symbols 中
    if ph in symbols:
        # 返回 ph
        return ph
    # 如果 ph 不在 symbols 中
    if ph not in symbols:
        # 将 ph 替换为 "UNK"
        ph = "UNK"
    # 返回 ph
    return ph

# 定义替换映射表 rep_map
rep_map = {
    "：": ",",
    "；": ",",
    # ... 其他映射
    "(": "'",
}
    ")": "'",  # 将 ")" 替换为 "'"
    "《": "'",  # 将 "《" 替换为 "'"
    "》": "'",  # 将 "》" 替换为 "'"
    "【": "'",  # 将 "【" 替换为 "'"
    "】": "'",  # 将 "】" 替换为 "'"
    "[": "'",   # 将 "[" 替换为 "'"
    "]": "'",   # 将 "]" 替换为 "'"
    "—": "-",   # 将 "—" 替换为 "-"
    "−": "-",   # 将 "−" 替换为 "-"
    "～": "-",   # 将 "～" 替换为 "-"
    "~": "-",    # 将 "~" 替换为 "-"
    "「": "'",   # 将 "「" 替换为 "'"
    "」": "'",   # 将 "」" 替换为 "'"
# 替换文本中的标点符号
def replace_punctuation(text):
    # 创建正则表达式模式，用于匹配需要替换的标点符号
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    
    # 使用正则表达式模式替换文本中的标点符号
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    # 返回替换后的文本
    return replaced_text


# 读取字典文件
def read_dict():
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                # 去除行首尾的空白字符
                line = line.strip()
                # 按空格分割单词和音节
                word_split = line.split("  ")
                word = word_split[0]

                # 按照特定格式分割音节
                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    # 返回读取的字典
    return g2p_dict


# 缓存字典到文件
def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


# 获取字典数据
def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict()
        cache_dict(g2p_dict, CACHE_PATH)

    return g2p_dict


# 获取英文字典数据
eng_dict = get_dict()


# 优化音节
def refine_ph(phn):
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    else:
        tone = 3
    return phn.lower(), tone


# 优化音节列表
def refine_syllables(syllables):
    tones = []
    phonemes = []
    for phn_list in syllables:
        for i in range(len(phn_list)):
            phn = phn_list[i]
            phn, tone = refine_ph(phn)
            phonemes.append(phn)
            tones.append(tone)
    return phonemes, tones


# 导入正则表达式模块和inflect模块
import re
import inflect
# 创建一个 inflect 引擎对象，用于处理英文单词的复数、序数等变换
_inflect = inflect.engine()
# 编译正则表达式，用于匹配包含逗号的数字
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
# 编译正则表达式，用于匹配小数
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
# 编译正则表达式，用于匹配包含英镑符号的数字
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
# 编译正则表达式，用于匹配包含美元符号的数字
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
# 编译正则表达式，用于匹配序数
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
# 编译正则表达式，用于匹配数字
_number_re = re.compile(r"[0-9]+")

# 缩写的正则表达式替换列表
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]

# 懒惰的国际音标替换列表
_lazy_ipa = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        ("r", "ɹ"),
        ("æ", "e"),
        ("ɑ", "a"),
        ("ɔ", "o"),
        ("ð", "z"),
        ("θ", "s"),
        ("ɛ", "e"),
        ("ɪ", "i"),
        ("ʊ", "u"),
        ("ʒ", "ʥ"),
        ("ʤ", "ʥ"),
        ("ˈ", "↓"),
    ]
]

# 懒惰的国际音标2替换列表
_lazy_ipa2 = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        ("r", "ɹ"),
        ("ð", "z"),
        ("θ", "s"),
        ("ʒ", "ʑ"),
        ("ʤ", "dʑ"),
        ("ˈ", "↓"),
    ]
]

# 国际音标到懒惰国际音标2的替换列表
_ipa_to_ipa2 = [
    (re.compile("%s" % x[0]), x[1]) for x in [("r", "ɹ"), ("ʤ", "dʒ"), ("ʧ", "tʃ")]
]

# 扩展美元符号的函数
def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # 意外的格式
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    # 如果有美元和美分
    if dollars and cents:
        # 根据美元数量确定美元的单位
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        # 根据美分数量确定美分的单位
        cent_unit = "cent" if cents == 1 else "cents"
        # 返回美元和美分的数量和单位
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    # 如果只有美元
    elif dollars:
        # 根据美元数量确定美元的单位
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        # 返回美元的数量和单位
        return "%s %s" % (dollars, dollar_unit)
    # 如果只有美分
    elif cents:
        # 根据美分数量确定美分的单位
        cent_unit = "cent" if cents == 1 else "cents"
        # 返回美分的数量和单位
        return "%s %s" % (cents, cent_unit)
    # 如果既没有美元也没有美分
    else:
        # 返回零美元
        return "zero dollars"
# 定义一个函数，用于去除文本中的逗号
def _remove_commas(m):
    return m.group(1).replace(",", "")

# 定义一个函数，用于将序数词扩展为完整的英文单词
def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))

# 定义一个函数，用于将数字扩展为完整的英文单词
def _expand_number(m):
    num = int(m.group(0))
    # 判断数字范围并进行相应的扩展
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(
                num, andword="", zero="oh", group=2
            ).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")

# 定义一个函数，用于将小数点扩展为完整的英文单词
def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")

# 对文本中的数字进行标准化处理
def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text

# 对文本进行标准化处理
def text_normalize(text):
    text = normalize_numbers(text)
    text = replace_punctuation(text)
    text = re.sub(r"([,;.\?\!])([\w])", r"\1 \2", text)
    return text

# 根据电话数量和单词数量分配电话
def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word

# 将文本分割成单词
def sep_text(text):
    words = re.split(r"([,;.\?\!\s+])", text)
    words = [word for word in words if word.strip() != ""]
    return words

# 将文本转换为单词
def text_to_words(text):
    tokens = tokenizer.tokenize(text)
    words = []
    # 遍历 tokens 列表，同时获取索引和元素
    for idx, t in enumerate(tokens):
        # 如果元素以 "▁" 开头
        if t.startswith("▁"):
            # 将元素去掉 "▁" 后添加到 words 列表中作为新的单词
            words.append([t[1:]])
        else:
            # 如果元素不是以 "▁" 开头
            if t in punctuation:
                # 如果当前元素是标点符号并且是最后一个元素
                if idx == len(tokens) - 1:
                    # 将当前元素作为一个单独的单词添加到 words 列表中
                    words.append([f"{t}"])
                else:
                    # 如果当前元素是标点符号并且后面还有元素
                    if (
                        not tokens[idx + 1].startswith("▁")
                        and tokens[idx + 1] not in punctuation
                    ):
                        # 如果下一个元素不是以 "▁" 开头并且不是标点符号
                        if idx == 0:
                            # 如果当前元素是第一个元素，则添加一个空列表到 words 列表中
                            words.append([])
                        # 将当前元素添加到 words 列表中最后一个列表中
                        words[-1].append(f"{t}")
                    else:
                        # 如果下一个元素是以 "▁" 开头或者是标点符号，则将当前元素作为一个单独的单词添加到 words 列表中
                        words.append([f"{t}"])
            else:
                # 如果当前元素不是以 "▁" 开头并且不是标点符号
                if idx == 0:
                    # 如果当前元素是第一个元素，则添加一个空列表到 words 列表中
                    words.append([])
                # 将当前元素添加到 words 列表中最后一个列表中
                words[-1].append(f"{t}")
    # 返回处理后的 words 列表
    return words
def g2p(text):
    phones = []  # 存储音素的列表
    tones = []  # 存储音调的列表
    phone_len = []  # 存储每个单词的音素长度

    # 将文本转换为单词列表
    words = text_to_words(text)

    # 遍历每个单词
    for word in words:
        temp_phones, temp_tones = [], []

        # 如果单词长度大于1且包含撇号，则将其合并为一个单词
        if len(word) > 1:
            if "'" in word:
                word = ["".join(word)]

        # 遍历单词中的每个字母
        for w in word:
            # 如果是标点符号，则直接添加到音素列表中，音调为0
            if w in punctuation:
                temp_phones.append(w)
                temp_tones.append(0)
                continue
            # 如果是大写字母，则从英文词典中获取对应的音素和音调
            if w.upper() in eng_dict:
                phns, tns = refine_syllables(eng_dict[w.upper()])
                temp_phones += [post_replace_ph(i) for i in phns]
                temp_tones += tns
            # 否则，从音素转换函数中获取音素列表和音调列表
            else:
                phone_list = list(filter(lambda p: p != " ", _g2p(w)))
                phns = []
                tns = []
                for ph in phone_list:
                    if ph in arpa:
                        ph, tn = refine_ph(ph)
                        phns.append(ph)
                        tns.append(tn)
                    else:
                        phns.append(ph)
                        tns.append(0)
                temp_phones += [post_replace_ph(i) for i in phns]
                temp_tones += tns

        # 将临时音素和音调列表添加到总的音素和音调列表中
        phones += temp_phones
        tones += temp_tones
        phone_len.append(len(temp_phones))

    # 根据音素长度和单词长度分配音素
    word2ph = []
    for token, pl in zip(words, phone_len):
        word_len = len(token)
        aaa = distribute_phone(pl, word_len)
        word2ph += aaa

    # 在音素和音调列表的开头和结尾添加占位符
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]

    # 断言确保音素和音调列表的长度与单词到音素的映射列表的长度一致
    assert len(phones) == len(tones), text
    assert len(phones) == sum(word2ph), text

    return phones, tones, word2ph


def get_bert_feature(text, word2ph):
    from text import english_bert_mock

    # 调用英文 BERT 特征提取函数
    return english_bert_mock.get_bert_feature(text, word2ph)


if __name__ == "__main__":
    # 打印调用 get_dict() 函数的结果
    # 打印调用 eng_word_to_phoneme("hello") 函数的结果
    # 打印调用 g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder.") 函数的结果
    # 创建一个空集合 all_phones
    # 遍历 eng_dict 字典的键值对
    # 遍历每个键值对中的值（音节列表）
    # 遍历每个音节列表中的音素
    # 将每个音素添加到 all_phones 集合中
    # 打印 all_phones 集合的内容
```