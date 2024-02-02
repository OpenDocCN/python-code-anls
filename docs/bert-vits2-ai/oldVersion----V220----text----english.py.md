# `Bert-VITS2\oldVersion\V220\text\english.py`

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
# 从当前目录下的 symbols 模块中导入所有内容
from . import symbols

# 获取当前文件的路径
current_file_path = os.path.dirname(__file__)
# 拼接路径，得到 CMU_DICT_PATH
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
# 拼接路径，得到 CACHE_PATH
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")
# 创建 G2p 对象
_g2p = G2p()
# 设置 LOCAL_PATH 为 "./bert/deberta-v3-large"
LOCAL_PATH = "./bert/deberta-v3-large"
# 从预训练模型 LOCAL_PATH 中加载 DebertaV2Tokenizer 对象
tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)

# 创建 arpa 集合，包含一系列字符串
arpa = {
    "AH0",
    "S",
    # ... 其他字符串
}

# 定义函数 post_replace_ph，用于替换音素
def post_replace_ph(ph):
    # 定义替换映射表
    rep_map = {
        "：": ",",
        "；": ",",
        # ... 其他映射
    }
    # 如果 ph 在替换映射表中
    if ph in rep_map.keys():
        # 用映射表中的值替换 ph
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

# 定义替换映射表
rep_map = {
    "：": ",",
    "；": ",",
    # ... 其他映射
}
    # 将中文标点符号替换为英文标点符号
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
# 替换文本中的标点符号
def replace_punctuation(text):
    # 根据替换映射表创建正则表达式模式
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    # 使用正则表达式模式替换文本中的标点符号
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    # 返回替换后的文本
    return replaced_text


# 读取字典文件内容并返回字典对象
def read_dict():
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0]
                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)
            line_index = line_index + 1
            line = f.readline()
    # 返回读取的字典对象
    return g2p_dict


# 将字典对象缓存到文件中
def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


# 获取字典对象，如果缓存文件存在则从缓存中读取，否则重新读取字典文件并缓存
def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict()
        cache_dict(g2p_dict, CACHE_PATH)
    # 返回获取到的字典对象
    return g2p_dict


# 获取英文单词到音素的字典对象
eng_dict = get_dict()


# 根据音素调整音节
def refine_ph(phn):
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    return phn.lower(), tone


# 根据音节列表调整音素和音调
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

# 创建inflect引擎对象
_inflect = inflect.engine()
# 匹配逗号分隔的数字
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
# 匹配小数
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
# 匹配英镑符号
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
# 匹配美元符号
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
# 匹配序数词
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
# 匹配数字
_number_re = re.compile(r"[0-9]+")

# 缩写的正则表达式替换对列表
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

# 懒惰的国际音标对列表
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

# 懒惰的国际音标对列表2
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

# 国际音标到懒惰国际音标2的对列表
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
        # 如果美元为1，则单位为"dollar"，否则为"dollars"
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        # 如果美分为1，则单位为"cent"，否则为"cents"
        cent_unit = "cent" if cents == 1 else "cents"
        # 返回美元和美分的字符串表示
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    # 如果只有美元
    elif dollars:
        # 如果美元为1，则单位为"dollar"，否则为"dollars"
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        # 返回美元的字符串表示
        return "%s %s" % (dollars, dollar_unit)
    # 如果只有美分
    elif cents:
        # 如果美分为1，则单位为"cent"，否则为"cents"
        cent_unit = "cent" if cents == 1 else "cents"
        # 返回美分的字符串表示
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


# 定义一个函数，用于对文本中的数字进行标准化处理
def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


# 定义一个函数，用于对文本进行标准化处理
def text_normalize(text):
    text = normalize_numbers(text)
    text = replace_punctuation(text)
    text = re.sub(r"([,;.\?\!])([\w])", r"\1 \2", text)
    return text


# 定义一个函数，用于将电话号码分配给单词
def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


# 定义一个函数，用于将文本分割成单词
def sep_text(text):
    words = re.split(r"([,;.\?\!\s+])", text)
    words = [word for word in words if word.strip() != ""]
    return words


# 定义一个函数，用于将文本转换为音素
def g2p(text):
    phones = []
    tones = []
    # word2ph = []
    words = sep_text(text)
    tokens = [tokenizer.tokenize(i) for i in words]
    # 遍历单词列表中的每个单词
    for word in words:
        # 如果单词的大写形式在英语词典中
        if word.upper() in eng_dict:
            # 从英语词典中获取音节数和音调信息
            phns, tns = refine_syllables(eng_dict[word.upper()])
            # 将音节数列表添加到phones列表中
            phones.append([post_replace_ph(i) for i in phns])
            # 将音调信息添加到tones列表中
            tones.append(tns)
            # 将单词的音节数添加到word2ph列表中
            # word2ph.append(len(phns))
        else:
            # 如果单词不在英语词典中
            # 将单词转换为音素列表
            phone_list = list(filter(lambda p: p != " ", _g2p(word)))
            phns = []
            tns = []
            # 遍历音素列表
            for ph in phone_list:
                # 如果音素在arpa中
                if ph in arpa:
                    # 从arpa中获取音素和音调信息
                    ph, tn = refine_ph(ph)
                    phns.append(ph)
                    tns.append(tn)
                else:
                    phns.append(ph)
                    tns.append(0)
            # 将音素列表添加到phones列表中
            phones.append([post_replace_ph(i) for i in phns])
            # 将音调信息添加到tones列表中
            tones.append(tns)
            # 将单词的音节数添加到word2ph列表中
            # word2ph.append(len(phns))
    # 计算每个单词的音节数
    word2ph = []
    for token, phoneme in zip(tokens, phones):
        phone_len = len(phoneme)
        word_len = len(token)
        # 将每个单词的音节数添加到word2ph列表中
        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa
    # 将所有音素列表合并成一个大的音素列表
    phones = ["_"] + [j for i in phones for j in i] + ["_"]
    # 将所有音调信息合并成一个大的音调信息列表
    tones = [0] + [j for i in tones for j in i] + [0]
    # 将所有单词的音节数合并成一个大的音节数列表
    word2ph = [1] + word2ph + [1]
    # 检查音素列表和音调信息列表的长度是否相等
    assert len(phones) == len(tones), text
    # 检查音素列表的长度是否等于所有单词的音节数之和
    assert len(phones) == sum(word2ph), text
    # 返回音素列表、音调信息列表和单词的音节数列表
    return phones, tones, word2ph
# 定义一个函数，用于获取文本的 BERT 特征，需要传入文本和单词到音素的映射表
def get_bert_feature(text, word2ph):
    # 从文本模块中导入英文 BERT 模拟器
    from text import english_bert_mock
    # 调用英文 BERT 模拟器的函数，获取文本的 BERT 特征
    return english_bert_mock.get_bert_feature(text, word2ph)

# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 打印获取字典的结果
    # print(get_dict())
    # 打印将英文单词转换为音素的结果
    # print(eng_word_to_phoneme("hello"))
    # 打印将文本转换为音素的结果
    print(g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."))
    # 创建一个空集合用于存储所有的音素
    # all_phones = set()
    # 遍历英文字典中的每个单词和音节
    # for k, syllables in eng_dict.items():
    #     # 遍历每个音节中的音素
    #     for group in syllables:
    #         for ph in group:
    #             # 将音素添加到集合中
    #             all_phones.add(ph)
    # 打印所有的音素
    # print(all_phones)
```