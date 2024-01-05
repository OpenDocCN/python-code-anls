# `d:/src/tocomm/Bert-VITS2\oldVersion\V220\text\english.py`

```
import pickle  # 导入 pickle 模块，用于序列化和反序列化 Python 对象
import os  # 导入 os 模块，用于与操作系统交互
import re  # 导入 re 模块，用于处理正则表达式
from g2p_en import G2p  # 从 g2p_en 模块中导入 G2p 类
from transformers import DebertaV2Tokenizer  # 从 transformers 模块中导入 DebertaV2Tokenizer 类

from . import symbols  # 从当前目录中导入 symbols 模块

current_file_path = os.path.dirname(__file__)  # 获取当前文件所在目录的路径
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")  # 拼接路径，指向 cmudict.rep 文件
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")  # 拼接路径，指向 cmudict_cache.pickle 文件
_g2p = G2p()  # 创建 G2p 对象
LOCAL_PATH = "./bert/deberta-v3-large"  # 设置本地路径
tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)  # 从本地路径加载 DebertaV2Tokenizer 对象

arpa = {  # 创建一个包含字符串的集合
    "AH0",
    "S",
    "AH1",
    "EY2",
# 创建一个包含字符串的列表
phonemes = [
    "AE2",
    "EH0",
    "OW2",
    "UH0",
    "NG",
    "B",
    "G",
    "AY0",
    "M",
    "AA0",
    "F",
    "AO0",
    "ER2",
    "UH1",
    "IY1",
    "AH2",
    "DH",
    "IY0",
    "EY1",
    "IH0",
]
    "K",  # 表示字母 K
    "N",  # 表示字母 N
    "W",  # 表示字母 W
    "IY2",  # 表示音标为 IY2 的音节
    "T",  # 表示字母 T
    "AA1",  # 表示音标为 AA1 的音节
    "ER1",  # 表示音标为 ER1 的音节
    "EH2",  # 表示音标为 EH2 的音节
    "OY0",  # 表示音标为 OY0 的音节
    "UH2",  # 表示音标为 UH2 的音节
    "UW1",  # 表示音标为 UW1 的音节
    "Z",  # 表示字母 Z
    "AW2",  # 表示音标为 AW2 的音节
    "AW1",  # 表示音标为 AW1 的音节
    "V",  # 表示字母 V
    "UW2",  # 表示音标为 UW2 的音节
    "AA2",  # 表示音标为 AA2 的音节
    "ER",  # 表示字母 ER
    "AW0",  # 表示音标为 AW0 的音节
    "UW0",  # 表示音标为 UW0 的音节
    "R",  # 字符串 "R"
    "OW1",  # 字符串 "OW1"
    "EH1",  # 字符串 "EH1"
    "ZH",  # 字符串 "ZH"
    "AE0",  # 字符串 "AE0"
    "IH2",  # 字符串 "IH2"
    "IH",  # 字符串 "IH"
    "Y",  # 字符串 "Y"
    "JH",  # 字符串 "JH"
    "P",  # 字符串 "P"
    "AY1",  # 字符串 "AY1"
    "EY0",  # 字符串 "EY0"
    "OY2",  # 字符串 "OY2"
    "TH",  # 字符串 "TH"
    "HH",  # 字符串 "HH"
    "D",  # 字符串 "D"
    "ER0",  # 字符串 "ER0"
    "CH",  # 字符串 "CH"
    "AO1",  # 字符串 "AO1"
    "AE1",  # 字符串 "AE1"
    "AO2",  # 代表音素 AO2
    "OY1",  # 代表音素 OY1
    "AY2",  # 代表音素 AY2
    "IH1",  # 代表音素 IH1
    "OW0",  # 代表音素 OW0
    "L",    # 代表音素 L
    "SH",   # 代表音素 SH
}


def post_replace_ph(ph):
    # 定义替换映射关系的字典
    rep_map = {
        "：": ",",   # 将中文冒号替换为英文逗号
        "；": ",",   # 将中文分号替换为英文逗号
        "，": ",",   # 将中文逗号替换为英文逗号
        "。": ".",   # 将中文句号替换为英文句号
        "！": "!",   # 将中文感叹号替换为英文感叹号
        "？": "?",   # 将中文问号替换为英文问号
        "\n": ".",   # 将换行符替换为英文句号
        "·": ",",    # 将中文间隔符替换为英文逗号
# 定义一个替换映射表，将一些特殊字符替换为标准字符
rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "、": ",",
    "…": "...",
    "···": "...",
    "・・・": "...",
    "v": "V",
}

# 如果输入的字符在替换映射表中，则将其替换为映射表中对应的值
if ph in rep_map.keys():
    ph = rep_map[ph]

# 如果输入的字符是标点符号，则直接返回该字符
if ph in symbols:
    return ph

# 如果输入的字符不是标点符号，则将其替换为"UNK"表示未知字符
if ph not in symbols:
    ph = "UNK"

# 返回处理后的字符
return ph
    "！": "!",  # 将中文感叹号替换为英文感叹号
    "？": "?",  # 将中文问号替换为英文问号
    "\n": ".",  # 将换行符替换为句号
    "．": ".",  # 将中文句号替换为英文句号
    "…": "...",  # 将中文省略号替换为英文省略号
    "···": "...",  # 将中文省略号替换为英文省略号
    "・・・": "...",  # 将中文省略号替换为英文省略号
    "·": ",",  # 将中文顿号替换为英文逗号
    "・": ",",  # 将中文顿号替换为英文逗号
    "、": ",",  # 将中文顿号替换为英文逗号
    "$": ".",  # 将美元符号替换为句号
    "“": "'",  # 将中文双引号替换为英文单引号
    "”": "'",  # 将中文双引号替换为英文单引号
    '"': "'",  # 将双引号替换为单引号
    "‘": "'",  # 将中文单引号替换为英文单引号
    "’": "'",  # 将中文单引号替换为英文单引号
    "（": "'",  # 将中文左括号替换为英文单引号
    "）": "'",  # 将中文右括号替换为英文单引号
    "(": "'",  # 将左括号替换为单引号
    ")": "'",  # 将右括号替换为单引号
    "《": "'",  # 将中文书名号《》替换为英文单引号'
    "》": "'",  # 将中文书名号《》替换为英文单引号'
    "【": "'",  # 将中文方括号【】替换为英文单引号'
    "】": "'",  # 将中文方括号【】替换为英文单引号'
    "[": "'",   # 将英文方括号[]替换为英文单引号'
    "]": "'",   # 将英文方括号[]替换为英文单引号'
    "—": "-",   # 将中文破折号—替换为英文减号-
    "−": "-",   # 将数学符号−替换为英文减号-
    "～": "-",  # 将中文波浪号～替换为英文减号-
    "~": "-",   # 将英文波浪号~替换为英文减号-
    "「": "'",  # 将中文书名号「」替换为英文单引号'
    "」": "'",  # 将中文书名号「」替换为英文单引号'
}

def replace_punctuation(text):
    # 创建正则表达式模式，用于匹配需要替换的标点符号
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    # 使用正则表达式模式替换文本中的标点符号
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    # replaced_text = re.sub(
    #     r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    #     + "".join(punctuation)
    #     + r"]+",
    #     "",
    #     replaced_text,
    # )
    # 使用正则表达式替换文本中的非指定字符为""
    return replaced_text


def read_dict():
    # 创建空字典
    g2p_dict = {}
    # 设置起始行号
    start_line = 49
    # 打开 CMU 字典文件
    with open(CMU_DICT_PATH) as f:
        # 读取文件的一行
        line = f.readline()
        # 初始化行号
        line_index = 1
        # 循环读取文件的每一行
        while line:
            # 如果行号大于等于起始行号
            if line_index >= start_line:
                # 去除行首尾的空白字符
                line = line.strip()
# 将每行文本按照空格分割，获取单词
word_split = line.split("  ")
word = word_split[0]

# 将单词的音节部分按照" - "分割，获取音节列表
syllable_split = word_split[1].split(" - ")

# 初始化单词到音节列表的字典
g2p_dict[word] = []

# 遍历音节列表，将每个音节按空格分割，获取音素列表，然后将音素列表添加到单词对应的音节列表中
for syllable in syllable_split:
    phone_split = syllable.split(" ")
    g2p_dict[word].append(phone_split)

# 递增行索引
line_index = line_index + 1

# 读取下一行文本
line = f.readline()

# 返回单词到音素列表的字典
return g2p_dict

# 将单词到音素列表的字典缓存到文件中
def cache_dict(g2p_dict, file_path):
    # 以二进制写模式打开文件
    with open(file_path, "wb") as pickle_file:
        # 使用 pickle 序列化字典，并将序列化后的数据写入文件
        pickle.dump(g2p_dict, pickle_file)
# 定义一个函数用于获取字典数据
def get_dict():
    # 如果缓存路径下的文件存在
    if os.path.exists(CACHE_PATH):
        # 以二进制形式打开缓存文件
        with open(CACHE_PATH, "rb") as pickle_file:
            # 从缓存文件中加载数据到 g2p_dict
            g2p_dict = pickle.load(pickle_file)
    else:
        # 如果缓存文件不存在，则调用 read_dict() 函数获取数据
        g2p_dict = read_dict()
        # 将获取的数据缓存到 CACHE_PATH
        cache_dict(g2p_dict, CACHE_PATH)

    # 返回获取的字典数据
    return g2p_dict

# 调用 get_dict() 函数获取英文字典数据
eng_dict = get_dict()

# 定义一个函数用于处理音素
def refine_ph(phn):
    tone = 0
    # 如果音素以数字结尾
    if re.search(r"\d$", phn):
        # 获取音素的音调
        tone = int(phn[-1]) + 1
        # 去除音素的音调部分
        phn = phn[:-1]
    # 将音素转换为小写，并返回音素和音调
    return phn.lower(), tone
# 定义一个函数，用于处理音节列表
def refine_syllables(syllables):
    # 创建空列表用于存储音调
    tones = []
    # 创建空列表用于存储音素
    phonemes = []
    # 遍历音节列表
    for phn_list in syllables:
        # 遍历每个音节
        for i in range(len(phn_list)):
            # 获取当前音节
            phn = phn_list[i]
            # 调用 refine_ph 函数处理音节，获取处理后的音节和音调
            phn, tone = refine_ph(phn)
            # 将处理后的音节添加到音素列表
            phonemes.append(phn)
            # 将音调添加到音调列表
            tones.append(tone)
    # 返回处理后的音素列表和音调列表
    return phonemes, tones

# 导入 re 模块，用于处理正则表达式
import re
# 导入 inflect 模块，用于处理英文数字转换
import inflect

# 创建 inflect 引擎对象
_inflect = inflect.engine()
# 编译正则表达式，用于匹配带逗号的数字
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
# 编译正则表达式，用于匹配小数
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")  # 匹配以英镑符号开头的金额
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")  # 匹配以美元符号开头的金额
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")  # 匹配序数词
_number_re = re.compile(r"[0-9]+")  # 匹配数字

# 缩写的正则表达式替换对列表
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])  # 匹配缩写词并替换为全称
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
# List of abbreviation and their full forms
_abbreviations = [
    ("hon", "honorable"),
    ("sgt", "sergeant"),
    ("capt", "captain"),
    ("esq", "esquire"),
    ("ltd", "limited"),
    ("col", "colonel"),
    ("ft", "fort"),
]

# List of (ipa, lazy ipa) pairs for phonetic transcription
_lazy_ipa = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        ("r", "ɹ"),
        ("æ", "e"),
        ("ɑ", "a"),
        ("ɔ", "o"),
        ("ð", "z"),
```

这段代码定义了两个列表，一个是缩写词和它们的全称，另一个是国际音标和它们的懒惰国际音标对。
# List of (ipa, ipa2) pairs:
# 创建一个包含 IPA 和对应的转换后的 IPA2 的元组列表

_ipa2 = [
    # 元组中的第一个元素是原始的 IPA，第二个元素是转换后的 IPA2
    ("r", "ɹ"),
    ("ð", "z"),
    ("θ", "s"),
    ("ɛ", "e"),
    ("ɪ", "i"),
    ("ʊ", "u"),
    ("ʒ", "ʥ"),
    ("ʤ", "ʥ"),
    ("ˈ", "↓"),
]

# List of (ipa, lazy ipa2) pairs:
# 创建一个包含 IPA 和对应的懒惰转换后的 IPA2 的元组列表

_lazy_ipa2 = [
    # 使用正则表达式将原始的 IPA 转换为懒惰的 IPA2
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
    ]
]
# 上面是一个未闭合的列表，需要找到对应的开括号来闭合它

# List of (ipa, ipa2) pairs
_ipa_to_ipa2 = [
    (re.compile("%s" % x[0]), x[1]) for x in [("r", "ɹ"), ("ʤ", "dʒ"), ("ʧ", "tʃ")]
]
# 创建一个包含正则表达式和对应替换字符串的元组列表，用于将特定的音标替换为另一种音标

def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
```
这段代码定义了一个函数，用于将匹配到的字符串中的数字转换为美元的格式。
    elif dollars:  # 如果有美元
        dollar_unit = "dollar" if dollars == 1 else "dollars"  # 如果美元数量为1，则单位为"dollar"，否则为"dollars"
        return "%s %s" % (dollars, dollar_unit)  # 返回美元数量和单位的字符串
    elif cents:  # 如果有美分
        cent_unit = "cent" if cents == 1 else "cents"  # 如果美分数量为1，则单位为"cent"，否则为"cents"
        return "%s %s" % (cents, cent_unit)  # 返回美分数量和单位的字符串
    else:  # 如果既没有美元也没有美分
        return "zero dollars"  # 返回"zero dollars"表示零美元


def _remove_commas(m):  # 定义一个函数，用于去除匹配到的逗号
    return m.group(1).replace(",", "")  # 返回去除逗号后的字符串


def _expand_ordinal(m):  # 定义一个函数，用于将匹配到的序数词扩展为完整的英文单词
    return _inflect.number_to_words(m.group(0))  # 使用_inflect库将匹配到的序数词转换为完整的英文单词


def _expand_number(m):  # 定义一个函数，用于将匹配到的数字转换为整数
    num = int(m.group(0))  # 将匹配到的数字转换为整数
    if num > 1000 and num < 3000:  # 如果数字大于1000且小于3000
        if num == 2000:  # 如果数字等于2000
            return "two thousand"  # 返回字符串 "two thousand"
        elif num > 2000 and num < 2010:  # 如果数字大于2000且小于2010
            return "two thousand " + _inflect.number_to_words(num % 100)  # 返回 "two thousand " 加上数字的英文表示
        elif num % 100 == 0:  # 如果数字能被100整除
            return _inflect.number_to_words(num // 100) + " hundred"  # 返回数字除以100的英文表示再加上 " hundred"
        else:  # 其他情况
            return _inflect.number_to_words(
                num, andword="", zero="oh", group=2
            ).replace(", ", " ")  # 返回数字的英文表示，替换逗号和空格
    else:  # 如果数字不在1000到3000之间
        return _inflect.number_to_words(num, andword="")  # 返回数字的英文表示，不带 "and" 连接词


def _expand_decimal_point(m):  # 定义一个函数，用于替换小数点
    return m.group(1).replace(".", " point ")  # 将小数点替换为 " point "


def normalize_numbers(text):  # 定义一个函数，用于规范化数字
    text = re.sub(_comma_number_re, _remove_commas, text)  # 用空格替换文本中的逗号
    text = re.sub(_pounds_re, r"\1 pounds", text)  # 将文本中的英镑符号替换为单词 "pounds"
    text = re.sub(_dollars_re, _expand_dollars, text)  # 将文本中的美元符号替换为相应的数字
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)  # 将文本中的小数点替换为相应的数字
    text = re.sub(_ordinal_re, _expand_ordinal, text)  # 将文本中的序数词替换为相应的数字
    text = re.sub(_number_re, _expand_number, text)  # 将文本中的数字替换为相应的数字
    return text  # 返回处理后的文本


def text_normalize(text):
    text = normalize_numbers(text)  # 调用 normalize_numbers 函数对文本中的数字进行规范化处理
    text = replace_punctuation(text)  # 调用 replace_punctuation 函数对文本中的标点进行处理
    text = re.sub(r"([,;.\?\!])([\w])", r"\1 \2", text)  # 在标点符号和单词之间添加空格
    return text  # 返回处理后的文本


def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word  # 创建一个长度为 n_word 的列表，每个元素初始化为 0
    for task in range(n_phone):  # 遍历 n_phone 次
        min_tasks = min(phones_per_word)  # 找到 phones_per_word 列表中的最小值
        min_index = phones_per_word.index(min_tasks)  # 找到列表 phones_per_word 中最小值的索引
        phones_per_word[min_index] += 1  # 将最小值对应的位置的值加1
    return phones_per_word  # 返回更新后的列表 phones_per_word


def sep_text(text):
    words = re.split(r"([,;.\?\!\s+])", text)  # 使用正则表达式将文本 text 按照标点符号和空格分割成单词列表
    words = [word for word in words if word.strip() != ""]  # 去除单词列表中的空字符串
    return words  # 返回处理后的单词列表


def g2p(text):
    phones = []  # 初始化音素列表
    tones = []  # 初始化音调列表
    # word2ph = []  # 初始化单词到音素的映射列表
    words = sep_text(text)  # 使用 sep_text 函数将文本 text 分割成单词列表
    tokens = [tokenizer.tokenize(i) for i in words]  # 使用 tokenizer 对单词列表中的每个单词进行分词
    for word in words:  # 遍历单词列表
        if word.upper() in eng_dict:  # 如果单词的大写形式在 eng_dict 中
            phns, tns = refine_syllables(eng_dict[word.upper()])  # 调用 refine_syllables 函数获取单词的音素和音调
            phones.append([post_replace_ph(i) for i in phns])  # 将经过 post_replace_ph 函数处理后的 phns 列表添加到 phones 列表中
            tones.append(tns)  # 将 tns 列表添加到 tones 列表中
            # word2ph.append(len(phns))  # 将 phns 列表的长度添加到 word2ph 列表中
        else:
            phone_list = list(filter(lambda p: p != " ", _g2p(word)))  # 使用 _g2p 函数将 word 转换为 phone_list 列表，并过滤掉空格
            phns = []  # 初始化 phns 列表
            tns = []  # 初始化 tns 列表
            for ph in phone_list:  # 遍历 phone_list 列表
                if ph in arpa:  # 如果 ph 在 arpa 中
                    ph, tn = refine_ph(ph)  # 使用 refine_ph 函数处理 ph，并将结果添加到 phns 和 tns 列表中
                    phns.append(ph)
                    tns.append(tn)
                else:  # 否则
                    phns.append(ph)  # 将 ph 添加到 phns 列表中
                    tns.append(0)  # 将 0 添加到 tns 列表中
            phones.append([post_replace_ph(i) for i in phns])  # 将经过 post_replace_ph 函数处理后的 phns 列表添加到 phones 列表中
            tones.append(tns)  # 将 tns 列表添加到 tones 列表中
            # word2ph.append(len(phns))  # 将 phns 列表的长度添加到 word2ph 列表中
    # phones = [post_replace_ph(i) for i in phones]  # 对 phones 列表中的每个元素经过 post_replace_ph 函数处理后重新赋值给 phones 列表
    word2ph = []  # 初始化一个空列表，用于存储单词到音素的映射关系
    for token, phoneme in zip(tokens, phones):  # 遍历单词和对应的音素
        phone_len = len(phoneme)  # 获取音素的长度
        word_len = len(token)  # 获取单词的长度

        aaa = distribute_phone(phone_len, word_len)  # 调用 distribute_phone 函数，计算单词和音素的映射关系
        word2ph += aaa  # 将计算得到的映射关系添加到 word2ph 列表中

    phones = ["_"] + [j for i in phones for j in i] + ["_"]  # 对 phones 列表进行处理，添加特殊符号
    tones = [0] + [j for i in tones for j in i] + [0]  # 对 tones 列表进行处理，添加特殊符号
    word2ph = [1] + word2ph + [1]  # 对 word2ph 列表进行处理，添加特殊符号
    assert len(phones) == len(tones), text  # 检查 phones 和 tones 的长度是否相等
    assert len(phones) == sum(word2ph), text  # 检查 phones 的长度是否等于 word2ph 中元素的总和

    return phones, tones, word2ph  # 返回处理后的 phones、tones 和 word2ph 列表


def get_bert_feature(text, word2ph):  # 定义函数 get_bert_feature，接受文本和单词到音素的映射关系作为参数
    from text import english_bert_mock  # 从 text 模块中导入 english_bert_mock 函数
    return english_bert_mock.get_bert_feature(text, word2ph)
```
这行代码调用了english_bert_mock模块中的get_bert_feature函数，传入了text和word2ph作为参数，并返回函数的结果。

```python
if __name__ == "__main__":
```
这行代码检查当前模块是否作为主程序运行。

```python
    # print(get_dict())
    # print(eng_word_to_phoneme("hello"))
    print(g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."))
```
这行代码调用了g2p函数，传入了"In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."作为参数，并打印函数的结果。

```python
    # all_phones = set()
    # for k, syllables in eng_dict.items():
    #     for group in syllables:
    #         for ph in group:
    #             all_phones.add(ph)
    # print(all_phones)
```
这段代码被注释掉了，原本是用来遍历eng_dict中的音节，并将其添加到all_phones集合中，然后打印出集合的内容。
```