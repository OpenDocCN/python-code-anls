# `d:/src/tocomm/Bert-VITS2\text\english.py`

```
import pickle  # 导入 pickle 模块，用于序列化和反序列化对象
import os  # 导入 os 模块，用于与操作系统交互
import re  # 导入 re 模块，用于处理正则表达式
from g2p_en import G2p  # 从 g2p_en 模块中导入 G2p 类
from transformers import DebertaV2Tokenizer  # 从 transformers 模块中导入 DebertaV2Tokenizer 类

from text import symbols  # 从 text 模块中导入 symbols
from text.symbols import punctuation  # 从 text.symbols 模块中导入 punctuation

current_file_path = os.path.dirname(__file__)  # 获取当前文件所在目录的路径
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")  # 拼接路径，得到 CMU_DICT_PATH
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")  # 拼接路径，得到 CACHE_PATH
_g2p = G2p()  # 创建 G2p 对象
LOCAL_PATH = "./bert/deberta-v3-large"  # 设置 LOCAL_PATH 变量为指定路径
tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)  # 使用指定路径初始化 tokenizer

arpa = {  # 创建 arpa 集合
    "AH0",  # 添加字符串 "AH0" 到集合中
    "S",  # 添加字符串 "S" 到集合中
    "AH1",  # 添加字符串 "AH1" 到集合中
    "EY2",  # 表示音素 EY2
    "AE2",  # 表示音素 AE2
    "EH0",  # 表示音素 EH0
    "OW2",  # 表示音素 OW2
    "UH0",  # 表示音素 UH0
    "NG",   # 表示音素 NG
    "B",    # 表示音素 B
    "G",    # 表示音素 G
    "AY0",  # 表示音素 AY0
    "M",    # 表示音素 M
    "AA0",  # 表示音素 AA0
    "F",    # 表示音素 F
    "AO0",  # 表示音素 AO0
    "ER2",  # 表示音素 ER2
    "UH1",  # 表示音素 UH1
    "IY1",  # 表示音素 IY1
    "AH2",  # 表示音素 AH2
    "DH",   # 表示音素 DH
    "IY0",  # 表示音素 IY0
    "EY1",  # 表示音素 EY1
    "IH0",  # 代表元音音素 /ɪ/，无声
    "K",    # 代表辅音音素 /k/
    "N",    # 代表辅音音素 /n/
    "W",    # 代表辅音音素 /w/
    "IY2",  # 代表元音音素 /i/，重读
    "T",    # 代表辅音音素 /t/
    "AA1",  # 代表元音音素 /ɑ/，重读
    "ER1",  # 代表元音音素 /ɜr/，重读
    "EH2",  # 代表元音音素 /ɛ/，重读
    "OY0",  # 代表双元音音素 /ɔɪ/，无重音
    "UH2",  # 代表元音音素 /ʊ/，重读
    "UW1",  # 代表元音音素 /u/，重读
    "Z",    # 代表辅音音素 /z/
    "AW2",  # 代表双元音音素 /aʊ/，重读
    "AW1",  # 代表双元音音素 /aʊ/，重读
    "V",    # 代表辅音音素 /v/
    "UW2",  # 代表元音音素 /u/，重读
    "AA2",  # 代表元音音素 /ɑ/，重读
    "ER",   # 代表元音音素 /ɜr/
    "AW0",  # 代表双元音音素 /aʊ/，无重音
    "UW0",  # 代表元音音素 /u/，发音为 /uː/
    "R",    # 代表辅音音素 /r/，发音为 /r/
    "OW1",  # 代表元音音素 /oʊ/，发音为 /əʊ/
    "EH1",  # 代表元音音素 /e/，发音为 /eɪ/
    "ZH",   # 代表浊辅音音素 /ʒ/，发音为 /ʒ/
    "AE0",  # 代表元音音素 /æ/，发音为 /æ/
    "IH2",  # 代表元音音素 /ɪ/，发音为 /ɪ/
    "IH",   # 代表元音音素 /ɪ/，发音为 /ɪ/
    "Y",    # 代表半元音音素 /j/，发音为 /j/
    "JH",   # 代表浊辅音音素 /dʒ/，发音为 /dʒ/
    "P",    # 代表清辅音音素 /p/，发音为 /p/
    "AY1",  # 代表元音音素 /aɪ/，发音为 /aɪ/
    "EY0",  # 代表元音音素 /eɪ/，发音为 /eɪ/
    "OY2",  # 代表元音音素 /ɔɪ/，发音为 /ɔɪ/
    "TH",   # 代表清辅音音素 /θ/，发音为 /θ/
    "HH",   # 代表清辅音音素 /h/，发音为 /h/
    "D",    # 代表清辅音音素 /d/，发音为 /d/
    "ER0",  # 代表元音音素 /ɜː/，发音为 /ɜː/
    "CH",   # 代表清辅音音素 /tʃ/，发音为 /tʃ/
    "AO1",  # 代表元音音素 /ɔ/，发音为 /ɔː/
    "AE1",  # 表示元音音素 AE1
    "AO2",  # 表示元音音素 AO2
    "OY1",  # 表示元音音素 OY1
    "AY2",  # 表示元音音素 AY2
    "IH1",  # 表示元音音素 IH1
    "OW0",  # 表示元音音素 OW0
    "L",    # 表示辅音音素 L
    "SH",   # 表示辅音音素 SH
}


def post_replace_ph(ph):
    # 定义替换映射表
    rep_map = {
        "：": ",",  # 将中文冒号替换为英文逗号
        "；": ",",  # 将中文分号替换为英文逗号
        "，": ",",  # 将中文逗号替换为英文逗号
        "。": ".",  # 将中文句号替换为英文句号
        "！": "!",  # 将中文感叹号替换为英文感叹号
        "？": "?",  # 将中文问号替换为英文问号
        "\n": ".",  # 将换行符替换为句号
        "·": ",",  # 将中文中的句号替换为英文逗号
        "、": ",",  # 将中文中的顿号替换为英文逗号
        "…": "...",  # 将中文中的省略号替换为英文省略号
        "···": "...",  # 将中文中的多个句号替换为英文省略号
        "・・・": "...",  # 将中文中的多个句号替换为英文省略号
        "v": "V",  # 将小写的 v 替换为大写的 V
    }
    if ph in rep_map.keys():  # 如果输入的字符在替换映射表中
        ph = rep_map[ph]  # 则将其替换为映射表中对应的值
    if ph in symbols:  # 如果替换后的字符在符号集合中
        return ph  # 则直接返回该字符
    if ph not in symbols:  # 如果替换后的字符不在符号集合中
        ph = "UNK"  # 则将其替换为未知字符标识
    return ph  # 返回处理后的字符


rep_map = {
    "：": ",",  # 将中文中的冒号替换为英文逗号
    "；": ",",  # 将中文中的分号替换为英文逗号
    "，": ",",  # 将中文中的逗号替换为英文逗号
    "。": ".",  # 将中文句号替换为英文句号
    "！": "!",  # 将中文感叹号替换为英文感叹号
    "？": "?",  # 将中文问号替换为英文问号
    "\n": ".",  # 将换行符替换为英文句号
    "．": ".",  # 将中文全角句号替换为英文句号
    "…": "...",  # 将中文省略号替换为英文省略号
    "···": "...",  # 将中文多个点号替换为英文省略号
    "・・・": "...",  # 将中文多个点号替换为英文省略号
    "·": ",",  # 将中文间隔点替换为英文逗号
    "・": ",",  # 将中文间隔点替换为英文逗号
    "、": ",",  # 将中文顿号替换为英文逗号
    "$": ".",  # 将美元符号替换为英文句号
    "“": "'",  # 将中文左双引号替换为英文单引号
    "”": "'",  # 将中文右双引号替换为英文单引号
    '"': "'",  # 将双引号替换为英文单引号
    "‘": "'",  # 将中文左单引号替换为英文单引号
    "’": "'",  # 将中文右单引号替换为英文单引号
    "（": "'",  # 将中文左括号替换为英文单引号
    "）": "'",  # 将中文右括号替换为英文单引号
    "(": "'",  # 将左括号替换为英文单引号
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
}

# 定义一个函数，用于替换文本中的标点符号
def replace_punctuation(text):
    # 创建一个正则表达式模式，用于匹配需要替换的标点符号
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    
    # 使用正则表达式模式替换文本中的标点符号
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    # 使用正则表达式替换文本中的特定字符
    # 将非日语、中文、标点符号以外的字符替换为空字符串
    # replaced_text = re.sub(
    #     r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    #     + "".join(punctuation)
    #     + r"]+",
    #     "",
    #     replaced_text,
    # )

    # 返回替换后的文本
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
        # 循环读取文件内容
        while line:
            # 判断是否达到起始行
            if line_index >= start_line:
                line = line.strip()  # 去除行首尾的空白字符
                word_split = line.split("  ")  # 以两个空格为分隔符，将行分割成单词和音节数组
                word = word_split[0]  # 获取单词

                syllable_split = word_split[1].split(" - ")  # 以" - "为分隔符，将音节数组分割成音节数组
                g2p_dict[word] = []  # 为单词创建空的音素列表
                for syllable in syllable_split:  # 遍历音节数组
                    phone_split = syllable.split(" ")  # 以空格为分隔符，将音节分割成音素数组
                    g2p_dict[word].append(phone_split)  # 将音素数组添加到单词的音素列表中

            line_index = line_index + 1  # 行索引自增1
            line = f.readline()  # 读取下一行

    return g2p_dict  # 返回单词到音素列表的字典


def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:  # 以二进制写模式打开文件
        pickle.dump(g2p_dict, pickle_file)  # 将字典对象序列化并写入文件
# 定义一个函数，用于获取字典数据
def get_dict():
    # 如果缓存路径下的文件存在
    if os.path.exists(CACHE_PATH):
        # 以二进制读取模式打开缓存文件
        with open(CACHE_PATH, "rb") as pickle_file:
            # 从缓存文件中加载数据到 g2p_dict
            g2p_dict = pickle.load(pickle_file)
    else:
        # 如果缓存文件不存在，则调用 read_dict() 函数获取数据
        g2p_dict = read_dict()
        # 将获取的数据缓存到 CACHE_PATH 路径下
        cache_dict(g2p_dict, CACHE_PATH)

    # 返回获取的字典数据
    return g2p_dict

# 调用 get_dict() 函数获取英文字典数据
eng_dict = get_dict()

# 定义一个函数，用于处理音标数据
def refine_ph(phn):
    # 初始化音调为 0
    tone = 0
    # 如果音标字符串末尾包含数字
    if re.search(r"\d$", phn):
        # 获取末尾数字并加一作为音调
        tone = int(phn[-1]) + 1
        # 去除音标字符串末尾的数字
        phn = phn[:-1]
    else:
        tone = 3  # 如果不符合任何条件，则将音调设置为3
    return phn.lower(), tone  # 返回小写的音素和音调


def refine_syllables(syllables):
    tones = []  # 创建一个空列表用于存储音调
    phonemes = []  # 创建一个空列表用于存储音素
    for phn_list in syllables:  # 遍历音节列表
        for i in range(len(phn_list)):  # 遍历每个音节
            phn = phn_list[i]  # 获取当前音节
            phn, tone = refine_ph(phn)  # 调用refine_ph函数处理音节，获取处理后的音素和音调
            phonemes.append(phn)  # 将处理后的音素添加到列表中
            tones.append(tone)  # 将处理后的音调添加到列表中
    return phonemes, tones  # 返回处理后的音素列表和音调列表


import re  # 导入re模块，用于处理正则表达式
import inflect  # 导入inflect模块，用于处理英文单词的复数形式等
# 导入 inflect 模块，用于处理英文单词的复数、序数等变换
_inflect = inflect.engine()
# 编译正则表达式，用于匹配逗号分隔的数字
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
# 编译正则表达式，用于匹配小数
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
# 编译正则表达式，用于匹配英镑符号和数字
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
# 编译正则表达式，用于匹配美元符号和数字
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
# 编译正则表达式，用于匹配序数词
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
# 编译正则表达式，用于匹配数字
_number_re = re.compile(r"[0-9]+")

# 缩写替换的正则表达式和替换字符串的列表
_abbreviations = [
    # 编译正则表达式，用于匹配缩写单词并替换为全称
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
        ("drs", "doctors"),  # 将 "drs" 替换为 "doctors"
        ("rev", "reverend"),  # 将 "rev" 替换为 "reverend"
        ("lt", "lieutenant"),  # 将 "lt" 替换为 "lieutenant"
        ("hon", "honorable"),  # 将 "hon" 替换为 "honorable"
        ("sgt", "sergeant"),  # 将 "sgt" 替换为 "sergeant"
        ("capt", "captain"),  # 将 "capt" 替换为 "captain"
        ("esq", "esquire"),  # 将 "esq" 替换为 "esquire"
        ("ltd", "limited"),  # 将 "ltd" 替换为 "limited"
        ("col", "colonel"),  # 将 "col" 替换为 "colonel"
        ("ft", "fort"),  # 将 "ft" 替换为 "fort"
    ]
]


# List of (ipa, lazy ipa) pairs:
_lazy_ipa = [
    (re.compile("%s" % x[0]), x[1])  # 使用正则表达式将 x[0] 替换为 x[1]，并添加到 _lazy_ipa 列表中
    for x in [
        ("r", "ɹ"),  # 将 "r" 替换为 "ɹ"
        ("æ", "e"),  # 将 "æ" 替换为 "e"
# List of (ipa, lazy ipa) pairs:
# 创建一个包含 IPA 和懒惰 IPA 对的列表

_lazy_ipa = [
    # 将每个元组中的 IPA 和懒惰 IPA 对应起来
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

# List of (ipa, lazy ipa2) pairs:
# 创建一个包含 IPA 和懒惰 IPA2 对的列表

_lazy_ipa2 = [
    # 使用列表推导式将每个元组中的 IPA 和懒惰 IPA2 对应起来
    (re.compile("%s" % x[0]), x[1])
    for x in [
        ("r", "ɹ"),
        ("ð", "z"),
        ("θ", "s"),
    ]
]
        ("ʒ", "ʑ"),  # 将字符"ʒ"替换为"ʑ"
        ("ʤ", "dʑ"),  # 将字符"ʤ"替换为"dʑ"
        ("ˈ", "↓"),  # 将字符"ˈ"替换为"↓"
    ]
]

# List of (ipa, ipa2) pairs
_ipa_to_ipa2 = [
    (re.compile("%s" % x[0]), x[1]) for x in [("r", "ɹ"), ("ʤ", "dʒ"), ("ʧ", "tʃ")]  # 使用正则表达式将字符替换为另一个字符
]


def _expand_dollars(m):
    match = m.group(1)  # 获取匹配的字符串
    parts = match.split(".")  # 以"."分割字符串
    if len(parts) > 2:
        return match + " dollars"  # 如果分割后的部分大于2，则返回原字符串加上"dollars"
    dollars = int(parts[0]) if parts[0] else 0  # 如果第一个部分存在，则转换为整数，否则为0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0  # 如果第二个部分存在，则转换为整数，否则为0
    if dollars and cents:  # 如果dollars和cents都存在
        dollar_unit = "dollar" if dollars == 1 else "dollars"  # 根据美元数量判断使用单数还是复数形式的单位
        cent_unit = "cent" if cents == 1 else "cents"  # 根据美分数量判断使用单数还是复数形式的单位
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)  # 返回美元和美分的数量以及对应的单位
    elif dollars:  # 如果只有美元数量
        dollar_unit = "dollar" if dollars == 1 else "dollars"  # 根据美元数量判断使用单数还是复数形式的单位
        return "%s %s" % (dollars, dollar_unit)  # 返回美元的数量以及对应的单位
    elif cents:  # 如果只有美分数量
        cent_unit = "cent" if cents == 1 else "cents"  # 根据美分数量判断使用单数还是复数形式的单位
        return "%s %s" % (cents, cent_unit)  # 返回美分的数量以及对应的单位
    else:  # 如果美元和美分数量都为0
        return "zero dollars"  # 返回零美元


def _remove_commas(m):
    return m.group(1).replace(",", "")  # 用空字符串替换匹配到的逗号


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))  # 将匹配到的序数词转换为对应的英文单词
# 定义一个函数，用于将匹配到的数字字符串转换为对应的英文表示
def _expand_number(m):
    # 将匹配到的数字字符串转换为整数
    num = int(m.group(0))
    # 判断数字范围，进行相应的转换
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

# 定义一个函数，用于将匹配到的小数点字符串替换为英文表示
def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")
def normalize_numbers(text):
    # 用空格替换文本中的逗号
    text = re.sub(_comma_number_re, _remove_commas, text)
    # 将英镑符号替换为 pounds
    text = re.sub(_pounds_re, r"\1 pounds", text)
    # 将美元符号替换为完整的 dollars 表示
    text = re.sub(_dollars_re, _expand_dollars, text)
    # 将小数点替换为完整的表示
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    # 将序数词替换为完整的表示
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    # 将数字替换为完整的表示
    text = re.sub(_number_re, _expand_number, text)
    return text


def text_normalize(text):
    # 对文本中的数字进行标准化处理
    text = normalize_numbers(text)
    # 替换文本中的标点符号
    text = replace_punctuation(text)
    # 在标点符号后面添加空格
    text = re.sub(r"([,;.\?\!])([\w])", r"\1 \2", text)
    return text


def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word  # 创建一个长度为 n_word 的列表，每个元素初始化为 0，用于记录每个单词的电话数量

    for task in range(n_phone):  # 遍历 n_phone 次，表示每个任务需要处理的电话数量
        min_tasks = min(phones_per_word)  # 找到 phones_per_word 中的最小值
        min_index = phones_per_word.index(min_tasks)  # 找到最小值的索引
        phones_per_word[min_index] += 1  # 将最小值对应的单词的电话数量加 1

    return phones_per_word  # 返回每个单词的电话数量列表


def sep_text(text):
    words = re.split(r"([,;.\?\!\s+])", text)  # 使用正则表达式将文本分割成单词列表
    words = [word for word in words if word.strip() != ""]  # 去除空白单词
    return words  # 返回处理后的单词列表


def text_to_words(text):
    tokens = tokenizer.tokenize(text)  # 使用分词器将文本分割成 tokens
    words = []  # 创建一个空列表，用于存储单词
    for idx, t in enumerate(tokens):  # 遍历 tokens
        if t.startswith("▁"):  # 如果 token 以 "▁" 开头
            words.append([t[1:]])  # 将去除 "▁" 后的部分作为单词添加到列表中
        else:  # 如果不是特殊情况，执行以下操作
            if t in punctuation:  # 如果当前标记在标点符号列表中
                if idx == len(tokens) - 1:  # 如果当前标记是最后一个
                    words.append([f"{t}"])  # 将当前标记作为一个单独的词添加到单词列表中
                else:  # 如果当前标记不是最后一个
                    if (  # 如果下一个标记不是以"▁"开头并且不是标点符号
                        not tokens[idx + 1].startswith("▁")
                        and tokens[idx + 1] not in punctuation
                    ):
                        if idx == 0:  # 如果当前标记是第一个
                            words.append([])  # 添加一个空列表到单词列表中
                        words[-1].append(f"{t}")  # 将当前标记添加到最后一个单词列表中
                    else:  # 如果下一个标记是以"▁"开头或者是标点符号
                        words.append([f"{t}"])  # 将当前标记作为一个单独的词添加到单词列表中
            else:  # 如果当前标记不是标点符号
                if idx == 0:  # 如果当前标记是第一个
                    words.append([])  # 添加一个空列表到单词列表中
                words[-1].append(f"{t}")  # 将当前标记添加到最后一个单词列表中
    return words  # 返回单词列表
def g2p(text):  # 定义一个函数 g2p，接受一个文本参数
    phones = []  # 初始化一个空列表 phones，用于存储音素
    tones = []  # 初始化一个空列表 tones，用于存储音调
    phone_len = []  # 初始化一个空列表 phone_len，用于存储音素长度
    # words = sep_text(text)  # 将文本分割成单词
    # tokens = [tokenizer.tokenize(i) for i in words]  # 对单词进行分词处理
    words = text_to_words(text)  # 将文本转换成单词列表

    for word in words:  # 遍历单词列表
        temp_phones, temp_tones = [], []  # 初始化临时列表 temp_phones 和 temp_tones
        if len(word) > 1:  # 如果单词长度大于1
            if "'" in word:  # 如果单词中包含撇号
                word = ["".join(word)]  # 将单词连接成字符串
        for w in word:  # 遍历单词中的每个字符
            if w in punctuation:  # 如果字符是标点符号
                temp_phones.append(w)  # 将字符添加到 temp_phones 列表中
                temp_tones.append(0)  # 将音调设为0，并添加到 temp_tones 列表中
                continue  # 继续下一个字符的处理
            if w.upper() in eng_dict:  # 如果字符的大写形式在英文词典中
                phns, tns = refine_syllables(eng_dict[w.upper()])  # 从英文单词的字典中获取音节和音调
                temp_phones += [post_replace_ph(i) for i in phns]  # 将音节经过处理后添加到临时音节列表中
                temp_tones += tns  # 将音调添加到临时音调列表中
                # w2ph.append(len(phns))  # 将音节数量添加到w2ph列表中
            else:
                phone_list = list(filter(lambda p: p != " ", _g2p(w)))  # 从英文单词转换为音素列表
                phns = []  # 初始化音节列表
                tns = []  # 初始化音调列表
                for ph in phone_list:  # 遍历音素列表
                    if ph in arpa:  # 如果音素在arpa中
                        ph, tn = refine_ph(ph)  # 对音素进行处理，获取音节和音调
                        phns.append(ph)  # 将处理后的音节添加到列表中
                        tns.append(tn)  # 将音调添加到列表中
                    else:
                        phns.append(ph)  # 将未处理的音节添加到列表中
                        tns.append(0)  # 将音调设为0
                temp_phones += [post_replace_ph(i) for i in phns]  # 将音节经过处理后添加到临时音节列表中
                temp_tones += tns  # 将音调添加到临时音调列表中
        phones += temp_phones  # 将临时音节列表添加到总音节列表中
        tones += temp_tones  # 将临时音调列表添加到总音调列表中
        phone_len.append(len(temp_phones))  # 将临时电话列表的长度添加到phone_len列表中

        # phones = [post_replace_ph(i) for i in phones]  # 对phones列表中的每个元素进行替换处理后重新赋值给phones列表

    word2ph = []  # 初始化word2ph列表

    for token, pl in zip(words, phone_len):  # 遍历words和phone_len列表中的元素
        word_len = len(token)  # 获取token的长度

        aaa = distribute_phone(pl, word_len)  # 调用distribute_phone函数，将返回值赋给aaa
        word2ph += aaa  # 将aaa列表中的元素添加到word2ph列表中

    phones = ["_"] + phones + ["_"]  # 在phones列表的开头和结尾添加下划线
    tones = [0] + tones + [0]  # 在tones列表的开头和结尾添加0
    word2ph = [1] + word2ph + [1]  # 在word2ph列表的开头和结尾添加1
    assert len(phones) == len(tones), text  # 断言phones列表和tones列表的长度相等，如果不相等则抛出异常并显示text
    assert len(phones) == sum(word2ph), text  # 断言phones列表的长度和word2ph列表元素之和相等，如果不相等则抛出异常并显示text

    return phones, tones, word2ph  # 返回phones、tones和word2ph列表


def get_bert_feature(text, word2ph):  # 定义get_bert_feature函数，接受text和word2ph两个参数
    from text import english_bert_mock  # 从 text 模块中导入 english_bert_mock 函数

    return english_bert_mock.get_bert_feature(text, word2ph)  # 调用 english_bert_mock 模块中的 get_bert_feature 函数并返回结果


if __name__ == "__main__":
    # print(get_dict())  # 打印 get_dict 函数的结果
    # print(eng_word_to_phoneme("hello"))  # 打印 eng_word_to_phoneme 函数对 "hello" 的结果
    print(g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."))  # 打印 g2p 函数对给定文本的结果
    # all_phones = set()  # 创建一个空集合
    # for k, syllables in eng_dict.items():  # 遍历 eng_dict 字典的键值对
    #     for group in syllables:  # 遍历每个值中的列表
    #         for ph in group:  # 遍历列表中的元素
    #             all_phones.add(ph)  # 将元素添加到集合中
    # print(all_phones)  # 打印集合中的所有元素
```