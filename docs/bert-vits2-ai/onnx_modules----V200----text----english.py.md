# `Bert-VITS2\onnx_modules\V200\text\english.py`

```
# 导入 pickle 模块
import pickle
# 导入 os 模块
import os
# 导入 re 模块
import re
# 从 g2p_en 模块中导入 G2p 类
from g2p_en import G2p
# 从当前目录中的 symbols 模块中导入所有内容
from . import symbols

# 获取当前文件所在目录的路径
current_file_path = os.path.dirname(__file__)
# 拼接得到 CMU 字典文件的路径
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
# 拼接得到缓存文件的路径
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")
# 创建 G2p 对象
_g2p = G2p()

# 定义 arpa 集合，包含一系列字符串
arpa = {
    "AH0",
    "S",
    # ... 其他字符串
}

# 定义函数，用于替换音素中的特殊字符
def post_replace_ph(ph):
    # 定义替换映射表
    rep_map = {
        "：": ",",
        "；": ",",
        # ... 其他替换规则
    }
    # 如果音素在替换映射表中，则进行替换
    if ph in rep_map.keys():
        ph = rep_map[ph]
    # 如果音素在 symbols 集合中，则返回原音素
    if ph in symbols:
        return ph
    # 如果音素不在 symbols 集合中，则替换为 "UNK"
    if ph not in symbols:
        ph = "UNK"
    return ph

# 定义函数，用于读取字典
def read_dict():
    # 创建空字典 g2p_dict
    g2p_dict = {}
    # 定义起始行号为 49
    start_line = 49
    # 打开 CMU 字典文件
    with open(CMU_DICT_PATH) as f:
        # 读取文件的一行
        line = f.readline()
        # 初始化行号
        line_index = 1
        # 循环读取文件的每一行
        while line:
            # 如果行号大于等于指定的起始行号
            if line_index >= start_line:
                # 去除行首尾的空白字符
                line = line.strip()
                # 以两个空格为分隔符，分割单词和音节数组
                word_split = line.split("  ")
                # 获取单词
                word = word_split[0]
    
                # 以" - "为分隔符，分割音节数组
                syllable_split = word_split[1].split(" - ")
                # 初始化单词到音节数组的映射
                g2p_dict[word] = []
                # 遍历音节数组
                for syllable in syllable_split:
                    # 以空格为分隔符，分割音素
                    phone_split = syllable.split(" ")
                    # 将音素添加到单词的音节数组中
                    g2p_dict[word].append(phone_split)
    
            # 行号加一
            line_index = line_index + 1
            # 读取下一行
            line = f.readline()
    
    # 返回单词到音节数组的映射
    return g2p_dict
# 将字典对象保存到指定的文件路径中
def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


# 获取字典对象，如果缓存文件存在则从缓存文件中读取，否则重新生成字典并保存到缓存文件中
def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict()
        cache_dict(g2p_dict, CACHE_PATH)

    return g2p_dict


# 从缓存中获取字典对象
eng_dict = get_dict()


# 对音节进行处理，去除音调信息
def refine_ph(phn):
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    return phn.lower(), tone


# 对音节列表进行处理，去除音调信息
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


# 导入正则表达式和inflect模块
import re
import inflect

# 创建inflect引擎对象
_inflect = inflect.engine()

# 编译正则表达式，用于匹配逗号分隔的数字
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")

# 编译正则表达式，用于匹配小数
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+")

# 编译正则表达式，用于匹配英镑符号
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")

# 编译正则表达式，用于匹配美元符号
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")

# 编译正则表达式，用于匹配序数词
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")

# 编译正则表达式，用于匹配数字
_number_re = re.compile(r"[0-9]+")

# 编译正则表达式，用于匹配缩写词并替换为全称
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

# 编译正则表达式，用于匹配懒惰的国际音标并替换为标准的国际音标
_lazy_ipa = [
    (re.compile("%s" % x[0]), x[1])
    # 遍历列表中的每个元组
    for x in [
        ("r", "ɹ"),  # 将字符 "r" 替换为 "ɹ"
        ("æ", "e"),  # 将字符 "æ" 替换为 "e"
        ("ɑ", "a"),  # 将字符 "ɑ" 替换为 "a"
        ("ɔ", "o"),  # 将字符 "ɔ" 替换为 "o"
        ("ð", "z"),  # 将字符 "ð" 替换为 "z"
        ("θ", "s"),  # 将字符 "θ" 替换为 "s"
        ("ɛ", "e"),  # 将字符 "ɛ" 替换为 "e"
        ("ɪ", "i"),  # 将字符 "ɪ" 替换为 "i"
        ("ʊ", "u"),  # 将字符 "ʊ" 替换为 "u"
        ("ʒ", "ʥ"),  # 将字符 "ʒ" 替换为 "ʥ"
        ("ʤ", "ʥ"),  # 将字符 "ʤ" 替换为 "ʥ"
        ("ˈ", "↓"),   # 将字符 "ˈ" 替换为 "↓"
    ]
# List of (ipa, lazy ipa2) pairs:
# 存储了 IPA 和懒惰的 IPA2 的列表
_lazy_ipa2 = [
    # 使用正则表达式将每个元组的第一个元素转换为字符串，并与第二个元素组成元组
    (re.compile("%s" % x[0]), x[1])
    for x in [
        # 遍历元组列表，将每个元组的第一个元素替换为字符串
        ("r", "ɹ"),
        ("ð", "z"),
        ("θ", "s"),
        ("ʒ", "ʑ"),
        ("ʤ", "dʑ"),
        ("ˈ", "↓"),
    ]
]

# List of (ipa, ipa2) pairs
# 存储了 IPA 和 IPA2 的列表
_ipa_to_ipa2 = [
    # 使用正则表达式将每个元组的第一个元素转换为字符串，并与第二个元素组成元组
    (re.compile("%s" % x[0]), x[1]) for x in [("r", "ɹ"), ("ʤ", "dʒ"), ("ʧ", "tʃ")]
]


def _expand_dollars(m):
    # 获取匹配的字符串
    match = m.group(1)
    # 将匹配的字符串按照"."分割
    parts = match.split(".")
    # 如果分割后的部分大于2，则返回原字符串 + " dollars"
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    # 如果第一个部分存在，则将其转换为整数，否则为0
    dollars = int(parts[0]) if parts[0] else 0
    # 如果有第二个部分且不为空，则将其转换为整数，否则为0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    # 根据不同情况返回相应的字符串
    if dollars and cents:
        # 根据不同情况返回相应的字符串
    elif dollars:
        # 根据不同情况返回相应的字符串
    elif cents:
        # 根据不同情况返回相应的字符串
    else:
        return "zero dollars"


def _remove_commas(m):
    # 返回匹配字符串中去掉逗号的部分


def _expand_ordinal(m):
    # 将匹配的序数词转换为对应的英文单词


def _expand_number(m):
    # 将匹配的数字转换为对应的英文单词


def _expand_decimal_point(m):
    # 将匹配的小数点替换为对应的英文单词


def normalize_numbers(text):
    # 使用正则表达式替换文本中的逗号
    text = re.sub(_comma_number_re, _remove_commas, text)
    # 使用正则表达式替换文本中的英镑符号为 "pounds"
    text = re.sub(_pounds_re, r"\1 pounds", text)
    # 使用正则表达式替换文本中的美元符号为扩展的美元形式
    text = re.sub(_dollars_re, _expand_dollars, text)
    # 使用正则表达式替换文本中的小数点为扩展的小数形式
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    # 使用正则表达式替换文本中的序数词为扩展的序数形式
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    # 使用正则表达式替换文本中的数字为扩展的数字形式
    text = re.sub(_number_re, _expand_number, text)
    # 返回替换后的文本
    return text
# 对文本进行数字标准化处理
def text_normalize(text):
    text = normalize_numbers(text)
    return text


# 将文本转换为音素序列
def g2p(text):
    phones = []  # 存储音素
    tones = []  # 存储音调
    word2ph = []  # 存储每个单词对应的音素数量
    words = re.split(r"([,;.\-\?\!\s+])", text)  # 使用标点符号和空格分割文本
    words = [word for word in words if word.strip() != ""]  # 去除空单词
    for word in words:
        if word.upper() in eng_dict:  # 如果单词在英语字典中
            phns, tns = refine_syllables(eng_dict[word.upper()])  # 获取单词的音素和音调
            phones += phns  # 将音素添加到列表中
            tones += tns  # 将音调添加到列表中
            word2ph.append(len(phns))  # 记录单词对应的音素数量
        else:
            # 对于不在字典中的单词，将其转换为音素序列
            phone_list = list(filter(lambda p: p != " ", _g2p(word))
            for ph in phone_list:
                if ph in arpa:  # 如果音素在arpa中
                    ph, tn = refine_ph(ph)  # 对音素进行处理
                    phones.append(ph)  # 添加处理后的音素
                    tones.append(tn)  # 添加音调
                else:
                    phones.append(ph)  # 添加原始音素
                    tones.append(0)  # 添加默认音调
            word2ph.append(len(phone_list))  # 记录单词对应的音素数量

    phones = [post_replace_ph(i) for i in phones]  # 对音素进行后处理

    phones = ["_"] + phones + ["_"]  # 在音素序列前后添加占位符
    tones = [0] + tones + [0]  # 在音调序列前后添加默认音调
    word2ph = [1] + word2ph + [1]  # 在单词对应音素数量序列前后添加默认值

    return phones, tones, word2ph  # 返回音素序列、音调序列和单词对应音素数量序列


# 获取文本的BERT特征
def get_bert_feature(text, word2ph):
    from text import english_bert_mock  # 导入英语BERT模型的模拟

    return english_bert_mock.get_bert_feature(text, word2ph)  # 调用模拟的英语BERT模型获取文本的BERT特征


if __name__ == "__main__":
    # print(get_dict())  # 打印字典内容
    # print(eng_word_to_phoneme("hello"))  # 打印单词对应的音素
    print(g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."))  # 打印给定文本的音素序列
    # all_phones = set()
    # for k, syllables in eng_dict.items():
    #     for group in syllables:
    #         for ph in group:
    #             all_phones.add(ph)
    # print(all_phones)  # 打印所有音素
```