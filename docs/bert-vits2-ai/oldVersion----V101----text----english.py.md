# `Bert-VITS2\oldVersion\V101\text\english.py`

```
# 导入 pickle、os、re 和 G2p 模块
import pickle
import os
import re
from g2p_en import G2p

# 从 text 模块中导入 symbols
from text import symbols

# 获取当前文件所在路径
current_file_path = os.path.dirname(__file__)
# 拼接路径，得到 CMU 字典文件路径
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
# 拼接路径，得到缓存文件路径
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")
# 创建 G2p 对象
_g2p = G2p()

# 定义 arpa 字符集
arpa = {
    # ... （此处省略了 arpa 字符集的具体内容）
}

# 定义函数，用于替换音素中的特殊字符
def post_replace_ph(ph):
    # 定义替换映射表
    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
        "v": "V",
    }
    # 如果音素在替换映射表中，则进行替换
    if ph in rep_map.keys():
        ph = rep_map[ph]
    # 如果音素在 symbols 中，则直接返回
    if ph in symbols:
        return ph
    # 如果音素不在 symbols 中，则替换为 "UNK"
    if ph not in symbols:
        ph = "UNK"
    return ph

# 定义函数，用于读取字典
def read_dict():
    # 创建空字典
    g2p_dict = {}
    # 定义起始行号
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
# 将字典对象 g2p_dict 缓存到文件中
def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


# 获取字典对象，如果缓存文件存在则从缓存文件中读取，否则调用 read_dict() 方法获取字典并缓存到文件中
def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict()
        cache_dict(g2p_dict, CACHE_PATH)

    return g2p_dict


# 获取英文字典对象
eng_dict = get_dict()


# 对音素进行处理，如果音素末尾有数字则提取出来作为音调，然后将音素转换为小写形式
def refine_ph(phn):
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    return phn.lower(), tone


# 对音节进行处理，将每个音素进行 refine_ph 处理，得到音素列表和音调列表
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


# 对文本进行规范化处理
def text_normalize(text):
    # todo: eng text normalize
    return text


# 将文本转换为音素和音调的列表
def g2p(text):
    phones = []
    tones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.upper() in eng_dict:
            phns, tns = refine_syllables(eng_dict[w.upper()])
            phones += phns
            tones += tns
        else:
            phone_list = list(filter(lambda p: p != " ", _g2p(w)))
            for ph in phone_list:
                if ph in arpa:
                    ph, tn = refine_ph(ph)
                    phones.append(ph)
                    tones.append(tn)
                else:
                    phones.append(ph)
                    tones.append(0)
    # todo: implement word2ph
    word2ph = [1 for i in phones]

    phones = [post_replace_ph(i) for i in phones]
    return phones, tones, word2ph


# 主程序入口
if __name__ == "__main__":
    # print(get_dict())
    # print(eng_word_to_phoneme("hello"))
    # 对给定文本进行音素和音调的转换
    print(g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."))
    # all_phones = set()
    # for k, syllables in eng_dict.items():
    #     for group in syllables:
    # 遍历 group 中的每个元素 ph
    # 将 ph 添加到 all_phones 集合中
    # 打印 all_phones 集合
```