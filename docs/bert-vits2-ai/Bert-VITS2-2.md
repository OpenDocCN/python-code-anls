# Bert-VITS2 源码解析 2

# `D:\src\Bert-VITS2\monotonic_align\core.py`

```python
import numba  # Import the numba library

@numba.jit(  # Decorator for the function to be compiled with numba
    numba.void(  # The function returns nothing
        numba.int32[:, :, ::1],  # 3D array of 32-bit integers
        numba.float32[:, :, ::1],  # 3D array of 32-bit floats
        numba.int32[::1],  # 1D array of 32-bit integers
        numba.int32[::1],  # 1D array of 32-bit integers
    ),
    nopython=True,  # Compile the function in nopython mode
    nogil=True,  # Release the GIL (Global Interpreter Lock)
)
def maximum_path_jit(paths, values, t_ys, t_xs):  # Define the function with the specified arguments
    b = paths.shape[0]  # Get the size of the first dimension of the paths array
    max_neg_val = -1e9  # Initialize a variable with a large negative value
    for i in range(int(b)):  # Iterate over the range of b
        path = paths[i]  # Get the i-th path
        value = values[i]  # Get the i-th value
        t_y = t_ys[i]  # Get the i-th t_y
        t_x = t_xs[i]  # Get the i-th t_x

        v_prev = v_cur = 0.0  # Initialize v_prev and v_cur to 0.0
        index = t_x - 1  # Set index to t_x - 1

        for y in range(t_y):  # Iterate over the range of t_y
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):  # Iterate over the range of x
                if x == y:  # Check if x is equal to y
                    v_cur = max_neg_val  # Set v_cur to max_neg_val
                else:  # If x is not equal to y
                    v_cur = value[y - 1, x]  # Set v_cur to the value at position (y-1, x)
                if x == 0:  # Check if x is 0
                    if y == 0:  # Check if y is 0
                        v_prev = 0.0  # Set v_prev to 0.0
                    else:  # If y is not 0
                        v_prev = max_neg_val  # Set v_prev to max_neg_val
                else:  # If x is not 0
                    v_prev = value[y - 1, x - 1]  # Set v_prev to the value at position (y-1, x-1)
                value[y, x] += max(v_prev, v_cur)  # Add the maximum of v_prev and v_cur to the value at position (y, x)

        for y in range(t_y - 1, -1, -1):  # Iterate over the range of t_y-1 to -1 with step -1
            path[y, index] = 1  # Set the value at position (y, index) in path to 1
            if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):  # Check the conditions
                index = index - 1  # Update the value of index
```

# `D:\src\Bert-VITS2\monotonic_align\__init__.py`

```python
from numpy import zeros, int32, float32  # 导入numpy库中的zeros、int32和float32函数
from torch import from_numpy  # 从torch库中导入from_numpy函数

from .core import maximum_path_jit  # 从当前目录下的core模块中导入maximum_path_jit函数


def maximum_path(neg_cent, mask):
    device = neg_cent.device  # 获取neg_cent的设备信息
    dtype = neg_cent.dtype  # 获取neg_cent的数据类型
    neg_cent = neg_cent.data.cpu().numpy().astype(float32)  # 将neg_cent的数据转换为numpy数组，并转换为float32类型
    path = zeros(neg_cent.shape, dtype=int32)  # 创建一个与neg_cent形状相同的int32类型的全零数组

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(int32)  # 计算mask在第一维度上的和，并转换为int32类型
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(int32)  # 计算mask在第二维度上的和，并转换为int32类型
    maximum_path_jit(path, neg_cent, t_t_max, t_s_max)  # 调用maximum_path_jit函数
    return from_numpy(path).to(device=device, dtype=dtype)  # 将path转换为torch张量，并设置设备和数据类型
```

# `D:\src\Bert-VITS2\oldVersion\__init__.py`

```py
"""
# 定义一个字符串变量，表示老版本模型推理兼容
""" 
version = "老版本模型推理兼容"

"""
# 打印输出变量version的值
"""
print(version)
```

# `D:\src\Bert-VITS2\oldVersion\V101\models.py`

```python
import math  # import the math module
import torch  # import the torch module
from torch import nn  # import the nn module from torch
from torch.nn import functional as F  # import the functional module from torch.nn

import commons  # import the commons module
import modules  # import the modules module
import attentions  # import the attentions module
import monotonic_align  # import the monotonic_align module

from torch.nn import Conv1d, ConvTranspose1d, Conv2d  # import Conv1d, ConvTranspose1d, Conv2d from torch.nn
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm  # import weight_norm, remove_weight_norm, spectral_norm from torch.nn.utils

from commons import init_weights, get_padding  # import init_weights, get_padding from commons
from .text import symbols, num_tones, num_languages  # import symbols, num_tones, num_languages from text
```

# `D:\src\Bert-VITS2\oldVersion\V101\__init__.py`

```python
"""
1.0.1 版本兼容
https://github.com/fishaudio/Bert-VITS2/releases/tag/1.0.1
"""
import torch  # 导入torch模块
import commons  # 导入commons模块
from .text.cleaner import clean_text  # 从text模块中的cleaner子模块导入clean_text函数
from .text import cleaned_text_to_sequence  # 从text模块导入cleaned_text_to_sequence函数
from oldVersion.V111.text import get_bert  # 从oldVersion.V111.text模块导入get_bert函数

def get_text(text, language_str, hps, device):  # 定义get_text函数，接受text, language_str, hps, device四个参数
    norm_text, phone, tone, word2ph = clean_text(text, language_str)  # 调用clean_text函数，将返回值分别赋给norm_text, phone, tone, word2ph
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)  # 调用cleaned_text_to_sequence函数，将返回值分别赋给phone, tone, language

    if hps.data.add_blank:  # 如果hps.data.add_blank为真
        phone = commons.intersperse(phone, 0)  # 调用commons模块的intersperse函数，将返回值赋给phone
        tone = commons.intersperse(tone, 0)  # 调用commons模块的intersperse函数，将返回值赋给tone
        language = commons.intersperse(language, 0)  # 调用commons模块的intersperse函数，将返回值赋给language
        for i in range(len(word2ph)):  # 遍历word2ph的长度
            word2ph[i] = word2ph[i] * 2  # 将word2ph[i]的值乘以2
        word2ph[0] += 1  # 将word2ph[0]的值加1
    bert = get_bert(norm_text, word2ph, language_str, device)  # 调用get_bert函数，将返回值赋给bert
    del word2ph  # 删除word2ph

    assert bert.shape[-1] == len(phone)  # 断言bert的最后一个维度等于phone的长度

    phone = torch.LongTensor(phone)  # 将phone转换为torch的LongTensor类型
    tone = torch.LongTensor(tone)  # 将tone转换为torch的LongTensor类型
    language = torch.LongTensor(language)  # 将language转换为torch的LongTensor类型

    return bert, phone, tone, language  # 返回bert, phone, tone, language

def infer(
    text,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    sid,
    hps,
    net_g,
    device,
):  # 定义infer函数，接受text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, hps, net_g, device九个参数
    bert, phones, tones, lang_ids = get_text(text, "ZH", hps, device)  # 调用get_text函数，将返回值分别赋给bert, phones, tones, lang_ids
    with torch.no_grad():  # 使用torch.no_grad()上下文管理器
        x_tst = phones.to(device).unsqueeze(0)  # 将phones转移到device上并增加一个维度
        tones = tones.to(device).unsqueeze(0)  # 将tones转移到device上并增加一个维度
        lang_ids = lang_ids.to(device).unsqueeze(0)  # 将lang_ids转移到device上并增加一个维度
        bert = bert.to(device).unsqueeze(0)  # 将bert转移到device上并增加一个维度
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)  # 创建一个LongTensor类型的张量，并将其转移到device上
        del phones  # 删除phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)  # 创建一个LongTensor类型的张量，并将其转移到device上
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )  # 调用net_g的infer方法，将返回值赋给audio
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers  # 删除x_tst, tones, lang_ids, bert, x_tst_lengths, speakers
        if torch.cuda.is_available():  # 如果torch.cuda.is_available()为真
            torch.cuda.empty_cache()  # 清空cuda缓存
        return audio  # 返回audio
```

# `D:\src\Bert-VITS2\oldVersion\V101\text\chinese.py`

```python
import os  # 导入os模块
import re  # 导入re模块
import cn2an  # 导入cn2an模块
from pypinyin import lazy_pinyin, Style  # 从pypinyin模块中导入lazy_pinyin和Style

from .symbols import punctuation  # 从symbols模块中导入punctuation
from .tone_sandhi import ToneSandhi  # 从tone_sandhi模块中导入ToneSandhi

current_file_path = os.path.dirname(__file__)  # 获取当前文件路径
pinyin_to_symbol_map = {  # 创建pinyin_to_symbol_map字典
    line.split("\t")[0]: line.strip().split("\t")[1]  # 从文件中读取内容并构建字典
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

import jieba.posseg as psg  # 导入jieba.posseg模块并重命名为psg

rep_map = {  # 创建rep_map字典
    "：": ",",  # 键值对
    "；": ",",  # 键值对
    # ... 其他键值对
}

tone_modifier = ToneSandhi()  # 创建ToneSandhi对象

# ... 其他函数定义

if __name__ == "__main__":
    from text.chinese_bert import get_bert_feature  # 从text.chinese_bert模块中导入get_bert_feature

    text = "啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = text_normalize(text)  # 对文本进行规范化处理
    print(text)  # 打印处理后的文本
    phones, tones, word2ph = g2p(text)  # 调用g2p函数处理文本
    bert = get_bert_feature(text, word2ph)  # 调用get_bert_feature函数获取bert特征

    print(phones, tones, word2ph, bert.shape)  # 打印处理后的phones、tones、word2ph和bert特征
```

# `D:\src\Bert-VITS2\oldVersion\V101\text\chinese_bert.py`

```python
import torch  # 导入PyTorch库
word_level_feature = torch.rand(38, 1024)  # 12个词,每个词1024维特征
word2phone = [  # 定义word2phone列表
    1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1
]

# 计算总帧数
total_frames = sum(word2phone)  # 计算word2phone列表中所有元素的总和
print(word_level_feature.shape)  # 打印word_level_feature的形状
print(word2phone)  # 打印word2phone列表
phone_level_feature = []  # 定义phone_level_feature列表
for i in range(len(word2phone)):  # 遍历word2phone列表
    print(word_level_feature[i].shape)  # 打印word_level_feature[i]的形状

    # 对每个词重复word2phone[i]次
    repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)  # 重复word_level_feature[i]的特征值word2phone[i]次
    phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature列表中

phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 沿着0维度拼接phone_level_feature列表中的张量
print(phone_level_feature.shape)  # 打印phone_level_feature的形状  # torch.Size([36, 1024])
```

# `D:\src\Bert-VITS2\oldVersion\V101\text\cleaner.py`

```python
# 导入chinese和cleaned_text_to_sequence模块
from . import chinese, cleaned_text_to_sequence

# 创建语言模块映射
language_module_map = {"ZH": chinese}

# 清洗文本的函数
def clean_text(text, language):
    # 根据语言选择对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取音素、音调和词转音素映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph

# 使用BERT模型清洗文本的函数
def clean_text_bert(text, language):
    # 根据语言选择对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取音素、音调和词转音素映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 获取BERT特征
    bert = language_module.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert

# 将文本转换为序列的函数
def text_to_sequence(text, language):
    # 清洗文本
    norm_text, phones, tones, word2ph = clean_text(text, language)
    # 将音素和音调转换为序列
    return cleaned_text_to_sequence(phones, tones, language)

# 主函数
if __name__ == "__main__":
    pass
```

# `D:\src\Bert-VITS2\oldVersion\V101\text\english.py`

```python
import pickle  # 导入pickle模块
import os  # 导入os模块
import re  # 导入re模块
from g2p_en import G2p  # 从g2p_en模块导入G2p类

from text import symbols  # 从text模块导入symbols变量

current_file_path = os.path.dirname(__file__)  # 获取当前文件路径
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")  # 拼接文件路径
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")  # 拼接文件路径
_g2p = G2p()  # 创建G2p对象

arpa = {  # 定义arpa集合
    "AH0",
    "S",
    "AH1",
    "EY2",
    ...
    "0",
    "L",
    "SH",
}


def post_replace_ph(ph):  # 定义函数post_replace_ph，参数为ph
    rep_map = {  # 定义rep_map字典
        "：": ",",
        "；": ",",
        ...
        "v": "V",
    }
    if ph in rep_map.keys():  # 如果ph在rep_map的键中
        ph = rep_map[ph]  # 将ph替换为rep_map中对应的值
    if ph in symbols:  # 如果ph在symbols中
        return ph  # 返回ph
    if ph not in symbols:  # 如果ph不在symbols中
        ph = "UNK"  # 将ph替换为"UNK"
    return ph  # 返回ph


def read_dict():  # 定义函数read_dict
    g2p_dict = {}  # 创建空字典g2p_dict
    start_line = 49  # 定义start_line变量为49
    with open(CMU_DICT_PATH) as f:  # 打开文件CMU_DICT_PATH
        line = f.readline()  # 读取文件的一行
        line_index = 1  # 定义line_index变量为1
        while line:  # 循环，直到line为空
            if line_index >= start_line:  # 如果line_index大于等于start_line
                line = line.strip()  # 去除行首尾的空白字符
                word_split = line.split("  ")  # 以两个空格分割行
                word = word_split[0]  # 获取分割后的第一个元素

                syllable_split = word_split[1].split(" - ")  # 以" - "分割第二个元素
                g2p_dict[word] = []  # 将word作为键，空列表作为值存入g2p_dict
                for syllable in syllable_split:  # 遍历syllable_split
                    phone_split = syllable.split(" ")  # 以空格分割syllable
                    g2p_dict[word].append(phone_split)  # 将phone_split添加到g2p_dict[word]中

            line_index = line_index + 1  # line_index加1
            line = f.readline()  # 读取下一行

    return g2p_dict  # 返回g2p_dict


def cache_dict(g2p_dict, file_path):  # 定义函数cache_dict，参数为g2p_dict和file_path
    with open(file_path, "wb") as pickle_file:  # 以二进制写模式打开file_path
        pickle.dump(g2p_dict, pickle_file)  # 将g2p_dict序列化并写入pickle_file


def get_dict():  # 定义函数get_dict
    if os.path.exists(CACHE_PATH):  # 如果CACHE_PATH文件存在
        with open(CACHE_PATH, "rb") as pickle_file:  # 以二进制读模式打开CACHE_PATH
            g2p_dict = pickle.load(pickle_file)  # 从pickle_file中反序列化出g2p_dict
    else:  # 如果CACHE_PATH文件不存在
        g2p_dict = read_dict()  # 调用read_dict函数获取g2p_dict
        cache_dict(g2p_dict, CACHE_PATH)  # 调用cache_dict函数缓存g2p_dict到CACHE_PATH

    return g2p_dict  # 返回g2p_dict


eng_dict = get_dict()  # 调用get_dict函数获取eng_dict


def refine_ph(phn):  # 定义函数refine_ph，参数为phn
    tone = 0  # 定义tone变量为0
    if re.search(r"\d$", phn):  # 如果phn以数字结尾
        tone = int(phn[-1]) + 1  # 获取数字并加1赋值给tone
        phn = phn[:-1]  # 去除phn的最后一个字符
    return phn.lower(), tone  # 返回phn的小写形式和tone


def refine_syllables(syllables):  # 定义函数refine_syllables，参数为syllables
    tones = []  # 创建空列表tones
    phonemes = []  # 创建空列表phonemes
    for phn_list in syllables:  # 遍历syllables
        for i in range(len(phn_list)):  # 遍历phn_list的索引
            phn = phn_list[i]  # 获取phn_list的元素
            phn, tone = refine_ph(phn)  # 调用refine_ph函数处理phn
            phonemes.append(phn)  # 将处理后的phn添加到phonemes
            tones.append(tone)  # 将tone添加到tones
    return phonemes, tones  # 返回phonemes和tones


def text_normalize(text):  # 定义函数text_normalize，参数为text
    # todo: eng text normalize
    return text  # 返回text


def g2p(text):  # 定义函数g2p，参数为text
    phones = []  # 创建空列表phones
    tones = []  # 创建空列表tones
    words = re.split(r"([,;.\-\?\!\s+])", text)  # 以标点符号和空格分割text
    for w in words:  # 遍历words
        if w.upper() in eng_dict:  # 如果w的大写形式在eng_dict中
            phns, tns = refine_syllables(eng_dict[w.upper()])  # 调用refine_syllables处理eng_dict[w.upper()]
            phones += phns  # 将phns添加到phones
            tones += tns  # 将tns添加到tones
        else:  # 如果w的大写形式不在eng_dict中
            phone_list = list(filter(lambda p: p != " ", _g2p(w)))  # 通过_g2p处理w并过滤空格
            for ph in phone_list:  # 遍历phone_list
                if ph in arpa:  # 如果ph在arpa中
                    ph, tn = refine_ph(ph)  # 调用refine_ph处理ph
                    phones.append(ph)  # 将处理后的ph添加到phones
                    tones.append(tn)  # 将tn添加到tones
                else:  # 如果ph不在arpa中
                    phones.append(ph)  # 将ph添加到phones
                    tones.append(0)  # 将0添加到tones
    # todo: implement word2ph
    word2ph = [1 for i in phones]  # 创建长度与phones相同的列表，元素为1

    phones = [post_replace_ph(i) for i in phones]  # 调用post_replace_ph处理phones
    return phones, tones, word2ph  # 返回phones、tones和word2ph


if __name__ == "__main__":  # 如果模块是直接运行的
    # print(get_dict())
    # print(eng_word_to_phoneme("hello"))
    print(g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."))  # 调用g2p函数并打印结果
    # all_phones = set()
    # for k, syllables in eng_dict.items():
    #     for group in syllables:
    #         for ph in group:
    #             all_phones.add(ph)
    # print(all_phones)
```

# `D:\src\Bert-VITS2\oldVersion\V101\text\english_bert_mock.py`

```python
import torch  # 导入torch模块

def get_bert_feature(norm_text, word2ph):
    return torch.zeros(1024, sum(word2ph))  # 返回一个1024x(sum(word2ph))的全零张量
```

# `D:\src\Bert-VITS2\oldVersion\V101\text\japanese.py`

```python
# Regular expression matching Japanese without punctuation marks:
_japanese_characters = re.compile(
    r"[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# Regular expression matching non-Japanese characters or punctuation marks:
_japanese_marks = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# List of (symbol, Japanese) pairs for marks:
_symbols_to_japanese = [(re.compile("%s" % x[0]), x[1]) for x in [("％", "パーセント")]]

# List of (consonant, sokuon) pairs:
_real_sokuon = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        (r"Q([↑↓]*[kg])", r"k#\1"),
        (r"Q([↑↓]*[tdjʧ])", r"t#\1"),
        (r"Q([↑↓]*[sʃ])", r"s\1"),
        (r"Q([↑↓]*[pb])", r"p#\1"),
    ]
]

# List of (consonant, hatsuon) pairs:
_real_hatsuon = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        (r"N([↑↓]*[pbm])", r"m\1"),
        (r"N([↑↓]*[ʧʥj])", r"n^\1"),
        (r"N([↑↓]*[tdn])", r"n\1"),
        (r"N([↑↓]*[kg])", r"ŋ\1"),
    ]
]

def post_replace_ph(ph):
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
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "UNK"
    return ph

def symbols_to_japanese(text):
    for regex, replacement in _symbols_to_japanese:
        text = re.sub(regex, replacement, text)
    return text

def preprocess_jap(text):
    """Reference https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html"""
    text = symbols_to_japanese(text)
    sentences = re.split(_japanese_marks, text)
    marks = re.findall(_japanese_marks, text)
    text = []
    for i, sentence in enumerate(sentences):
        if re.match(_japanese_characters, sentence):
            p = pyopenjtalk.g2p(sentence)
            text += p.split(" ")

        if i < len(marks):
            text += [marks[i].replace(" ", "")]
    return text

def text_normalize(text):
    # todo: jap text normalize
    return text

def g2p(norm_text):
    phones = preprocess_jap(norm_text)
    phones = [post_replace_ph(i) for i in phones]
    # todo: implement tones and word2ph
    tones = [0 for i in phones]
    word2ph = [1 for i in phones]
    return phones, tones, word2ph
```

# `D:\src\Bert-VITS2\oldVersion\V101\text\symbols.py`

```python
punctuation = ["!", "?", "…", ",", ".", "'", "-"]  # 创建一个包含标点符号的列表
pu_symbols = punctuation + ["SP", "UNK"]  # 将标点符号列表与额外的字符串列表合并
pad = "_"  # 创建一个下划线字符串

zh_symbols = [  # 创建一个包含中文字符的列表
    "E",
    "En",
    ...
    "OO",
]
num_zh_tones = 6  # 设置中文音调数量为6

ja_symbols = [  # 创建一个包含日文字符的列表
    "I",
    "N",
    ...
    "z",
]
num_ja_tones = 1  # 设置日文音调数量为1

en_symbols = [  # 创建一个包含英文字符的列表
    "aa",
    "ae",
    ...
    "zh",
]
num_en_tones = 4  # 设置英文音调数量为4

normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))  # 将所有语言的字符合并并去重排序
symbols = [pad] + normal_symbols + pu_symbols  # 创建一个包含所有字符和标点符号的列表
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]  # 获取标点符号在列表中的索引

num_tones = num_zh_tones + num_ja_tones + num_en_tones  # 计算所有语言的音调数量之和

language_id_map = {"ZH": 0, "JA": 1, "EN": 2}  # 创建一个语言到ID的映射字典
num_languages = len(language_id_map.keys())  # 获取语言数量

language_tone_start_map = {  # 创建一个语言到音调起始位置的映射字典
    "ZH": 0,
    "JA": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
}

if __name__ == "__main__":
    a = set(zh_symbols)  # 创建一个包含中文字符的集合
    b = set(en_symbols)  # 创建一个包含英文字符的集合
    print(sorted(a & b))  # 打印中英文字符的交集
```

# `D:\src\Bert-VITS2\oldVersion\V101\text\tone_sandhi.py`

```python
# the meaning of jieba pos tag: https://blog.csdn.net/weixin_44174352/article/details/113731041
# e.g.
# word: "家里"
# pos: "s"
# finals: ['ia1', 'i3']
def _neural_sandhi(self, word: str, pos: str, finals: List[str]) -> List[str]:
    # reduplication words for n. and v. e.g. 奶奶, 试试, 旺旺
    for j, item in enumerate(word):
        if (
            j - 1 >= 0
            and item == word[j - 1]
            and pos[0] in {"n", "v", "a"}
            and word not in self.must_not_neural_tone_words
        ):
            finals[j] = finals[j][:-1] + "5"
    ge_idx = word.find("个")
    if len(word) >= 1 and word[-1] in "吧呢啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶":
        finals[-1] = finals[-1][:-1] + "5"
    elif len(word) >= 1 and word[-1] in "的地得":
        finals[-1] = finals[-1][:-1] + "5"
    # e.g. 走了, 看着, 去过
    # elif len(word) == 1 and word in "了着过" and pos in {"ul", "uz", "ug"}:
    #     finals[-1] = finals[-1][:-1] + "5"
    elif (
        len(word) > 1
        and word[-1] in "们子"
        and pos in {"r", "n"}
        and word not in self.must_not_neural_tone_words
    ):
        finals[-1] = finals[-1][:-1] + "5"
    # e.g. 桌上, 地下, 家里
    elif len(word) > 1 and word[-1] in "上下里" and pos in {"s", "l", "f"}:
        finals[-1] = finals[-1][:-1] + "5"
    # e.g. 上来, 下去
    elif len(word) > 1 and word[-1] in "来去" and word[-2] in "上下进出回过起开":
        finals[-1] = finals[-1][:-1] + "5"
    # 个做量词
    elif (
        ge_idx >= 1
        and (word[ge_idx - 1].isnumeric() or word[ge_idx - 1] in "几有两半多各整每做是")
    ) or word == "个":
        finals[ge_idx] = finals[ge_idx][:-1] + "5"
    else:
        if (
            word in self.must_neural_tone_words
            or word[-2:] in self.must_neural_tone_words
        ):
            finals[-1] = finals[-1][:-1] + "5"

    word_list = self._split_word(word)
    finals_list = [finals[: len(word_list[0])], finals[len(word_list[0]) :]]
    for i, word in enumerate(word_list):
        # conventional neural in Chinese
        if (
            word in self.must_neural_tone_words
            or word[-2:] in self.must_neural_tone_words
        ):
            finals_list[i][-1] = finals_list[i][-1][:-1] + "5"
    finals = sum(finals_list, [])
    return finals
```

# `D:\src\Bert-VITS2\oldVersion\V101\text\__init__.py`

```python
# Import all symbols from the symbols module
from .symbols import *

# Create a dictionary that maps symbols to their corresponding IDs
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

# Define a function that converts a string of text to a sequence of IDs corresponding to the symbols in the text
def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    # Convert each symbol in the cleaned text to its corresponding ID using the _symbol_to_id dictionary
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    # Calculate the tone start index based on the language
    tone_start = language_tone_start_map[language]
    # Add the tone start index to each tone in the input
    tones = [i + tone_start for i in tones]
    # Get the language ID based on the language
    lang_id = language_id_map[language]
    # Create a list of language IDs corresponding to the number of phones
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids

# Define a function that retrieves BERT features for normalized text and word-to-phoneme mapping based on the language
def get_bert(norm_text, word2ph, language):
    # Import the get_bert_feature function from the chinese_bert module for Chinese language
    from .chinese_bert import get_bert_feature as zh_bert
    # Import the get_bert_feature function from the english_bert_mock module for English language
    from .english_bert_mock import get_bert_feature as en_bert

    # Create a mapping of language to the corresponding BERT feature function
    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert}
    # Retrieve the BERT features for the normalized text and word-to-phoneme mapping based on the language
    bert = lang_bert_func_map[language](norm_text, word2ph)
    return bert
```

# `D:\src\Bert-VITS2\oldVersion\V110\models.py`

```python
import math  # import the math module
import torch  # import the torch module
from torch import nn  # import the nn module from the torch module
from torch.nn import functional as F  # import the functional module from the nn module

import commons  # import the commons module
import modules  # import the modules module
import attentions  # import the attentions module
import monotonic_align  # import the monotonic_align module

from torch.nn import Conv1d, ConvTranspose1d, Conv2d  # import Conv1d, ConvTranspose1d, Conv2d from the torch.nn module
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm  # import weight_norm, remove_weight_norm, spectral_norm from the torch.nn.utils module

from commons import init_weights, get_padding  # import init_weights, get_padding from the commons module
from .text import symbols, num_tones, num_languages  # import symbols, num_tones, num_languages from the text module
```

# `D:\src\Bert-VITS2\oldVersion\V110\__init__.py`

```python
"""
1.1 版本兼容
https://github.com/fishaudio/Bert-VITS2/releases/tag/1.1
"""
import torch  # 导入PyTorch库
import commons  # 导入自定义的commons库
from .text.cleaner import clean_text  # 从text包中的cleaner模块导入clean_text函数
from .text import cleaned_text_to_sequence  # 从text包中导入cleaned_text_to_sequence函数
from oldVersion.V111.text import get_bert  # 从oldVersion.V111.text模块导入get_bert函数

def get_text(text, language_str, hps, device):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)  # 调用clean_text函数，返回规范化文本、音素、音调和word2ph
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)  # 调用cleaned_text_to_sequence函数，返回音素、音调和语言

    if hps.data.add_blank:  # 如果hps.data.add_blank为真
        phone = commons.intersperse(phone, 0)  # 调用commons库中的intersperse函数，将phone中的元素间插入0
        tone = commons.intersperse(tone, 0)  # 调用commons库中的intersperse函数，将tone中的元素间插入0
        language = commons.intersperse(language, 0)  # 调用commons库中的intersperse函数，将language中的元素间插入0
        for i in range(len(word2ph)):  # 遍历word2ph的长度
            word2ph[i] = word2ph[i] * 2  # 将word2ph中的每个元素乘以2
        word2ph[0] += 1  # 将word2ph的第一个元素加1
    bert = get_bert(norm_text, word2ph, language_str, device)  # 调用get_bert函数，返回bert
    del word2ph  # 删除word2ph
    assert bert.shape[-1] == len(phone), phone  # 断言bert的最后一个维度长度等于phone的长度

    if language_str == "ZH":  # 如果language_str为"ZH"
        bert = bert  # 将bert赋值给bert
        ja_bert = torch.zeros(768, len(phone))  # 创建一个形状为(768, len(phone))的全0张量
    elif language_str == "JP":  # 如果language_str为"JP"
        ja_bert = bert  # 将bert赋值给ja_bert
        bert = torch.zeros(1024, len(phone))  # 创建一个形状为(1024, len(phone))的全0张量
    else:  # 否则
        bert = torch.zeros(1024, len(phone))  # 创建一个形状为(1024, len(phone))的全0张量
        ja_bert = torch.zeros(768, len(phone))  # 创建一个形状为(768, len(phone))的全0张量
    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"  # 断言bert的最后一个维度长度等于phone的长度

    phone = torch.LongTensor(phone)  # 将phone转换为LongTensor类型
    tone = torch.LongTensor(tone)  # 将tone转换为LongTensor类型
    language = torch.LongTensor(language)  # 将language转换为LongTensor类型
    return bert, ja_bert, phone, tone, language  # 返回bert, ja_bert, phone, tone, language

def infer(
    text,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    sid,
    language,
    hps,
    net_g,
    device,
):
    bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps, device)  # 调用get_text函数，返回bert, ja_bert, phones, tones, lang_ids
    with torch.no_grad():  # 禁用梯度计算
        x_tst = phones.to(device).unsqueeze(0)  # 将phones转换为device上的张量并在0维度上增加维度
        tones = tones.to(device).unsqueeze(0)  # 将tones转换为device上的张量并在0维度上增加维度
        lang_ids = lang_ids.to(device).unsqueeze(0)  # 将lang_ids转换为device上的张量并在0维度上增加维度
        bert = bert.to(device).unsqueeze(0)  # 将bert转换为device上的张量并在0维度上增加维度
        ja_bert = ja_bert.to(device).unsqueeze(0)  # 将ja_bert转换为device上的张量并在0维度上增加维度
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)  # 创建一个包含phones长度的LongTensor张量
        del phones  # 删除phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)  # 创建一个包含sid对应的spk2id的张量
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )  # 调用net_g的infer方法，返回音频
        del x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, ja_bert  # 删除x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, ja_bert
        if torch.cuda.is_available():  # 如果CUDA可用
            torch.cuda.empty_cache()  # 清空CUDA缓存
        return audio  # 返回音频
```

# `D:\src\Bert-VITS2\oldVersion\V110\text\chinese.py`

```python
import os  # 导入os模块
import re  # 导入re模块
import cn2an  # 导入cn2an模块
from pypinyin import lazy_pinyin, Style  # 从pypinyin模块中导入lazy_pinyin和Style
from .symbols import punctuation  # 从symbols模块中导入punctuation
from .tone_sandhi import ToneSandhi  # 从tone_sandhi模块中导入ToneSandhi

current_file_path = os.path.dirname(__file__)  # 获取当前文件所在目录的路径
pinyin_to_symbol_map = {  # 创建pinyin_to_symbol_map字典
    line.split("\t")[0]: line.strip().split("\t")[1]  # 以"\t"分割行并创建字典键值对
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()  # 读取文件内容并遍历行
}

import jieba.posseg as psg  # 导入jieba.posseg模块并重命名为psg

rep_map = {  # 创建rep_map字典
    "：": ",",  # 键值对
    "；": ",",  # 键值对
    # ... 其他键值对
}

tone_modifier = ToneSandhi()  # 创建ToneSandhi类的实例对象

# 定义replace_punctuation函数
def replace_punctuation(text):
    # 函数实现内容
    # ...
    return replaced_text  # 返回替换后的文本

# 定义g2p函数
def g2p(text):
    # 函数实现内容
    # ...
    return phones, tones, word2ph  # 返回phones, tones, word2ph

# 定义_get_initials_finals函数
def _get_initials_finals(word):
    # 函数实现内容
    # ...
    return initials, finals  # 返回initials, finals

# 定义_g2p函数
def _g2p(segments):
    # 函数实现内容
    # ...
    return phones_list, tones_list, word2ph  # 返回phones_list, tones_list, word2ph

# 定义text_normalize函数
def text_normalize(text):
    # 函数实现内容
    # ...
    return text  # 返回处理后的文本

# 定义get_bert_feature函数
def get_bert_feature(text, word2ph):
    # 函数实现内容
    # ...
    return chinese_bert.get_bert_feature(text, word2ph)  # 返回bert特征

# 主程序
if __name__ == "__main__":
    # 主程序实现内容
    # ...
    print(phones, tones, word2ph, bert.shape)  # 打印phones, tones, word2ph, bert.shape
```

# `D:\src\Bert-VITS2\oldVersion\V110\text\chinese_bert.py`

```python
import torch  # 导入PyTorch库
import sys  # 导入sys库
from transformers import AutoTokenizer, AutoModelForMaskedLM  # 从transformers库中导入AutoTokenizer和AutoModelForMaskedLM类

tokenizer = AutoTokenizer.from_pretrained("./bert/chinese-roberta-wwm-ext-large")  # 使用预训练模型初始化tokenizer

def get_bert_feature(text, word2ph, device=None):  # 定义名为get_bert_feature的函数，接受text、word2ph和device三个参数
    if (  # 如果条件判断
        sys.platform == "darwin"  # 当前操作系统为darwin
        and torch.backends.mps.is_available()  # 并且torch的mps后端可用
        and device == "cpu"  # 并且device为cpu
    ):  # 则执行以下代码
        device = "mps"  # 将device设置为"mps"
    if not device:  # 如果device不存在
        device = "cuda"  # 将device设置为"cuda"
    model = AutoModelForMaskedLM.from_pretrained(  # 使用预训练模型初始化model
        "./bert/chinese-roberta-wwm-ext-large"
    ).to(device)  # 并将其移动到指定的device
    with torch.no_grad():  # 使用torch.no_grad()上下文管理器
        inputs = tokenizer(text, return_tensors="pt")  # 使用tokenizer对text进行编码
        for i in inputs:  # 遍历inputs
            inputs[i] = inputs[i].to(device)  # 将inputs[i]移动到指定的device
        res = model(**inputs, output_hidden_states=True)  # 使用model对inputs进行预测
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 对预测结果进行处理
    assert len(word2ph) == len(text) + 2  # 断言word2ph的长度等于text的长度加2
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 初始化phone_level_feature为空列表
    for i in range(len(word2phone)):  # 遍历word2phone
        repeat_feature = res[i].repeat(word2phone[i], 1)  # 对res[i]进行重复操作
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature中
    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 对phone_level_feature进行拼接
    return phone_level_feature.T  # 返回phone_level_feature的转置

if __name__ == "__main__":  # 如果当前脚本作为主程序运行
    import torch  # 导入PyTorch库

    word_level_feature = torch.rand(38, 1024)  # 创建一个38x1024的张量并初始化为随机数
    word2phone = [  # 定义名为word2phone的列表
        1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1
    ]

    total_frames = sum(word2phone)  # 计算word2phone列表中所有元素的和
    print(word_level_feature.shape)  # 打印word_level_feature的形状
    print(word2phone)  # 打印word2phone列表
    phone_level_feature = []  # 初始化phone_level_feature为空列表
    for i in range(len(word2phone)):  # 遍历word2phone列表
        print(word_level_feature[i].shape)  # 打印word_level_feature[i]的形状
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)  # 对word_level_feature[i]进行重复操作
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature中
    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 对phone_level_feature进行拼接
    print(phone_level_feature.shape)  # 打印phone_level_feature的形状
```

# `D:\src\Bert-VITS2\oldVersion\V110\text\cleaner.py`

```python
# 导入模块 chinese, japanese, cleaned_text_to_sequence
from . import chinese, japanese, cleaned_text_to_sequence

# 创建语言模块映射字典
language_module_map = {"ZH": chinese, "JP": japanese}

# 定义函数 clean_text，用于清洗文本
def clean_text(text, language):
    # 根据语言选择对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取文本的音素、音调和词转音素映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 返回规范化文本、音素、音调和词转音素映射
    return norm_text, phones, tones, word2ph

# 定义函数 clean_text_bert，用于对文本进行 BERT 特征提取
def clean_text_bert(text, language):
    # 根据语言选择对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取文本的音素、音调和词转音素映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 获取文本的 BERT 特征
    bert = language_module.get_bert_feature(norm_text, word2ph)
    # 返回音素、音调和 BERT 特征
    return phones, tones, bert

# 定义函数 text_to_sequence，用于将文本转换为序列
def text_to_sequence(text, language):
    # 获取规范化文本、音素、音调和词转音素映射
    norm_text, phones, tones, word2ph = clean_text(text, language)
    # 调用 cleaned_text_to_sequence 函数将音素、音调和语言作为参数，返回序列
    return cleaned_text_to_sequence(phones, tones, language)

# 判断是否为主程序入口
if __name__ == "__main__":
    pass
```

# `D:\src\Bert-VITS2\oldVersion\V110\text\english.py`

```python
import pickle  # 导入pickle模块
import os  # 导入os模块
import re  # 导入re模块
from g2p_en import G2p  # 从g2p_en模块导入G2p类

from . import symbols  # 从当前目录导入symbols模块

current_file_path = os.path.dirname(__file__)  # 获取当前文件所在目录的路径
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")  # 拼接文件路径
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")  # 拼接文件路径
_g2p = G2p()  # 创建G2p对象

# 定义arpa集合
arpa = {
    "AH0",
    "S",
    # ... (省略部分内容)
    "L",
    "SH",
}

# 定义替换函数
def post_replace_ph(ph):
    # 定义替换映射
    rep_map = {
        "：": ",",
        "；": ",",
        # ... (省略部分内容)
        "OY2": "OY1",
        "TH": "TH",
        "HH": "HH",
        "D": "D",
        "ER0": "ER0",
        "CH": "CH",
        "AO1": "AO1",
        "AE1": "AE1",
        "AO2": "AO2",
        "OY1": "OY1",
        "AY2": "AY2",
        "IH1": "IH1",
        "OW0": "OW0",
        "L": "L",
        "SH": "SH",
    }
    # 替换ph
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "UNK"
    return ph

# 读取字典
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

    return g2p_dict

# 缓存字典
def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)

# 获取字典
def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict()
        cache_dict(g2p_dict, CACHE_PATH)

    return g2p_dict

# 获取字典
eng_dict = get_dict()

# 优化ph
def refine_ph(phn):
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    return phn.lower(), tone

# 优化音节
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

# 文本规范化
def text_normalize(text):
    # todo: eng text normalize
    return text

# g2p转换
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

if __name__ == "__main__":
    # print(get_dict())
    # print(eng_word_to_phoneme("hello"))
    print(g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."))
    # all_phones = set()
    # for k, syllables in eng_dict.items():
    #     for group in syllables:
    #         for ph in group:
    #             all_phones.add(ph)
    # print(all_phones)
```

# `D:\src\Bert-VITS2\oldVersion\V110\text\english_bert_mock.py`

```python
import torch  # 导入torch模块

def get_bert_feature(norm_text, word2ph):
    return torch.zeros(1024, sum(word2ph))  # 返回一个大小为1024x(sum(word2ph))的全零张量
```

# `D:\src\Bert-VITS2\oldVersion\V110\text\japanese.py`

```python
# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
import re
import unicodedata

from transformers import AutoTokenizer

from . import punctuation, symbols

try:
    import MeCab
except ImportError as e:
    raise ImportError("Japanese requires mecab-python3 and unidic-lite.") from e
from num2words import num2words

# ... (skipping the rest of the code for brevity)
```

# `D:\src\Bert-VITS2\oldVersion\V110\text\japanese_bert.py`

```python
import torch  # 导入PyTorch库
from transformers import AutoTokenizer, AutoModelForMaskedLM  # 从transformers库中导入AutoTokenizer和AutoModelForMaskedLM类
import sys  # 导入sys模块

tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")  # 使用预训练的tokenizer初始化tokenizer对象


def get_bert_feature(text, word2ph, device=None):  # 定义名为get_bert_feature的函数，接受text、word2ph和device三个参数
    if (
        sys.platform == "darwin"  # 如果操作系统是macOS
        and torch.backends.mps.is_available()  # 并且PyTorch的多进程服务可用
        and device == "cpu"  # 并且设备是CPU
    ):
        device = "mps"  # 将设备设置为"mps"
    if not device:  # 如果设备未指定
        device = "cuda"  # 将设备设置为"cuda"
    model = AutoModelForMaskedLM.from_pretrained("./bert/bert-base-japanese-v3").to(
        device  # 使用预训练的模型初始化model对象，并将其移动到指定的设备上
    )
    with torch.no_grad():  # 使用torch.no_grad()上下文管理器
        inputs = tokenizer(text, return_tensors="pt")  # 使用tokenizer对文本进行编码，并返回PyTorch张量
        for i in inputs:  # 遍历inputs中的键
            inputs[i] = inputs[i].to(device)  # 将inputs中的值移动到指定的设备上
        res = model(**inputs, output_hidden_states=True)  # 使用模型对输入进行推理，输出隐藏状态
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 将模型的输出进行拼接和处理，并移动到CPU上
    assert inputs["input_ids"].shape[-1] == len(word2ph)  # 断言输入的input_ids的最后一个维度长度等于word2ph的长度
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 初始化phone_level_feature为空列表
    for i in range(len(word2phone)):  # 遍历word2phone的长度
        repeat_feature = res[i].repeat(word2phone[i], 1)  # 将res[i]重复word2phone[i]次，并沿着指定维度进行重复
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 将phone_level_feature进行拼接，并沿着指定维度进行拼接

    return phone_level_feature.T  # 返回phone_level_feature的转置
```

# `D:\src\Bert-VITS2\oldVersion\V110\text\symbols.py`

```python
punctuation = ["!", "?", "…", ",", ".", "'", "-"]  # 创建一个包含标点符号的列表
pu_symbols = punctuation + ["SP", "UNK"]  # 创建一个包含标点符号和特殊标记的列表
pad = "_"  # 创建一个填充符号

# chinese
zh_symbols = [  # 创建一个包含中文音节的列表
    "E",
    "En",
    ...
    "OO",
]
num_zh_tones = 6  # 中文音节的数量

# japanese
ja_symbols = [  # 创建一个包含日文音节的列表
    "N",
    "a",
    ...
    "zy",
]
num_ja_tones = 1  # 日文音节的数量

# English
en_symbols = [  # 创建一个包含英文音节的列表
    "aa",
    "ae",
    ...
    "zh",
]
num_en_tones = 4  # 英文音节的数量

# combine all symbols
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))  # 合并所有音节并去重排序
symbols = [pad] + normal_symbols + pu_symbols  # 创建一个包含填充符号、所有音节和标点符号的列表
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]  # 获取标点符号在列表中的索引

# combine all tones
num_tones = num_zh_tones + num_ja_tones + num_en_tones  # 计算所有语言的音节数量

# language maps
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}  # 创建一个语言到ID的映射
num_languages = len(language_id_map.keys())  # 计算语言的数量

language_tone_start_map = {  # 创建一个语言到音节起始位置的映射
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
}

if __name__ == "__main__":
    a = set(zh_symbols)  # 创建一个包含中文音节的集合
    b = set(en_symbols)  # 创建一个包含英文音节的集合
    print(sorted(a & b))  # 打印中英文音节的交集
```

# `D:\src\Bert-VITS2\oldVersion\V110\text\tone_sandhi.py`

```python
# the meaning of jieba pos tag: https://blog.csdn.net/weixin_44174352/article/details/113731041
# e.g.
# word: "家里"
# pos: "s"
# finals: ['ia1', 'i3']
def _neural_sandhi(self, word: str, pos: str, finals: List[str]) -> List[str]:
    # reduplication words for n. and v. e.g. 奶奶, 试试, 旺旺
    for j, item in enumerate(word):
        if (
            j - 1 >= 0
            and item == word[j - 1]
            and pos[0] in {"n", "v", "a"}
            and word not in self.must_not_neural_tone_words
        ):
            finals[j] = finals[j][:-1] + "5"
    ge_idx = word.find("个")
    if len(word) >= 1 and word[-1] in "吧呢啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶":
        finals[-1] = finals[-1][:-1] + "5"
    elif len(word) >= 1 and word[-1] in "的地得":
        finals[-1] = finals[-1][:-1] + "5"
    # e.g. 走了, 看着, 去过
    # elif len(word) == 1 and word in "了着过" and pos in {"ul", "uz", "ug"}:
    #     finals[-1] = finals[-1][:-1] + "5"
    elif (
        len(word) > 1
        and word[-1] in "们子"
        and pos in {"r", "n"}
        and word not in self.must_not_neural_tone_words
    ):
        finals[-1] = finals[-1][:-1] + "5"
    # e.g. 桌上, 地下, 家里
    elif len(word) > 1 and word[-1] in "上下里" and pos in {"s", "l", "f"}:
        finals[-1] = finals[-1][:-1] + "5"
    # e.g. 上来, 下去
    elif len(word) > 1 and word[-1] in "来去" and word[-2] in "上下进出回过起开":
        finals[-1] = finals[-1][:-1] + "5"
    # 个做量词
    elif (
        ge_idx >= 1
        and (word[ge_idx - 1].isnumeric() or word[ge_idx - 1] in "几有两半多各整每做是")
    ) or word == "个":
        finals[ge_idx] = finals[ge_idx][:-1] + "5"
    else:
        if (
            word in self.must_neural_tone_words
            or word[-2:] in self.must_neural_tone_words
        ):
            finals[-1] = finals[-1][:-1] + "5"

    word_list = self._split_word(word)
    finals_list = [finals[: len(word_list[0])], finals[len(word_list[0]) :]]
    for i, word in enumerate(word_list):
        # conventional neural in Chinese
        if (
            word in self.must_neural_tone_words
            or word[-2:] in self.must_neural_tone_words
        ):
            finals_list[i][-1] = finals_list[i][-1][:-1] + "5"
    finals = sum(finals_list, [])
    return finals
```

# `D:\src\Bert-VITS2\oldVersion\V110\text\__init__.py`

```python
# Import all symbols from the symbols module
from .symbols import *

# Create a dictionary that maps symbols to their corresponding IDs
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

# Define a function that converts a string of text to a sequence of IDs corresponding to the symbols in the text
def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    # Convert each symbol in the cleaned text to its corresponding ID using the _symbol_to_id dictionary
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    # Calculate the tone start index based on the language
    tone_start = language_tone_start_map[language]
    # Add the tone start index to each tone in the input
    tones = [i + tone_start for i in tones]
    # Get the language ID based on the language
    lang_id = language_id_map[language]
    # Create a list of language IDs corresponding to the number of symbols in the input
    lang_ids = [lang_id for i in phones]
    # Return the list of symbol IDs, tones, and language IDs
    return phones, tones, lang_ids

# Define a function that retrieves BERT features for a given normalized text, word-to-phoneme mapping, language, and device
def get_bert(norm_text, word2ph, language, device):
    # Import BERT feature extraction functions for Chinese, English, and Japanese
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert_mock import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert

    # Create a mapping of language codes to their corresponding BERT feature extraction functions
    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}
    # Retrieve the BERT features for the given language using the corresponding BERT feature extraction function
    bert = lang_bert_func_map[language](norm_text, word2ph, device)
    # Return the BERT features
    return bert
```

# `D:\src\Bert-VITS2\oldVersion\V111\models.py`

```python
import math  # import the math module
import torch  # import the torch module
from torch import nn  # import the nn module from the torch module
from torch.nn import functional as F  # import the functional module from the nn module

import commons  # import the commons module
import modules  # import the modules module
import attentions  # import the attentions module
import monotonic_align  # import the monotonic_align module

from torch.nn import Conv1d, ConvTranspose1d, Conv2d  # import Conv1d, ConvTranspose1d, Conv2d from the torch.nn module
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm  # import weight_norm, remove_weight_norm, spectral_norm from the torch.nn.utils module

from commons import init_weights, get_padding  # import init_weights, get_padding from the commons module
from .text import symbols, num_tones, num_languages  # import symbols, num_tones, num_languages from the text module
```

# `D:\src\Bert-VITS2\oldVersion\V111\__init__.py`

```python
"""
1.1.1版本兼容
https://github.com/fishaudio/Bert-VITS2/releases/tag/1.1.1
"""
```

# `D:\src\Bert-VITS2\oldVersion\V111\text\chinese.py`

```python
import os  # 导入os模块
import re  # 导入re模块
import cn2an  # 导入cn2an模块
from pypinyin import lazy_pinyin, Style  # 从pypinyin模块中导入lazy_pinyin和Style
from .symbols import punctuation  # 从symbols模块中导入punctuation
from .tone_sandhi import ToneSandhi  # 从tone_sandhi模块中导入ToneSandhi

current_file_path = os.path.dirname(__file__)  # 获取当前文件路径
pinyin_to_symbol_map = {  # 创建pinyin_to_symbol_map字典
    line.split("\t")[0]: line.strip().split("\t")[1]  # 从文件中读取内容并生成字典
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

import jieba.posseg as psg  # 导入jieba.posseg模块并重命名为psg

rep_map = {  # 创建rep_map字典
    "：": ",",  # 键值对
    "；": ",",  # 键值对
    # ... 其他键值对
}

tone_modifier = ToneSandhi()  # 创建ToneSandhi实例

# 定义replace_punctuation函数
def replace_punctuation(text):
    # 函数实现内容
    return replaced_text  # 返回替换后的文本

# 定义g2p函数
def g2p(text):
    # 函数实现内容
    return phones, tones, word2ph  # 返回phones, tones, word2ph

# 定义_get_initials_finals函数
def _get_initials_finals(word):
    # 函数实现内容
    return initials, finals  # 返回initials, finals

# 定义_g2p函数
def _g2p(segments):
    # 函数实现内容
    return phones_list, tones_list, word2ph  # 返回phones_list, tones_list, word2ph

# 定义text_normalize函数
def text_normalize(text):
    # 函数实现内容
    return text  # 返回处理后的文本

# 定义get_bert_feature函数
def get_bert_feature(text, word2ph):
    # 函数实现内容
    return chinese_bert.get_bert_feature(text, word2ph)  # 返回bert特征

if __name__ == "__main__":
    # 主程序
    text = "啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = text_normalize(text)  # 文本规范化
    print(text)  # 打印处理后的文本
    phones, tones, word2ph = g2p(text)  # 获取音素、音调和word2ph
    bert = get_bert_feature(text, word2ph)  # 获取bert特征
    print(phones, tones, word2ph, bert.shape)  # 打印phones, tones, word2ph和bert特征
```

# `D:\src\Bert-VITS2\oldVersion\V111\text\chinese_bert.py`

```python
word_level_feature = torch.rand(38, 1024)  # 12个词,每个词1024维特征
word2phone = [
    1,
    2,
    1,
    2,
    2,
    1,
    2,
    2,
    1,
    2,
    2,
    1,
    2,
    2,
    2,
    2,
    2,
    1,
    1,
    2,
    2,
    1,
    2,
    2,
    2,
    2,
    1,
    2,
    2,
    2,
    2,
    2,
    1,
    2,
    2,
    2,
    2,
    1,
]

# 计算总帧数
total_frames = sum(word2phone)
print(word_level_feature.shape)
print(word2phone)
phone_level_feature = []
for i in range(len(word2phone)):
    print(word_level_feature[i].shape)

    # 对每个词重复word2phone[i]次
    repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)
    phone_level_feature.append(repeat_feature)

phone_level_feature = torch.cat(phone_level_feature, dim=0)
print(phone_level_feature.shape)  # torch.Size([36, 1024])
```