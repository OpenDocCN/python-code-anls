# Bert-VITS2 源码解析 7

# `D:\src\Bert-VITS2\onnx_modules\V230\models_onnx.py`

```python
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
from . import attentions_onnx


from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from commons import init_weights, get_padding
from .text import symbols, num_tones, num_languages


class DurationDiscriminator(nn.Module):  # vits2
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels  # input channel
        self.filter_channels = filter_channels  # filter channel
        self.kernel_size = kernel_size  # kernel size
        self.p_dropout = p_dropout  # dropout probability
        self.gin_channels = gin_channels  # gin channels

        self.drop = nn.Dropout(p_dropout)  # dropout layer
        self.conv_1 = nn.Conv1d(  # 1D convolutional layer
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)  # layer normalization
        self.conv_2 = nn.Conv1d(  # 1D convolutional layer
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)  # layer normalization
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)  # 1D convolutional layer

        self.LSTM = nn.LSTM(  # LSTM layer
            2 * filter_channels, filter_channels, batch_first=True, bidirectional=True
        )

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)  # 1D convolutional layer

        self.output_layer = nn.Sequential(  # Sequential layer
            nn.Linear(2 * filter_channels, 1), nn.Sigmoid()  # Linear layer, Sigmoid activation function
        )

    def forward_probability(self, x, dur):
        dur = self.dur_proj(dur)  # dur projection
        x = torch.cat([x, dur], dim=1)  # concatenate
        x = x.transpose(1, 2)  # transpose
        x, _ = self.LSTM(x)  # LSTM layer
        output_prob = self.output_layer(x)  # output layer
        return output_prob

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)  # detach tensor
        if g is not None:
            g = torch.detach(g)  # detach tensor
            x = x + self.cond(g)  # add tensor
        x = self.conv_1(x * x_mask)  # convolutional layer
        x = torch.relu(x)  # ReLU activation function
        x = self.norm_1(x)  # layer normalization
        x = self.drop(x)  # dropout layer
        x = self.conv_2(x * x_mask)  # convolutional layer
        x = torch.relu(x)  # ReLU activation function
        x = self.norm_2(x)  # layer normalization
        x = self.drop(x)  # dropout layer

        output_probs = []  # empty list
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, dur)  # forward probability
            output_probs.append(output_prob)  # append output probability

        return output_probs  # return output probabilities
```

# `D:\src\Bert-VITS2\onnx_modules\V230\__init__.py`

```python
from .text.symbols import symbols  # 从text模块中的symbols文件中导入symbols变量
from .models_onnx import SynthesizerTrn  # 从models_onnx模块中导入SynthesizerTrn类

__all__ = ["symbols", "SynthesizerTrn"]  # 定义__all__变量，包含symbols和SynthesizerTrn，用于模块导入时指定可导入的内容
```

# `D:\src\Bert-VITS2\onnx_modules\V230\text\symbols.py`

```python
punctuation = ["!", "?", "…", ",", ".", "'", "-"]  # Define a list of punctuation marks
pu_symbols = punctuation + ["SP", "UNK"]  # Combine the punctuation list with special symbols
pad = "_"  # Define a padding symbol

# Define Chinese symbols
zh_symbols = [
    # ... (list of Chinese symbols)
]
num_zh_tones = 6  # Define the number of Chinese tones

# Define Japanese symbols
ja_symbols = [
    # ... (list of Japanese symbols)
]
num_ja_tones = 2  # Define the number of Japanese tones

# Define English symbols
en_symbols = [
    # ... (list of English symbols)
]
num_en_tones = 4  # Define the number of English tones

# Combine all symbols
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))  # Combine all language symbols and sort them
symbols = [pad] + normal_symbols + pu_symbols  # Combine symbols with padding and special symbols
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]  # Get the index of special symbols in the combined symbols list

# Combine all tones
num_tones = num_zh_tones + num_ja_tones + num_en_tones  # Calculate the total number of tones

# Define language maps
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}  # Map language IDs to numerical values
num_languages = len(language_id_map.keys())  # Get the total number of languages

language_tone_start_map = {
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
}  # Map the starting index of tones for each language

if __name__ == "__main__":
    a = set(zh_symbols)  # Create a set of Chinese symbols
    b = set(en_symbols)  # Create a set of English symbols
    print(sorted(a & b))  # Print the intersection of Chinese and English symbols
```

# `D:\src\Bert-VITS2\onnx_modules\V230\text\__init__.py`

```python
from .symbols import *  # 从symbols模块中导入所有的变量和函数
```

# `D:\src\Bert-VITS2\onnx_modules\V230_OnnxInference\__init__.py`

```python
import numpy as np  # 导入numpy库
import onnxruntime as ort  # 导入onnxruntime库


def convert_pad_shape(pad_shape):  # 定义函数convert_pad_shape
    layer = pad_shape[::-1]  # 反转pad_shape
    pad_shape = [item for sublist in layer for item in sublist]  # 将layer展开成一维数组
    return pad_shape  # 返回pad_shape


def sequence_mask(length, max_length=None):  # 定义函数sequence_mask
    if max_length is None:  # 如果max_length为空
        max_length = length.max()  # max_length取length的最大值
    x = np.arange(max_length, dtype=length.dtype)  # 生成一个长度为max_length的数组x
    return np.expand_dims(x, 0) < np.expand_dims(length, 1)  # 返回x和length的比较结果


def generate_path(duration, mask):  # 定义函数generate_path
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape  # 获取mask的形状
    cum_duration = np.cumsum(duration, -1)  # 对duration进行累加

    cum_duration_flat = cum_duration.reshape(b * t_x)  # 将cum_duration展开成一维数组
    path = sequence_mask(cum_duration_flat, t_y)  # 生成path
    path = path.reshape(b, t_x, t_y)  # 调整path的形状
    path = path ^ np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]  # 对path进行异或操作
    path = np.expand_dims(path, 1).transpose(0, 1, 3, 2)  # 调整path的形状
    return path  # 返回path


class OnnxInferenceSession:  # 定义类OnnxInferenceSession
    def __init__(self, path, Providers=["CPUExecutionProvider"]):  # 定义初始化函数
        self.enc = ort.InferenceSession(path["enc"], providers=Providers)  # 初始化self.enc
        self.emb_g = ort.InferenceSession(path["emb_g"], providers=Providers)  # 初始化self.emb_g
        self.dp = ort.InferenceSession(path["dp"], providers=Providers)  # 初始化self.dp
        self.sdp = ort.InferenceSession(path["sdp"], providers=Providers)  # 初始化self.sdp
        self.flow = ort.InferenceSession(path["flow"], providers=Providers)  # 初始化self.flow
        self.dec = ort.InferenceSession(path["dec"], providers=Providers)  # 初始化self.dec

    def __call__(  # 定义__call__函数
        self,
        seq,
        tone,
        language,
        bert_zh,
        bert_jp,
        bert_en,
        sid,
        seed=114514,
        seq_noise_scale=0.8,
        sdp_noise_scale=0.6,
        length_scale=1.0,
        sdp_ratio=0.0,
    ):
        if seq.ndim == 1:  # 如果seq的维度为1
            seq = np.expand_dims(seq, 0)  # 将seq扩展为二维���组
        if tone.ndim == 1:  # 如果tone的维度为1
            tone = np.expand_dims(tone, 0)  # 将tone扩展为二维数组
        if language.ndim == 1:  # 如果language的维度为1
            language = np.expand_dims(language, 0)  # 将language扩展为二维数组
        assert (seq.ndim == 2, tone.ndim == 2, language.ndim == 2)  # 断言seq、tone、language的维度为2
        g = self.emb_g.run(  # 运行self.emb_g
            None,
            {
                "sid": sid.astype(np.int64),  # 传入参数sid
            },
        )[0]  # 获取返回值的第一个元素
        g = np.expand_dims(g, -1)  # 将g扩展为三维数组
        enc_rtn = self.enc.run(  # 运行self.enc
            None,
            {
                "x": seq.astype(np.int64),  # 传入参数seq
                "t": tone.astype(np.int64),  # 传入参数tone
                "language": language.astype(np.int64),  # 传入参数language
                "bert_0": bert_zh.astype(np.float32),  # 传入参数bert_zh
                "bert_1": bert_jp.astype(np.float32),  # 传入参数bert_jp
                "bert_2": bert_en.astype(np.float32),  # 传入参数bert_en
                "g": g.astype(np.float32),  # 传入参数g
            },
        )
        x, m_p, logs_p, x_mask = enc_rtn[0], enc_rtn[1], enc_rtn[2], enc_rtn[3]  # 获取enc_rtn的四个元素
        np.random.seed(seed)  # 设置随机数种子
        zinput = np.random.randn(x.shape[0], 2, x.shape[2]) * sdp_noise_scale  # 生成zinput
        logw = self.sdp.run(  # 运行self.sdp
            None, {"x": x, "x_mask": x_mask, "zin": zinput.astype(np.float32), "g": g}
        )[0] * (sdp_ratio) + self.dp.run(None, {"x": x, "x_mask": x_mask, "g": g})[
            0
        ] * (
            1 - sdp_ratio
        )  # 计算logw
        w = np.exp(logw) * x_mask * length_scale  # 计算w
        w_ceil = np.ceil(w)  # 对w进行向上取整
        y_lengths = np.clip(np.sum(w_ceil, (1, 2)), a_min=1.0, a_max=100000).astype(
            np.int64
        )  # 计算y_lengths
        y_mask = np.expand_dims(sequence_mask(y_lengths, None), 1)  # 生成y_mask
        attn_mask = np.expand_dims(x_mask, 2) * np.expand_dims(y_mask, -1)  # 生成attn_mask
        attn = generate_path(w_ceil, attn_mask)  # 生成attn
        m_p = np.matmul(attn.squeeze(1), m_p.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )  # 计算m_p
        logs_p = np.matmul(attn.squeeze(1), logs_p.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )  # 计算logs_p

        z_p = (
            m_p
            + np.random.randn(m_p.shape[0], m_p.shape[1], m_p.shape[2])
            * np.exp(logs_p)
            * seq_noise_scale
        )  # 计算z_p

        z = self.flow.run(  # 运行self.flow
            None,
            {
                "z_p": z_p.astype(np.float32),  # 传入参数z_p
                "y_mask": y_mask.astype(np.float32),  # 传入参数y_mask
                "g": g,  # 传入参数g
            },
        )[0]  # 获取返回值的第一个元素

        return self.dec.run(None, {"z_in": z.astype(np.float32), "g": g})[0]  # 运行self.dec并返回结果
```

# `D:\src\Bert-VITS2\text\bert_utils.py`

```python
from pathlib import Path  # 导入Path类

from huggingface_hub import hf_hub_download  # 从huggingface_hub模块导入hf_hub_download函数

from config import config  # 从config模块导入config对象


MIRROR: str = config.mirror  # 从config对象中获取mirror属性，并赋值给MIRROR变量


def _check_bert(repo_id, files, local_path):  # 定义_check_bert函数，接受repo_id、files、local_path三个参数
    for file in files:  # 遍历files列表
        if not Path(local_path).joinpath(file).exists():  # 如果local_path下的file文件不存在
            if MIRROR.lower() == "openi":  # 如果MIRROR的值转换为小写后等于"openi"
                import openi  # 导入openi模块

                openi.model.download_model(  # 调用openi.model.download_model函数
                    "Stardust_minus/Bert-VITS2", repo_id.split("/")[-1], "./bert"  # 传入三个参数
                )
            else:  # 否则
                hf_hub_download(  # 调用hf_hub_download函数
                    repo_id, file, local_dir=local_path, local_dir_use_symlinks=False  # 传入四个参数
                )
```

# `D:\src\Bert-VITS2\text\chinese.py`

```python
import os  # 导入os模块
import re  # 导入re模块
import cn2an  # 导入cn2an模块
from pypinyin import lazy_pinyin, Style  # 从pypinyin模块中导入lazy_pinyin和Style
from text.symbols import punctuation  # 从text.symbols模块中导入punctuation
from text.tone_sandhi import ToneSandhi  # 从text.tone_sandhi模块中导入ToneSandhi

current_file_path = os.path.dirname(__file__)  # 获取当前文件路径
pinyin_to_symbol_map = {  # 创建pinyin_to_symbol_map字典
    line.split("\t")[0]: line.strip().split("\t")[1]  # 从文件中读取每行并添加到字典中
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

import jieba.posseg as psg  # 导入jieba.posseg模块并重命名为psg

rep_map = {  # 创建rep_map字典
    "：": ",",  # 添加键值对
    "；": ",",  # 添加键值对
    # ... 其他键值对
}

tone_modifier = ToneSandhi()  # 创建ToneSandhi对象

# ... 其他函数定义

if __name__ == "__main__":
    from text.chinese_bert import get_bert_feature  # 从text.chinese_bert模块中导入get_bert_feature

    text = "啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = text_normalize(text)  # 对文本进行规范化处理
    print(text)  # 打印处理后的文本
    phones, tones, word2ph = g2p(text)  # 对文本进行g2p处理
    bert = get_bert_feature(text, word2ph)  # 获取文本的bert特征

    print(phones, tones, word2ph, bert.shape)  # 打印处理后的phones、tones、word2ph和bert特征
```

# `D:\src\Bert-VITS2\text\chinese_bert.py`

```python
if __name__ == "__main__":
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

# `D:\src\Bert-VITS2\text\cleaner.py`

```python
# 导入text模块中的chinese、japanese、english、cleaned_text_to_sequence函数
from text import chinese, japanese, english, cleaned_text_to_sequence

# 创建语言模块映射字典
language_module_map = {"ZH": chinese, "JP": japanese, "EN": english}

# 定义clean_text函数，用于清洗文本
def clean_text(text, language):
    # 根据语言选择对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取文本的音素、音调和词转音素映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph

# 定义clean_text_bert函数，用于对文本进行BERT特征提取
def clean_text_bert(text, language):
    # 根据语言选择对应的语言模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取文本的音素、音调和词转音素映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 获取文本的BERT特征
    bert = language_module.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert

# 定义text_to_sequence函数，用于将文本转换为序列
def text_to_sequence(text, language):
    # 对文本进行清洗处理
    norm_text, phones, tones, word2ph = clean_text(text, language)
    # 将清洗后的文本转换为序列
    return cleaned_text_to_sequence(phones, tones, language)

# 判断是否为主程序入口
if __name__ == "__main__":
    pass
```

# `D:\src\Bert-VITS2\text\english.py`

```python
import pickle  # 导入pickle模块
import os  # 导入os模块
import re  # 导入re模块
from g2p_en import G2p  # 从g2p_en模块导入G2p类
from transformers import DebertaV2Tokenizer  # 从transformers模块导入DebertaV2Tokenizer类
from text import symbols  # 从text模块导入symbols
from text.symbols import punctuation  # 从text.symbols模块导入punctuation

current_file_path = os.path.dirname(__file__)  # 获取当前文件路径
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")  # 拼接文件路径
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")  # 拼接文件路径
_g2p = G2p()  # 创建G2p对象
LOCAL_PATH = "./bert/deberta-v3-large"  # 设置本地路径
tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)  # 从预训练模型中加载tokenizer

# 定义arpa集合
arpa = {
    "AH0",
    "S",
    # ... (省略部分内容)
    "ʤ",
}

# 定义post_replace_ph函数，用于替换音素
def post_replace_ph(ph):
    # 定义替换映射表
    rep_map = {
        "：": ",",
        "；": ",",
        # ... (省略部分内容)
        "v": "V",
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "UNK"
    return ph

# 定义替换标点符号的函数
def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    return replaced_text

# 读取词典
def read_dict():
    # ... (函数实现部分)

# 缓存词典
def cache_dict(g2p_dict, file_path):
    # ... (函数实现部分)

# 获取词典
def get_dict():
    # ... (函数实现部分)

eng_dict = get_dict()  # 获取英文词典

# 优化音素
def refine_ph(phn):
    # ... (函数实现部分)

# 优化音节
def refine_syllables(syllables):
    # ... (函数实现部分)

# 定义文本规范化函数
def text_normalize(text):
    # ... (函数实现部分)

# 分配音素
def distribute_phone(n_phone, n_word):
    # ... (函数实现部分)

# 分割文本
def sep_text(text):
    # ... (函数实现部分)

# 文本转换为单词
def text_to_words(text):
    # ... (函数实现部分)

# 文本转换为音素
def g2p(text):
    # ... (函数实现部分)

# 获取BERT特征
def get_bert_feature(text, word2ph):
    # ... (函数实现部分)

# 执行测试
if __name__ == "__main__":
    print(g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."))  # 打印文本的音素表示
```

# `D:\src\Bert-VITS2\text\english_bert_mock.py`

```python
import sys  # 导入sys模块

import torch  # 导入torch模块
from transformers import DebertaV2Model, DebertaV2Tokenizer  # 从transformers模块中导入DebertaV2Model和DebertaV2Tokenizer类

from config import config  # 从config模块中导入config类

LOCAL_PATH = "./bert/deberta-v3-large"  # 设置本地路径为"./bert/deberta-v3-large"

tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)  # 使用DebertaV2Tokenizer类从预训练模型中加载tokenizer

models = dict()  # 创建一个空字典models

def get_bert_feature(
    text,
    word2ph,
    device=config.bert_gen_config.device,
    style_text=None,
    style_weight=0.7,
):
    if (
        sys.platform == "darwin"  # 如果系统平台是darwin
        and torch.backends.mps.is_available()  # 并且torch后端支持MPS
        and device == "cpu"  # 并且设备是CPU
    ):
        device = "mps"  # 将设备设置为"mps"
    if not device:  # 如果设备不存在
        device = "cuda"  # 将设备设置为"cuda"
    if device not in models.keys():  # 如果设备不在models的键中
        models[device] = DebertaV2Model.from_pretrained(LOCAL_PATH).to(device)  # 从预训练模型中加载DebertaV2Model，并将其移动到指定设备
    with torch.no_grad():  # 使用torch.no_grad()上下文管理器
        inputs = tokenizer(text, return_tensors="pt")  # 使用tokenizer对文本进行编码
        for i in inputs:  # 遍历inputs
            inputs[i] = inputs[i].to(device)  # 将inputs的值移动到指定设备
        res = models[device](**inputs, output_hidden_states=True)  # 使用models对inputs进行预测，输出隐藏状态
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 将隐藏状态拼接并移动到CPU
        if style_text:  # 如果存在style_text
            style_inputs = tokenizer(style_text, return_tensors="pt")  # 使用tokenizer对style_text进行编码
            for i in style_inputs:  # 遍历style_inputs
                style_inputs[i] = style_inputs[i].to(device)  # 将style_inputs的值移动到指定设备
            style_res = models[device](**style_inputs, output_hidden_states=True)  # 使用models对style_inputs进行预测，输出隐藏状态
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()  # 将隐藏状态拼接并移动到CPU
            style_res_mean = style_res.mean(0)  # 计算style_res的均值
    assert len(word2ph) == res.shape[0], (text, res.shape[0], len(word2ph))  # 断言word2ph的长度等于res的行数
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 创建一个空列表phone_level_feature
    for i in range(len(word2phone)):  # 遍历word2phone的长度
        if style_text:  # 如果存在style_text
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)  # 计算重复特征
                + style_res_mean.repeat(word2phone[i], 1) * style_weight  # 计算重复特征
            )
        else:  # 如果不存在style_text
            repeat_feature = res[i].repeat(word2phone[i], 1)  # 计算重复特征
        phone_level_feature.append(repeat_feature)  # 将重复特征添加到phone_level_feature中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 沿着行方向拼接phone_level_feature

    return phone_level_feature.T  # 返回phone_level_feature的转置
```

# `D:\src\Bert-VITS2\text\japanese.py`

```python
# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
import re  # Import the regular expression module
import unicodedata  # Import the unicodedata module

from transformers import AutoTokenizer  # Import the AutoTokenizer class from the transformers module

from text import punctuation, symbols  # Import punctuation and symbols from the text module

from num2words import num2words  # Import the num2words function from the num2words module

import pyopenjtalk  # Import the pyopenjtalk module
import jaconv  # Import the jaconv module

# Define the function kata2phoneme
def kata2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""
    # ... (function implementation)

# Define the function hira2kata
def hira2kata(text: str) -> str:
    # ... (function implementation)

# Define the regular expression pattern for marks
_MARKS = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# Define the function text2kata
def text2kata(text: str) -> str:
    # ... (function implementation)

# Define the function text2sep_kata
def text2sep_kata(text: str) -> (list, list):
    # ... (function implementation)

# Define the function get_accent
def get_accent(parsed):
    # ... (function implementation)

# Define the _ALPHASYMBOL_YOMI dictionary
_ALPHASYMBOL_YOMI = {
    # ... (dictionary content)
}

# Define the regular expression pattern for numbers with separators
_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")
# Define the _CURRENCY_MAP dictionary
_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
# Define the regular expression pattern for currency
_CURRENCY_RX = re.compile(r"([$¥£€])([0-9.]*[0-9])")
# Define the regular expression pattern for numbers
_NUMBER_RX = re.compile(r"[0-9]+(\.[0-9]+)?")

# Define the function japanese_convert_numbers_to_words
def japanese_convert_numbers_to_words(text: str) -> str:
    # ... (function implementation)

# Define the function japanese_convert_alpha_symbols_to_words
def japanese_convert_alpha_symbols_to_words(text: str) -> str:
    # ... (function implementation)

# Define the function japanese_text_to_phonemes
def japanese_text_to_phonemes(text: str) -> str:
    # ... (function implementation)

# Define the function is_japanese_character
def is_japanese_character(char):
    # ... (function implementation)

# Define the rep_map dictionary
rep_map = {
    # ... (dictionary content)
}

# Define the function replace_punctuation
def replace_punctuation(text):
    # ... (function implementation)

# Define the function text_normalize
def text_normalize(text):
    # ... (function implementation)

# Define the function distribute_phone
def distribute_phone(n_phone, n_word):
    # ... (function implementation)

# Define the function handle_long
def handle_long(sep_phonemes):
    # ... (function implementation)

# Create an instance of AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese-char-wwm")

# Define the function align_tones
def align_tones(phones, tones):
    # ... (function implementation)

# Define the function rearrange_tones
def rearrange_tones(tones, phones):
    # ... (function implementation)

# Define the function g2p
def g2p(norm_text):
    # ... (function implementation)

# Check if the script is being run as the main program
if __name__ == "__main__":
    # Create an instance of AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese")
    text = "hello,こんにちは、世界ー！……"
    from text.japanese_bert import get_bert_feature

    text = text_normalize(text)
    print(text)

    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)

    print(phones, tones, word2ph, bert.shape)
```

# `D:\src\Bert-VITS2\text\japanese_bert.py`

```python
import sys  # 导入sys模块

import torch  # 导入torch模块
from transformers import AutoModelForMaskedLM, AutoTokenizer  # 从transformers模块中导入AutoModelForMaskedLM和AutoTokenizer类

from config import config  # 从config模块中导入config类
from text.japanese import text2sep_kata  # 从text.japanese模块中导入text2sep_kata函数

LOCAL_PATH = "./bert/deberta-v2-large-japanese-char-wwm"  # 设置LOCAL_PATH变量为"./bert/deberta-v2-large-japanese-char-wwm"

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)  # 使用AutoTokenizer类从预训练模型路径LOCAL_PATH中加载tokenizer

models = dict()  # 创建一个空的字典models

def get_bert_feature(
    text,
    word2ph,
    device=config.bert_gen_config.device,
    style_text=None,
    style_weight=0.7,
):
    text = "".join(text2sep_kata(text)[0])  # 将text转换为片假名并连接成字符串
    if style_text:
        style_text = "".join(text2sep_kata(style_text)[0])  # 如果style_text存在，将style_text转换为片假名并连接成字符串
    if (
        sys.platform == "darwin"  # 如果系统平台是darwin
        and torch.backends.mps.is_available()  # 并且torch的mps后端可用
        and device == "cpu"  # 并且设备是cpu
    ):
        device = "mps"  # 将device设置为"mps"
    if not device:  # 如果device不存在
        device = "cuda"  # 将device设置为"cuda"
    if device not in models.keys():  # 如果device不在models的键中
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)  # 从预训练模型路径LOCAL_PATH中加载AutoModelForMaskedLM模型，并将其移到device上
    with torch.no_grad():  # 使用torch.no_grad()上下文管理器
        inputs = tokenizer(text, return_tensors="pt")  # 使用tokenizer对text进行编码并返回张量
        for i in inputs:  # 遍历inputs
            inputs[i] = inputs[i].to(device)  # 将inputs[i]移动到device上
        res = models[device](**inputs, output_hidden_states=True)  # 使用models[device]对inputs进行预测，并返回隐藏状态
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 将隐藏状态拼接并转移到CPU上
        if style_text:  # 如果style_text存在
            style_inputs = tokenizer(style_text, return_tensors="pt")  # 使用tokenizer对style_text进行编码并返回张量
            for i in style_inputs:  # 遍历style_inputs
                style_inputs[i] = style_inputs[i].to(device)  # 将style_inputs[i]移动到device上
            style_res = models[device](**style_inputs, output_hidden_states=True)  # 使用models[device]对style_inputs进行预测，并返回隐藏状态
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()  # 将隐藏状态拼接并转移到CPU上
            style_res_mean = style_res.mean(0)  # 计算style_res的均值

    assert len(word2ph) == len(text) + 2  # 断言word2ph的长度等于text的长度加2
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 创建一个空列表phone_level_feature
    for i in range(len(word2phone)):  # 遍历word2phone的长度
        if style_text:  # 如果style_text存在
            repeat_feature = (  # 计算repeat_feature
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)  # 重复res[i]并乘以(1 - style_weight)
                + style_res_mean.repeat(word2phone[i], 1) * style_weight  # 加上重复style_res_mean并乘以style_weight
            )
        else:  # 如果style_text不存在
            repeat_feature = res[i].repeat(word2phone[i], 1)  # 重复res[i]
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 沿着0维度拼接phone_level_feature

    return phone_level_feature.T  # 返回phone_level_feature的转置
```

# `D:\src\Bert-VITS2\text\symbols.py`

```python
punctuation = ["!", "?", "…", ",", ".", "'", "-"]  # 创建一个包含标点符号的列表
pu_symbols = punctuation + ["SP", "UNK"]  # 创建一个包含标点符号和特殊标记的列表
pad = "_"  # 创建一个填充标记

# chinese
zh_symbols = [  # 创建一个包含中文音节的列表
    "E",
    "En",
    ...
    "OO",
]
num_zh_tones = 6  # 设置中文音节的音调数量为6

# japanese
ja_symbols = [  # 创建一个包含日文音节的列表
    "N",
    "a",
    ...
    "zy",
]
num_ja_tones = 2  # 设置日文音节的音调数量为2

# English
en_symbols = [  # 创建一个包含英文音素的列表
    "aa",
    "ae",
    ...
    "zh",
]
num_en_tones = 4  # 设置英文音素的音调数量为4

# combine all symbols
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))  # 合并所有音节并去重排序
symbols = [pad] + normal_symbols + pu_symbols  # 创建一个包含填充标记、所有音节和特殊标记的列表
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]  # 获取特殊标记在列表中的索引

# combine all tones
num_tones = num_zh_tones + num_ja_tones + num_en_tones  # 计算所有音节的总音调数量

# language maps
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}  # 创建一个语言到ID的映射字典
num_languages = len(language_id_map.keys())  # 获取语言数量

language_tone_start_map = {  # 创建一个语言到音调起始位置的映射字典
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
}

if __name__ == "__main__":
    a = set(zh_symbols)  # 创建一个包含中文音节的集合
    b = set(en_symbols)  # 创建一个包含英文音素的集合
    print(sorted(a & b))  # 打印中英文共有的音节
```

# `D:\src\Bert-VITS2\text\tone_sandhi.py`

```python
from typing import List
from typing import Tuple

import jieba
from pypinyin import lazy_pinyin
from pypinyin import Style


class ToneSandhi:
    def __init__(self):
        self.must_neural_tone_words = {
            "麻烦",
            "麻利",
            # ... (truncated for brevity)
            "咖喱",
            "扫把",
            "惦记",
        }
        self.must_not_neural_tone_words = {
            "男子",
            "女子",
            # ... (truncated for brevity)
            "量子",
            "莲子",
        }
        self.punc = "：，；。？！“”‘’':,;.?!"

    def _neural_sandhi(self, word: str, pos: str, finals: List[str]) -> List[str]:
        # ... (function explanation truncated for brevity)

    def _bu_sandhi(self, word: str, finals: List[str]) -> List[str]:
        # ... (function explanation truncated for brevity)

    def _yi_sandhi(self, word: str, finals: List[str]) -> List[str]:
        # ... (function explanation truncated for brevity)

    def _split_word(self, word: str) -> List[str]:
        # ... (function explanation truncated for brevity)

    def _three_sandhi(self, word: str, finals: List[str]) -> List[str]:
        # ... (function explanation truncated for brevity)

    def _all_tone_three(self, finals: List[str]) -> bool:
        # ... (function explanation truncated for brevity)

    def _merge_bu(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # ... (function explanation truncated for brevity)

    def _merge_yi(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # ... (function explanation truncated for brevity)

    def _merge_continuous_three_tones(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # ... (function explanation truncated for brevity)

    def _merge_continuous_three_tones_2(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # ... (function explanation truncated for brevity)

    def _is_reduplication(self, word: str) -> bool:
        # ... (function explanation truncated for brevity)

    def _merge_er(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # ... (function explanation truncated for brevity)

    def _merge_reduplication(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # ... (function explanation truncated for brevity)

    def pre_merge_for_modify(self, seg: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        # ... (function explanation truncated for brevity)

    def modified_tone(self, word: str, pos: str, finals: List[str]) -> List[str]:
        # ... (function explanation truncated for brevity)
```

# `D:\src\Bert-VITS2\text\__init__.py`

```python
from text.symbols import *  # Import all symbols from the text.symbols module

_symbol_to_id = {s: i for i, s in enumerate(symbols)}  # Create a dictionary mapping symbols to their corresponding IDs


def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]  # Convert each symbol in the cleaned text to its corresponding ID
    tone_start = language_tone_start_map[language]  # Get the tone start value based on the language
    tones = [i + tone_start for i in tones]  # Add the tone start value to each tone in the list
    lang_id = language_id_map[language]  # Get the language ID based on the language
    lang_ids = [lang_id for i in phones]  # Create a list of language IDs corresponding to the symbols in the text
    return phones, tones, lang_ids  # Return the lists of phones, tones, and language IDs


def get_bert(norm_text, word2ph, language, device, style_text=None, style_weight=0.7):
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert_mock import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert

    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}  # Create a map of language codes to BERT feature functions
    bert = lang_bert_func_map[language](  # Get the BERT feature using the appropriate language BERT function
        norm_text, word2ph, device, style_text, style_weight
    )
    return bert  # Return the BERT feature


def check_bert_models():
    import json
    from pathlib import Path
    from config import config
    from .bert_utils import _check_bert

    if config.mirror.lower() == "openi":  # Check if the mirror is set to "openi"
        import openi  # Import the openi module

        kwargs = {"token": config.openi_token} if config.openi_token else {}  # Set the kwargs dictionary with the openi token if available
        openi.login(**kwargs)  # Log in to openi using the provided token

    with open("./bert/bert_models.json", "r") as fp:  # Open the bert_models.json file for reading
        models = json.load(fp)  # Load the JSON data from the file
        for k, v in models.items():  # Iterate through the models in the JSON data
            local_path = Path("./bert").joinpath(k)  # Create the local path for the model
            _check_bert(v["repo_id"], v["files"], local_path)  # Check the BERT model using the provided repository ID, files, and local path


def init_openjtalk():
    import platform

    if platform.platform() == "Linux":  # Check if the platform is Linux
        import pyopenjtalk  # Import the pyopenjtalk module

        pyopenjtalk.g2p("こんにちは，世界。")  # Perform g2p conversion for the given text


init_openjtalk()  # Initialize openjtalk
check_bert_models()  # Check BERT models
```

# `D:\src\Bert-VITS2\tools\classify_language.py`

```python
import regex as re  # 导入regex模块

try:
    from config import config  # 导入config模块中的config对象

    LANGUAGE_IDENTIFICATION_LIBRARY = (
        config.webui_config.language_identification_library
    )  # 从config对象中获取webui_config.language_identification_library的值
except:
    LANGUAGE_IDENTIFICATION_LIBRARY = "langid"  # 如果导入失败，则将LANGUAGE_IDENTIFICATION_LIBRARY设置为"langid"

module = LANGUAGE_IDENTIFICATION_LIBRARY.lower()  # 将LANGUAGE_IDENTIFICATION_LIBRARY的值转换为小写

langid_languages = [  # 创建一个包含多种语言的列表
    "af",
    "am",
    ...
    "zu",
]


def classify_language(text: str, target_languages: list = None) -> str:  # 定义一个函数，用于识别文本的语言
    if module == "fastlid" or module == "fasttext":  # 如果module的值为"fastlid"或"fasttext"
        from fastlid import fastlid, supported_langs  # 从fastlid模块中导入fastlid和supported_langs对象

        classifier = fastlid  # 将classifier设置为fastlid
        if target_languages != None:  # 如果target_languages不为空
            target_languages = [
                lang for lang in target_languages if lang in supported_langs
            ]  # 将target_languages中的元素筛选出在supported_langs中的元素
            fastlid.set_languages = target_languages  # 设置fastlid的语言为target_languages
    elif module == "langid":  # 如果module的值为"langid"
        import langid  # 导入langid模块

        classifier = langid.classify  # 将classifier设置为langid.classify
        if target_languages != None:  # 如果target_languages不为空
            target_languages = [
                lang for lang in target_languages if lang in langid_languages
            ]  # 将target_languages中的元素筛选出在langid_languages中的元素
            langid.set_languages(target_languages)  # 设置langid的语言为target_languages
    else:
        raise ValueError(f"Wrong module {module}")  # 抛出一个值错误，提示module的值错误

    lang = classifier(text)[0]  # 识别文本的语言

    return lang  # 返回识别出的语言


def classify_zh_ja(text: str) -> str:  # 定义一个函数，用于识别中文和日文
    for idx, char in enumerate(text):  # 遍历文本中的字符
        unicode_val = ord(char)  # 获取字符的Unicode值

        # 检测日语字符
        if 0x3040 <= unicode_val <= 0x309F or 0x30A0 <= unicode_val <= 0x30FF:  # 如果字符是日语字符
            return "ja"  # 返回"ja"

        # 检测汉字字符
        if 0x4E00 <= unicode_val <= 0x9FFF:  # 如果字符是汉字字符
            # 检查周围的字符
            next_char = text[idx + 1] if idx + 1 < len(text) else None  # 获取下一个字符

            if next_char and (
                0x3040 <= ord(next_char) <= 0x309F or 0x30A0 <= ord(next_char) <= 0x30FF
            ):  # 如果下一个字符是日语字符
                return "ja"  # 返回"ja"

    return "zh"  # 返回"zh"


def split_alpha_nonalpha(text, mode=1):  # 定义一个函数，用于按照指定模式分割文本
    if mode == 1:  # 如果mode的值为1
        pattern = r"(?<=[\u4e00-\u9fff\u3040-\u30FF\d\s])(?=[\p{Latin}])|(?<=[\p{Latin}\s])(?=[\u4e00-\u9fff\u3040-\u30FF\d])"  # 设置pattern的值
    elif mode == 2:  # 如果mode的值为2
        pattern = r"(?<=[\u4e00-\u9fff\u3040-\u30FF\s])(?=[\p{Latin}\d])|(?<=[\p{Latin}\d\s])(?=[\u4e00-\u9fff\u3040-\u30FF])"  # 设置pattern的值
    else:
        raise ValueError("Invalid mode. Supported modes are 1 and 2.")  # 抛出一个值错误，提示mode的值无效

    return re.split(pattern, text)  # 使用pattern对text进行分割


if __name__ == "__main__":  # 如果当前脚本被直接执行
    text = "这是一个测试文本"
    print(classify_language(text))  # 输出识别出的语言
    print(classify_zh_ja(text))  # 输出"zh"

    text = "これはテストテキストです"
    print(classify_language(text))  # 输出识别出的语言
    print(classify_zh_ja(text))  # 输出"ja"

    text = "vits和Bert-VITS2是tts模型。花费3days.花费3天。Take 3 days"

    print(split_alpha_nonalpha(text, mode=1))  # 输出按照模式1分割后的结果
    # output: ['vits', '和', 'Bert-VITS', '2是', 'tts', '模型。花费3', 'days.花费3天。Take 3 days']

    print(split_alpha_nonalpha(text, mode=2))  # 输出按照模式2分割后的结果
    # output: ['vits', '和', 'Bert-VITS2', '是', 'tts', '模型。花费', '3days.花费', '3', '天。Take 3 days']

    text = "vits 和 Bert-VITS2 是 tts 模型。花费3days.花费3天。Take 3 days"
    print(split_alpha_nonalpha(text, mode=1))  # 输出按照模式1分割后的结果
    # output: ['vits ', '和 ', 'Bert-VITS', '2 ', '是 ', 'tts ', '模型。花费3', 'days.花费3天。Take ', '3 ', 'days']

    text = "vits 和 Bert-VITS2 是 tts 模型。花费3days.花费3天。Take 3 days"
    print(split_alpha_nonalpha(text, mode=2))  # 输出按照模式2分割后的结果
    # output: ['vits ', '和 ', 'Bert-VITS2 ', '是 ', 'tts ', '模型。花费', '3days.花费', '3', '天。Take ', '3 ', 'days']
```

# `D:\src\Bert-VITS2\tools\log.py`

```python
from loguru import logger  # 导入loguru模块中的logger类
import sys  # 导入sys模块

logger.remove()  # 移除所有默认的处理器

log_format = (  # 自定义日志格式
    "<g>{time:MM-DD HH:mm:ss}</g> <lvl>{level:<9}</lvl>| {file}:{line} | {message}"
)

logger.add(sys.stdout, format=log_format, backtrace=True, diagnose=True)  # 添加自定义格式的日志处理器到标准输出
```

# `D:\src\Bert-VITS2\tools\sentence.py`

```python
import logging  # 导入logging模块

import regex as re  # 导入regex模块并重命名为re

from tools.classify_language import classify_language, split_alpha_nonalpha  # 从tools.classify_language模块中导入classify_language和split_alpha_nonalpha函数


def check_is_none(item) -> bool:  # 定义函数check_is_none，参数为item，返回布尔值
    """none -> True, not none -> False"""  # 函数的文档字符串
    return (  # 返回一个布尔值
        item is None  # 判断item是否为None
        or (isinstance(item, str) and str(item).isspace())  # 判断item是否为字符串且是否为空白
        or str(item) == ""  # 判断item是否为空字符串
    )


def markup_language(text: str, target_languages: list = None) -> str:  # 定义函数markup_language，参数为text和target_languages，返回字符串
    pattern = (  # 定义pattern变量
        r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`"  # 匹配各种标点符号
        r"\！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」"  # 匹配各种全角标点符号
        r"『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+"  # 匹配各种其他符号
    )
    sentences = re.split(pattern, text)  # 使用正则表达式pattern对text进行分割，得到sentences列表

    pre_lang = ""  # 初始化pre_lang为空字符串
    p = 0  # 初始化p为0

    if target_languages is not None:  # 如果target_languages不为None
        sorted_target_languages = sorted(target_languages)  # 对target_languages进行排序
        if sorted_target_languages in [["en", "zh"], ["en", "ja"], ["en", "ja", "zh"]]:  # 如果sorted_target_languages在指定的列表中
            new_sentences = []  # 初始化new_sentences为空列表
            for sentence in sentences:  # 遍历sentences列表
                new_sentences.extend(split_alpha_nonalpha(sentence))  # 将split_alpha_nonalpha函数处理后的结果添加到new_sentences中
            sentences = new_sentences  # 将sentences更新为new_sentences

    for sentence in sentences:  # 遍历sentences列表
        if check_is_none(sentence):  # 如果sentence满足check_is_none函数的条件
            continue  # 跳过当前循环

        lang = classify_language(sentence, target_languages)  # 调用classify_language函数对sentence进行分类，结果赋值给lang

        if pre_lang == "":  # 如果pre_lang为空字符串
            text = text[:p] + text[p:].replace(  # 替换text中的内容
                sentence, f"[{lang.upper()}]{sentence}", 1  # 将sentence替换为带有语言标记的sentence
            )
            p += len(f"[{lang.upper()}]")  # 更新p的值
        elif pre_lang != lang:  # 如果pre_lang不等于lang
            text = text[:p] + text[p:].replace(  # 替换text中的内容
                sentence, f"[{pre_lang.upper()}][{lang.upper()}]{sentence}", 1  # 将sentence替换为带有语言标记的sentence
            )
            p += len(f"[{pre_lang.upper()}][{lang.upper()}]")  # 更新p的值
        pre_lang = lang  # 更新pre_lang的值
        p += text[p:].index(sentence) + len(sentence)  # 更新p的值
    text += f"[{pre_lang.upper()}]"  # 在text末尾添加带有语言标记的内容

    return text  # 返回处理后的text


def split_by_language(text: str, target_languages: list = None) -> list:  # 定义函数split_by_language，参数为text和target_languages，返回列表
    pattern = (  # 定义pattern变量
        r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`"  # 匹配各种标点符号
        r"\！？\。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」"  # 匹配各种全角标点符号
        r"『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+"  # 匹配各种其他符号
    )
    sentences = re.split(pattern, text)  # 使用正则表达式pattern对text进行分割，得到sentences列表

    pre_lang = ""  # 初始化pre_lang为空字符串
    start = 0  # 初始化start为0
    end = 0  # 初始化end为0
    sentences_list = []  # 初始化sentences_list为空列表

    if target_languages is not None:  # 如果target_languages不为None
        sorted_target_languages = sorted(target_languages)  # 对target_languages进行排序
        if sorted_target_languages in [["en", "zh"], ["en", "ja"], ["en", "ja", "zh"]]:  # 如果sorted_target_languages在指定的列表中
            new_sentences = []  # 初始化new_sentences为空列表
            for sentence in sentences:  # 遍历sentences列表
                new_sentences.extend(split_alpha_nonalpha(sentence))  # 将split_alpha_nonalpha函数处理后的结果添加到new_sentences中
            sentences = new_sentences  # 将sentences更新为new_sentences

    for sentence in sentences:  # 遍历sentences列表
        if check_is_none(sentence):  # 如果sentence满足check_is_none函数的条件
            continue  # 跳过当前循环

        lang = classify_language(sentence, target_languages)  # 调用classify_language函数对sentence进行分类，结果赋值给lang

        end += text[end:].index(sentence)  # 更新end的值
        if pre_lang != "" and pre_lang != lang:  # 如果pre_lang不为空字符串且不等于lang
            sentences_list.append((text[start:end], pre_lang))  # 将文本和语言添加到sentences_list中
            start = end  # 更新start的值
        end += len(sentence)  # 更新end的值
        pre_lang = lang  # 更新pre_lang的值
    sentences_list.append((text[start:], pre_lang))  # 将剩余的文本和语言添加到sentences_list中

    return sentences_list  # 返回处理后的sentences_list


def sentence_split(text: str, max: int) -> list:  # 定义函数sentence_split，参数为text和max，返回列表
    pattern = r"[!(),—+\-.:;?？。，、；：]+"  # 定义pattern变量
    sentences = re.split(pattern, text)  # 使用正则表达式pattern对text进行分割，得到sentences列表
    discarded_chars = re.findall(pattern, text)  # 使用正则表达式pattern在text中查找匹配的内容，得到discarded_chars列表

    sentences_list, count, p = [], 0, 0  # 初始化sentences_list为空列表，count为0，p为0

    # 按被分割的符号遍历
    for i, discarded_chars in enumerate(discarded_chars):  # 遍历discarded_chars列表
        count += len(sentences[i]) + len(discarded_chars)  # 更新count的值
        if count >= max:  # 如果count大于等于max
            sentences_list.append(text[p : p + count].strip())  # 将指定范围内的文本添加到sentences_list中
            p += count  # 更新p的值
            count = 0  # 重置count为0

    # 加入最后剩余的文本
    if p < len(text):  # 如果p小于text的长度
        sentences_list.append(text[p:])  # 将剩余的文本添加到sentences_list中

    return sentences_list  # 返回处理后的sentences_list


def sentence_split_and_markup(text, max=50, lang="auto", speaker_lang=None):  # 定义函数sentence_split_and_markup，参数为text、max、lang和speaker_lang
    # 如果该speaker只支持一种语言
    if speaker_lang is not None and len(speaker_lang) == 1:  # 如果speaker_lang不为None且长度为1
        if lang.upper() not in ["AUTO", "MIX"] and lang.lower() != speaker_lang[0]:  # 如果lang不在指定的列表中且不等于speaker_lang的第一个���素
            logging.debug(  # 输出调试信息
                f'lang "{lang}" is not in speaker_lang {speaker_lang},automatically set lang={speaker_lang[0]}'  # 格式化输出信息
            )
        lang = speaker_lang[0]  # 更新lang的值

    sentences_list = []  # 初始化sentences_list为空列表
    if lang.upper() != "MIX":  # 如果lang不为MIX
        if max <= 0:  # 如果max小于等于0
            sentences_list.append(  # 将结果添加到sentences_list中
                markup_language(text, speaker_lang)  # 调用markup_language函数处理text
                if lang.upper() == "AUTO"  # 如果lang为AUTO
                else f"[{lang.upper()}]{text}[{lang.upper()}]"  # 否则在text两侧添加语言标记
            )
        else:
            for i in sentence_split(text, max):  # 遍历sentence_split函数处理后的结果
                if check_is_none(i):  # 如果i满足check_is_none函数的条件
                    continue  # 跳过当前循环
                sentences_list.append(  # 将结果添加到sentences_list中
                    markup_language(i, speaker_lang)  # 调用markup_language函数处理i
                    if lang.upper() == "AUTO"  # 如果lang为AUTO
                    else f"[{lang.upper()}]{i}[{lang.upper()}]"  # 否则在i两侧添加语言标记
                )
    else:
        sentences_list.append(text)  # 将text添加到sentences_list中

    for i in sentences_list:  # 遍历sentences_list
        logging.debug(i)  # 输出调试信息

    return sentences_list  # 返回处理后的sentences_list


if __name__ == "__main__":  # 如果当前脚本为主程序
    text = "这几天心里颇不宁静。今晚在院子里坐着乘凉，忽然想起日日走过的荷塘，在这满月的光里，总该另有一番样子吧。月亮渐渐地升高了，墙外马路上孩子们的欢笑，已经听不见了；妻在屋里拍着闰儿，迷迷糊糊地哼着眠歌。我悄悄地披了大衫，带上门出去。"
    print(markup_language(text, target_languages=None))  # 调用markup_language函数处理text并输出结果
    print(sentence_split(text, max=50))  # 调用sentence_split函数处理text并输出结果
    print(sentence_split_and_markup(text, max=50, lang="auto", speaker_lang=None))  # 调用sentence_split_and_markup函数处理text并输出结果

    text = "你好，这是一段用来测试自动标注的文本。こんにちは,これは自動ラベリングのテスト用テキストです.Hello, this is a piece of text to test autotagging.你好！今天我们要介绍VITS项目，其重点是使用了GAN Duration predictor和transformer flow,并且接入了Bert模型来提升韵律。Bert embedding会在稍后介绍。"
    print(split_by_language(text, ["zh", "ja", "en"]))  # 调用split_by_language函数处理text并输出结果

    text = "vits和Bert-VITS2是tts模型。花费3days.花费3天。Take 3 days"
    print(split_by_language(text, ["zh", "ja", "en"]))  # 调用split_by_language函数处理text并输出结果
    # output: [('vits', 'en'), ('和', 'ja'), ('Bert-VITS', 'en'), ('2是', 'zh'), ('tts', 'en'), ('模型。花费3', 'zh'), ('days.', 'en'), ('花费3天。', 'zh'), ('Take 3 days', 'en')]

    print(split_by_language(text, ["zh", "en"]))  # 调用split_by_language函数处理text并输出结果
    # output: [('vits', 'en'), ('和', 'zh'), ('Bert-VITS', 'en'), ('2是', 'zh'), ('tts', 'en'), ('模型。花费3', 'zh'), ('days.', 'en'), ('花费3天。', 'zh'), ('Take 3 days', 'en')]

    text = "vits 和 Bert-VITS2 是 tts 模型。花费 3 days. 花费 3天。Take 3 days"
    print(split_by_language(text, ["zh", "en"]))  # 调用split_by_language函数处理text并输出结果
    # output: [('vits ', 'en'), ('和 ', 'zh'), ('Bert-VITS2 ', 'en'), ('是 ', 'zh'), ('tts ', 'en'), ('模型。花费 ', 'zh'), ('3 days. ', 'en'), ('花费 3天。', 'zh'), ('Take 3 days', 'en')]
```

# `D:\src\Bert-VITS2\tools\translate.py`

```python
"""
翻译api
"""
from config import config  # 导入config模块

import random  # 导入random模块
import hashlib  # 导入hashlib模块
import requests  # 导入requests模块


def translate(Sentence: str, to_Language: str = "jp", from_Language: str = ""):
    """
    :param Sentence: 待翻译语句
    :param from_Language: 待翻译语句语言
    :param to_Language: 目标语言
    :return: 翻译后语句 出错时返回None

    常见语言代码：中文 zh 英语 en 日语 jp
    """
    appid = config.translate_config.app_key  # 从config模块中获取app_key
    key = config.translate_config.secret_key  # 从config模块中获取secret_key
    if appid == "" or key == "":  # 如果app_key或secret_key为空
        return "请开发者在config.yml中配置app_key与secret_key"  # 返回提示信息
    url = "https://fanyi-api.baidu.com/api/trans/vip/translate"  # 设置请求的URL
    texts = Sentence.splitlines()  # 将待翻译语句按行分割
    outTexts = []  # 初始化翻译后的语句列表
    for t in texts:  # 遍历每行待翻译语句
        if t != "":  # 如果行不为空
            # 签名计算 参考文档 https://api.fanyi.baidu.com/product/113
            salt = str(random.randint(1, 100000))  # 生成随机数作为盐值
            signString = appid + t + salt + key  # 构造签名字符串
            hs = hashlib.md5()  # 创建md5对象
            hs.update(signString.encode("utf-8"))  # 更新md5对象的内容
            signString = hs.hexdigest()  # 获取md5对象的十六进制表示
            if from_Language == "":  # 如果待翻译语句语言为空
                from_Language = "auto"  # 设置待翻译语句语言为自动检测
            headers = {"Content-Type": "application/x-www-form-urlencoded"}  # 设置请求头
            payload = {  # 构造请求参数
                "q": t,
                "from": from_Language,
                "to": to_Language,
                "appid": appid,
                "salt": salt,
                "sign": signString,
            }
            # 发送请求
            try:
                response = requests.post(  # 发送POST请求
                    url=url, data=payload, headers=headers, timeout=3
                )
                response = response.json()  # 将响应转换为JSON格式
                if "trans_result" in response.keys():  # 如果响应中包含trans_result
                    result = response["trans_result"][0]  # 获取翻译结果
                    if "dst" in result.keys():  # 如果翻译结果中包含dst
                        dst = result["dst"]  # 获取翻译后的语句
                        outTexts.append(dst)  # 将翻译后的语句添加到列表中
            except Exception:  # 捕获异常
                return Sentence  # 返回原始待翻译语句
        else:  # 如果行为空
            outTexts.append(t)  # 将空行添加到列表中
    return "\n".join(outTexts)  # 返回翻译后的语句
```

# `D:\src\Bert-VITS2\tools\__init__.py`

```py
"""
# 导入工具包
"""
```