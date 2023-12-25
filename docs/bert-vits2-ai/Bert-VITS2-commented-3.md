# Bert-VITS2 源码解析 3

# `D:\src\Bert-VITS2\oldVersion\V111\text\cleaner.py`

```python
# 导入模块 chinese, japanese, cleaned_text_to_sequence
from . import chinese, japanese, cleaned_text_to_sequence
# 导入模块 fix 中的 japanese 并重命名为 japanese_fix
from .fix import japanese as japanese_fix

# 创建语言模块映射字典
language_module_map = {"ZH": chinese, "JP": japanese}
# 创建修复后的语言模块映射字典
language_module_map_fix = {"ZH": chinese, "JP": japanese_fix}

# 定义函数 clean_text，用于清洗文本
def clean_text(text, language):
    # 从语言模块映射字典中获取对应语言的模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取文本的音素、音调和词转音素映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 返回规范化文本、音素、音调和词转音素映射
    return norm_text, phones, tones, word2ph

# 定义函数 clean_text_fix，用于使用修复后的语言模块清洗文本
def clean_text_fix(text, language):
    """使用dev分支修复"""
    # 从修复后的语言模块映射字典中获取对应语言的模块
    language_module = language_module_map_fix[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取文本的音素、音调和词转音素映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 返回规范化文本、音素、音调和词转音素映射
    return norm_text, phones, tones, word2ph

# 定义函数 clean_text_bert，用于使用 BERT 特征清洗文本
def clean_text_bert(text, language):
    # 从语言模块映射字典中获取对应语言的模块
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
    # 对文本进行清洗
    norm_text, phones, tones, word2ph = clean_text(text, language)
    # 将清洗后的文本转换为序列
    return cleaned_text_to_sequence(phones, tones, language)

# 如果作为主程序运行，则执行 pass
if __name__ == "__main__":
    pass
```

# `D:\src\Bert-VITS2\oldVersion\V111\text\english.py`

```python
import pickle  # 导入pickle模块
import os  # 导入os模块
import re  # 导入re模块
from g2p_en import G2p  # 从g2p_en模块导入G2p类

from . import symbols  # 从当前目录导入symbols模块

current_file_path = os.path.dirname(__file__)  # 获取当前文件路径
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")  # 拼接文件路径
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")  # 拼接文件路径
_g2p = G2p()  # 创建G2p对象

# 定义arpa集合
arpa = {
    "AH0",
    "S",
    ...
    "L",
    "SH",
}

# 定义替换函数
def post_replace_ph(ph):
    ...

# 读取字典
def read_dict():
    ...

# 缓存字典
def cache_dict(g2p_dict, file_path):
    ...

# 获取字典
def get_dict():
    ...

eng_dict = get_dict()  # 获取字典

# 优化音素
def refine_ph(phn):
    ...

# 优化音节
def refine_syllables(syllables):
    ...

# 文本规范化
def text_normalize(text):
    ...

# 文本转音素
def g2p(text):
    ...

if __name__ == "__main__":
    print(g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."))  # 打印文本的音素表示
```

# `D:\src\Bert-VITS2\oldVersion\V111\text\english_bert_mock.py`

```python
import torch  # 导入torch模块

def get_bert_feature(norm_text, word2ph):
    return torch.zeros(1024, sum(word2ph))  # 返回一个1024x(sum(word2ph))的全零张量
```

# `D:\src\Bert-VITS2\oldVersion\V111\text\japanese.py`

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

# `D:\src\Bert-VITS2\oldVersion\V111\text\japanese_bert.py`

```python
import torch  # 导入PyTorch库
from transformers import AutoTokenizer, AutoModelForMaskedLM  # 从transformers库中导入AutoTokenizer和AutoModelForMaskedLM
import sys  # 导入sys库

tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")  # 使用预训练的tokenizer

models = dict()  # 创建一个空字典用于存储模型

def get_bert_feature(text, word2ph, device=None):  # 定义一个名为get_bert_feature的函数，接受text、word2ph和device三个参数
    if (  # 如果条件判断语句开始
        sys.platform == "darwin"  # 判断当前操作系统是否为darwin
        and torch.backends.mps.is_available()  # 判断是否支持MPS
        and device == "cpu"  # 判断设备是否为CPU
    ):  # 条件判断语句结束
        device = "mps"  # 将device设置为"mps"
    if not device:  # 如果device为空
        device = "cuda"  # 将device设置为"cuda"
    if device not in models.keys():  # 如果device不在models的键中
        models[device] = AutoModelForMaskedLM.from_pretrained(  # 使用预训练的模型
            "./bert/bert-base-japanese-v3"
        ).to(device)  # 将模型移动到指定的设备
    with torch.no_grad():  # 使用torch.no_grad()上下文管理器
        inputs = tokenizer(text, return_tensors="pt")  # 使用tokenizer对文本进行编码
        for i in inputs:  # 遍历inputs
            inputs[i] = inputs[i].to(device)  # 将inputs中的数据移动到指定的设备
        res = models[device](**inputs, output_hidden_states=True)  # 使用模型对输入进行预测
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 对模型的输出进行处理
    assert inputs["input_ids"].shape[-1] == len(word2ph)  # 断言输入的单词数量和音素数量是否一致
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 创建一个空列表用于存储特征
    for i in range(len(word2phone)):  # 遍历word2phone的长度
        repeat_feature = res[i].repeat(word2phone[i], 1)  # 将res[i]重复word2phone[i]次
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 沿着指定维度拼接phone_level_feature

    return phone_level_feature.T  # 返回phone_level_feature的转置
```

# `D:\src\Bert-VITS2\oldVersion\V111\text\symbols.py`

```python
punctuation = ["!", "?", "…", ",", ".", "'", "-"]  # 创建标点符号列表
pu_symbols = punctuation + ["SP", "UNK"]  # 创建标点符号列表并添加特殊标记
pad = "_"  # 创建填充符号

# chinese
zh_symbols = [  # 创建中文音节列表
    "E",
    "En",
    ...
    "OO",
]
num_zh_tones = 6  # 中文音调数量

# japanese
ja_symbols = [  # 创建日文音节列表
    "N",
    "a",
    ...
    "zy",
]
num_ja_tones = 1  # 日文音调数量

# English
en_symbols = [  # 创建英文音节列表
    "aa",
    "ae",
    ...
    "zh",
]
num_en_tones = 4  # 英文音调数量

# combine all symbols
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))  # 合并所有音节并去重排序
symbols = [pad] + normal_symbols + pu_symbols  # 创建所有音节和标点符号的列表
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]  # 获取标点符号在列表中的索引

# combine all tones
num_tones = num_zh_tones + num_ja_tones + num_en_tones  # 计算所有语言的音调数量

# language maps
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}  # 创建语言到ID的映射
num_languages = len(language_id_map.keys())  # 计算语言数量

language_tone_start_map = {  # 创建语言到音调起始位置的映射
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
}

if __name__ == "__main__":
    a = set(zh_symbols)  # 创建中文音节集合
    b = set(en_symbols)  # 创建英文音节集合
    print(sorted(a & b))  # 打印中英文音节的交集
```

# `D:\src\Bert-VITS2\oldVersion\V111\text\tone_sandhi.py`

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

# `D:\src\Bert-VITS2\oldVersion\V111\text\__init__.py`

```python
from .symbols import *

# Create a dictionary that maps symbols to their corresponding IDs
_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    # Convert symbols in the cleaned text to their corresponding IDs
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    # Adjust tones based on the language's tone start map
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    # Create a list of language IDs corresponding to the phones
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids


def get_bert(norm_text, word2ph, language, device):
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert_mock import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert

    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}
    # Get BERT features based on the language using the corresponding BERT function
    bert = lang_bert_func_map[language](norm_text, word2ph, device)
    return bert


def get_bert_fix(norm_text, word2ph, language, device):
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert_mock import get_bert_feature as en_bert
    from .fix.japanese_bert import get_bert_feature as jp_bert

    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}
    # Get BERT features based on the language using the corresponding BERT function
    bert = lang_bert_func_map[language](norm_text, word2ph, device)
    return bert
```

# `D:\src\Bert-VITS2\oldVersion\V111\text\fix\japanese.py`

```python
# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
import re  # Import the regular expression module
import unicodedata  # Import the unicodedata module

from transformers import AutoTokenizer  # Import the AutoTokenizer class from the transformers module

from .. import punctuation, symbols  # Import punctuation and symbols from the parent package

from num2words import num2words  # Import the num2words function from the num2words module

import pyopenjtalk  # Import the pyopenjtalk module
import jaconv  # Import the jaconv module

# Define a function to convert katakana text to phonemes
def kata2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""
    # ... (function implementation)

# Define a function to convert hiragana to katakana
def hira2kata(text: str) -> str:
    return jaconv.hira2kata(text)

# Define a regular expression pattern for matching marks
_MARKS = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# Define a function to convert text to katakana
def text2kata(text: str) -> str:
    # ... (function implementation)

# Define a function to convert text to separated katakana
def text2sep_kata(text: str) -> (list, list):
    # ... (function implementation)

# Define a dictionary mapping alpha symbols to their corresponding pronunciations
_ALPHASYMBOL_YOMI = {
    # ... (dictionary content)
}

# Define regular expression patterns for matching numbers and currency symbols
_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")
_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
_CURRENCY_RX = re.compile(r"([$¥£€])([0-9.]*[0-9])")
_NUMBER_RX = re.compile(r"[0-9]+(\.[0-9]+)?")

# Define a function to convert numbers in text to words
def japanese_convert_numbers_to_words(text: str) -> str:
    # ... (function implementation)

# Define a function to convert alpha symbols in text to words
def japanese_convert_alpha_symbols_to_words(text: str) -> str:
    return "".join([_ALPHASYMBOL_YOMI.get(ch, ch) for ch in text.lower()])

# Define a function to convert Japanese text to phonemes
def japanese_text_to_phonemes(text: str) -> str:
    """Convert Japanese text to phonemes."""
    # ... (function implementation)

# Define a function to check if a character is a Japanese character
def is_japanese_character(char):
    # ... (function implementation)

# Define a dictionary for replacing punctuation in text
rep_map = {
    # ... (dictionary content)
}

# Define a function to replace punctuation in text
def replace_punctuation(text):
    # ... (function implementation)

# Define a function to normalize text
def text_normalize(text):
    # ... (function implementation)

# Define a function to distribute phones to words
def distribute_phone(n_phone, n_word):
    # ... (function implementation)

# Define a function to handle long phonemes
def handle_long(sep_phonemes):
    # ... (function implementation)

# Create an instance of the AutoTokenizer class
tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")

# Define a function to convert text to phonemes using a pre-trained model
def g2p(norm_text):
    # ... (function implementation)

# Main program
if __name__ == "__main__":
    # ... (main program implementation)
```

# `D:\src\Bert-VITS2\oldVersion\V111\text\fix\japanese_bert.py`

```python
import torch  # 导入PyTorch库
from transformers import AutoTokenizer, AutoModelForMaskedLM  # 从transformers库中导入AutoTokenizer和AutoModelForMaskedLM
import sys  # 导入sys库
from .japanese import text2sep_kata  # 从japanese模块中导入text2sep_kata函数
from config import config  # 从config模块中导入config变量

tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")  # 使用预训练的tokenizer初始化tokenizer对象

models = dict()  # 初始化空字典models


def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):  # 定义名为get_bert_feature的函数，接受text、word2ph和device三个参数
    sep_text, _ = text2sep_kata(text)  # 调用text2sep_kata函数，将返回的结果赋值给sep_text
    sep_tokens = [tokenizer.tokenize(t) for t in sep_text]  # 对sep_text中的每个元素进行tokenize操作，将结果存储在sep_tokens中
    sep_ids = [tokenizer.convert_tokens_to_ids(t) for t in sep_tokens]  # 对sep_tokens中的每个元素进行convert_tokens_to_ids操作，将结果存储在sep_ids中
    sep_ids = [2] + [item for sublist in sep_ids for item in sublist] + [3]  # 对sep_ids进行处理，将结果重新赋值给sep_ids
    return get_bert_feature_with_token(sep_ids, word2ph, device)  # 调用get_bert_feature_with_token函数，返回结果


def get_bert_feature_with_token(tokens, word2ph, device=config.bert_gen_config.device):  # 定义名为get_bert_feature_with_token的函数，接受tokens、word2ph和device三个参数
    if (  # 如果以下条件成立
        sys.platform == "darwin"  # 当前操作系统为darwin
        and torch.backends.mps.is_available()  # 并且torch的mps后端可用
        and device == "cpu"  # 并且device为cpu
    ):  # 则执行以下操作
        device = "mps"  # 将device重新赋值为"mps"
    if not device:  # 如果device不存在
        device = "cuda"  # 将device重新赋值为"cuda"
    if device not in models.keys():  # 如果device不在models的键中
        models[device] = AutoModelForMaskedLM.from_pretrained(  # 将device作为键，对应的AutoModelForMaskedLM对象作为值，存储到models中
            "./bert/bert-base-japanese-v3"
        ).to(device)
    with torch.no_grad():  # 使用torch.no_grad()上下文管理器
        inputs = torch.tensor(tokens).to(device).unsqueeze(0)  # 将tokens转换为tensor，并移动到指定的device上，然后在第0维度上增加一个维度
        token_type_ids = torch.zeros_like(inputs).to(device)  # 生成与inputs形状相同的全零tensor，并移动到指定的device上
        attention_mask = torch.ones_like(inputs).to(device)  # 生成与inputs形状相同的全一tensor，并移动到指定的device上
        inputs = {  # 创建名为inputs的字典
            "input_ids": inputs,  # 将input_ids键对应的值设为inputs
            "token_type_ids": token_type_ids,  # 将token_type_ids键对应的值设为token_type_ids
            "attention_mask": attention_mask,  # 将attention_mask键对应的值设为attention_mask
        }
        res = models[device](**inputs, output_hidden_states=True)  # 调用models中device对应的AutoModelForMaskedLM对象，传入inputs和output_hidden_states=True参数，将结果赋值给res
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 对res中的hidden_states进行处理，将结果重新赋值给res
    assert inputs["input_ids"].shape[-1] == len(word2ph)  # 断言inputs中input_ids键对应的值的最后一个维度的长度等于word2ph的长度
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 初始化空列表phone_level_feature
    for i in range(len(word2phone)):  # 遍历word2phone的索引
        repeat_feature = res[i].repeat(word2phone[i], 1)  # 对res中的第i个元素进行repeat操作，将结果存储在repeat_feature中
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature中
    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 对phone_level_feature进行处理，将结果重新赋值给phone_level_feature
    return phone_level_feature.T  # 返回phone_level_feature的转置
```

# `D:\src\Bert-VITS2\oldVersion\V111\text\fix\__init__.py`

```python
# 创建一个名为my_list的空列表
my_list = []

# 向my_list列表中添加一个整数10
my_list.append(10)

# 向my_list列表中添加一个字符串"hello"
my_list.append("hello")

# 打印my_list列表的内容
print(my_list)
```

```python
# 创建一个名为my_list的空列表
my_list = []

# 向my_list列表中添加一个整数10
my_list.append(10)

# 向my_list列表中添加一个字符串"hello"
my_list.append("hello")

# 打印my_list列表的内容
print(my_list)
```

# `D:\src\Bert-VITS2\oldVersion\V200\models.py`

```py
# 添加注释
```python
import math  # 导入数学库
import torch  # 导入torch库
from torch import nn  # 从torch库中导入nn模块
from torch.nn import functional as F  # 从torch.nn中导入functional模块

import commons  # 导入commons模块
import modules  # 导入modules模块
import attentions  # 导入attentions模块
import monotonic_align  # 导入monotonic_align模块

from torch.nn import Conv1d, ConvTranspose1d, Conv2d  # 从torch.nn中导入Conv1d, ConvTranspose1d, Conv2d模块
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm  # 从torch.nn.utils中导入weight_norm, remove_weight_norm, spectral_norm模块
from commons import init_weights, get_padding  # 从commons中导入init_weights, get_padding模块
from text import symbols, num_tones, num_languages  # 从text中导入symbols, num_tones, num_languages模块
```

# `D:\src\Bert-VITS2\oldVersion\V200\__init__.py`

```python
"""
@Desc: 2.0版本兼容 对应2.0.1 2.0.2-fix
"""
import torch  # 导入torch模块
import commons  # 导入commons模块
from .text import cleaned_text_to_sequence, get_bert  # 从text模块中导入cleaned_text_to_sequence和get_bert函数
from .text.cleaner import clean_text  # 从text模块中导入clean_text函数


def get_text(text, language_str, hps, device):  # 定义get_text函数，接受text, language_str, hps, device四个参数
    # 在此处实现当前版本的get_text
    norm_text, phone, tone, word2ph = clean_text(text, language_str)  # 调用clean_text函数，将返回值分别赋给norm_text, phone, tone, word2ph
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)  # 调用cleaned_text_to_sequence函数，将返回值分别赋给phone, tone, language

    if hps.data.add_blank:  # 如果hps.data.add_blank为真
        phone = commons.intersperse(phone, 0)  # 调用commons.intersperse函数，将返回值赋给phone
        tone = commons.intersperse(tone, 0)  # 调用commons.intersperse函数，将返回值赋给tone
        language = commons.intersperse(language, 0)  # 调用commons.intersperse函数，将返回值赋给language
        for i in range(len(word2ph)):  # 遍历word2ph的长度
            word2ph[i] = word2ph[i] * 2  # 将word2ph[i]的值乘以2
        word2ph[0] += 1  # 将word2ph[0]的值加1
    bert_ori = get_bert(norm_text, word2ph, language_str, device)  # 调用get_bert函数，将返回值赋给bert_ori
    del word2ph  # 删除word2ph
    assert bert_ori.shape[-1] == len(phone), phone  # 断言bert_ori的最后一个维度等于phone的长度

    if language_str == "ZH":  # 如果language_str等于"ZH"
        bert = bert_ori  # 将bert_ori的值赋给bert
        ja_bert = torch.zeros(1024, len(phone))  # 创建一个1024*len(phone)的全零张量，赋给ja_bert
        en_bert = torch.zeros(1024, len(phone))  # 创建一个1024*len(phone)的全零张量，赋给en_bert
    elif language_str == "JP":  # 如果language_str等于"JP"
        bert = torch.zeros(1024, len(phone))  # 创建一个1024*len(phone)的全零张量，赋给bert
        ja_bert = bert_ori  # 将bert_ori的值赋给ja_bert
        en_bert = torch.zeros(1024, len(phone))  # 创建一个1024*len(phone)的全零张量，赋给en_bert
    elif language_str == "EN":  # 如果language_str等于"EN"
        bert = torch.zeros(1024, len(phone))  # 创建一个1024*len(phone)的全零张量，赋给bert
        ja_bert = torch.zeros(1024, len(phone))  # 创建一个1024*len(phone)的全零张量，赋给ja_bert
        en_bert = bert_ori  # 将bert_ori的值赋给en_bert
    else:  # 否则
        raise ValueError("language_str should be ZH, JP or EN")  # 抛出ValueError异常，提示"language_str should be ZH, JP or EN"

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"  # 断言bert的最后一个维度等于phone的长度，如果不等于则抛出异常

    phone = torch.LongTensor(phone)  # 将phone转换为LongTensor类型，赋给phone
    tone = torch.LongTensor(tone)  # 将tone转换为LongTensor类型，赋给tone
    language = torch.LongTensor(language)  # 将language转换为LongTensor类型，赋给language
    return bert, ja_bert, en_bert, phone, tone, language  # 返回bert, ja_bert, en_bert, phone, tone, language


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
):  # 定义infer函数，接受text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, language, hps, net_g, device十个参数
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text, language, hps, device
    )  # 调用get_text函数，将返回值分别赋给bert, ja_bert, en_bert, phones, tones, lang_ids
    with torch.no_grad():  # 进入torch.no_grad()上下文管理器
        x_tst = phones.to(device).unsqueeze(0)  # 将phones转换为device类型，并在0维度上增加一个维度，赋给x_tst
        tones = tones.to(device).unsqueeze(0)  # 将tones转换为device类型，并在0维度上增加一个维度，赋给tones
        lang_ids = lang_ids.to(device).unsqueeze(0)  # 将lang_ids转换为device类型，并在0维度上增加一个维度，赋给lang_ids
        bert = bert.to(device).unsqueeze(0)  # 将bert转换为device类型，并在0维度上增加一个维度，赋给bert
        ja_bert = ja_bert.to(device).unsqueeze(0)  # 将ja_bert转换为device类型，并在0维度上增加一个维度，赋给ja_bert
        en_bert = en_bert.to(device).unsqueeze(0)  # 将en_bert转换为device类型，并在0维度上增加一个维度，赋给en_bert
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)  # 创建一个包含phones.size(0)的LongTensor类型张量，赋给x_tst_lengths
        del phones  # 删除phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)  # 创建一个包含hps.data.spk2id[sid]的LongTensor类型张量，赋给speakers
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                en_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )  # 调用net_g.infer函数，将返回值赋给audio
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers, ja_bert, en_bert  # 删除x_tst, tones, lang_ids, bert, x_tst_lengths, speakers, ja_bert, en_bert
        if torch.cuda.is_available():  # 如果torch.cuda.is_available()为真
            torch.cuda.empty_cache()  # 清空cuda缓存
        return audio  # 返回audio
```

# `D:\src\Bert-VITS2\oldVersion\V200\text\bert_utils.py`

```python
from pathlib import Path  # 导入Path类

from huggingface_hub import hf_hub_download  # 从huggingface_hub模块中导入hf_hub_download函数

from config import config  # 从config模块中导入config对象


MIRROR: str = config.mirror  # 从config对象中获取mirror属性，并赋值给MIRROR变量


def _check_bert(repo_id, files, local_path):  # 定义_check_bert函数，接受repo_id、files和local_path三个参数
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

# `D:\src\Bert-VITS2\oldVersion\V200\text\chinese.py`

```python
import os  # 导入os模块
import re  # 导入re模块
import cn2an  # 导入cn2an模块
from pypinyin import lazy_pinyin, Style  # 从pypinyin模块中导入lazy_pinyin和Style
from .symbols import punctuation  # 从symbols模块中导入punctuation
from .tone_sandhi import ToneSandhi  # 从tone_sandhi模块中导入ToneSandhi

current_file_path = os.path.dirname(__file__)  # 获取当前文件路径
pinyin_to_symbol_map = {  # 创建pinyin_to_symbol_map字典
    line.split("\t")[0]: line.strip().split("\t")[1]  # 从文件中读取内容并创建字典
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
    # 替换文本中的特定词语
    # 创建正则表达式模式
    # 使用正则表达式替换文本中的特定词语
    # 使用正则表达式替换文本中的特定词语
    return replaced_text  # 返回替换后的文本

# 定义g2p函数
def g2p(text):
    # 创建正则表达式模式
    # 对文本进行分割
    # 调用_g2p函数处理分割后的文本
    return phones, tones, word2ph  # 返回处理后的结果

# 定义_get_initials_finals函数
def _get_initials_finals(word):
    # 创建初始和韵母列表
    # 调用lazy_pinyin函数获取初始和韵母
    return initials, finals  # 返回初始和韵母列表

# 定义_g2p函数
def _g2p(segments):
    # 创建空列表
    # 遍历分词后的句子
    # 调用_get_initials_finals函数处理分词后的句子
    return phones_list, tones_list, word2ph  # 返回处理后的结果

# 定义text_normalize函数
def text_normalize(text):
    # 使用正则表达式查找数字
    # 遍历数字列表
    # 替换文本中的特定词语
    return text  # 返回处理后的文本

# 定义get_bert_feature函数
def get_bert_feature(text, word2ph):
    # 导入chinese_bert模块
    return chinese_bert.get_bert_feature(text, word2ph)  # 返回bert特征

if __name__ == "__main__":
    from text.chinese_bert import get_bert_feature  # 从chinese_bert模块中导入get_bert_feature

    text = "啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = text_normalize(text)  # 对文本进行规范化处理
    print(text)  # 打印处理后的文本
    phones, tones, word2ph = g2p(text)  # 调用g2p函数处理文本
    bert = get_bert_feature(text, word2ph)  # 调用get_bert_feature函数获取bert特征
    print(phones, tones, word2ph, bert.shape)  # 打印处理后的结果
```

# `D:\src\Bert-VITS2\oldVersion\V200\text\chinese_bert.py`

```python
import sys  # 导入sys模块

import torch  # 导入torch模块
from transformers import AutoModelForMaskedLM, AutoTokenizer  # 从transformers模块中导入AutoModelForMaskedLM和AutoTokenizer类

from config import config  # 从config模块中导入config类

LOCAL_PATH = "./bert/chinese-roberta-wwm-ext-large"  # 设置LOCAL_PATH变量为"./bert/chinese-roberta-wwm-ext-large"

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)  # 使用AutoTokenizer类从预训练模型路径LOCAL_PATH中加载tokenizer

models = dict()  # 创建一个空字典models

def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):  # 定义函数get_bert_feature，接受text、word2ph和device三个参数
    if (  # 如果条件判断
        sys.platform == "darwin"  # 当前操作系统为darwin
        and torch.backends.mps.is_available()  # torch后端支持mps
        and device == "cpu"  # device为cpu
    ):  # 条件判断结束
        device = "mps"  # 将device设置为"mps"
    if not device:  # 如果device不存在
        device = "cuda"  # 将device设置为"cuda"
    if device not in models.keys():  # 如果device不在models的键中
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)  # 将models[device]设置为从预训练模型路径LOCAL_PATH中加载的AutoModelForMaskedLM类，并转移到device上
    with torch.no_grad():  # 使用torch.no_grad()上下文管理器
        inputs = tokenizer(text, return_tensors="pt")  # 使用tokenizer对text进行编码，返回PyTorch张量
        for i in inputs:  # 遍历inputs
            inputs[i] = inputs[i].to(device)  # 将inputs[i]转移到device上
        res = models[device](**inputs, output_hidden_states=True)  # 使用models[device]对inputs进行预测，输出隐藏状态
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 将res["hidden_states"]的倒数第3到倒数第2个元素进行拼接，然后取第一个元素，转移到CPU上

    assert len(word2ph) == len(text) + 2  # 断言word2ph的长度等于text的长度加2
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 创建一个空列表phone_level_feature
    for i in range(len(word2phone)):  # 遍历word2phone的长度
        repeat_feature = res[i].repeat(word2phone[i], 1)  # 将res[i]重复word2phone[i]次，沿着第1维度
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 对phone_level_feature进行拼接，沿着第0维度

    return phone_level_feature.T  # 返回phone_level_feature的转置

if __name__ == "__main__":  # 如果模块是直接运行的
    word_level_feature = torch.rand(38, 1024)  # 创建一个38x1024的随机张量
    word2phone = [  # 创建一个名为word2phone的列表
        1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1
    ]

    total_frames = sum(word2phone)  # 计算word2phone列表中所有元素的和
    print(word_level_feature.shape)  # 打印word_level_feature的形状
    print(word2phone)  # 打印word2phone列表
    phone_level_feature = []  # 创建一个空列表phone_level_feature
    for i in range(len(word2phone)):  # 遍历word2phone的长度
        print(word_level_feature[i].shape)  # 打印word_level_feature[i]的形状

        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)  # 将word_level_feature[i]重复word2phone[i]次，沿着第1维度
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 对phone_level_feature进行拼接，沿着第0维度
    print(phone_level_feature.shape)  # 打印phone_level_feature的形状
```

# `D:\src\Bert-VITS2\oldVersion\V200\text\cleaner.py`

```python
# 导入模块 chinese, japanese, english, cleaned_text_to_sequence
from . import chinese, japanese, english, cleaned_text_to_sequence

# 创建语言模块映射
language_module_map = {"ZH": chinese, "JP": japanese, "EN": english}

# 定义函数 clean_text，用于清洗文本
def clean_text(text, language):
    # 获取对应语言的模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取音素、音调和词转音素映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph

# 定义函数 clean_text_bert，用于对文本进行 BERT 处理
def clean_text_bert(text, language):
    # 获取对应语言的模块
    language_module = language_module_map[language]
    # 对文本进行规范化处理
    norm_text = language_module.text_normalize(text)
    # 获取音素、音调和词转音素映射
    phones, tones, word2ph = language_module.g2p(norm_text)
    # 获取 BERT 特征
    bert = language_module.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert

# 定义函数 text_to_sequence，用于将文本转换为序列
def text_to_sequence(text, language):
    # 对文本进行清洗
    norm_text, phones, tones, word2ph = clean_text(text, language)
    # 将清洗后的文本转换为序列
    return cleaned_text_to_sequence(phones, tones, language)

# 主程序入口
if __name__ == "__main__":
    pass
```

# `D:\src\Bert-VITS2\oldVersion\V200\text\english.py`

```python
import pickle  # 导入pickle模块
import os  # 导入os模块
import re  # 导入re模块
from g2p_en import G2p  # 从g2p_en模块导入G2p类

from . import symbols  # 从当前目录导入symbols模块

current_file_path = os.path.dirname(__file__)  # 获取当前文件路径
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")  # 拼接文件路径
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")  # 拼接文件路径
_g2p = G2p()  # 创建G2p对象

# arpa列表
arpa = {
    "AH0",
    "S",
    "AH1",
    ...
    "ZH",
    "AE0",
    "IH2",
    "IH",
    "Y",
    "JH",
    "P",
    "AY1",
    "EY0",
    "OY2",
    "TH",
    "HH",
    "D",
    "ER0",
    "CH",
    "AO1",
    "AE1",
    "AO2",
    "OY1",
    "AY2",
    "IH1",
    "OW0",
    "L",
    "SH",
}

# 替换音素
def post_replace_ph(ph):
    rep_map = {
        "：": ",",
        "；": ",",
        ...
        "v": "V",
    }
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

eng_dict = get_dict()  # 获取字典

# 优化音素
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

# 正则表达式和数字处理
_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")
# ...
```

# `D:\src\Bert-VITS2\oldVersion\V200\text\english_bert_mock.py`

```python
import sys  # 导入sys模块

import torch  # 导入torch模块
from transformers import DebertaV2Model, DebertaV2Tokenizer  # 从transformers模块中导入DebertaV2Model和DebertaV2Tokenizer类

from config import config  # 从config模块中导入config类

LOCAL_PATH = "./bert/deberta-v3-large"  # 设置本地路径

tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)  # 使用本地路径初始化DebertaV2Tokenizer对象

models = dict()  # 创建一个空字典

def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):  # 定义一个名为get_bert_feature的函数，接受text、word2ph和device三个参数
    if (  # 如果条件判断
        sys.platform == "darwin"  # 当前操作系统为"darwin"
        and torch.backends.mps.is_available()  # torch后端支持多进程
        and device == "cpu"  # 设备为CPU
    ):  # 条件判断结束
        device = "mps"  # 将device设置为"mps"
    if not device:  # 如果device不存在
        device = "cuda"  # 将device设置为"cuda"
    if device not in models.keys():  # 如果device不在models的键中
        models[device] = DebertaV2Model.from_pretrained(LOCAL_PATH).to(device)  # 使用本地路径初始化DebertaV2Model对象，并将其移到指定的device上
    with torch.no_grad():  # 使用torch.no_grad()上下文管理器
        inputs = tokenizer(text, return_tensors="pt")  # 使用tokenizer对text进行编码，返回PyTorch张量
        for i in inputs:  # 遍历inputs
            inputs[i] = inputs[i].to(device)  # 将inputs中的每个张量移到指定的device上
        res = models[device](**inputs, output_hidden_states=True)  # 使用models中对应device的模型对inputs进行预测，输出隐藏状态
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 将隐藏状态的最后三层拼接起来，然后取第一个样本，并将其移到CPU上
    # assert len(word2ph) == len(text)+2  # 断言word2ph的长度等于text的长度加2
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 创建一个空列表phone_level_feature
    for i in range(len(word2phone)):  # 遍历word2phone的长度
        repeat_feature = res[i].repeat(word2phone[i], 1)  # 将res中的第i个元素重复word2phone[i]次，沿着第二个维度重复
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 将phone_level_feature中的元素沿着第一个维度拼接起来

    return phone_level_feature.T  # 返回phone_level_feature的转置
```

# `D:\src\Bert-VITS2\oldVersion\V200\text\japanese.py`

```python
# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
import re  # Import the regular expression module
import unicodedata  # Import the unicodedata module

from transformers import AutoTokenizer  # Import the AutoTokenizer class from the transformers module

from . import punctuation, symbols  # Import punctuation and symbols from the current package

from num2words import num2words  # Import the num2words function from the num2words module

import pyopenjtalk  # Import the pyopenjtalk module
import jaconv  # Import the jaconv module

# Define a function to convert katakana text to phonemes
def kata2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""
    # ... (function implementation)

# Define a function to convert hiragana to katakana
def hira2kata(text: str) -> str:
    # ... (function implementation)

# Define a regular expression pattern for matching marks
_MARKS = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# Define a function to convert text to katakana
def text2kata(text: str) -> str:
    # ... (function implementation)

# Define a function to convert text to separated katakana
def text2sep_kata(text: str) -> (list, list):
    # ... (function implementation)

# Define a function to get accent from parsed text
def get_accent(parsed):
    # ... (function implementation)

# Define a dictionary of alpha symbols and their corresponding pronunciations
_ALPHASYMBOL_YOMI = {
    # ... (dictionary content)
}

# Define regular expression patterns for matching numbers and currency symbols
_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")
_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
_CURRENCY_RX = re.compile(r"([$¥£€])([0-9.]*[0-9])")
_NUMBER_RX = re.compile(r"[0-9]+(\.[0-9]+)?")

# Define a function to convert numbers to words in Japanese
def japanese_convert_numbers_to_words(text: str) -> str:
    # ... (function implementation)

# Define a function to convert alpha symbols to words in Japanese
def japanese_convert_alpha_symbols_to_words(text: str) -> str:
    # ... (function implementation)

# Define a function to convert Japanese text to phonemes
def japanese_text_to_phonemes(text: str) -> str:
    # ... (function implementation)

# Define a function to check if a character is a Japanese character
def is_japanese_character(char):
    # ... (function implementation)

# Define a dictionary for replacing punctuation
rep_map = {
    # ... (dictionary content)
}

# Define a function to replace punctuation in text
def replace_punctuation(text):
    # ... (function implementation)

# Define a function to normalize text
def text_normalize(text):
    # ... (function implementation)

# Define a function to distribute phones to words
def distribute_phone(n_phone, n_word):
    # ... (function implementation)

# Define a function to handle long phonemes
def handle_long(sep_phonemes):
    # ... (function implementation)

# Create an instance of the AutoTokenizer class
tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese")

# Define a function to align tones with phonemes
def align_tones(phones, tones):
    # ... (function implementation)

# Define a function to convert text to phonemes using g2p
def g2p(norm_text):
    # ... (function implementation)

# If the script is executed as the main program
if __name__ == "__main__":
    # Create an instance of the AutoTokenizer class
    tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese")
    # Define a sample text
    text = "hello,こんにちは、世界ー！……"
    # Import the get_bert_feature function from the japanese_bert module
    from text.japanese_bert import get_bert_feature
    # Normalize the sample text
    text = text_normalize(text)
    # Print the normalized text
    print(text)
    # Convert the normalized text to phonemes using the g2p function
    phones, tones, word2ph = g2p(text)
    # Get the BERT feature for the normalized text
    bert = get_bert_feature(text, word2ph)
    # Print the phonemes, tones, word2ph, and the shape of the BERT feature
    print(phones, tones, word2ph, bert.shape)
```

# `D:\src\Bert-VITS2\oldVersion\V200\text\japanese_bert.py`

```python
import sys  # 导入sys模块

import torch  # 导入torch模块
from transformers import AutoModelForMaskedLM, AutoTokenizer  # 从transformers模块中导入AutoModelForMaskedLM和AutoTokenizer类

from config import config  # 从config模块中导入config变量
from .japanese import text2sep_kata  # 从japanese模块中导入text2sep_kata函数

LOCAL_PATH = "./bert/deberta-v2-large-japanese"  # 设置LOCAL_PATH变量为"./bert/deberta-v2-large-japanese"

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)  # 使用AutoTokenizer类从预训练模型路径LOCAL_PATH中加载tokenizer

models = dict()  # 创建一个空的字典models

def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):  # 定义get_bert_feature函数，接受text、word2ph和device三个参数
    sep_text, _, _ = text2sep_kata(text)  # 调用text2sep_kata函数，将结果赋值给sep_text
    sep_tokens = [tokenizer.tokenize(t) for t in sep_text]  # 使用tokenizer对sep_text中的每个文本进行分词，将结果存储在sep_tokens列表中
    sep_ids = [tokenizer.convert_tokens_to_ids(t) for t in sep_tokens]  # 将sep_tokens中的每个分词转换为对应的id，将结果存储在sep_ids列表中
    sep_ids = [2] + [item for sublist in sep_ids for item in sublist] + [3]  # 对sep_ids进行处理，将结果重新赋值给sep_ids
    return get_bert_feature_with_token(sep_ids, word2ph, device)  # 调用get_bert_feature_with_token函数，返回结果

def get_bert_feature_with_token(tokens, word2ph, device=config.bert_gen_config.device):  # 定义get_bert_feature_with_token函数，接受tokens、word2ph和device三个参数
    if (  # 如果条件判断
        sys.platform == "darwin"  # 当前操作系统为"darwin"
        and torch.backends.mps.is_available()  # torch的mps后端可用
        and device == "cpu"  # device为"cpu"
    ):  # 条件判断结束
        device = "mps"  # 将device赋值为"mps"
    if not device:  # 如果device不存在
        device = "cuda"  # 将device赋值为"cuda"
    if device not in models.keys():  # 如果device不在models的键中
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)  # 使用AutoModelForMaskedLM类从预训练模型路径LOCAL_PATH中加载模型，并将其移动到device上
    with torch.no_grad():  # 使用torch.no_grad()上下文管理器
        inputs = torch.tensor(tokens).to(device).unsqueeze(0)  # 将tokens转换为tensor，并移动到device上，然后在第0维度上增加一个维度
        token_type_ids = torch.zeros_like(inputs).to(device)  # 创建与inputs相同形状的全零tensor，并移动到device上
        attention_mask = torch.ones_like(inputs).to(device)  # 创建与inputs相同形状的全一tensor，并移动到device上
        inputs = {  # 创建inputs字典
            "input_ids": inputs,  # "input_ids"键对应的值为inputs
            "token_type_ids": token_type_ids,  # "token_type_ids"键对应的值为token_type_ids
            "attention_mask": attention_mask,  # "attention_mask"键对应的值为attention_mask
        }  # 字典创建结束

        # for i in inputs:
        #     inputs[i] = inputs[i].to(device)
        res = models[device](**inputs, output_hidden_states=True)  # 使用models中对应device的模型对inputs进行预测，返回结果
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 对模型预测结果进行处理，将结果赋值给res
    assert inputs["input_ids"].shape[-1] == len(word2ph)  # 断言inputs["input_ids"]的最后一个维度长度等于word2ph的长度
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 创建一个空列表phone_level_feature
    for i in range(len(word2phone)):  # 遍历word2phone的长度
        repeat_feature = res[i].repeat(word2phone[i], 1)  # 对res[i]进行重复，将结果存储在repeat_feature中
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature列表中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 对phone_level_feature进行拼接，将结果赋值给phone_level_feature
    return phone_level_feature.T  # 返回phone_level_feature的转置
```

# `D:\src\Bert-VITS2\oldVersion\V200\text\symbols.py`

```python
punctuation = ["!", "?", "…", ",", ".", "'", "-"]  # List of punctuation marks
pu_symbols = punctuation + ["SP", "UNK"]  # Combine punctuation marks with special symbols
pad = "_"  # Define padding symbol

# chinese
zh_symbols = [  # List of Chinese symbols
    "E",
    "En",
    "a",
    ...
    "OO",
]
num_zh_tones = 6  # Number of Chinese tones

# japanese
ja_symbols = [  # List of Japanese symbols
    "N",
    "a",
    ...
    "zy",
]
num_ja_tones = 2  # Number of Japanese tones

# English
en_symbols = [  # List of English symbols
    "aa",
    "ae",
    ...
    "zh",
]
num_en_tones = 4  # Number of English tones

# combine all symbols
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))  # Combine all symbols and sort them
symbols = [pad] + normal_symbols + pu_symbols  # Combine symbols with padding and special symbols
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]  # Get the indices of special symbols

# combine all tones
num_tones = num_zh_tones + num_ja_tones + num_en_tones  # Calculate total number of tones

# language maps
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}  # Map language IDs to numbers
num_languages = len(language_id_map.keys())  # Get the total number of languages

language_tone_start_map = {  # Map language to starting tone index
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
}

if __name__ == "__main__":
    a = set(zh_symbols)  # Create a set of Chinese symbols
    b = set(en_symbols)  # Create a set of English symbols
    print(sorted(a & b))  # Print the intersection of Chinese and English symbols
```

# `D:\src\Bert-VITS2\oldVersion\V200\text\tone_sandhi.py`

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

# `D:\src\Bert-VITS2\oldVersion\V200\text\__init__.py`

```python
from .symbols import *

# Create a dictionary that maps each symbol to its corresponding ID
_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    # Convert each symbol in the cleaned text to its corresponding ID
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    # Add tone start to each tone
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    # Create a list of language IDs corresponding to the symbols
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids


def get_bert(norm_text, word2ph, language, device):
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert_mock import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert

    # Create a mapping of language to BERT feature function
    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}
    # Get BERT feature using the corresponding language's BERT feature function
    bert = lang_bert_func_map[language](norm_text, word2ph, device)
    return bert


def check_bert_models():
    import json
    from pathlib import Path

    from config import config
    from .bert_utils import _check_bert

    if config.mirror.lower() == "openi":
        import openi

        kwargs = {"token": config.openi_token} if config.openi_token else {}
        openi.login(**kwargs)

    with open("./bert/bert_models.json", "r") as fp:
        models = json.load(fp)
        for k, v in models.items():
            local_path = Path("./bert").joinpath(k)
            # Check if the BERT model files are available locally
            _check_bert(v["repo_id"], v["files"], local_path)
```

# `D:\src\Bert-VITS2\oldVersion\V210\emo_gen.py`

```python
import librosa  # 导入librosa库
import numpy as np  # 导入numpy库
import torch  # 导入torch库
import torch.nn as nn  # 导入torch.nn库
from torch.utils.data import Dataset  # 从torch.utils.data库中导入Dataset类
from transformers import Wav2Vec2Processor  # 从transformers库中导入Wav2Vec2Processor类
from transformers.models.wav2vec2.modeling_wav2vec2 import (  # 从transformers.models.wav2vec2.modeling_wav2vec2库中导入Wav2Vec2Model和Wav2Vec2PreTrainedModel类
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from config import config  # 从config库中导入config模块


class RegressionHead(nn.Module):  # 定义RegressionHead类，继承自nn.Module类
    r"""Classification head."""

    def __init__(self, config):  # 定义初始化方法
        super().__init__()  # 调用父类的初始化方法

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 定义全连接层
        self.dropout = nn.Dropout(config.final_dropout)  # 定义dropout层
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)  # 定义全连接层

    def forward(self, features, **kwargs):  # 定义前向传播方法
        x = features  # 将features赋值给x
        x = self.dropout(x)  # dropout层
        x = self.dense(x)  # 全连接层
        x = torch.tanh(x)  # tanh激活函数
        x = self.dropout(x)  # dropout层
        x = self.out_proj(x)  # 全连接层

        return x  # 返回x


class EmotionModel(Wav2Vec2PreTrainedModel):  # 定义EmotionModel类，继承自Wav2Vec2PreTrainedModel类
    r"""Speech emotion classifier."""

    def __init__(self, config):  # 定义初始化方法
        super().__init__(config)  # 调用父类的初始化方法

        self.config = config  # 将config赋值给self.config
        self.wav2vec2 = Wav2Vec2Model(config)  # 定义Wav2Vec2Model模型
        self.classifier = RegressionHead(config)  # 定义RegressionHead分类器
        self.init_weights()  # 初始化权重

    def forward(  # 定义前向传播方法
        self,
        input_values,
    ):
        outputs = self.wav2vec2(input_values)  # Wav2Vec2Model模型的前向传播
        hidden_states = outputs[0]  # 获取输出的隐藏状态
        hidden_states = torch.mean(hidden_states, dim=1)  # 对隐藏状态进行平均
        logits = self.classifier(hidden_states)  # 使用分类器进行分类

        return hidden_states, logits  # 返回隐藏状态和logits


class AudioDataset(Dataset):  # 定义AudioDataset类，继承自Dataset类
    def __init__(self, list_of_wav_files, sr, processor):  # 定义初始化方法
        self.list_of_wav_files = list_of_wav_files  # 将list_of_wav_files赋值给self.list_of_wav_files
        self.processor = processor  # 将processor赋值给self.processor
        self.sr = sr  # 将sr赋值给self.sr

    def __len__(self):  # 定义__len__方法
        return len(self.list_of_wav_files)  # 返回list_of_wav_files的长度

    def __getitem__(self, idx):  # 定义__getitem__方法
        wav_file = self.list_of_wav_files[idx]  # 获取指定索引的wav文件
        audio_data, _ = librosa.load(wav_file, sr=self.sr)  # 加载音频数据
        processed_data = self.processor(audio_data, sampling_rate=self.sr)[  # 处理音频数据
            "input_values"
        ][0]
        return torch.from_numpy(processed_data)  # 返回处理后的数据


device = config.emo_gen_config.device  # 获取设备
model_name = "./emotional/wav2vec2-large-robust-12-ft-emotion-msp-dim"  # 模型名称
processor = Wav2Vec2Processor.from_pretrained(model_name)  # 从预训练模型中加载Wav2Vec2Processor
model = EmotionModel.from_pretrained(model_name).to(device)  # 从预训练模型中加载EmotionModel，并移动到指定设备


def process_func(  # 定义process_func函数
    x: np.ndarray,
    sampling_rate: int,
    model: EmotionModel,
    processor: Wav2Vec2Processor,
    device: str,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""
    model = model.to(device)  # 将模型移动到指定设备
    y = processor(x, sampling_rate=sampling_rate)  # 处理音频数据
    y = y["input_values"][0]  # 获取处理后的数据
    y = torch.from_numpy(y).unsqueeze(0).to(device)  # 将数据转换为张量并移动到指定设备

    # run through model
    with torch.no_grad():  # 不计算梯度
        y = model(y)[0 if embeddings else 1]  # 通过模型进行前向传播

    # convert to numpy
    y = y.detach().cpu().numpy()  # 将张量转换为numpy数组

    return y  # 返回结果


def get_emo(path):  # 定义get_emo函数
    wav, sr = librosa.load(path, 16000)  # 加载音频数据
    return process_func(  # 返回process_func函数的结果
        np.expand_dims(wav, 0).astype(np.float64),
        sr,
        model,
        processor,
        device,
        embeddings=True,
    ).squeeze(0)
```