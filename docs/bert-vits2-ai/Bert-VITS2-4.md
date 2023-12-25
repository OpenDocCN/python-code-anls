# Bert-VITS2 源码解析 4

# `D:\src\Bert-VITS2\oldVersion\V210\models.py`

```py
# The given code is a PyTorch implementation of a synthesizer for training. The model is quite complex and consists of multiple components such as TextEncoder, Generator, PosteriorEncoder, StochasticDurationPredictor, DurationPredictor, ReferenceEncoder, and others. The model is used for training and inference.

# The code is quite long and consists of multiple classes and methods. It is used to train a synthesizer model and perform inference. The model takes input features and produces output features. The model is quite complex and consists of multiple components such as TextEncoder, Generator, PosteriorEncoder, StochasticDurationPredictor, DurationPredictor, ReferenceEncoder, and others. The model is used for training and inference.
```

# `D:\src\Bert-VITS2\oldVersion\V210\__init__.py`

```python
"""
@Desc: 2.1版本兼容 对应版本 v2.1 Emo and muti-lang optimize
"""
import torch  # 导入torch模块
import commons  # 导入commons模块
from .text import cleaned_text_to_sequence, get_bert  # 从text模块导入cleaned_text_to_sequence和get_bert函数
from .text.cleaner import clean_text  # 从text模块的cleaner子模块导入clean_text函数

def get_text(text, language_str, hps, device, style_text=None, style_weight=0.7):
    # 在此处实现当前版本的get_text
    norm_text, phone, tone, word2ph = clean_text(text, language_str)  # 调用clean_text函数，获取norm_text, phone, tone, word2ph
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)  # 调用cleaned_text_to_sequence函数，获取phone, tone, language

    if hps.data.add_blank:  # 如果hps.data.add_blank为真
        phone = commons.intersperse(phone, 0)  # 调用commons模块的intersperse函数
        tone = commons.intersperse(tone, 0)  # 调用commons模块的intersperse函数
        language = commons.intersperse(language, 0)  # 调用commons模块的intersperse函数
        for i in range(len(word2ph)):  # 遍历word2ph的长度
            word2ph[i] = word2ph[i] * 2  # 对word2ph的每个元素乘以2
        word2ph[0] += 1  # 对word2ph的第一个元素加1
    bert_ori = get_bert(  # 调用get_bert函数
        norm_text, word2ph, language_str, device, style_text, style_weight
    )
    del word2ph  # 删除word2ph
    assert bert_ori.shape[-1] == len(phone), phone  # 断言bert_ori的最后一个维度等于phone的长度

    if language_str == "ZH":  # 如果language_str为"ZH"
        bert = bert_ori  # bert等于bert_ori
        ja_bert = torch.zeros(1024, len(phone))  # 创建一个1024xlen(phone)的全零张量
        en_bert = torch.zeros(1024, len(phone))  # 创建一个1024xlen(phone)的全零张量
    elif language_str == "JP":  # 如果language_str为"JP"
        bert = torch.zeros(1024, len(phone))  # 创建一个1024xlen(phone)的全零张量
        ja_bert = bert_ori  # ja_bert等于bert_ori
        en_bert = torch.zeros(1024, len(phone))  # 创建一个1024xlen(phone)的全零张量
    elif language_str == "EN":  # 如果language_str为"EN"
        bert = torch.zeros(1024, len(phone))  # 创建一个1024xlen(phone)的全零张量
        ja_bert = torch.zeros(1024, len(phone))  # 创建一个1024xlen(phone)的全零张量
        en_bert = bert_ori  # en_bert等于bert_ori
    else:  # 否则
        raise ValueError("language_str should be ZH, JP or EN")  # 抛出ValueError异常��提示language_str应为ZH、JP或EN

    assert bert.shape[-1] == len(  # 断言bert的最后一个维度等于
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"  # phone的长度

    phone = torch.LongTensor(phone)  # 将phone转换为LongTensor类型
    tone = torch.LongTensor(tone)  # 将tone转换为LongTensor类型
    language = torch.LongTensor(language)  # 将language转换为LongTensor类型
    return bert, ja_bert, en_bert, phone, tone, language  # 返回bert, ja_bert, en_bert, phone, tone, language

def get_emo_(reference_audio, emotion):
    from .emo_gen import get_emo  # 从emo_gen模块导入get_emo函数

    emo = (  # emo等于
        torch.from_numpy(get_emo(reference_audio))  # 调用get_emo函数并转换为torch张量
        if reference_audio  # 如果reference_audio存在
        else torch.Tensor([emotion])  # 否则为一个包含emotion的张量
    )
    return emo  # 返回emo

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
    reference_audio=None,
    emotion=None,
    skip_start=False,
    skip_end=False,
    style_text=None,
    style_weight=0.7,
):
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(  # 调用get_text函数
        text, language, hps, device, style_text, style_weight
    )
    emo = get_emo_(reference_audio, emotion)  # 调用get_emo_函数
    if skip_start:  # 如果skip_start为真
        phones = phones[1:]  # phones从第二个元素开始切片
        tones = tones[1:]  # tones从第二个元素开始切片
        lang_ids = lang_ids[1:]  # lang_ids从第二个元素开始切片
        bert = bert[:, 1:]  # bert的第二个维度开始切片
        ja_bert = ja_bert[:, 1:]  # ja_bert的第二个维度开始切片
        en_bert = en_bert[:, 1:]  # en_bert的第二个维度开始切片
    if skip_end:  # 如果skip_end为真
        phones = phones[:-1]  # phones切片到倒数第二个元素
        tones = tones[:-1]  # tones切片到倒数第二个元素
        lang_ids = lang_ids[:-1]  # lang_ids切片到倒数第二个元素
        bert = bert[:, :-1]  # bert切片到倒数第二个维度
        ja_bert = ja_bert[:, :-1]  # ja_bert切片到倒数第二个维度
        en_bert = en_bert[:, :-1]  # en_bert切片到倒数第二个维度
    with torch.no_grad():  # 禁用梯度计算
        x_tst = phones.to(device).unsqueeze(0)  # 将phones转换为device类型并增加一个维度
        tones = tones.to(device).unsqueeze(0)  # 将tones转换为device类型并增加一个维度
        lang_ids = lang_ids.to(device).unsqueeze(0)  # 将lang_ids转换为device类型并增加一个维度
        bert = bert.to(device).unsqueeze(0)  # 将bert转换为device类型并增加一个维度
        ja_bert = ja_bert.to(device).unsqueeze(0)  # 将ja_bert转换为device类型并增加一个维度
        en_bert = en_bert.to(device).unsqueeze(0)  # 将en_bert转换为device类型并增加一个维度
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)  # 创建一个包含phones长度的LongTensor类型张量
        emo = emo.to(device).unsqueeze(0)  # 将emo转换为device类型并增加一个维度
        del phones  # 删除phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)  # 创建一个包含hps.data.spk2id[sid]的LongTensor类型张量
        audio = (  # audio等于
            net_g.infer(  # 调用net_g的infer方法
                x_tst,  # 传入参数x_tst
                x_tst_lengths,  # 传入参数x_tst_lengths
                speakers,  # 传入参数speakers
                tones,  # 传入参数tones
                lang_ids,  # 传入参数lang_ids
                bert,  # 传入参数bert
                ja_bert,  # 传入参数ja_bert
                en_bert,  # 传入参数en_bert
                emo,  # 传入参数emo
                sdp_ratio=sdp_ratio,  # 传入参数sdp_ratio
                noise_scale=noise_scale,  # 传入参数noise_scale
                noise_scale_w=noise_scale_w,  # 传入参数noise_scale_w
                length_scale=length_scale,  # 传入参数length_scale
            )[0][0, 0]  # 获取net_g.infer方法的返回值
            .data.cpu()  # 将返回值转移到cpu上
            .float()  # 转换为float类型
            .numpy()  # 转换为numpy数组
        )
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers, ja_bert, en_bert, emo  # 删除变量
        if torch.cuda.is_available():  # 如果cuda可用
            torch.cuda.empty_cache()  # 清空缓存
        return audio  # 返回audio

def infer_multilang(
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
    reference_audio=None,
    emotion=None,
    skip_start=False,
    skip_end=False,
):
    bert, ja_bert, en_bert, phones, tones, lang_ids = [], [], [], [], [], []  # 初始化bert, ja_bert, en_bert, phones, tones, lang_ids为空列表
    emo = get_emo_(reference_audio, emotion)  # 调用get_emo_函数
    for idx, (txt, lang) in enumerate(zip(text, language)):  # 遍历text和language的元素
        skip_start = (idx != 0) or (skip_start and idx == 0)  # 计算skip_start
        skip_end = (idx != len(text) - 1) or (skip_end and idx == len(text) - 1)  # 计算skip_end
        (
            temp_bert,
            temp_ja_bert,
            temp_en_bert,
            temp_phones,
            temp_tones,
            temp_lang_ids,
        ) = get_text(txt, lang, hps, device)  # 调用get_text函数
        if skip_start:  # 如果skip_start为真
            temp_bert = temp_bert[:, 1:]  # temp_bert的第二个维度开始切片
            temp_ja_bert = temp_ja_bert[:, 1:]  # temp_ja_bert的第二个维度开始切片
            temp_en_bert = temp_en_bert[:, 1:]  # temp_en_bert的第二个维度开始切片
            temp_phones = temp_phones[1:]  # temp_phones从第二个元素开始切片
            temp_tones = temp_tones[1:]  # temp_tones从第二个元素开始切片
            temp_lang_ids = temp_lang_ids[1:]  # temp_lang_ids从第二个元素开始切片
        if skip_end:  # 如果skip_end为真
            temp_bert = temp_bert[:, :-1]  # temp_bert切片到倒数第二个维度
            temp_ja_bert = temp_ja_bert[:, :-1]  # temp_ja_bert切片到倒数第二个维度
            temp_en_bert = temp_en_bert[:, :-1]  # temp_en_bert切片到倒数第二个维度
            temp_phones = temp_phones[:-1]  # temp_phones切片到倒数第二个元素
            temp_tones = temp_tones[:-1]  # temp_tones切片到倒数第二个元素
            temp_lang_ids = temp_lang_ids[:-1]  # temp_lang_ids切片到倒数第二个元素
        bert.append(temp_bert)  # 将temp_bert添加到bert列表中
        ja_bert.append(temp_ja_bert)  # 将temp_ja_bert添加到ja_bert列表中
        en_bert.append(temp_en_bert)  # 将temp_en_bert添加到en_bert列表中
        phones.append(temp_phones)  # 将temp_phones添加到phones列表中
        tones.append(temp_tones)  # 将temp_tones添加到tones列表中
        lang_ids.append(temp_lang_ids)  # 将temp_lang_ids添加到lang_ids列表中
    bert = torch.concatenate(bert, dim=1)  # 在bert列表上进行维度拼接
    ja_bert = torch.concatenate(ja_bert, dim=1)  # 在ja_bert列表上进行维度拼接
    en_bert = torch.concatenate(en_bert, dim=1)  # 在en_bert列表上进行维度拼接
    phones = torch.concatenate(phones, dim=0)  # 在phones列表上进行维度拼接
    tones = torch.concatenate(tones, dim=0)  # 在tones列表上进行维度拼接
    lang_ids = torch.concatenate(lang_ids, dim=0)  # 在lang_ids列表上进行维度拼接
    with torch.no_grad():  # 禁用梯度计算
        x_tst = phones.to(device).unsqueeze(0)  # 将phones转换为device类型并增加一个维度
        tones = tones.to(device).unsqueeze(0)  # 将tones转换为device类型并增加一个维度
        lang_ids = lang_ids.to(device).unsqueeze(0)  # 将lang_ids转换为device类型并增加一个维度
        bert = bert.to(device).unsqueeze(0)  # 将bert转换为device类型并增加一个维度
        ja_bert = ja_bert.to(device).unsqueeze(0)  # 将ja_bert转换为device类型并增加一个维度
        en_bert = en_bert.to(device).unsqueeze(0)  # 将en_bert转换为device类型并增加一个维度
        emo = emo.to(device).unsqueeze(0)  # 将emo转换为device类型并增加一个维度
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)  # 创建一个包含phones长度的LongTensor类型张量
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)  # 创建一个包含hps.data.spk2id[sid]的LongTensor类型张量
        audio = (  # audio等于
            net_g.infer(  # 调用net_g的infer方法
                x_tst,  # 传入参数x_tst
                x_tst_lengths,  # 传入参数x_tst_lengths
                speakers,  # 传入参数speakers
                tones,  # 传入参数tones
                lang_ids,  # 传入参数lang_ids
                bert,  # 传入参数bert
                ja_bert,  # 传入参数ja_bert
                en_bert,  # 传入参数en_bert
                emo,  # 传入参数emo
                sdp_ratio=sdp_ratio,  # 传入参数sdp_ratio
                noise_scale=noise_scale,  # 传入参数noise_scale
                noise_scale_w=noise_scale_w,  # 传入参数noise_scale_w
                length_scale=length_scale,  # 传入参数length_scale
            )[0][0, 0]  # 获取net_g.infer方法的返回值
            .data.cpu()  # 将返回值转移到cpu上
            .float()  # 转换为float类型
            .numpy()  # 转换为numpy数组
        )
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers, ja_bert, en_bert, emo  # 删除变量
        if torch.cuda.is_available():  # 如果cuda可用
            torch.cuda.empty_cache()  # 清空缓存
        return audio  # 返回audio
```

# `D:\src\Bert-VITS2\oldVersion\V210\text\bert_utils.py`

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

# `D:\src\Bert-VITS2\oldVersion\V210\text\chinese.py`

```python
import os  # 导入os模块
import re  # 导入re模块
import cn2an  # 导入cn2an模块
from pypinyin import lazy_pinyin, Style  # 从pypinyin模块中导入lazy_pinyin和Style
from .symbols import punctuation  # 从symbols模块中导入punctuation
from .tone_sandhi import ToneSandhi  # 从tone_sandhi模块中导入ToneSandhi

current_file_path = os.path.dirname(__file__)  # 获取当前文件所在目录的路径
pinyin_to_symbol_map = {  # 创建pinyin_to_symbol_map字典
    line.split("\t")[0]: line.strip().split("\t")[1]  # 以"\t"分割行并创建键值对
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()  # 读取opencpop-strict.txt文件的每一行
}

import jieba.posseg as psg  # 导入jieba.posseg模块并重命名为psg

rep_map = {  # 创建rep_map字典
    "：": ",",  # 键值对
    "；": ",",  # 键值对
    # ... 其他键值对
}

tone_modifier = ToneSandhi()  # 创建ToneSandhi类的实例对象


def replace_punctuation(text):  # 定义函数replace_punctuation，参数为text
    # 函数内部代码


def g2p(text):  # 定义函数g2p，参数为text
    # 函数内部代码


def _get_initials_finals(word):  # 定义函数_get_initials_finals，参数为word
    # 函数内部代码


def _g2p(segments):  # 定义函数_g2p，参数为segments
    # 函数内部代码


def text_normalize(text):  # 定义函数text_normalize，参数为text
    # 函数内部代码


def get_bert_feature(text, word2ph):  # 定义函数get_bert_feature，参数为text和word2ph
    # 函数内部代码


if __name__ == "__main__":  # 如果当前模块是主程序
    from text.chinese_bert import get_bert_feature  # 从text.chinese_bert模块中导入get_bert_feature

    # 函数内部代码
```

# `D:\src\Bert-VITS2\oldVersion\V210\text\chinese_bert.py`

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

# `D:\src\Bert-VITS2\oldVersion\V210\text\cleaner.py`

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

# `D:\src\Bert-VITS2\oldVersion\V210\text\english.py`

```python
import pickle  # Import the pickle module
import os  # Import the os module
import re  # Import the re module
from g2p_en import G2p  # Import the G2p class from the g2p_en module
from transformers import DebertaV2Tokenizer  # Import the DebertaV2Tokenizer class from the transformers module
from . import symbols  # Import the symbols module from the current package

current_file_path = os.path.dirname(__file__)  # Get the directory path of the current file
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")  # Join the current file path with "cmudict.rep"
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")  # Join the current file path with "cmudict_cache.pickle"
_g2p = G2p()  # Create an instance of the G2p class
LOCAL_PATH = "./bert/deberta-v3-large"  # Set the LOCAL_PATH variable to "./bert/deberta-v3-large"
tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)  # Initialize the tokenizer using the DebertaV2Tokenizer class and the LOCAL_PATH

# ... (skipping the long list of "arpa" and other functions)

if __name__ == "__main__":
    print(g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."))  # Print the result of the g2p function with the given text
```
Note: The comments added explain the purpose of each statement in the code.
```

# `D:\src\Bert-VITS2\oldVersion\V210\text\english_bert_mock.py`

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

# `D:\src\Bert-VITS2\oldVersion\V210\text\japanese.py`

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

# `D:\src\Bert-VITS2\oldVersion\V210\text\japanese_bert.py`

```python
import sys  # 导入sys模块

import torch  # 导入torch模块
from transformers import AutoModelForMaskedLM, AutoTokenizer  # 从transformers模块中导入AutoModelForMaskedLM和AutoTokenizer类

from config import config  # 从config模块中导入config变量
from .japanese import text2sep_kata  # 从当前目录下的japanese模块中导入text2sep_kata函数

LOCAL_PATH = "./bert/deberta-v2-large-japanese-char-wwm"  # 设置LOCAL_PATH变量为"./bert/deberta-v2-large-japanese-char-wwm"

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)  # 使用AutoTokenizer类从预训练模型路径LOCAL_PATH中实例化tokenizer对象

models = dict()  # 创建一个空字典models

def get_bert_feature(
    text,
    word2ph,
    device=config.bert_gen_config.device,
    style_text=None,
    style_weight=0.7,
):
    text = "".join(text2sep_kata(text)[0])  # 将text转换为片假名形式
    if style_text:
        style_text = "".join(text2sep_kata(style_text)[0])  # 如果style_text存在，则将其转换为片假名形式
    if (
        sys.platform == "darwin"  # 如果当前操作系统是macOS
        and torch.backends.mps.is_available()  # 并且torch后端支持MPS
        and device == "cpu"  # 并且设备为CPU
    ):
        device = "mps"  # 将设备设置为"mps"
    if not device:  # 如果设备不存在
        device = "cuda"  # 将设备设置为"cuda"
    if device not in models.keys():  # 如果设备不在models字典的键中
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)  # 从预训练模型路径LOCAL_PATH中实例化AutoModelForMaskedLM对象，并将其移到指定设备
    with torch.no_grad():  # 使用torch.no_grad()上下文管理器
        inputs = tokenizer(text, return_tensors="pt")  # 使用tokenizer对text进行编码，并返回PyTorch张量
        for i in inputs:  # 遍历inputs中的键
            inputs[i] = inputs[i].to(device)  # 将inputs中的值移动到指定设备
        res = models[device](**inputs, output_hidden_states=True)  # 使用models中对应设备的模型对inputs进行预测，并返回隐藏状态
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 将隐藏状态的倒数第3到倒数第2层的输出拼接起来，并移动到CPU
        if style_text:  # 如果style_text存在
            style_inputs = tokenizer(style_text, return_tensors="pt")  # 使用tokenizer对style_text进行编码，并返回PyTorch张量
            for i in style_inputs:  # 遍历style_inputs中的键
                style_inputs[i] = style_inputs[i].to(device)  # 将style_inputs中的值移动到指定设备
            style_res = models[device](**style_inputs, output_hidden_states=True)  # 使用models中对应设备的模型对style_inputs进行预测，并返回隐藏状态
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()  # 将隐藏状态的倒数第3到倒数第2层的输出拼接起来，并移动到CPU
            style_res_mean = style_res.mean(0)  # 计算style_res的均值

    assert len(word2ph) == len(text) + 2  # 断言word2ph的长度等于text的长度加2
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 创建一个空列表phone_level_feature
    for i in range(len(word2phone)):  # 遍历word2phone的长度
        if style_text:  # 如果style_text存在
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)  # 重复res[i]，并根据style_weight进行加权
                + style_res_mean.repeat(word2phone[i], 1) * style_weight  # 重复style_res_mean，并根据style_weight进行加权
            )
        else:  # 如果style_text不存在
            repeat_feature = res[i].repeat(word2phone[i], 1)  # 重复res[i]
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 沿着0维度拼接phone_level_feature

    return phone_level_feature.T  # 返回phone_level_feature的转置
```

# `D:\src\Bert-VITS2\oldVersion\V210\text\symbols.py`

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

# `D:\src\Bert-VITS2\oldVersion\V210\text\tone_sandhi.py`

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

# `D:\src\Bert-VITS2\oldVersion\V210\text\__init__.py`

```python
# Import symbols from symbols module
from .symbols import *

# Create a dictionary mapping symbols to their corresponding IDs
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

# Define a function to convert cleaned text to a sequence of IDs corresponding to the symbols in the text
def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids

# Define a function to get BERT features for a given text
def get_bert(norm_text, word2ph, language, device, style_text, style_weight):
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert_mock import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert

    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}
    bert = lang_bert_func_map[language](
        norm_text, word2ph, device, style_text, style_weight
    )
    return bert

# Define a function to check BERT models
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
            _check_bert(v["repo_id"], v["files"], local_path)

# Call the check_bert_models function
check_bert_models()
```

# `D:\src\Bert-VITS2\oldVersion\V220\clap_gen.py`

```python
import argparse  # 导入argparse库
from multiprocessing import Pool, cpu_count  # 从multiprocessing库中导入Pool和cpu_count

import torch  # 导入torch库
import torch.multiprocessing as mp  # 导入torch.multiprocessing库并重命名为mp
from tqdm import tqdm  # 从tqdm库中导入tqdm

import utils  # 导入utils库
from config import config  # 从config库中导入config
from .clap_wrapper import get_clap_audio_feature  # 从clap_wrapper库中导入get_clap_audio_feature
import librosa  # 导入librosa库
import os  # 导入os库

os.environ["OMP_NUM_THREADS"] = "1"  # 设置环境变量OMP_NUM_THREADS为1
os.environ["MKL_NUM_THREADS"] = "1"  # 设置环境变量MKL_NUM_THREADS为1

# 定义函数process_line，参数为line
def process_line(line):
    device = config.emo_gen_config.device  # 从config.emo_gen_config中获取device
    if config.emo_gen_config.use_multi_device:  # 如果config.emo_gen_config.use_multi_device为True
        rank = mp.current_process()._identity  # 获取当前进程的标识
        rank = rank[0] if len(rank) > 0 else 0  # 如果rank的长度大于0，取rank[0]，否则为0
        if torch.cuda.is_available():  # 如果torch.cuda可用
            gpu_id = rank % torch.cuda.device_count()  # 计算gpu_id
            device = torch.device(f"cuda:{gpu_id}")  # 设置device为cuda:gpu_id
        else:  # 否则
            device = torch.device("cpu")  # 设置device为cpu
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")  # 从line中分割出wav_path, _, language_str, text, phones, tone, word2ph

    clap_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".emo.npy")  # 将wav_path中的.wav替换为.emo.npy
    if os.path.isfile(clap_path):  # 如果clap_path对应的文件存在
        return  # 返回

    audio = librosa.load(wav_path, 48000)[0]  # 以48000的采样率加载wav_path对应的音频文件，并取第一个返回值
    # audio = librosa.resample(audio, 44100, 48000)  # 对音频进行重采样

    clap = get_clap_audio_feature(audio, device)  # 调用get_clap_audio_feature函数，传入audio和device作为参数，并将返回值赋给clap
    torch.save(clap, clap_path)  # 保存clap到clap_path

# 如果当前脚本为主程序
if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建ArgumentParser对象parser
    parser.add_argument(  # 添加参数
        "-c", "--config", type=str, default=config.emo_gen_config.config_path  # 参数名为-c或--config，类型为str，缺省值为config.emo_gen_config.config_path
    )
    parser.add_argument(  # 添加参数
        "--num_processes", type=int, default=config.emo_gen_config.num_processes  # 参数名为--num_processes，类型为int，缺省值为config.emo_gen_config.num_processes
    )
    args, _ = parser.parse_known_args()  # 解析命令行参数
    config_path = args.config  # 从args中获取config参数，赋值给config_path
    hps = utils.get_hparams_from_file(config_path)  # 调用get_hparams_from_file函数，传入config_path作为参数，并将返回值赋给hps
    lines = []  # 创建空列表lines
    with open(hps.data.training_files, encoding="utf-8") as f:  # 打开hps.data.training_files文件
        lines.extend(f.readlines())  # 将文件中的每一行添加到lines列表中

    with open(hps.data.validation_files, encoding="utf-8") as f:  # 打开hps.data.validation_files文件
        lines.extend(f.readlines())  # 将文件中的每一行添加到lines列表中
    if len(lines) != 0:  # 如果lines列表不为空
        num_processes = min(args.num_processes, cpu_count())  # 取args.num_processes和cpu_count()的最小值，赋值给num_processes
        with Pool(processes=num_processes) as pool:  # 创建进程池pool
            for _ in tqdm(pool.imap_unordered(process_line, lines), total=len(lines)):  # 遍历进程池的无序映射结果
                pass  # 什么也不做

    print(f"clap生成完毕!, 共有{len(lines)}个emo.pt生成!")  # 打印clap生成完毕的信息和emo.pt生成的数量
```

# `D:\src\Bert-VITS2\oldVersion\V220\clap_wrapper.py`

```python
import sys  # 导入sys模块

import torch  # 导入torch模块
from transformers import ClapModel, ClapProcessor  # 从transformers模块中导入ClapModel和ClapProcessor类

from config import config  # 从config模块中导入config类

models = dict()  # 创建一个空字典models
processor = ClapProcessor.from_pretrained("./emotional/clap-htsat-fused")  # 从预训练模型中加载ClapProcessor

def get_clap_audio_feature(audio_data, device=config.bert_gen_config.device):  # 定义函数get_clap_audio_feature，参数为audio_data和device
    if (  # 如果条件判断
        sys.platform == "darwin"  # 当前操作系统为darwin
        and torch.backends.mps.is_available()  # 并且torch的mps后端可用
        and device == "cpu"  # 并且device为cpu
    ):  # 条件判断结束
        device = "mps"  # 将device设置为"mps"
    if not device:  # 如果device不存在
        device = "cuda"  # 将device设置为"cuda"
    if device not in models.keys():  # 如果device不在models的键中
        models[device] = ClapModel.from_pretrained("./emotional/clap-htsat-fused").to(  # 将models[device]设置为从预训练模型中加载的ClapModel
            device  # 设置device
        )  # 结束赋值
    with torch.no_grad():  # 使用torch的no_grad上下文管理器
        inputs = processor(  # 将processor应用于
            audios=audio_data,  # audio_data
            return_tensors="pt",  # 返回张量类型为"pt"
            sampling_rate=48000  # 采样率为48000
        ).to(device)  # 将结果转移到device
        emb = models[device].get_audio_features(**inputs)  # 将models[device]的音频特征提取应用于inputs
    return emb.T  # 返回emb的转置

def get_clap_text_feature(text, device=config.bert_gen_config.device):  # 定义函数get_clap_text_feature，参数为text和device
    if (  # 如果条件判断
        sys.platform == "darwin"  # 当前操作系统为darwin
        and torch.backends.mps.is_available()  # 并且torch的mps后端可用
        and device == "cpu"  # 并且device为cpu
    ):  # 条件判断结束
        device = "mps"  # 将device设置为"mps"
    if not device:  # 如果device不存在
        device = "cuda"  # 将device设置为"cuda"
    if device not in models.keys():  # 如果device不在models的键中
        models[device] = ClapModel.from_pretrained("./emotional/clap-htsat-fused").to(  # 将models[device]设置为从预训练模型中加载的ClapModel
            device  # 设置device
        )  # 结束赋值
    with torch.no_grad():  # 使用torch的no_grad上下文管理器
        inputs = processor(text=text, return_tensors="pt").to(device)  # 将processor应用于text，返回张量类型为"pt"，并转移到device
        emb = models[device].get_text_features(**inputs)  # 将models[device]的文本特征提取应用于inputs
    return emb.T  # 返回emb的转置
```

# `D:\src\Bert-VITS2\oldVersion\V220\models.py`

```python
import math  # import the math module
import torch  # import the torch module
from torch import nn  # import the nn module from torch
from torch.nn import functional as F  # import the F module from torch.nn

import commons  # import the commons module
import modules  # import the modules module
import attentions  # import the attentions module
import monotonic_align  # import the monotonic_align module

from torch.nn import Conv1d, ConvTranspose1d, Conv2d  # import Conv1d, ConvTranspose1d, Conv2d from torch.nn
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm  # import weight_norm, remove_weight_norm, spectral_norm from torch.nn.utils

from commons import init_weights, get_padding  # import init_weights, get_padding from commons
from text import symbols, num_tones, num_languages  # import symbols, num_tones, num_languages from text

from vector_quantize_pytorch import VectorQuantize  # import VectorQuantize from vector_quantize_pytorch
```

# `D:\src\Bert-VITS2\oldVersion\V220\__init__.py`

```python
"""
@Desc: 2.2版本兼容 对应版本 v2.2 Clap-Enhanced prompt audio generation
"""
import numpy as np  # 导入numpy库
import torch  # 导入torch库
import commons  # 导入commons模块
from .text import cleaned_text_to_sequence, get_bert  # 从text模块导入cleaned_text_to_sequence, get_bert函数
from .text.cleaner import clean_text  # 从text模块的cleaner模块导入clean_text函数
from .clap_wrapper import get_clap_audio_feature, get_clap_text_feature  # 从clap_wrapper模块导入get_clap_audio_feature, get_clap_text_feature函数

def get_text(text, language_str, hps, device, style_text=None, style_weight=0.7):
    # 在此处实现当前版本的get_text
    norm_text, phone, tone, word2ph = clean_text(text, language_str)  # 调用clean_text函数
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)  # 调用cleaned_text_to_sequence函数

    if hps.data.add_blank:  # 判断hps.data.add_blank是否为真
        phone = commons.intersperse(phone, 0)  # 调用commons模块的intersperse函数
        tone = commons.intersperse(tone, 0)  # 调用commons模块的intersperse函数
        language = commons.intersperse(language, 0)  # 调用commons模块的intersperse函数
        for i in range(len(word2ph)):  # 遍历word2ph列表
            word2ph[i] = word2ph[i] * 2  # 对word2ph列表中的元素进行操作
        word2ph[0] += 1  # 对word2ph列表中的第一个元素进行操作
    bert_ori = get_bert(  # 调用get_bert函数
        norm_text, word2ph, language_str, device, style_text=None, style_weight=0.7
    )
    del word2ph  # 删除word2ph变量
    assert bert_ori.shape[-1] == len(phone), phone  # 断言语句

    if language_str == "ZH":  # 判断language_str是否为"ZH"
        bert = bert_ori  # 赋值操作
        ja_bert = torch.rand(1024, len(phone))  # 生成指定大小的张量
        en_bert = torch.rand(1024, len(phone))  # 生成指定大小的张量
    elif language_str == "JP":  # 判断language_str是否为"JP"
        bert = torch.rand(1024, len(phone))  # 生成指定大小的张量
        ja_bert = bert_ori  # 赋值操作
        en_bert = torch.rand(1024, len(phone))  # 生成指定大小的张量
    elif language_str == "EN":  # 判断language_str是否为"EN"
        bert = torch.rand(1024, len(phone))  # 生成指定大小的张量
        ja_bert = torch.rand(1024, len(phone))  # 生成指定大小的张量
        en_bert = bert_ori  # 赋值操作
    else:
        raise ValueError("language_str should be ZH, JP or EN")  # 抛出异常

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"  # 断言语句

    phone = torch.LongTensor(phone)  # 转换为张量
    tone = torch.LongTensor(tone)  # 转换为张量
    language = torch.LongTensor(language)  # 转换为张量
    return bert, ja_bert, en_bert, phone, tone, language  # 返回结果
```

# `D:\src\Bert-VITS2\oldVersion\V220\text\bert_utils.py`

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

# `D:\src\Bert-VITS2\oldVersion\V220\text\chinese.py`

```python
import os  # 导入os模块
import re  # 导入re模块
import cn2an  # 导入cn2an模块
from pypinyin import lazy_pinyin, Style  # 从pypinyin模块中导入lazy_pinyin和Style
from .symbols import punctuation  # 从symbols模块中导入punctuation
from .tone_sandhi import ToneSandhi  # 从tone_sandhi模块中导入ToneSandhi

current_file_path = os.path.dirname(__file__)  # 获取当前文件所在目录的路径
pinyin_to_symbol_map = {  # 创建pinyin_to_symbol_map字典
    line.split("\t")[0]: line.strip().split("\t")[1]  # 以"\t"分割行并创建键值对
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()  # 读取opencpop-strict.txt文件的每一行
}

import jieba.posseg as psg  # 导入jieba.posseg模块并重命名为psg

rep_map = {  # 创建rep_map字典
    "：": ",",  # 键值对
    "；": ",",  # 键值对
    # ... 其他键值对
}

tone_modifier = ToneSandhi()  # 创建ToneSandhi类的实例对象


def replace_punctuation(text):  # 定义函数replace_punctuation，参数为text
    # 函数内部代码


def g2p(text):  # 定义函数g2p，参数为text
    # 函数内部代码


def _get_initials_finals(word):  # 定义函数_get_initials_finals，参数为word
    # 函数内部代码


def _g2p(segments):  # 定义函数_g2p，参数为segments
    # 函数内部代码


def text_normalize(text):  # 定义函数text_normalize，参数为text
    # 函数内部代码


def get_bert_feature(text, word2ph):  # 定义函数get_bert_feature，参数为text和word2ph
    # 函数内部代码


if __name__ == "__main__":  # 如果当前模块是主程序
    from text.chinese_bert import get_bert_feature  # 从text.chinese_bert模块中导入get_bert_feature

    # 函数内部代码
```

# `D:\src\Bert-VITS2\oldVersion\V220\text\chinese_bert.py`

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

# `D:\src\Bert-VITS2\oldVersion\V220\text\cleaner.py`

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

# `D:\src\Bert-VITS2\oldVersion\V220\text\english.py`

```python
import pickle  # Import the pickle module
import os  # Import the os module
import re  # Import the re module
from g2p_en import G2p  # Import the G2p class from the g2p_en module
from transformers import DebertaV2Tokenizer  # Import the DebertaV2Tokenizer class from the transformers module
from . import symbols  # Import the symbols module from the current package

current_file_path = os.path.dirname(__file__)  # Get the directory path of the current file
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")  # Join the current file path with "cmudict.rep"
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")  # Join the current file path with "cmudict_cache.pickle"
_g2p = G2p()  # Create an instance of the G2p class
LOCAL_PATH = "./bert/deberta-v3-large"  # Set the LOCAL_PATH variable to "./bert/deberta-v3-large"
tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)  # Initialize the tokenizer using the DebertaV2Tokenizer class and the LOCAL_PATH

# ... (skipping the long list of "arpa" and other functions)

if __name__ == "__main__":
    print(g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."))  # Print the result of the g2p function with the given text
```
Note: The comments added explain the purpose of each statement in the code.
```

# `D:\src\Bert-VITS2\oldVersion\V220\text\english_bert_mock.py`

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

# `D:\src\Bert-VITS2\oldVersion\V220\text\japanese.py`

```python
# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
import re  # Importing regular expression module
import unicodedata  # Importing unicodedata module

from transformers import AutoTokenizer  # Importing AutoTokenizer from transformers module

from . import punctuation, symbols  # Importing punctuation and symbols from current directory

from num2words import num2words  # Importing num2words module

import pyopenjtalk  # Importing pyopenjtalk module
import jaconv  # Importing jaconv module

# Function to convert katakana text to phonemes
def kata2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""
    text = text.strip()  # Remove leading and trailing whitespaces
    if text == "ー":  # If text is "ー"
        return ["ー"]  # Return a list containing "ー"
    elif text.startswith("ー"):  # If text starts with "ー"
        return ["ー"] + kata2phoneme(text[1:])  # Return a list containing "ー" concatenated with the result of kata2phoneme function called with text[1:]
    res = []  # Initialize an empty list
    prev = None  # Initialize prev to None
    while text:  # While text is not empty
        if re.match(_MARKS, text):  # If text matches the regular expression _MARKS
            res.append(text)  # Append text to res
            text = text[1:]  # Update text to exclude the first character
            continue
        if text.startswith("ー"):  # If text starts with "ー"
            if prev:  # If prev is not None
                res.append(prev[-1])  # Append the last character of prev to res
            text = text[1:]  # Update text to exclude the first character
            continue
        res += pyopenjtalk.g2p(text).lower().replace("cl", "q").split(" ")  # Append the result of g2p function called with text to res after converting to lowercase and replacing "cl" with "q" and splitting by space
        break
    return res  # Return res

# Function to convert hiragana text to katakana
def hira2kata(text: str) -> str:
    return jaconv.hira2kata(text)  # Return the result of hira2kata function called with text

# Set of symbol tokens
_SYMBOL_TOKENS = set(list("・、。？！"))
# Set of tokens without yomi
_NO_YOMI_TOKENS = set(list("「」『』―（）［］[]"))
# Regular expression for matching marks
_MARKS = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# Function to convert text to katakana
def text2kata(text: str) -> str:
    parsed = pyopenjtalk.run_frontend(text)  # Parse the text using run_frontend function

    res = []  # Initialize an empty list
    for parts in parsed:  # For each part in parsed
        word, yomi = replace_punctuation(parts["string"]), parts["pron"].replace(
            "’", ""
        )  # Get word and yomi after replacing punctuation and removing "’"
        if yomi:  # If yomi is not empty
            if re.match(_MARKS, yomi):  # If yomi matches the regular expression _MARKS
                if len(word) > 1:  # If length of word is greater than 1
                    word = [replace_punctuation(i) for i in list(word)]  # Replace punctuation in each character of word
                    yomi = word  # Update yomi to word
                    res += yomi  # Extend res with yomi
                    sep += word  # Extend sep with word
                    continue
                elif word not in rep_map.keys() and word not in rep_map.values():  # If word is not in rep_map keys and values
                    word = ","  # Update word to ","
                yomi = word  # Update yomi to word
            res.append(yomi)  # Append yomi to res
        else:  # If yomi is empty
            if word in _SYMBOL_TOKENS:  # If word is in _SYMBOL_TOKENS
                res.append(word)  # Append word to res
            elif word in ("っ", "ッ"):  # If word is "っ" or "ッ"
                res.append("ッ")  # Append "ッ" to res
            elif word in _NO_YOMI_TOKENS:  # If word is in _NO_YOMI_TOKENS
                pass  # Do nothing
            else:  # For all other cases
                res.append(word)  # Append word to res
    return hira2kata("".join(res))  # Return the result of hira2kata function called with the concatenation of res

# Function to convert text to separated katakana
def text2sep_kata(text: str) -> (list, list):
    parsed = pyopenjtalk.run_frontend(text)  # Parse the text using run_frontend function

    res = []  # Initialize an empty list
    sep = []  # Initialize an empty list
    for parts in parsed:  # For each part in parsed
        word, yomi = replace_punctuation(parts["string"]), parts["pron"].replace(
            "’", ""
        )  # Get word and yomi after replacing punctuation and removing "’"
        if yomi:  # If yomi is not empty
            if re.match(_MARKS, yomi):  # If yomi matches the regular expression _MARKS
                if len(word) > 1:  # If length of word is greater than 1
                    word = [replace_punctuation(i) for i in list(word)]  # Replace punctuation in each character of word
                    yomi = word  # Update yomi to word
                    res += yomi  # Extend res with yomi
                    sep += word  # Extend sep with word
                    continue
                elif word not in rep_map.keys() and word not in rep_map.values():  # If word is not in rep_map keys and values
                    word = ","  # Update word to ","
                yomi = word  # Update yomi to word
            res.append(yomi)  # Append yomi to res
        else:  # If yomi is empty
            if word in _SYMBOL_TOKENS:  # If word is in _SYMBOL_TOKENS
                res.append(word)  # Append word to res
            elif word in ("っ", "ッ"):  # If word is "っ" or "ッ"
                res.append("ッ")  # Append "ッ" to res
            elif word in _NO_YOMI_TOKENS:  # If word is in _NO_YOMI_TOKENS
                pass  # Do nothing
            else:  # For all other cases
                res.append(word)  # Append word to res
        sep.append(word)  # Append word to sep
    return sep, [hira2kata(i) for i in res], get_accent(parsed)  # Return sep, hira2kata of res, and the result of get_accent function called with parsed

# Function to get accent from parsed data
def get_accent(parsed):
    labels = pyopenjtalk.make_label(parsed)  # Make labels from parsed data

    phonemes = []  # Initialize an empty list
    accents = []  # Initialize an empty list
    for n, label in enumerate(labels):  # For each index n and label in labels
        phoneme = re.search(r"\-([^\+]*)\+", label).group(1)  # Search for phoneme in label
        if phoneme not in ["sil", "pau"]:  # If phoneme is not "sil" or "pau"
            phonemes.append(phoneme.replace("cl", "q").lower())  # Append the lowercase of phoneme with "cl" replaced by "q" to phonemes
        else:  # For all other cases
            continue  # Continue to the next iteration
        a1 = int(re.search(r"/A:(\-?[0-9]+)\+", label).group(1))  # Get a1 from label
        a2 = int(re.search(r"\+(\d+)\+", label).group(1))  # Get a2 from label
        if re.search(r"\-([^\+]*)\+", labels[n + 1]).group(1) in ["sil", "pau"]:  # If the next phoneme is "sil" or "pau"
            a2_next = -1  # Set a2_next to -1
        else:  # For all other cases
            a2_next = int(re.search(r"\+(\d+)\+", labels[n + 1]).group(1))  # Get a2_next from the next label
        # Falling
        if a1 == 0 and a2_next == a2 + 1:  # If a1 is 0 and a2_next is a2 + 1
            accents.append(-1)  # Append -1 to accents
        # Rising
        elif a2 == 1 and a2_next == 2:  # If a2 is 1 and a2_next is 2
            accents.append(1)  # Append 1 to accents
        else:  # For all other cases
            accents.append(0)  # Append 0 to accents
    return list(zip(phonemes, accents))  # Return a list of tuples containing phonemes and accents

# Dictionary of alpha symbols and their yomi
_ALPHASYMBOL_YOMI = {
    # ... (omitted for brevity)
}

# Regular expression for matching numbers with separator
_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")
# Dictionary of currency symbols and their names
_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
# Regular expression for matching currency symbols and numbers
_CURRENCY_RX = re.compile(r"([$¥£€])([0-9.]*[0-9])")
# Regular expression for matching numbers
_NUMBER_RX = re.compile(r"[0-9]+(\.[0-9]+)?")

# Function to convert Japanese numbers to words
def japanese_convert_numbers_to_words(text: str) -> str:
    res = _NUMBER_WITH_SEPARATOR_RX.sub(lambda m: m[0].replace(",", ""), text)  # Replace numbers with separator in text
    res = _CURRENCY_RX.sub(lambda m: m[2] + _CURRENCY_MAP.get(m[1], m[1]), res)  # Replace currency symbols and numbers in text
    res = _NUMBER_RX.sub(lambda m: num2words(m[0], lang="ja"), res)  # Replace numbers in text with their words
    return res  # Return res

# Function to convert alpha symbols to words
def japanese_convert_alpha_symbols_to_words(text: str) -> str:
    return "".join([_ALPHASYMBOL_YOMI.get(ch, ch) for ch in text.lower()])  # Return the result of joining the yomi of alpha symbols in lowercase text

# Function to convert Japanese text to phonemes
def japanese_text_to_phonemes(text: str) -> str:
    res = unicodedata.normalize("NFKC", text)  # Normalize text using NFKC
    res = japanese_convert_numbers_to_words(res)  # Convert numbers to words in res
    # res = japanese_convert_alpha_symbols_to_words(res)  # Convert alpha symbols to words in res
    res = text2kata(res)  # Convert res to katakana
    res = kata2phoneme(res)  # Convert res to phonemes
    return res  # Return res

# Function to check if a character is a Japanese character
def is_japanese_character(char):
    # ... (omitted for brevity)
    return False  # Return False

# Dictionary of replacement mappings for punctuation
rep_map = {
    # ... (omitted for brevity)
}

# Function to replace punctuation in text
def replace_punctuation(text):
    # ... (omitted for brevity)
    return replaced_text  # Return replaced_text

# Function to normalize text
def text_normalize(text):
    res = unicodedata.normalize("NFKC", text)  # Normalize text using NFKC
    res = japanese_convert_numbers_to_words(res)  # Convert numbers to words in res
    # res = "".join([i for i in res if is_japanese_character(i)])  # Join characters in res if they are Japanese characters
    res = replace_punctuation(res)  # Replace punctuation in res
    res = res.replace("゙", "")  # Replace "゙" with empty string in res
    return res  # Return res

# Function to distribute phones to words
def distribute_phone(n_phone, n_word):
    # ... (omitted for brevity)
    return phones_per_word  # Return phones_per_word

# Function to handle long phonemes
def handle_long(sep_phonemes):
    # ... (omitted for brevity)
    return sep_phonemes  # Return sep_phonemes

# Load tokenizer from pretrained model
tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese-char-wwm")

# Function to align tones with phones
def align_tones(phones, tones):
    # ... (omitted for brevity)
    return res  # Return res

# Function to rearrange tones
def rearrange_tones(tones, phones):
    # ... (omitted for brevity)
    return res  # Return res

# Function to convert text to phones, tones, word2ph, and bert feature
def g2p(norm_text):
    # ... (omitted for brevity)
    return phones, tones, word2ph, bert.shape  # Return phones, tones, word2ph, and bert.shape

# Main function
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese")
    text = "hello,こんにちは、世界ー！……"
    from text.japanese_bert import get_bert_feature

    text = text_normalize(text)  # Normalize text
    print(text)  # Print text

    phones, tones, word2ph = g2p(text)  # Get phones, tones, and word2ph from text
    bert = get_bert_feature(text, word2ph)  # Get bert feature from text and word2ph

    print(phones, tones, word2ph, bert.shape)  # Print phones, tones, word2ph, and bert.shape
```