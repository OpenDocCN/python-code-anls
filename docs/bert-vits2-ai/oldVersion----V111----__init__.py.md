# `Bert-VITS2\oldVersion\V111\__init__.py`

```

"""
1.1.1版本兼容
https://github.com/fishaudio/Bert-VITS2/releases/tag/1.1.1
"""
# 导入 torch 库
import torch
# 导入 commons 模块
import commons
# 从 text 模块中导入 clean_text 和 clean_text_fix 函数
from .text.cleaner import clean_text, clean_text_fix
# 从 text 模块中导入 cleaned_text_to_sequence 函数
from .text import cleaned_text_to_sequence
# 从 text 模块中导入 get_bert 和 get_bert_fix 函数
from .text import get_bert, get_bert_fix

# 定义函数 get_text，接收文本、语言字符串、hps 和设备作为参数
def get_text(text, language_str, hps, device):
    # 调用 clean_text 函数，清理文本并返回规范化的文本、音素、音调和单词到音素的映射
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    # 将音素、音调和语言字符串转换为序列
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 如果 hps.data.add_blank 为真
    if hps.data.add_blank:
        # 在音素、音调和语言序列中插入空白符
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        # 对单词到音素的映射进行处理
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    # 获取规范化文本的 BERT 表示
    bert = get_bert(norm_text, word2ph, language_str, device)
    # 释放 word2ph 占用的内存
    del word2ph
    # 断言 BERT 表示的长度与音素序列的长度相等
    assert bert.shape[-1] == len(phone), phone

    # 根据语言字符串进行不同的处理
    if language_str == "ZH":
        bert = bert
        ja_bert = torch.zeros(768, len(phone))
    elif language_str == "JP":
        ja_bert = bert
        bert = torch.zeros(1024, len(phone))
    else:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))

    # 断言 BERT 表示的长度与音素序列的长度相等
    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    # 将音素、音调和语言序列转换为 LongTensor 类型
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    # 返回 BERT 表示、日语 BERT 表示、音素、音调和语言序列
    return bert, ja_bert, phone, tone, language

# 定义函数 get_text_fix，接收文本、语言字符串、hps 和设备作为参数
def get_text_fix(text, language_str, hps, device):
    # 调用 clean_text_fix 函数，清理文本并返回规范化的文本、音素、音调和单词到音素的映射
    norm_text, phone, tone, word2ph = clean_text_fix(text, language_str)
    # 将音素、音调和语言字符串转换为序列
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 如果 hps.data.add_blank 为真
    if hps.data.add_blank:
        # 在音素、音调和语言序列中插入空白符
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        # 对单词到音素的映射进行处理
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    # 获取规范化文本的 BERT 表示
    bert = get_bert_fix(norm_text, word2ph, language_str, device)
    # 释放 word2ph 占用的内存
    del word2ph
    # 断言 BERT 表示的长度与音素序列的长度相等
    assert bert.shape[-1] == len(phone), phone

    # 根据语言字符串进行不同的处理
    if language_str == "ZH":
        bert = bert
        ja_bert = torch.zeros(768, len(phone))
    elif language_str == "JP":
        ja_bert = bert
        bert = torch.zeros(1024, len(phone))
    else:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))

    # 断言 BERT 表示的长度与音素序列的长度相等
    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    # 将音素、音调和语言序列转换为 LongTensor 类型
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    # 返回 BERT 表示、日语 BERT 表示、音素、音调和语言序列
    return bert, ja_bert, phone, tone, language

# 定义函数 infer，接收文本、sdp_ratio、noise_scale、noise_scale_w、length_scale、sid、language、hps、net_g 和设备作为参数
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
    # 获取文本的 BERT 表示、日语 BERT 表示、音素、音调和语言序列
    bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps, device)
    # 禁用梯度计算
    with torch.no_grad():
        # 将音素转换为指定设备上的张量，并添加一个维度
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        # 释放 phones 占用的内存
        del phones
        # 将说话者 ID 转换为指定设备上的张量
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        # 生成音频
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
        )
        # 释放内存
        del x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, ja_bert
        # 如果 CUDA 可用，则清空缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 返回生成的音频
        return audio

# 定义函数 infer_fix，接收文本、sdp_ratio、noise_scale、noise_scale_w、length_scale、sid、language、hps、net_g 和设备作为参数
def infer_fix(
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
    # 获取文本的 BERT 表示、日语 BERT 表示、音素、音调和语言序列
    bert, ja_bert, phones, tones, lang_ids = get_text_fix(text, language, hps, device)
    # 禁用梯度计算
    with torch.no_grad():
        # 将音素转换为指定设备上的张量，并添加一个维度
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        # 释放 phones 占用的内存
        del phones
        # 将说话者 ID 转换为指定设备上的张量
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        # 生成音频
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
        )
        # 释放内存
        del x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, ja_bert
        # 如果 CUDA 可用，则清空缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 返回生成的音频
        return audio

```