# `Bert-VITS2\oldVersion\V220\__init__.py`

```py
# 导入所需的库
import numpy as np
import torch
import commons
from .text import cleaned_text_to_sequence, get_bert
from .text.cleaner import clean_text
from .clap_wrapper import get_clap_audio_feature, get_clap_text_feature

# 定义函数get_text，用于生成文本对应的特征
def get_text(text, language_str, hps, device, style_text=None, style_weight=0.7):
    # 清洗文本，获取规范化文本、音素、语调和单词到音素的映射
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    # 将音素、语调和语言转换为序列
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 如果数据中添加了空白符
    if hps.data.add_blank:
        # 在音素、语调和语言序列中插入空白符
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        # 将单词到音素的映射中的值扩大两倍，并在第一个值上加一
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    # 获取规范化文本对应的BERT特征
    bert_ori = get_bert(
        norm_text, word2ph, language_str, device, style_text=None, style_weight=0.7
    )
    # 释放word2ph的内存
    del word2ph
    # 断言BERT特征的长度与音素序列的长度相等
    assert bert_ori.shape[-1] == len(phone), phone

    # 根据语言类型生成对应的BERT特征
    if language_str == "ZH":
        bert = bert_ori
        ja_bert = torch.rand(1024, len(phone))
        en_bert = torch.rand(1024, len(phone))
    elif language_str == "JP":
        bert = torch.rand(1024, len(phone))
        ja_bert = bert_ori
        en_bert = torch.rand(1024, len(phone))
    elif language_str == "EN":
        bert = torch.rand(1024, len(phone))
        ja_bert = torch.rand(1024, len(phone))
        en_bert = bert_ori
    else:
        raise ValueError("language_str should be ZH, JP or EN")

    # 断言BERT特征的长度与音素序列的长度相等
    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    # 将音素、语调和语言转换为张量
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    # 返回生成的BERT特征和音素、语调、语言序列
    return bert, ja_bert, en_bert, phone, tone, language

# 定义推断函数infer，用于生成音频
def infer(
    text,
    emotion,
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
    skip_start=False,
    # 设置一个布尔值变量，用于控制是否跳过结尾
    skip_end=False,
    # 设置一个文本样式变量，用于存储文本的样式信息
    style_text=None,
    # 设置一个权重值变量，用于控制样式的权重
    style_weight=0.7,
    # 检查 reference_audio 是否为 numpy 数组，如果是则获取音频特征，否则获取文本特征
    if isinstance(reference_audio, np.ndarray):
        emo = get_clap_audio_feature(reference_audio, device)
    else:
        emo = get_clap_text_feature(emotion, device)
    # 压缩维度为1的张量
    emo = torch.squeeze(emo, dim=1)

    # 获取文本特征和相关信息
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text, language, hps, device
    )
    # 如果需要跳过开头部分，则去除前3个元素
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        bert = bert[:, 3:]
        ja_bert = ja_bert[:, 3:]
        en_bert = en_bert[:, 3:]
    # 如果需要跳过结尾部分，则去除末尾2个元素
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        bert = bert[:, :-2]
        ja_bert = ja_bert[:, :-2]
        en_bert = en_bert[:, :-2]
    # 使用无梯度计算环境，将数据转移到设备上，并添加一个维度
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        en_bert = en_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        emo = emo.to(device).unsqueeze(0)
        del phones
        # 将说话者信息转移到设备上
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        # 使用神经网络进行推断，获取音频数据
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
                emo,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        # 释放内存，返回音频数据
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers, ja_bert, en_bert, emo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio

# 多语言推断函数
def infer_multilang(
    text,
    sdp_ratio,
    # 噪声比例
    noise_scale,
    # 噪声比例权重
    noise_scale_w,
    # 长度比例
    length_scale,
    # sid
    sid,
    # 语言
    language,
    # hps
    hps,
    # 神经网络 G
    net_g,
    # 设备
    device,
    # 参考音频，默认为 None
    reference_audio=None,
    # 情感，默认为 None
    emotion=None,
    # 是否跳过开始部分，默认为 False
    skip_start=False,
    # 是否跳过结束部分，默认为 False
    skip_end=False,
    # 初始化空列表用于存储各种特征
    bert, ja_bert, en_bert, phones, tones, lang_ids = [], [], [], [], [], []
    # 根据输入的参考音频和情感获取特征
    if isinstance(reference_audio, np.ndarray):
        # 如果参考音频是 numpy 数组，则获取音频特征
        emo = get_clap_audio_feature(reference_audio, device)
    else:
        # 如果参考音频不是 numpy 数组，则获取文本特征
        emo = get_clap_text_feature(emotion, device)
    # 去除维度为1的维度
    emo = torch.squeeze(emo, dim=1)
    # 遍历文本和语言列表
    for idx, (txt, lang) in enumerate(zip(text, language)):
        # 判断是否跳过开头
        skip_start = (idx != 0) or (skip_start and idx == 0)
        # 判断是否跳过结尾
        skip_end = (idx != len(text) - 1) or (skip_end and idx == len(text) - 1)
        # 获取文本特征
        (
            temp_bert,
            temp_ja_bert,
            temp_en_bert,
            temp_phones,
            temp_tones,
            temp_lang_ids,
        ) = get_text(txt, lang, hps, device)
        # 如果需要跳过开头，则去除开头部分特征
        if skip_start:
            temp_bert = temp_bert[:, 3:]
            temp_ja_bert = temp_ja_bert[:, 3:]
            temp_en_bert = temp_en_bert[:, 3:]
            temp_phones = temp_phones[3:]
            temp_tones = temp_tones[3:]
            temp_lang_ids = temp_lang_ids[3:]
        # 如果需要跳过结尾，则去除结尾部分特征
        if skip_end:
            temp_bert = temp_bert[:, :-2]
            temp_ja_bert = temp_ja_bert[:, :-2]
            temp_en_bert = temp_en_bert[:, :-2]
            temp_phones = temp_phones[:-2]
            temp_tones = temp_tones[:-2]
            temp_lang_ids = temp_lang_ids[:-2]
        # 将获取的特征添加到对应的列表中
        bert.append(temp_bert)
        ja_bert.append(temp_ja_bert)
        en_bert.append(temp_en_bert)
        phones.append(temp_phones)
        tones.append(temp_tones)
        lang_ids.append(temp_lang_ids)
    # 沿指定维度拼接张量
    bert = torch.concatenate(bert, dim=1)
    ja_bert = torch.concatenate(ja_bert, dim=1)
    en_bert = torch.concatenate(en_bert, dim=1)
    phones = torch.concatenate(phones, dim=0)
    tones = torch.concatenate(tones, dim=0)
    lang_ids = torch.concatenate(lang_ids, dim=0)
    # 使用 torch.no_grad() 上下文管理器，关闭梯度计算
    with torch.no_grad():
        # 将 phones 转移到指定设备，并在第0维度增加一个维度
        x_tst = phones.to(device).unsqueeze(0)
        # 将 tones 转移到指定设备，并在第0维度增加一个维度
        tones = tones.to(device).unsqueeze(0)
        # 将 lang_ids 转移到指定设备，并在第0维度增加一个维度
        lang_ids = lang_ids.to(device).unsqueeze(0)
        # 将 bert 转移到指定设备，并在第0维度增加一个维度
        bert = bert.to(device).unsqueeze(0)
        # 将 ja_bert 转移到指定设备，并在第0维度增加一个维度
        ja_bert = ja_bert.to(device).unsqueeze(0)
        # 将 en_bert 转移到指定设备，并在第0维度增加一个维度
        en_bert = en_bert.to(device).unsqueeze(0)
        # 将 emo 转移到指定设备，并在第0维度增加一个维度
        emo = emo.to(device).unsqueeze(0)
        # 创建一个包含 phones 大小的 LongTensor，并转移到指定设备
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        # 释放 phones 占用的内存
        del phones
        # 创建一个包含 sid 对应的 speaker id 的 LongTensor，并转移到指定设备
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        # 使用 net_g 模型进行推断，得到音频数据
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
                emo,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        # 释放占用的内存
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers, ja_bert, en_bert, emo
        # 如果可用 CUDA，则清空 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 返回音频数据
        return audio
```