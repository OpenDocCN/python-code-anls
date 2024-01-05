# `d:/src/tocomm/Bert-VITS2\oldVersion\V220\__init__.py`

```
"""
@Desc: 2.2版本兼容 对应版本 v2.2 Clap-Enhanced prompt audio generation
"""
import numpy as np  # 导入numpy库，用于数值计算
import torch  # 导入torch库，用于构建神经网络
import commons  # 导入commons模块
from .text import cleaned_text_to_sequence, get_bert  # 从text模块中导入cleaned_text_to_sequence和get_bert函数
from .text.cleaner import clean_text  # 从text模块中导入clean_text函数
from .clap_wrapper import get_clap_audio_feature, get_clap_text_feature  # 从clap_wrapper模块中导入get_clap_audio_feature和get_clap_text_feature函数


def get_text(text, language_str, hps, device, style_text=None, style_weight=0.7):
    # 在此处实现当前版本的get_text
    norm_text, phone, tone, word2ph = clean_text(text, language_str)  # 调用clean_text函数，对文本进行清洗，得到规范化文本、音素、语调和word2ph映射
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)  # 调用cleaned_text_to_sequence函数，将音素、语调和语言字符串转换为序列

    if hps.data.add_blank:  # 如果hps.data.add_blank为True
        phone = commons.intersperse(phone, 0)  # 调用commons模块中的intersperse函数，在phone序列中插入0
        tone = commons.intersperse(tone, 0)  # 调用commons模块中的intersperse函数，在tone序列中插入0
        language = commons.intersperse(language, 0)  # 调用commons模块中的intersperse函数，在language序列中插入0
        # 遍历 word2ph 列表，将每个元素乘以2
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        # 将 word2ph 列表的第一个元素加1
        word2ph[0] += 1
    # 调用 get_bert 函数，传入参数 norm_text, word2ph, language_str, device, style_text=None, style_weight=0.7，并将返回值赋给 bert_ori
    bert_ori = get_bert(
        norm_text, word2ph, language_str, device, style_text=None, style_weight=0.7
    )
    # 删除 word2ph 列表
    del word2ph
    # 断言 bert_ori 的最后一个维度长度等于 phone 的长度，如果不等则抛出异常
    assert bert_ori.shape[-1] == len(phone), phone

    # 根据 language_str 的值进行不同的赋值操作
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
        en_bert = bert_ori  # 将变量 bert_ori 的值赋给 en_bert
    else:  # 如果条件不成立
        raise ValueError("language_str should be ZH, JP or EN")  # 抛出数值错误，提示 language_str 应该是 ZH、JP 或 EN

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"  # 断言，如果条件不成立则抛出异常，提示 bert 序列长度与 phone 长度不相等

    phone = torch.LongTensor(phone)  # 将 phone 转换为 LongTensor 类型
    tone = torch.LongTensor(tone)  # 将 tone 转换为 LongTensor 类型
    language = torch.LongTensor(language)  # 将 language 转换为 LongTensor 类型
    return bert, ja_bert, en_bert, phone, tone, language  # 返回 bert, ja_bert, en_bert, phone, tone, language 变量的值


def infer(
    text,
    emotion,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,  # 缩放比例
    sid,  # sid
    language,  # 语言
    hps,  # hps
    net_g,  # 网络g
    device,  # 设备
    reference_audio=None,  # 参考音频，默认为None
    skip_start=False,  # 是否跳过开头，默认为False
    skip_end=False,  # 是否跳过结尾，默认为False
    style_text=None,  # 风格文本，默认为None
    style_weight=0.7,  # 风格权重，默认为0.7
):
    if isinstance(reference_audio, np.ndarray):  # 如果参考音频是numpy数组
        emo = get_clap_audio_feature(reference_audio, device)  # 获取音频特征
    else:
        emo = get_clap_text_feature(emotion, device)  # 获取文本特征
    emo = torch.squeeze(emo, dim=1)  # 压缩维度为1的张量

    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(  # 获取文本信息
        text, language, hps, device  # 文本内容、语言、hps、设备
    )  # 如果存在 skip_start，则对 phones、tones、lang_ids、bert、ja_bert、en_bert 进行切片操作
    if skip_start:  # 如果 skip_start 存在
        phones = phones[3:]  # 对 phones 进行切片操作，去掉前3个元素
        tones = tones[3:]  # 对 tones 进行切片操作，去掉前3个元素
        lang_ids = lang_ids[3:]  # 对 lang_ids 进行切片操作，去掉前3个元素
        bert = bert[:, 3:]  # 对 bert 进行切片操作，去掉第一维的前3个元素
        ja_bert = ja_bert[:, 3:]  # 对 ja_bert 进行切片操作，去掉第一维的前3个元素
        en_bert = en_bert[:, 3:]  # 对 en_bert 进行切片操作，去掉第一维的前3个元素
    if skip_end:  # 如果 skip_end 存在
        phones = phones[:-2]  # 对 phones 进行切片操作，去掉最后2个元素
        tones = tones[:-2]  # 对 tones 进行切片操作，去掉最后2个元素
        lang_ids = lang_ids[:-2]  # 对 lang_ids 进行切片操作，去掉最后2个元素
        bert = bert[:, :-2]  # 对 bert 进行切片操作，去掉第一维的最后2个元素
        ja_bert = ja_bert[:, :-2]  # 对 ja_bert 进行切片操作，去掉第一维的最后2个元素
        en_bert = en_bert[:, :-2]  # 对 en_bert 进行切片操作，去掉第一维的最后2个元素
    with torch.no_grad():  # 使用 torch.no_grad() 上下文管理器
        x_tst = phones.to(device).unsqueeze(0)  # 将 phones 转移到指定设备，并在第一维度增加一个维度
        tones = tones.to(device).unsqueeze(0)  # 将 tones 转移到指定设备，并在第一维度增加一个维度
        lang_ids = lang_ids.to(device).unsqueeze(0)  # 将 lang_ids 转移到指定设备，并在第一维度增加一个维度
        bert = bert.to(device).unsqueeze(0)  # 将 bert 转移到指定设备，并在第一维度增加一个维度
        # 将ja_bert转换为指定设备上的张量，并在第0维上增加一个维度
        ja_bert = ja_bert.to(device).unsqueeze(0)
        # 将en_bert转换为指定设备上的张量，并在第0维上增加一个维度
        en_bert = en_bert.to(device).unsqueeze(0)
        # 创建一个包含phones长度的长整型张量，并转换为指定设备
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        # 将emo转换为指定设备上的张量，并在第0维上增加一个维度
        emo = emo.to(device).unsqueeze(0)
        # 删除phones变量
        del phones
        # 创建一个包含sid对应的speakers的长整型张量，并转换为指定设备
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        # 调用net_g的infer方法进行推断，传入相关参数
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
        length_scale=length_scale,  # 设置长度比例
    )[0][0, 0]  # 从结果中获取特定位置的值
    .data.cpu()  # 将数据移动到 CPU 上
    .float()  # 将数据类型转换为浮点型
    .numpy()  # 将数据转换为 NumPy 数组
)
del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers, ja_bert, en_bert, emo  # 删除不再需要的变量，释放内存
if torch.cuda.is_available():  # 检查是否有可用的 CUDA 设备
    torch.cuda.empty_cache()  # 清空 CUDA 缓存
return audio  # 返回音频数据


def infer_multilang(
    text,  # 文本输入
    sdp_ratio,  # SDP 比例
    noise_scale,  # 噪音比例
    noise_scale_w,  # 噪音比例权重
    length_scale,  # 长度比例
    sid,  # 会话 ID
    language,  # 语言
    hps,  # 参数hps
    net_g,  # 参数net_g
    device,  # 参数device
    reference_audio=None,  # 参数reference_audio，默认值为None
    emotion=None,  # 参数emotion，默认值为None
    skip_start=False,  # 参数skip_start，默认值为False
    skip_end=False,  # 参数skip_end，默认值为False
):
    bert, ja_bert, en_bert, phones, tones, lang_ids = [], [], [], [], [], []  # 初始化变量bert, ja_bert, en_bert, phones, tones, lang_ids为空列表
    # emo = get_emo_(reference_audio, emotion, sid)  # 调用get_emo_函数，获取情感特征
    if isinstance(reference_audio, np.ndarray):  # 判断reference_audio是否为numpy数组
        emo = get_clap_audio_feature(reference_audio, device)  # 调用get_clap_audio_feature函数，获取音频特征
    else:
        emo = get_clap_text_feature(emotion, device)  # 调用get_clap_text_feature函数，获取文本特征
    emo = torch.squeeze(emo, dim=1)  # 对emo进行维度压缩
    for idx, (txt, lang) in enumerate(zip(text, language)):  # 遍历text和language的元素
        skip_start = (idx != 0) or (skip_start and idx == 0)  # 更新skip_start的值
        skip_end = (idx != len(text) - 1) or (skip_end and idx == len(text) - 1)  # 更新skip_end的值
        (
            temp_bert,  # 临时变量temp_bert
        temp_ja_bert,  # 从 get_text 函数返回的结果中获取日语 BERT 数据
        temp_en_bert,  # 从 get_text 函数返回的结果中获取英语 BERT 数据
        temp_phones,  # 从 get_text 函数返回的结果中获取音素数据
        temp_tones,  # 从 get_text 函数返回的结果中获取音调数据
        temp_lang_ids,  # 从 get_text 函数返回的结果中获取语言 ID 数据
    ) = get_text(txt, lang, hps, device)  # 调用 get_text 函数获取文本数据

    if skip_start:  # 如果需要跳过开头部分
        temp_bert = temp_bert[:, 3:]  # 从开头部分去掉前3个元素
        temp_ja_bert = temp_ja_bert[:, 3:]  # 从开头部分去掉前3个元素
        temp_en_bert = temp_en_bert[:, 3:]  # 从开头部分去掉前3个元素
        temp_phones = temp_phones[3:]  # 从开头部分去掉前3个元素
        temp_tones = temp_tones[3:]  # 从开头部分去掉前3个元素
        temp_lang_ids = temp_lang_ids[3:]  # 从开头部分去掉前3个元素

    if skip_end:  # 如果需要跳过结尾部分
        temp_bert = temp_bert[:, :-2]  # 从结尾部分去掉最后2个元素
        temp_ja_bert = temp_ja_bert[:, :-2]  # 从结尾部分去掉最后2个元素
        temp_en_bert = temp_en_bert[:, :-2]  # 从结尾部分去掉最后2个元素
        temp_phones = temp_phones[:-2]  # 从结尾部分去掉最后2个元素
        temp_tones = temp_tones[:-2]  # 从结尾部分去掉最后2个元素
        temp_lang_ids = temp_lang_ids[:-2]  # 从结尾部分去掉最后2个元素
        # 将temp_bert添加到bert列表中
        bert.append(temp_bert)
        # 将temp_ja_bert添加到ja_bert列表中
        ja_bert.append(temp_ja_bert)
        # 将temp_en_bert添加到en_bert列表中
        en_bert.append(temp_en_bert)
        # 将temp_phones添加到phones列表中
        phones.append(temp_phones)
        # 将temp_tones添加到tones列表中
        tones.append(temp_tones)
        # 将temp_lang_ids添加到lang_ids列表中
        lang_ids.append(temp_lang_ids)
    # 在维度1上拼接bert列表中的张量
    bert = torch.concatenate(bert, dim=1)
    # 在维度1上拼接ja_bert列表中的张量
    ja_bert = torch.concatenate(ja_bert, dim=1)
    # 在维度1上拼接en_bert列表中的张量
    en_bert = torch.concatenate(en_bert, dim=1)
    # 在维度0上拼接phones列表中的张量
    phones = torch.concatenate(phones, dim=0)
    # 在维度0上拼接tones列表中的张量
    tones = torch.concatenate(tones, dim=0)
    # 在维度0上拼接lang_ids列表中的张量
    lang_ids = torch.concatenate(lang_ids, dim=0)
    # 在没有梯度的上下文中执行以下代码
    with torch.no_grad():
        # 将phones转移到设备上并在维度0上增加一个维度
        x_tst = phones.to(device).unsqueeze(0)
        # 将tones转移到设备上并在维度0上增加一个维度
        tones = tones.to(device).unsqueeze(0)
        # 将lang_ids转移到设备上并在维度0上增加一个维度
        lang_ids = lang_ids.to(device).unsqueeze(0)
        # 将bert转移到设备上并在维度0上增加一个维度
        bert = bert.to(device).unsqueeze(0)
        # 将ja_bert转移到设备上并在维度0上增加一个维度
        ja_bert = ja_bert.to(device).unsqueeze(0)
        # 将en_bert转移到设备上并在维度0上增加一个维度
        en_bert = en_bert.to(device).unsqueeze(0)
        # 将emo转移到设备上并在维度0上增加一个维度
        emo = emo.to(device).unsqueeze(0)
        # 创建一个 LongTensor，包含 phones 的长度，并将其移动到指定的设备上
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        # 释放 phones 占用的内存空间
        del phones
        # 创建一个 LongTensor，包含 speakers 对应的 id，并将其移动到指定的设备上
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        # 调用 net_g 的 infer 方法进行推理，得到音频数据
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
            # 将音频数据移动到 CPU 上
            .data.cpu()
        .float()  # 将数据转换为浮点数类型
        .numpy()  # 将数据转换为 NumPy 数组
        )  # 结束函数调用
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers, ja_bert, en_bert, emo  # 删除变量以释放内存
        if torch.cuda.is_available():  # 检查是否有可用的 CUDA 设备
            torch.cuda.empty_cache()  # 清空 CUDA 缓存
        return audio  # 返回 audio 变量的值
```