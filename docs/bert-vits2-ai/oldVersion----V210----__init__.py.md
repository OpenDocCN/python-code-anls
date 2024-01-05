# `d:/src/tocomm/Bert-VITS2\oldVersion\V210\__init__.py`

```
"""
@Desc: 2.1版本兼容 对应版本 v2.1 Emo and muti-lang optimize
"""
# 导入torch模块
import torch
# 导入commons模块
import commons
# 从当前目录下的text模块中导入cleaned_text_to_sequence和get_bert函数
from .text import cleaned_text_to_sequence, get_bert
# 从text.cleaner模块中导入clean_text函数
from .text.cleaner import clean_text

# 定义get_text函数，接受text, language_str, hps, device, style_text=None, style_weight=0.7等参数
def get_text(text, language_str, hps, device, style_text=None, style_weight=0.7):
    # 调用clean_text函数，返回norm_text, phone, tone, word2ph
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    # 调用cleaned_text_to_sequence函数，返回phone, tone, language
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 如果hps.data.add_blank为True
    if hps.data.add_blank:
        # 在phone列表中插入0
        phone = commons.intersperse(phone, 0)
        # 在tone列表中插入0
        tone = commons.intersperse(tone, 0)
        # 在language列表中插入0
        language = commons.intersperse(language, 0)
        # 遍历word2ph列表
        for i in range(len(word2ph)):
            # 将word2ph[i]乘以2
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1  # 增加 word2ph 列表中第一个元素的值
    bert_ori = get_bert(  # 调用 get_bert 函数，将返回值赋给 bert_ori
        norm_text, word2ph, language_str, device, style_text, style_weight  # get_bert 函数的参数
    )
    del word2ph  # 删除 word2ph 变量
    assert bert_ori.shape[-1] == len(phone), phone  # 断言 bert_ori 的最后一个维度长度等于 phone 的长度

    if language_str == "ZH":  # 如果 language_str 为 "ZH"
        bert = bert_ori  # 将 bert_ori 赋给 bert
        ja_bert = torch.zeros(1024, len(phone))  # 创建一个 1024xlen(phone) 的全零张量，赋给 ja_bert
        en_bert = torch.zeros(1024, len(phone))  # 创建一个 1024xlen(phone) 的全零张量，赋给 en_bert
    elif language_str == "JP":  # 如果 language_str 为 "JP"
        bert = torch.zeros(1024, len(phone))  # 创建一个 1024xlen(phone) 的全零张量，赋给 bert
        ja_bert = bert_ori  # 将 bert_ori 赋给 ja_bert
        en_bert = torch.zeros(1024, len(phone))  # 创建一个 1024xlen(phone) 的全零张量，赋给 en_bert
    elif language_str == "EN":  # 如果 language_str 为 "EN"
        bert = torch.zeros(1024, len(phone))  # 创建一个 1024xlen(phone) 的全零张量，赋给 bert
        ja_bert = torch.zeros(1024, len(phone))  # 创建一个 1024xlen(phone) 的全零张量，赋给 ja_bert
        en_bert = bert_ori  # 将 bert_ori 赋给 en_bert
    else:  # 如果 language_str 不是 "ZH", "JP", "EN"
        raise ValueError("language_str should be ZH, JP or EN")
        # 如果语言字符串不是ZH、JP或EN，则抛出数值错误

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"
    # 断言BERT的最后一个维度长度与phone的长度相等，如果不相等则抛出断言错误

    phone = torch.LongTensor(phone)
    # 将phone转换为长整型张量
    tone = torch.LongTensor(tone)
    # 将tone转换为长整型张量
    language = torch.LongTensor(language)
    # 将language转换为长整型张量
    return bert, ja_bert, en_bert, phone, tone, language
    # 返回bert, ja_bert, en_bert, phone, tone, language

def get_emo_(reference_audio, emotion):
    from .emo_gen import get_emo
    # 从emo_gen模块中导入get_emo函数

    emo = (
        torch.from_numpy(get_emo(reference_audio))
        if reference_audio
        else torch.Tensor([emotion])
    )
    # 如果reference_audio存在，则将其转换为张量，否则将emotion转换为张量
    return emo  # 返回情感推断结果

def infer(
    text,  # 输入的文本
    sdp_ratio,  # sdp_ratio 参数
    noise_scale,  # 噪音比例参数
    noise_scale_w,  # 噪音比例参数
    length_scale,  # 长度比例参数
    sid,  # sid 参数
    language,  # 语言参数
    hps,  # hps 参数
    net_g,  # 神经网络参数
    device,  # 设备参数
    reference_audio=None,  # 参考音频，默认为 None
    emotion=None,  # 情感，默认为 None
    skip_start=False,  # 是否跳过开头，默认为 False
    skip_end=False,  # 是否跳过结尾，默认为 False
    style_text=None,  # 风格文本，默认为 None
    style_weight=0.7,  # 风格权重，默认为 0.7
):
    # 调用 get_text 函数获取文本的 BERT 表示、语音特征等信息
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text, language, hps, device, style_text, style_weight
    )
    # 调用 get_emo_ 函数获取参考音频的情感表示
    emo = get_emo_(reference_audio, emotion)
    # 如果需要跳过开头部分，则将相应的信息进行裁剪
    if skip_start:
        phones = phones[1:]
        tones = tones[1:]
        lang_ids = lang_ids[1:]
        bert = bert[:, 1:]
        ja_bert = ja_bert[:, 1:]
        en_bert = en_bert[:, 1:]
    # 如果需要跳过结尾部分，则将相应的信息进行裁剪
    if skip_end:
        phones = phones[:-1]
        tones = tones[:-1]
        lang_ids = lang_ids[:-1]
        bert = bert[:, :-1]
        ja_bert = ja_bert[:, :-1]
        en_bert = en_bert[:, :-1]
    # 使用 torch.no_grad() 上下文管理器，确保在该上下文中不进行梯度计算
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)  # 将phones张量移动到指定设备上，并在第0维度上增加一个维度
        tones = tones.to(device).unsqueeze(0)  # 将tones张量移动到指定设备上，并在第0维度上增加一个维度
        lang_ids = lang_ids.to(device).unsqueeze(0)  # 将lang_ids张量移动到指定设备上，并在第0维度上增加一个维度
        bert = bert.to(device).unsqueeze(0)  # 将bert张量移动到指定设备上，并在第0维度上增加一个维度
        ja_bert = ja_bert.to(device).unsqueeze(0)  # 将ja_bert张量移动到指定设备上，并在第0维度上增加一个维度
        en_bert = en_bert.to(device).unsqueeze(0)  # 将en_bert张量移动到指定设备上，并在第0维度上增加一个维度
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)  # 创建一个包含phones张量大小的长整型张量，并将其移动到指定设备上
        emo = emo.to(device).unsqueeze(0)  # 将emo张量移动到指定设备上，并在第0维度上增加一个维度
        del phones  # 删除phones张量
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)  # 创建一个包含指定speakers的长整型张量，并将其移动到指定设备上
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
# 定义一个函数infer_multilang，接受text, sdp_ratio, noise_scale等参数
def infer_multilang(
    text,
    sdp_ratio,
    noise_scale,
    noise_scale_w,  # 噪声比例
    length_scale,  # 长度比例
    sid,  # sid
    language,  # 语言
    hps,  # hps
    net_g,  # 网络g
    device,  # 设备
    reference_audio=None,  # 参考音频，默认为None
    emotion=None,  # 情感，默认为None
    skip_start=False,  # 跳过开始，默认为False
    skip_end=False,  # 跳过结束，默认为False
):
    bert, ja_bert, en_bert, phones, tones, lang_ids = [], [], [], [], [], []  # 初始化空列表
    emo = get_emo_(reference_audio, emotion)  # 获取情感
    for idx, (txt, lang) in enumerate(zip(text, language)):  # 遍历文本和语言
        skip_start = (idx != 0) or (skip_start and idx == 0)  # 设置跳过开始的条件
        skip_end = (idx != len(text) - 1) or (skip_end and idx == len(text) - 1)  # 设置跳过结束的条件
        (
            temp_bert,  # 临时bert
            temp_ja_bert,  # 临时日文bert
        temp_en_bert,  # 从get_text函数返回的结果中获取英文BERT数据
        temp_phones,   # 从get_text函数返回的结果中获取音素数据
        temp_tones,    # 从get_text函数返回的结果中获取音调数据
        temp_lang_ids, # 从get_text函数返回的结果中获取语言ID数据
    ) = get_text(txt, lang, hps, device)  # 调用get_text函数获取文本数据

    if skip_start:  # 如果需要跳过开头部分
        temp_bert = temp_bert[:, 1:]  # 从temp_bert中去掉第一个元素
        temp_ja_bert = temp_ja_bert[:, 1:]  # 从temp_ja_bert中去掉第一个元素
        temp_en_bert = temp_en_bert[:, 1:]  # 从temp_en_bert中去掉第一个元素
        temp_phones = temp_phones[1:]  # 从temp_phones中去掉第一个元素
        temp_tones = temp_tones[1:]  # 从temp_tones中去掉第一个元素
        temp_lang_ids = temp_lang_ids[1:]  # 从temp_lang_ids中去掉第一个元素

    if skip_end:  # 如果需要跳过结尾部分
        temp_bert = temp_bert[:, :-1]  # 从temp_bert中去掉最后一个元素
        temp_ja_bert = temp_ja_bert[:, :-1]  # 从temp_ja_bert中去掉最后一个元素
        temp_en_bert = temp_en_bert[:, :-1]  # 从temp_en_bert中去掉最后一个元素
        temp_phones = temp_phones[:-1]  # 从temp_phones中去掉最后一个元素
        temp_tones = temp_tones[:-1]  # 从temp_tones中去掉最后一个元素
        temp_lang_ids = temp_lang_ids[:-1]  # 从temp_lang_ids中去掉最后一个元素

    bert.append(temp_bert)  # 将处理后的temp_bert添加到bert列表中
        ja_bert.append(temp_ja_bert)  # 将temp_ja_bert添加到ja_bert列表中
        en_bert.append(temp_en_bert)  # 将temp_en_bert添加到en_bert列表中
        phones.append(temp_phones)  # 将temp_phones添加到phones列表中
        tones.append(temp_tones)  # 将temp_tones添加到tones列表中
        lang_ids.append(temp_lang_ids)  # 将temp_lang_ids添加到lang_ids列表中
    bert = torch.concatenate(bert, dim=1)  # 沿着第一个维度将bert列表中的张量连接起来
    ja_bert = torch.concatenate(ja_bert, dim=1)  # 沿着第一个维度将ja_bert列表中的张量连接起来
    en_bert = torch.concatenate(en_bert, dim=1)  # 沿着第一个维度将en_bert列表中的张量连接起来
    phones = torch.concatenate(phones, dim=0)  # 沿着第零个维度将phones列表中的张量连接起来
    tones = torch.concatenate(tones, dim=0)  # 沿着第零个维度将tones列表中的张量连接起来
    lang_ids = torch.concatenate(lang_ids, dim=0)  # 沿着第零个维度将lang_ids列表中的张量连接起来
    with torch.no_grad():  # 进入不计算梯度的上下文
        x_tst = phones.to(device).unsqueeze(0)  # 将phones转移到指定设备并在第一个维度增加一个维度
        tones = tones.to(device).unsqueeze(0)  # 将tones转移到指定设备并在第一个维度增加一个维度
        lang_ids = lang_ids.to(device).unsqueeze(0)  # 将lang_ids转移到指定设备并在第一个维度增加一个维度
        bert = bert.to(device).unsqueeze(0)  # 将bert转移到指定设备并在第一个维度增加一个维度
        ja_bert = ja_bert.to(device).unsqueeze(0)  # 将ja_bert转移到指定设备并在第一个维度增加一个维度
        en_bert = en_bert.to(device).unsqueeze(0)  # 将en_bert转移到指定设备并在第一个维度增加一个维度
        emo = emo.to(device).unsqueeze(0)  # 将emo转移到指定设备并在第一个维度增加一个维度
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)  # 创建一个包含phones长度的张量并将其转移到指定设备
        # 删除变量 phones
        del phones
        # 使用语音说话人的ID来创建一个长整型张量，并将其移动到指定的设备上
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        # 使用神经网络进行推断，生成音频数据
        audio = (
            net_g.infer(
                x_tst,  # 输入的测试数据
                x_tst_lengths,  # 测试数据的长度
                speakers,  # 说话人ID
                tones,  # 音调
                lang_ids,  # 语言ID
                bert,  # BERT模型
                ja_bert,  # 日语BERT模型
                en_bert,  # 英语BERT模型
                emo,  # 情感
                sdp_ratio=sdp_ratio,  # SDP比例
                noise_scale=noise_scale,  # 噪音比例
                noise_scale_w=noise_scale_w,  # 噪音比例w
                length_scale=length_scale,  # 长度比例
            )[0][0, 0]  # 获取推断结果的第一个元素
            .data.cpu()  # 将数据移动到CPU上
            .float()  # 转换为浮点数
        .numpy()  # 将数据转换为 NumPy 数组
        )  # 结束函数调用
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers, ja_bert, en_bert, emo  # 删除变量以释放内存
        if torch.cuda.is_available():  # 检查是否有可用的 CUDA 设备
            torch.cuda.empty_cache()  # 清空 CUDA 缓存
        return audio  # 返回结果
```