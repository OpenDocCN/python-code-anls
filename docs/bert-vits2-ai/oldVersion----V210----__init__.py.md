# `Bert-VITS2\oldVersion\V210\__init__.py`

```py
"""
@Desc: 2.1版本兼容 对应版本 v2.1 Emo and muti-lang optimize
"""
# 导入torch模块
import torch
# 导入commons模块
import commons
# 从当前目录下的text模块中导入cleaned_text_to_sequence和get_bert函数
from .text import cleaned_text_to_sequence, get_bert
# 从text模块中的cleaner子模块中导入clean_text函数
from .text.cleaner import clean_text

# 定义get_text函数，接收text、language_str、hps、device、style_text和style_weight等参数
def get_text(text, language_str, hps, device, style_text=None, style_weight=0.7):
    # 调用clean_text函数，处理文本并返回处理后的文本、音素、语调和word2ph
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    # 调用cleaned_text_to_sequence函数，将音素、语调和语言转换为序列
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 如果hps.data.add_blank为True，则在phone、tone和language序列中插入0
    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    # 调用get_bert函数，获取BERT表示
    bert_ori = get_bert(
        norm_text, word2ph, language_str, device, style_text, style_weight
    )
    # 删除word2ph变量
    del word2ph
    # 断言bert_ori的最后一个维度与phone的长度相等
    assert bert_ori.shape[-1] == len(phone), phone

    # 根据语言类型，初始化bert、ja_bert和en_bert
    if language_str == "ZH":
        bert = bert_ori
        ja_bert = torch.zeros(1024, len(phone))
        en_bert = torch.zeros(1024, len(phone))
    elif language_str == "JP":
        bert = torch.zeros(1024, len(phone))
        ja_bert = bert_ori
        en_bert = torch.zeros(1024, len(phone))
    elif language_str == "EN":
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(1024, len(phone))
        en_bert = bert_ori
    else:
        raise ValueError("language_str should be ZH, JP or EN")

    # 断言bert的最后一个维度与phone的长度相等
    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    # 将phone、tone和language转换为LongTensor类型
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    # 返回bert、ja_bert、en_bert、phone、tone和language
    return bert, ja_bert, en_bert, phone, tone, language

# 定义get_emo_函数，接收reference_audio和emotion参数
def get_emo_(reference_audio, emotion):
    # 从emo_gen模块中导入get_emo函数
    from .emo_gen import get_emo
    # 如果reference_audio不为空，则调用get_emo函数获取情感表示，否则使用给定的emotion值
    emo = (
        torch.from_numpy(get_emo(reference_audio))
        if reference_audio
        else torch.Tensor([emotion])
    )
    # 返回情感表示
    return emo

# 定义infer函数，接收text、sdp_ratio、noise_scale、noise_scale_w和length_scale等参数
def infer(
    text,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    # 学生ID
    sid,
    # 语言
    language,
    # 语音合成参数
    hps,
    # 神经网络模型
    net_g,
    # 设备
    device,
    # 参考音频，默认为None
    reference_audio=None,
    # 情感，默认为None
    emotion=None,
    # 是否跳过开头，默认为False
    skip_start=False,
    # 是否跳过结尾，默认为False
    skip_end=False,
    # 风格文本，默认为None
    style_text=None,
    # 风格权重，默认为0.7
    style_weight=0.7,
    # 调用 get_text 函数获取文本的 BERT 表示、日语 BERT 表示、英语 BERT 表示、音素、语调和语言 ID
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text, language, hps, device, style_text, style_weight
    )
    # 根据参考音频和情感获取情感表示
    emo = get_emo_(reference_audio, emotion)
    # 如果需要跳过开头部分，则对各个变量进行切片操作
    if skip_start:
        phones = phones[1:]
        tones = tones[1:]
        lang_ids = lang_ids[1:]
        bert = bert[:, 1:]
        ja_bert = ja_bert[:, 1:]
        en_bert = en_bert[:, 1:]
    # 如果需要跳过结尾部分，则对各个变量进行切片操作
    if skip_end:
        phones = phones[:-1]
        tones = tones[:-1]
        lang_ids = lang_ids[:-1]
        bert = bert[:, :-1]
        ja_bert = ja_bert[:, :-1]
        en_bert = en_bert[:, :-1]
    # 使用 torch.no_grad() 上下文管理器，避免梯度计算
    with torch.no_grad():
        # 将 phones 转换为指定设备上的张量，并增加一个维度
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        en_bert = en_bert.to(device).unsqueeze(0)
        # 创建长度张量，指定 phones 的长度，并转移到指定设备上
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        # 将情感表示转移到指定设备上，并增加一个维度
        emo = emo.to(device).unsqueeze(0)
        # 释放内存，删除不再需要的变量
        del phones
        # 创建说话者张量，指定 sid 对应的 ID，并转移到指定设备上
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        # 调用 net_g 的 infer 方法，生成音频数据
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
        # 释放内存，删除不再需要的变量
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers, ja_bert, en_bert, emo
        # 如果有可用的 GPU，则清空 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 返回生成的音频数据
        return audio


# 定义 infer_multilang 函数，用于多语言推理
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
    # 定义一个变量 emotion，初始值为 None
    emotion=None,
    # 定义一个布尔变量 skip_start，初始值为 False，表示是否跳过开头部分
    skip_start=False,
    # 定义一个布尔变量 skip_end，初始值为 False，表示是否跳过结尾部分
    skip_end=False,
# 初始化空列表用于存储数据
bert, ja_bert, en_bert, phones, tones, lang_ids = [], [], [], [], [], []
# 获取参考音频的情感标签
emo = get_emo_(reference_audio, emotion)
# 遍历文本和语言列表
for idx, (txt, lang) in enumerate(zip(text, language)):
    # 设置跳过开头和结尾的标志
    skip_start = (idx != 0) or (skip_start and idx == 0)
    skip_end = (idx != len(text) - 1) or (skip_end and idx == len(text) - 1)
    # 调用函数获取文本的各种数据
    (
        temp_bert,
        temp_ja_bert,
        temp_en_bert,
        temp_phones,
        temp_tones,
        temp_lang_ids,
    ) = get_text(txt, lang, hps, device)
    # 如果需要跳过开头，则对数据进行处理
    if skip_start:
        temp_bert = temp_bert[:, 1:]
        temp_ja_bert = temp_ja_bert[:, 1:]
        temp_en_bert = temp_en_bert[:, 1:]
        temp_phones = temp_phones[1:]
        temp_tones = temp_tones[1:]
        temp_lang_ids = temp_lang_ids[1:]
    # 如果需要跳过结尾，则对数据进行处理
    if skip_end:
        temp_bert = temp_bert[:, :-1]
        temp_ja_bert = temp_ja_bert[:, :-1]
        temp_en_bert = temp_en_bert[:, :-1]
        temp_phones = temp_phones[:-1]
        temp_tones = temp_tones[:-1]
        temp_lang_ids = temp_lang_ids[:-1]
    # 将处理后的数据添加到对应的列表中
    bert.append(temp_bert)
    ja_bert.append(temp_ja_bert)
    en_bert.append(temp_en_bert)
    phones.append(temp_phones)
    tones.append(temp_tones)
    lang_ids.append(temp_lang_ids)
# 拼接列表中的数据
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