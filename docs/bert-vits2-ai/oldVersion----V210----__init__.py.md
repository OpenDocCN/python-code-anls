# `Bert-VITS2\oldVersion\V210\__init__.py`

```

# 导入torch模块
import torch
# 导入commons模块
import commons
# 从当前目录下的text模块中导入cleaned_text_to_sequence和get_bert函数
from .text import cleaned_text_to_sequence, get_bert
# 从text模块的cleaner子模块中导入clean_text函数
from .text.cleaner import clean_text

# 定义get_text函数，接收text、language_str、hps、device、style_text和style_weight等参数
def get_text(text, language_str, hps, device, style_text=None, style_weight=0.7):
    # 调用clean_text函数，获取规范化文本、音素、语调和word2ph
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    # 调用cleaned_text_to_sequence函数，将音素、语调和语言转换为序列
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 如果hps.data.add_blank为True，则在phone、tone、language和word2ph中插入0
    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    # 调用get_bert函数，获取bert_ori
    bert_ori = get_bert(
        norm_text, word2ph, language_str, device, style_text, style_weight
    )
    # 删除word2ph
    del word2ph
    # 断言bert_ori的最后一个维度与phone的长度相等
    assert bert_ori.shape[-1] == len(phone), phone

    # 根据language_str的不同，初始化bert、ja_bert和en_bert
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
    # 如果reference_audio不为空，则调用get_emo函数，否则返回包含emotion的Tensor
    emo = (
        torch.from_numpy(get_emo(reference_audio))
        if reference_audio
        else torch.Tensor([emotion])
    )
    return emo

# 定义infer函数，接收多个参数
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
    # 调用get_text函数，获取bert、ja_bert、en_bert、phones、tones和lang_ids
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text, language, hps, device, style_text, style_weight
    )
    # 调用get_emo_函数，获取emo
    emo = get_emo_(reference_audio, emotion)
    # 根据skip_start和skip_end的值，对phones、tones、lang_ids、bert、ja_bert和en_bert进行处理
    if skip_start:
        phones = phones[1:]
        tones = tones[1:]
        lang_ids = lang_ids[1:]
        bert = bert[:, 1:]
        ja_bert = ja_bert[:, 1:]
        en_bert = en_bert[:, 1:]
    if skip_end:
        phones = phones[:-1]
        tones = tones[:-1]
        lang_ids = lang_ids[:-1]
        bert = bert[:, :-1]
        ja_bert = ja_bert[:, :-1]
        en_bert = en_bert[:, :-1]
    # 在不计算梯度的情况下，将数据移动到设备上进行推理
    with torch.no_grad():
        # 对数据进行处理并调用net_g进行推理，返回音频数据
        # 最后释放不再需要的变量并清空GPU缓存
        return audio

# 定义infer_multilang函数，接收多个参数
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
    # 省略部分代码...

```