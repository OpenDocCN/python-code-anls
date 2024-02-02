# `Bert-VITS2\oldVersion\V110\__init__.py`

```py
"""
1.1 版本兼容
https://github.com/fishaudio/Bert-VITS2/releases/tag/1.1
"""
# 导入 torch 库
import torch
# 导入 commons 模块
import commons
# 从 text 模块中的 cleaner 导入 clean_text 函数
from .text.cleaner import clean_text
# 从 text 模块中导入 cleaned_text_to_sequence 函数
from .text import cleaned_text_to_sequence
# 从 oldVersion.V111.text 模块中导入 get_bert 函数

# 定义 get_text 函数，接受文本、语言字符串、hps、设备作为参数
def get_text(text, language_str, hps, device):
    # 调用 clean_text 函数，清理文本并返回规范化文本、音素、语调、单词到音素的映射
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    # 将音素、语调、语言字符串转换为序列
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 如果 hps.data.add_blank 为真
    if hps.data.add_blank:
        # 在音素、语调、语言序列中插入 0
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        # 对 word2ph 中的每个元素乘以 2，并在第一个元素上加 1
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    # 调用 get_bert 函数，获取规范化文本的 BERT 表示
    bert = get_bert(norm_text, word2ph, language_str, device)
    # 删除 word2ph
    del word2ph
    # 断言 bert 的最后一个维度等于音素的长度
    assert bert.shape[-1] == len(phone), phone

    # 如果语言字符串为 "ZH"
    if language_str == "ZH":
        # bert 不变，ja_bert 为长度为 len(phone) 的全零张量
        bert = bert
        ja_bert = torch.zeros(768, len(phone))
    # 如果语言字符串为 "JP"
    elif language_str == "JP":
        # ja_bert 不变，bert 为长度为 len(phone) 的全零张量
        ja_bert = bert
        bert = torch.zeros(1024, len(phone))
    # 否则
    else:
        # bert 和 ja_bert 都为长度为 len(phone) 的全零张量
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))

    # 断言 bert 的最后一个维度等于音素的长度
    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    # 将 phone、tone、language 转换为 LongTensor 类型
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    # 返回 bert、ja_bert、phone、tone、language
    return bert, ja_bert, phone, tone, language

# 定义 infer 函数，接受文本、sdp_ratio、noise_scale、noise_scale_w、length_scale、sid、language、hps、net_g、device 作为参数
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
    # 调用 get_text 函数，获取 bert、ja_bert、phones、tones、lang_ids
    bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps, device)
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
        # 创建一个包含 phones.size(0) 的长整型张量，并转移到指定设备
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        # 释放 phones 占用的内存
        del phones
        # 创建一个包含 hps.data.spk2id[sid] 的长整型张量，并转移到指定设备
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        # 使用 net_g.infer() 推断音频数据，并将结果转移到 CPU，转换为浮点型，再转换为 numpy 数组
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
        # 释放占用的内存
        del x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, ja_bert
        # 如果 CUDA 可用，则清空 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 返回音频数据
        return audio
```