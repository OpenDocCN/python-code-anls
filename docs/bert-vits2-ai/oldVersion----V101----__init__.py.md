# `Bert-VITS2\oldVersion\V101\__init__.py`

```
"""
1.0.1 版本兼容
https://github.com/fishaudio/Bert-VITS2/releases/tag/1.0.1
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

# 定义函数 get_text，接受 text, language_str, hps, device 四个参数
def get_text(text, language_str, hps, device):
    # 调用 clean_text 函数，返回规范化后的文本、音素、音调和单词到音素的映射
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    # 调用 cleaned_text_to_sequence 函数，返回音素、音调和语言的序列
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 如果 hps.data.add_blank 为真
    if hps.data.add_blank:
        # 在 phone、tone、language 序列中插入 0
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        # 对 word2ph 中的每个元素乘以 2
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        # 将 word2ph 的第一个元素加 1
        word2ph[0] += 1
    # 调用 get_bert 函数，返回 bert
    bert = get_bert(norm_text, word2ph, language_str, device)
    # 删除 word2ph
    del word2ph

    # 断言 bert 的最后一个维度等于 phone 的长度
    assert bert.shape[-1] == len(phone)

    # 将 phone、tone、language 转换为 torch 的 LongTensor 类型
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)

    # 返回 bert, phone, tone, language
    return bert, phone, tone, language

# 定义函数 infer，接受 text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, hps, net_g, device 九个参数
def infer(
    text,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    sid,
    hps,
    net_g,
    device,
):
    # 调用 get_text 函数，返回 bert, phones, tones, lang_ids
    bert, phones, tones, lang_ids = get_text(text, "ZH", hps, device)
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
        # 创建一个包含 phones.size(0) 的长整型张量，并转移到指定设备
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        # 释放 phones 占用的内存
        del phones
        # 创建一个包含 hps.data.spk2id[sid] 的长整型张量，并转移到指定设备
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        # 使用 net_g.infer() 方法生成音频数据
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()  # 将数据转移到 CPU
            .float()  # 转换数据类型为浮点型
            .numpy()  # 转换为 NumPy 数组
        )
        # 释放内存
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers
        # 如果 CUDA 可用，清空 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 返回生成的音频数据
        return audio
```