# `Bert-VITS2\oldVersion\V101\__init__.py`

```

"""
1.0.1 版本兼容
https://github.com/fishaudio/Bert-VITS2/releases/tag/1.0.1
"""
# 导入 torch 库
import torch
# 导入自定义的 commons 模块
import commons
# 从 text 模块中的 cleaner 导入 clean_text 函数
from .text.cleaner import clean_text
# 从 text 模块中导入 cleaned_text_to_sequence 函数
from .text import cleaned_text_to_sequence
# 从 oldVersion.V111.text 模块中导入 get_bert 函数

# 定义 get_text 函数，接受文本、语言字符串、hps 参数和设备作为输入
def get_text(text, language_str, hps, device):
    # 调用 clean_text 函数，清理文本并返回规范化后的文本、音素、语调和单词到音素的映射
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    # 将音素、语调和语言字符串转换为序列
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 如果 hps.data.add_blank 为真，则在音素、语调和语言序列中插入空白符
    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    # 调用 get_bert 函数，获取规范化文本、单词到音素的映射、语言字符串和设备相关的 BERT 表示
    bert = get_bert(norm_text, word2ph, language_str, device)
    # 释放 word2ph 变量的内存
    del word2ph

    # 断言 BERT 表示的最后一个维度长度与音素序列长度相等
    assert bert.shape[-1] == len(phone)

    # 将音素、语调和语言序列转换为 torch 张量
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)

    # 返回 BERT 表示、音素、语调和语言序列
    return bert, phone, tone, language

# 定义 infer 函数，接受文本、sdp_ratio、noise_scale、noise_scale_w、length_scale、sid、hps、net_g 和设备作为输入
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
    # 调用 get_text 函数，获取 BERT 表示、音素、语调和语言序列
    bert, phones, tones, lang_ids = get_text(text, "ZH", hps, device)
    # 禁用梯度计算
    with torch.no_grad():
        # 将音素、语调和语言序列移动到设备上，并添加一个维度
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        # 创建音素序列长度的 torch 张量
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        # 释放 phones 变量的内存
        del phones
        # 创建说话者的 torch 张量
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        # 使用 net_g 模型进行推理，生成音频
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
            .data.cpu()
            .float()
            .numpy()
        )
        # 释放内存
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers
        # 如果 CUDA 可用，则清空缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 返回生成的音频
        return audio

```