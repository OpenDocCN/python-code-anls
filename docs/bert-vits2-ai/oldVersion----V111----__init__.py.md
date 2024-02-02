# `Bert-VITS2\oldVersion\V111\__init__.py`

```py
"""
1.1.1版本兼容
https://github.com/fishaudio/Bert-VITS2/releases/tag/1.1.1
"""
# 导入 torch 库
import torch
# 导入自定义的 commons 模块
import commons
# 从 text 模块中导入 clean_text 和 clean_text_fix 函数
from .text.cleaner import clean_text, clean_text_fix
# 从 text 模块中导入 cleaned_text_to_sequence 函数
from .text import cleaned_text_to_sequence
# 从 text 模块中导入 get_bert 和 get_bert_fix 函数

# 定义 get_text 函数，接受文本、语言字符串、hps 和设备作为参数
def get_text(text, language_str, hps, device):
    # 调用 clean_text 函数，清理文本并返回规范化的文本、音素、音调和单词到音素的映射
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    # 将音素、音调和语言字符串转换为序列
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 如果 hps.data.add_blank 为真
    if hps.data.add_blank:
        # 在音素、音调和语言序列中插入 0
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        # 对单词到音素的映射进行处理
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    # 调用 get_bert 函数，获取规范化文本的 BERT 表示
    bert = get_bert(norm_text, word2ph, language_str, device)
    # 释放 word2ph 变量的内存
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

    # 将音素、音调和语言字符串转换为 LongTensor 类型，并返回结果
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language

# 定义 get_text_fix 函数，接受文本、语言字符串、hps 和设备作为参数
def get_text_fix(text, language_str, hps, device):
    # 调用 clean_text_fix 函数，清理文本并返回规范化的文本、音素、音调和单词到音素的映射
    norm_text, phone, tone, word2ph = clean_text_fix(text, language_str)
    # 将音素、音调和语言字符串转换为序列
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 如果 hps.data.add_blank 为真
    if hps.data.add_blank:
        # 在音素、音调和语言序列中插入 0
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        # 对单词到音素的映射进行处理
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    # 调用 get_bert_fix 函数，获取规范化文本的 BERT 表示
    bert = get_bert_fix(norm_text, word2ph, language_str, device)
    # 释放 word2ph 变量的内存
    del word2ph
    # 检查 BERT 的最后一个维度是否与电话号码的长度相等，如果不相等则抛出异常
    assert bert.shape[-1] == len(phone), phone

    # 根据语言类型进行不同的处理
    if language_str == "ZH":
        # 如果语言是中文，则不做任何处理
        bert = bert
        # 创建一个与电话号码长度相等的全零张量
        ja_bert = torch.zeros(768, len(phone))
    elif language_str == "JP":
        # 如果语言是日文，则将原始的 BERT 赋值给日文 BERT
        ja_bert = bert
        # 创建一个与电话号码长度相等的全零张量
        bert = torch.zeros(1024, len(phone))
    else:
        # 如果是其他语言，则创建一个与电话号码长度相等的全零张量
        bert = torch.zeros(1024, len(phone))
        # 创建一个与电话号码长度相等的全零张量
        ja_bert = torch.zeros(768, len(phone))

    # 再次检查 BERT 的最后一个维度是否与电话号码的长度相等，如果不相等则抛出异常
    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    # 将电话号码转换为长整型张量
    phone = torch.LongTensor(phone)
    # 将音调转换为长整型张量
    tone = torch.LongTensor(tone)
    # 将语言类型转换为长整型张量
    language = torch.LongTensor(language)
    # 返回处理后的结果
    return bert, ja_bert, phone, tone, language
# 推断函数，根据输入的文本和参数进行语音合成
def infer(
    text,  # 输入的文本
    sdp_ratio,  # sdp 比例
    noise_scale,  # 噪声比例
    noise_scale_w,  # 噪声比例 w
    length_scale,  # 长度比例
    sid,  # 说话人 ID
    language,  # 语言
    hps,  # 参数设置
    net_g,  # 生成器网络
    device,  # 设备
):
    # 获取文本的 BERT 表示、日语 BERT 表示、音素、音调和语言 ID
    bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps, device)
    # 禁用梯度计算
    with torch.no_grad():
        # 将音素转移到指定设备并增加一个维度
        x_tst = phones.to(device).unsqueeze(0)
        # 将音调转移到指定设备并增加一个维度
        tones = tones.to(device).unsqueeze(0)
        # 将语言 ID 转移到指定设备并增加一个维度
        lang_ids = lang_ids.to(device).unsqueeze(0)
        # 将 BERT 表示转移到指定设备并增加一个维度
        bert = bert.to(device).unsqueeze(0)
        # 将日语 BERT 表示转移到指定设备并增加一个维度
        ja_bert = ja_bert.to(device).unsqueeze(0)
        # 创建音素长度的张量并转移到指定设备
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        # 释放内存
        del phones
        # 创建说话人 ID 的张量并转移到指定设备
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        # 进行推断，生成音频
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


# 修复后的推断函数，与原推断函数类似，但使用了不同的文本处理函数
def infer_fix(
    text,  # 输入的文本
    sdp_ratio,  # sdp 比例
    noise_scale,  # 噪声比例
    noise_scale_w,  # 噪声比例 w
    length_scale,  # 长度比例
    sid,  # 说话人 ID
    language,  # 语言
    hps,  # 参数设置
    net_g,  # 生成器网络
    device,  # 设备
):
    # 获取修复后的文本的 BERT 表示、日语 BERT 表示、音素、音调和语言 ID
    bert, ja_bert, phones, tones, lang_ids = get_text_fix(text, language, hps, device)
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