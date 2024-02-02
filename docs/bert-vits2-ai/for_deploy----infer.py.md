# `Bert-VITS2\for_deploy\infer.py`

```py
"""
版本管理、兼容推理及模型加载实现。
版本说明：
    1. 版本号与github的release版本号对应，使用哪个release版本训练的模型即对应其版本号
    2. 请在模型的config.json中显示声明版本号，添加一个字段"version" : "你的版本号"
特殊版本说明：
    1.1.1-fix： 1.1.1版本训练的模型，但是在推理时使用dev的日语修复
    2.2：当前版本
"""
import torch  # 导入PyTorch库
import commons  # 导入自定义的commons模块
from text import cleaned_text_to_sequence  # 从text模块中导入cleaned_text_to_sequence函数
from text.cleaner import clean_text  # 从text.cleaner模块中导入clean_text函数
import utils  # 导入自定义的utils模块
import numpy as np  # 导入NumPy库

from models import SynthesizerTrn  # 从models模块中导入SynthesizerTrn类
from text.symbols import symbols  # 从text.symbols模块中导入symbols变量

from oldVersion.V210.models import SynthesizerTrn as V210SynthesizerTrn  # 从oldVersion.V210.models模块中导入SynthesizerTrn类，并重命名为V210SynthesizerTrn
from oldVersion.V210.text import symbols as V210symbols  # 从oldVersion.V210.text模块中导入symbols变量，并重命名为V210symbols
from oldVersion.V200.models import SynthesizerTrn as V200SynthesizerTrn  # 从oldVersion.V200.models模块中导入SynthesizerTrn类，并重命名为V200SynthesizerTrn
from oldVersion.V200.text import symbols as V200symbols  # 从oldVersion.V200.text模块中导入symbols变量，并重命名为V200symbols
from oldVersion.V111.models import SynthesizerTrn as V111SynthesizerTrn  # 从oldVersion.V111.models模块中导入SynthesizerTrn类，并重命名为V111SynthesizerTrn
from oldVersion.V111.text import symbols as V111symbols  # 从oldVersion.V111.text模块中导入symbols变量，并重命名为V111symbols
from oldVersion.V110.models import SynthesizerTrn as V110SynthesizerTrn  # 从oldVersion.V110.models模块中导入SynthesizerTrn类，并重命名为V110SynthesizerTrn
from oldVersion.V110.text import symbols as V110symbols  # 从oldVersion.V110.text模块中导入symbols变量，并重命名为V110symbols
from oldVersion.V101.models import SynthesizerTrn as V101SynthesizerTrn  # 从oldVersion.V101.models模块中导入SynthesizerTrn类，并重命名为V101SynthesizerTrn
from oldVersion.V101.text import symbols as V101symbols  # 从oldVersion.V101.text模块中导入symbols变量，并重命名为V101symbols

from oldVersion import V111, V110, V101, V200, V210  # 从oldVersion模块中导入V111, V110, V101, V200, V210模块

# 当前版本信息
latest_version = "2.2"  # 设置latest_version变量为"2.2"

# 版本兼容
SynthesizerTrnMap = {  # 创建SynthesizerTrnMap字典
    "2.1": V210SynthesizerTrn,  # 键为"2.1"，值为V210SynthesizerTrn
    "2.0.2-fix": V200SynthesizerTrn,  # 键为"2.0.2-fix"，值为V200SynthesizerTrn
    "2.0.1": V200SynthesizerTrn,  # 键为"2.0.1"，值为V200SynthesizerTrn
    "2.0": V200SynthesizerTrn,  # 键为"2.0"，值为V200SynthesizerTrn
    "1.1.1-fix": V111SynthesizerTrn,  # 键为"1.1.1-fix"，值为V111SynthesizerTrn
    "1.1.1": V111SynthesizerTrn,  # 键为"1.1.1"，值为V111SynthesizerTrn
    "1.1": V110SynthesizerTrn,  # 键为"1.1"，值为V110SynthesizerTrn
    "1.1.0": V110SynthesizerTrn,  # 键为"1.1.0"，值为V110SynthesizerTrn
    "1.0.1": V101SynthesizerTrn,  # 键为"1.0.1"，值为V101SynthesizerTrn
    "1.0": V101SynthesizerTrn,  # 键为"1.0"，值为V101SynthesizerTrn
    "1.0.0": V101SynthesizerTrn,  # 键为"1.0.0"，值为V101SynthesizerTrn
}

symbolsMap = {  # 创建symbolsMap字典
    "2.1": V210symbols,  # 键为"2.1"，值为V210symbols
    "2.0.2-fix": V200symbols,  # 键为"2.0.2-fix"，值为V200symbols
    "2.0.1": V200symbols,  # 键为"2.0.1"，值为V200symbols
    "2.0": V200symbols,  # 键为"2.0"，值为V200symbols
    "1.1.1-fix": V111symbols,  # 键为"1.1.1-fix"，值为V111symbols
    "1.1.1": V111symbols,  # 键为"1.1.1"，值为V111symbols
    "1.1": V110symbols,  # 键为"1.1"，值为V110symbols
    "1.1.0": V110symbols,  # 键为"1.1.0"，值为V110symbols
    "1.0.1": V101symbols,  # 键为"1.0.1"，值为V101symbols
    "1.0": V101symbols,  # 键为"1.0"，值为V101symbols
    "1.0.0": V101symbols,  # 键为"1.0.0"，值为V101symbols
}


# def get_emo_(reference_audio, emotion, sid):
#     emo = (
#         torch.from_numpy(get_emo(reference_audio))
# 如果 reference_audio 存在且情感为 -1，则使用 reference_audio
# 否则，使用 np.load(f"emo_clustering/{sid}/cluster_center_{emotion}.npy") 加载的数据创建 torch.FloatTensor 对象
# 返回结果为 emo

def get_net_g(model_path: str, version: str, device: str, hps):
    # 如果版本不是最新版本，则创建对应版本的 SynthesizerTrnMap 模型对象
    net_g = SynthesizerTrnMap[version](
        len(symbolsMap[version]),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    # 否则，创建当前版本的 SynthesizerTrn 模型对象
    else:
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    # 设置为评估模式
    _ = net_g.eval()
    # 加载模型参数
    _ = utils.load_checkpoint(model_path, net_g, None, skip_optimizer=True)
    # 返回模型对象
    return net_g

def get_text(text, language_str, bert, hps, device):
    # 清洗文本，获取规范化文本、音素、语调、word2ph
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    # 将清洗后的文本转换为序列
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 如果数据集中添加了空白符
    if hps.data.add_blank:
        # 在音素、语调、语言序列中插入空白符
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        # 对 word2ph 中的每个元素乘以 2，并将第一个元素加 1
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    # 获取规范化文本的 BERT 特征
    bert_ori = bert[language_str].get_bert_feature(norm_text, word2ph, device)
    # 删除 word2ph
    del word2ph
    # 断言 bert_ori 的最后一个维度长度与 phone 相等
    assert bert_ori.shape[-1] == len(phone), phone

    # 如果语言为中文
    if language_str == "ZH":
        # bert 使用 bert_ori
        bert = bert_ori
        # ja_bert 为随机生成的形状为 (1024, len(phone)) 的张量
        ja_bert = torch.randn(1024, len(phone))
        # en_bert 为随机生成的形状为 (1024, len(phone)) 的张量
        en_bert = torch.randn(1024, len(phone))
    # 如果语言为日语
    elif language_str == "JP":
        # bert 为随机生成的形状为 (1024, len(phone)) 的张量
        bert = torch.randn(1024, len(phone))
        # ja_bert 使用 bert_ori
        ja_bert = bert_ori
        # en_bert 为随机生成的形状为 (1024, len(phone)) 的张量
        en_bert = torch.randn(1024, len(phone))
    # 如果语言为英文，则使用随机生成的大小为(1024, len(phone))的张量作为bert
    elif language_str == "EN":
        bert = torch.randn(1024, len(phone))
        # 使用随机生成的大小为(1024, len(phone))的张量作为ja_bert
        ja_bert = torch.randn(1024, len(phone))
        # 使用bert_ori作为en_bert
        en_bert = bert_ori
    # 如果语言不是中文、日文或英文，则抛出数值错误
    else:
        raise ValueError("language_str should be ZH, JP or EN")

    # 断言bert的最后一个维度（即列数）与phone的长度相等，如果不相等则抛出断言错误
    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    # 将phone转换为长整型张量
    phone = torch.LongTensor(phone)
    # 将tone转换为长整型张量
    tone = torch.LongTensor(tone)
    # 将language转换为长整型张量
    language = torch.LongTensor(language)
    # 返回bert, ja_bert, en_bert, phone, tone, language
    return bert, ja_bert, en_bert, phone, tone, language
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
    bert=None,
    clap=None,
    reference_audio=None,
    skip_start=False,
    skip_end=False,
):
    # 2.2版本参数位置变了
    # 2.1 参数新增 emotion reference_audio skip_start skip_end
    # 定义一个字典，将版本号映射到对应的infer函数
    inferMap_V3 = {
        "2.1": V210.infer,
    }
    # 支持中日英三语版本
    # 定义一个字典，将版本号映射到对应的infer函数
    inferMap_V2 = {
        "2.0.2-fix": V200.infer,
        "2.0.1": V200.infer,
        "2.0": V200.infer,
        "1.1.1-fix": V111.infer_fix,
        "1.1.1": V111.infer,
        "1.1": V110.infer,
        "1.1.0": V110.infer,
    }
    # 仅支持中文版本
    # 定义一个字典，将版本号映射到对应的infer函数
    # 在测试中，并未发现两个版本的模型不能互相通用
    inferMap_V1 = {
        "1.0.1": V101.infer,
        "1.0": V101.infer,
        "1.0.0": V101.infer,
    }
    # 如果hps对象有version属性，则使用其版本号，否则使用latest_version
    version = hps.version if hasattr(hps, "version") else latest_version
    # 非当前版本，根据版本号选择合适的infer
    # 如果版本不是最新版本
    if version != latest_version:
        # 如果版本在推理映射 V3 的键中
        if version in inferMap_V3.keys():
            # 调用对应版本的推理函数
            return inferMap_V3[version](
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
                reference_audio,
                emotion,
                skip_start,
                skip_end,
            )
        # 如果版本在推理映射 V2 的键中
        if version in inferMap_V2.keys():
            # 调用对应版本的推理函数
            return inferMap_V2[version](
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
            )
        # 如果版本在推理映射 V1 的键中
        if version in inferMap_V1.keys():
            # 调用对应版本的推理函数
            return inferMap_V1[version](
                text,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                sid,
                hps,
                net_g,
                device,
            )
    # 如果版本是最新版本，则执行以下代码
    # 如果 reference_audio 是 numpy 数组
    if isinstance(reference_audio, np.ndarray):
        # 获取音频特征
        emo = clap.get_clap_audio_feature(reference_audio, device)
    else:
        # 获取文本特征
        emo = clap.get_clap_text_feature(emotion, device)
    # 压缩维度
    emo = torch.squeeze(emo, dim=1)

    # 获取文本特征
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text, language, bert, hps, device
    )
    # 如果需要跳过开头
    if skip_start:
        # 跳过前三个元素
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        bert = bert[:, 3:]
        ja_bert = ja_bert[:, 3:]
        en_bert = en_bert[:, 3:]
    # 如果需要跳过结尾
    if skip_end:
        # 跳过最后两个元素
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        bert = bert[:, :-2]
        ja_bert = ja_bert[:, :-2]
        en_bert = en_bert[:, :-2]
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
        # 创建一个包含 phones 大小的长整型张量，并转移到指定设备
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        # 将 emo 转移到指定设备，并在第0维度增加一个维度
        emo = emo.to(device).unsqueeze(0)
        # 释放 phones 占用的内存
        del phones
        # 创建一个包含 sid 对应的说话者 ID 的长整型张量，并转移到指定设备
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
        # 如果 CUDA 可用，则清空 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 返回音频数据
        return audio
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
    bert=None,
    clap=None,
    reference_audio=None,
    emotion=None,
    skip_start=False,
    skip_end=False,
):
    # 初始化空列表用于存储不同语言的文本特征
    bert, ja_bert, en_bert, phones, tones, lang_ids = [], [], [], [], [], []
    # 如果参考音频是一个 NumPy 数组，则获取音频特征
    if isinstance(reference_audio, np.ndarray):
        emo = clap.get_clap_audio_feature(reference_audio, device)
    # 否则，获取文本特征
    else:
        emo = clap.get_clap_text_feature(emotion, device)
    # 压缩维度
    emo = torch.squeeze(emo, dim=1)
    # 遍历文本和语言列表
    for idx, (txt, lang) in enumerate(zip(text, language)):
        # 根据参数设置是否跳过开头和结尾
        skip_start = (idx != 0) or (skip_start and idx == 0)
        skip_end = (idx != len(text) - 1) or (skip_end and idx == len(text) - 1)
        # 获取文本特征
        (
            temp_bert,
            temp_ja_bert,
            temp_en_bert,
            temp_phones,
            temp_tones,
            temp_lang_ids,
        ) = get_text(txt, lang, bert, hps, device)
        # 如果需要跳过开头，则移除开头部分的特征
        if skip_start:
            temp_bert = temp_bert[:, 3:]
            temp_ja_bert = temp_ja_bert[:, 3:]
            temp_en_bert = temp_en_bert[:, 3:]
            temp_phones = temp_phones[3:]
            temp_tones = temp_tones[3:]
            temp_lang_ids = temp_lang_ids[3:]
        # 如果需要跳过结尾，则移除结尾部分的特征
        if skip_end:
            temp_bert = temp_bert[:, :-2]
            temp_ja_bert = temp_ja_bert[:, :-2]
            temp_en_bert = temp_en_bert[:, :-2]
            temp_phones = temp_phones[:-2]
            temp_tones = temp_tones[:-2]
            temp_lang_ids = temp_lang_ids[:-2]
        # 将处理后的特征添加到对应的列表中
        bert.append(temp_bert)
        ja_bert.append(temp_ja_bert)
        en_bert.append(temp_en_bert)
        phones.append(temp_phones)
        tones.append(temp_tones)
        lang_ids.append(temp_lang_ids)
    # 沿指定维度拼接列表中的特征
    bert = torch.concatenate(bert, dim=1)
    ja_bert = torch.concatenate(ja_bert, dim=1)
    en_bert = torch.concatenate(en_bert, dim=1)
    # 将 phones 列表中的张量在维度0上进行拼接
    phones = torch.concatenate(phones, dim=0)
    # 将 tones 列表中的张量在维度0上进行拼接
    tones = torch.concatenate(tones, dim=0)
    # 将 lang_ids 列表中的张量在维度0上进行拼接
    lang_ids = torch.concatenate(lang_ids, dim=0)
    # 关闭梯度计算
    with torch.no_grad():
        # 将 phones 转移到设备上并在维度0上增加一个维度
        x_tst = phones.to(device).unsqueeze(0)
        # 将 tones 转移到设备上并在维度0上增加一个维度
        tones = tones.to(device).unsqueeze(0)
        # 将 lang_ids 转移到设备上并在维度0上增加一个维度
        lang_ids = lang_ids.to(device).unsqueeze(0)
        # 将 bert 转移到设备上并在维度0上增加一个维度
        bert = bert.to(device).unsqueeze(0)
        # 将 ja_bert 转移到设备上并在维度0上增加一个维度
        ja_bert = ja_bert.to(device).unsqueeze(0)
        # 将 en_bert 转移到设备上并在维度0上增加一个维度
        en_bert = en_bert.to(device).unsqueeze(0)
        # 将 emo 转移到设备上并在维度0上增加一个维度
        emo = emo.to(device).unsqueeze(0)
        # 创建一个包含 phones 大小的长整型张量并将其转移到设备上
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        # 删除 phones 变量
        del phones
        # 创建一个包含 hps.data.spk2id[sid] 的长整型张量并将其转移到设备上
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
            .data.cpu()
            .float()
            .numpy()
        )
        # 删除不再需要的变量
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers, ja_bert, en_bert, emo
        # 如果 CUDA 可用，则清空 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 返回音频数据
        return audio
```