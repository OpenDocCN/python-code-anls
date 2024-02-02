# `Bert-VITS2\infer.py`

```py
"""
版本管理、兼容推理及模型加载实现。
版本说明：
    1. 版本号与github的release版本号对应，使用哪个release版本训练的模型即对应其版本号
    2. 请在模型的config.json中显示声明版本号，添加一个字段"version" : "你的版本号"
特殊版本说明：
    1.1.1-fix： 1.1.1版本训练的模型，但是在推理时使用dev的日语修复
    2.3：当前版本
"""
import torch
import commons
from text import cleaned_text_to_sequence, get_bert

# from clap_wrapper import get_clap_audio_feature, get_clap_text_feature
from typing import Union
from text.cleaner import clean_text
import utils

from models import SynthesizerTrn
from text.symbols import symbols

from oldVersion.V220.models import SynthesizerTrn as V220SynthesizerTrn
from oldVersion.V220.text import symbols as V220symbols
from oldVersion.V210.models import SynthesizerTrn as V210SynthesizerTrn
from oldVersion.V210.text import symbols as V210symbols
from oldVersion.V200.models import SynthesizerTrn as V200SynthesizerTrn
from oldVersion.V200.text import symbols as V200symbols
from oldVersion.V111.models import SynthesizerTrn as V111SynthesizerTrn
from oldVersion.V111.text import symbols as V111symbols
from oldVersion.V110.models import SynthesizerTrn as V110SynthesizerTrn
from oldVersion.V110.text import symbols as V110symbols
from oldVersion.V101.models import SynthesizerTrn as V101SynthesizerTrn
from oldVersion.V101.text import symbols as V101symbols

from oldVersion import V111, V110, V101, V200, V210, V220

# 当前版本信息
latest_version = "2.3"

# 版本兼容
SynthesizerTrnMap = {
    "2.2": V220SynthesizerTrn,  # 将版本号2.2映射到对应的模型类
    "2.1": V210SynthesizerTrn,  # 将版本号2.1映射到对应的模型类
    "2.0.2-fix": V200SynthesizerTrn,  # 将版本号2.0.2-fix映射到对应的模型类
    "2.0.1": V200SynthesizerTrn,  # 将版本号2.0.1映射到对应的模型类
    "2.0": V200SynthesizerTrn,  # 将版本号2.0映射到对应的模型类
    "1.1.1-fix": V111SynthesizerTrn,  # 将版本号1.1.1-fix映射到对应的模型类
    "1.1.1": V111SynthesizerTrn,  # 将版本号1.1.1映射到对应的模型类
    "1.1": V110SynthesizerTrn,  # 将版本号1.1映射到对应的模型类
    "1.1.0": V110SynthesizerTrn,  # 将版本号1.1.0映射到对应的模型类
    "1.0.1": V101SynthesizerTrn,  # 将版本号1.0.1映射到对应的模型类
    "1.0": V101SynthesizerTrn,  # 将版本号1.0映射到对应的模型类
    "1.0.0": V101SynthesizerTrn,  # 将版本号1.0.0映射到对应的模型类
}

symbolsMap = {
    "2.2": V220symbols,  # 将版本号2.2映射到对应的符号集
    "2.1": V210symbols,  # 将版本号2.1映射到对应的符号集
    "2.0.2-fix": V200symbols,  # 将版本号2.0.2-fix映射到对应的符号集
    "2.0.1": V200symbols,  # 将版本号2.0.1映射到对应的符号集
    "2.0": V200symbols,  # 将版本号2.0映射到对应的符号集
    "1.1.1-fix": V111symbols,  # 将版本号1.1.1-fix映射到对应的符号集
    "1.1.1": V111symbols,  # 将版本号1.1.1映射到对应的符号集
    # 将版本号和对应的符号集合关联起来
    "1.1": V110symbols,
    "1.1.0": V110symbols,
    "1.0.1": V101symbols,
    "1.0": V101symbols,
    "1.0.0": V101symbols,
# 定义一个名为 get_emo_ 的函数，接受三个参数：reference_audio, emotion, sid
def get_emo_(reference_audio, emotion, sid):
    # 如果 reference_audio 存在且 emotion 为 -1，则将获取的情感数据转换为 torch 张量
    emo = (
        torch.from_numpy(get_emo(reference_audio))
        if reference_audio and emotion == -1
        else torch.FloatTensor(
            np.load(f"emo_clustering/{sid}/cluster_center_{emotion}.npy")
        )
    )
    # 返回获取的情感数据
    return emo


# 定义一个名为 get_net_g 的函数，接受四个参数：model_path, version, device, hps
def get_net_g(model_path: str, version: str, device: str, hps):
    # 如果版本号不是最新版本 latest_version，则创建对应版本的 SynthesizerTrnMap 模型
    if version != latest_version:
        net_g = SynthesizerTrnMap[version](
            len(symbolsMap[version]),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)
    else:
        # 否则创建当前版本的 SynthesizerTrn 模型
        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)
    # 设置 net_g 模型为评估模式
    _ = net_g.eval()
    # 加载模型参数
    _ = utils.load_checkpoint(model_path, net_g, None, skip_optimizer=True)
    # 返回创建的 net_g 模型
    return net_g


# 定义一个名为 get_text 的函数，接受六个参数：text, language_str, hps, device, style_text, style_weight
def get_text(text, language_str, hps, device, style_text=None, style_weight=0.7):
    # 如果 style_text 为空字符串，则将其设置为 None
    style_text = None if style_text == "" else style_text
    # 清理文本并获取规范化的文本、音素、语调、单词到音素的映射
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    # 将音素、语调、语言转换为序列
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 如果数据集中添加了空白符
    if hps.data.add_blank:
        # 在音素、语调、语言序列中插入空白符
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        # 将单词到音素的映射中的值乘以2，并在第一个值上加1
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    # 获取经过 BERT 处理后的文本表示
    bert_ori = get_bert(
        norm_text, word2ph, language_str, device, style_text, style_weight
    )
    # 删除 word2ph 变量
    del word2ph
    # 断言 bert_ori 的最后一个维度长度与音素序列长度相同
    assert bert_ori.shape[-1] == len(phone), phone

    # 如果语言为中文
    if language_str == "ZH":
        # 将 bert_ori 赋值给 bert
        bert = bert_ori
        # 创建一个形状为 (1024, len(phone)) 的随机张量，并赋值给 ja_bert 和 en_bert
        ja_bert = torch.randn(1024, len(phone))
        en_bert = torch.randn(1024, len(phone))
    # 如果语言为日语，则使用随机生成的大小为(1024, len(phone))的张量作为bert
    elif language_str == "JP":
        bert = torch.randn(1024, len(phone))
        # 将ja_bert设置为bert_ori
        ja_bert = bert_ori
        # 使用随机生成的大小为(1024, len(phone))的张量作为en_bert
        en_bert = torch.randn(1024, len(phone))
    # 如果语言为英语，则使用随机生成的大小为(1024, len(phone))的张量作为bert
    elif language_str == "EN":
        bert = torch.randn(1024, len(phone))
        # 使用随机生成的大小为(1024, len(phone))的张量作为ja_bert
        ja_bert = torch.randn(1024, len(phone))
        # 将en_bert设置为bert_ori
        en_bert = bert_ori
    # 如果语言既不是日语也不是英语，则抛出数值错误
    else:
        raise ValueError("language_str should be ZH, JP or EN")

    # 断言bert的最后一个维度（即列数）与phone的长度相等，如果不相等则抛出异常
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
    emotion: Union[int, str],
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
    skip_end=False,
    style_text=None,
    style_weight=0.7,
):
    # 2.2版本参数位置变了
    inferMap_V4 = {
        "2.2": V220.infer,
    }
    # 2.1 参数新增 emotion reference_audio skip_start skip_end
    inferMap_V3 = {
        "2.1": V210.infer,
    }
    # 支持中日英三语版本
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
    # 在测试中，并未发现两个版本的模型不能互相通用
    inferMap_V1 = {
        "1.0.1": V101.infer,
        "1.0": V101.infer,
        "1.0.0": V101.infer,
    }
    version = hps.version if hasattr(hps, "version") else latest_version
    # 非当前版本，根据版本号选择合适的infer
    # 如果版本不是最新版本
    if version != latest_version:
        # 如果版本在推理映射 V4 中
        if version in inferMap_V4.keys():
            # 返回 V4 版本的推理结果
            return inferMap_V4[version](
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
                reference_audio,
                skip_start,
                skip_end,
                style_text,
                style_weight,
            )
        # 如果版本在推理映射 V3 中
        if version in inferMap_V3.keys():
            # 返回 V3 版本的推理结果
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
                style_text,
                style_weight,
            )
        # 如果版本在推理映射 V2 中
        if version in inferMap_V2.keys():
            # 返回 V2 版本的推理结果
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
        # 如果版本在推理映射 V1 中
        if version in inferMap_V1.keys():
            # 返回 V1 版本的推理结果
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
    # 在此处实现当前版本的推理
    # 获取情感特征
    # emo = get_emo_(reference_audio, emotion, sid)
    # 如果 reference_audio 是 numpy 数组
    # if isinstance(reference_audio, np.ndarray):
    #     获取音频特征
    #     emo = get_clap_audio_feature(reference_audio, device)
    # else:
    #     其他情况
    # 调用 get_text 函数获取文本的特征
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text,
        language,
        hps,
        device,
        style_text=style_text,
        style_weight=style_weight,
    )
    # 如果需要跳过开头部分，则对获取的特征进行切片操作
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        bert = bert[:, 3:]
        ja_bert = ja_bert[:, 3:]
        en_bert = en_bert[:, 3:]
    # 如果需要跳过结尾部分，则对获取的特征进行切片操作
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        bert = bert[:, :-2]
        ja_bert = ja_bert[:, :-2]
        en_bert = en_bert[:, :-2]
    # 使用 torch.no_grad() 上下文管理器，关闭梯度计算
    with torch.no_grad():
        # 将 phones 转移到指定设备上，并增加一个维度
        x_tst = phones.to(device).unsqueeze(0)
        # 将 tones 转移到指定设备上，并增加一个维度
        tones = tones.to(device).unsqueeze(0)
        # 将 lang_ids 转移到指定设备上，并增加一个维度
        lang_ids = lang_ids.to(device).unsqueeze(0)
        # 将 bert 转移到指定设备上，并增加一个维度
        bert = bert.to(device).unsqueeze(0)
        # 将 ja_bert 转移到指定设备上，并增加一个维度
        ja_bert = ja_bert.to(device).unsqueeze(0)
        # 将 en_bert 转移到指定设备上，并增加一个维度
        en_bert = en_bert.to(device).unsqueeze(0)
        # 创建一个包含 phones 大小的长整型张量，并转移到指定设备上
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        # 根据 speakers 的 id 获取对应的张量，并转移到指定设备上
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
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        # 释放内存，删除不再需要的张量
        del (
            x_tst,
            tones,
            lang_ids,
            bert,
            x_tst_lengths,
            speakers,
            ja_bert,
            en_bert,
        )  # , emo
        # 如果有可用的 CUDA 设备，则清空 CUDA 缓存
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
    reference_audio=None,
    emotion=None,
    skip_start=False,
    skip_end=False,
):
    bert, ja_bert, en_bert, phones, tones, lang_ids = [], [], [], [], [], []
    # 初始化空列表用于存储各种特征数据
    for idx, (txt, lang) in enumerate(zip(text, language)):
        # 根据文本和语言类型获取文本特征
        _skip_start = (idx != 0) or (skip_start and idx == 0)
        _skip_end = (idx != len(language) - 1) or skip_end
        (
            temp_bert,
            temp_ja_bert,
            temp_en_bert,
            temp_phones,
            temp_tones,
            temp_lang_ids,
        ) = get_text(txt, lang, hps, device)
        # 根据条件判断是否需要跳过开头部分的特征
        if _skip_start:
            temp_bert = temp_bert[:, 3:]
            temp_ja_bert = temp_ja_bert[:, 3:]
            temp_en_bert = temp_en_bert[:, 3:]
            temp_phones = temp_phones[3:]
            temp_tones = temp_tones[3:]
            temp_lang_ids = temp_lang_ids[3:]
        # 根据条件判断是否需要跳过结尾部分的特征
        if _skip_end:
            temp_bert = temp_bert[:, :-2]
            temp_ja_bert = temp_ja_bert[:, :-2]
            temp_en_bert = temp_en_bert[:, :-2]
            temp_phones = temp_phones[:-2]
            temp_tones = temp_tones[:-2]
            temp_lang_ids = temp_lang_ids[:-2]
        # 将获取的特征数据添加到对应的列表中
        bert.append(temp_bert)
        ja_bert.append(temp_ja_bert)
        en_bert.append(temp_en_bert)
        phones.append(temp_phones)
        tones.append(temp_tones)
        lang_ids.append(temp_lang_ids)
    # 将列表中的特征数据拼接成张量
    bert = torch.concatenate(bert, dim=1)
    ja_bert = torch.concatenate(ja_bert, dim=1)
    en_bert = torch.concatenate(en_bert, dim=1)
    phones = torch.concatenate(phones, dim=0)
    tones = torch.concatenate(tones, dim=0)
    # 将多个张量在指定维度上拼接起来
    lang_ids = torch.concatenate(lang_ids, dim=0)
    # 关闭梯度计算
    with torch.no_grad():
        # 将 phones 转移到指定设备上，并增加一个维度
        x_tst = phones.to(device).unsqueeze(0)
        # 将 tones 转移到指定设备上，并增加一个维度
        tones = tones.to(device).unsqueeze(0)
        # 将 lang_ids 转移到指定设备上，并增加一个维度
        lang_ids = lang_ids.to(device).unsqueeze(0)
        # 将 bert 转移到指定设备上，并增加一个维度
        bert = bert.to(device).unsqueeze(0)
        # 将 ja_bert 转移到指定设备上，并增加一个维度
        ja_bert = ja_bert.to(device).unsqueeze(0)
        # 将 en_bert 转移到指定设备上，并增加一个维度
        en_bert = en_bert.to(device).unsqueeze(0)
        # 创建 x_tst_lengths 张量，并转移到指定设备上
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        # 释放 phones 占用的内存
        del phones
        # 创建 speakers 张量，并转移到指定设备上
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        # 调用 net_g.infer 方法生成音频数据
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
        del (
            x_tst,
            tones,
            lang_ids,
            bert,
            x_tst_lengths,
            speakers,
            ja_bert,
            en_bert,
        )  # , emo
        # 如果 CUDA 可用，则清空 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 返回生成的音频数据
        return audio
```