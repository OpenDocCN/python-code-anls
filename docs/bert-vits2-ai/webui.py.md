# `Bert-VITS2\webui.py`

```
# 禁用 flake8 对 E402 错误的检查
# 导入必要的模块
import os
import logging
import re_matching
from tools.sentence import split_by_language

# 设置日志级别
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# 配置日志格式
logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 导入必要的模块
import torch
import utils
from infer import infer, latest_version, get_net_g, infer_multilang
import gradio as gr
import webbrowser
import numpy as np
from config import config
from tools.translate import translate
import librosa

# 初始化 net_g 变量
net_g = None

# 获取设备配置
device = config.webui_config.device
# 如果设备为 "mps"，则设置环境变量
if device == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 定义生成音频的函数，接受多个参数
def generate_audio(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    reference_audio,
    emotion,
    style_text,
    style_weight,
    skip_start=False,
    skip_end=False,
):
    # 初始化音频列表
    audio_list = []
    # 创建长度为采样率一半的零数组，用于表示静音
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    # 使用 torch.no_grad() 上下文管理器，关闭梯度计算
    with torch.no_grad():
        # 遍历切片列表
        for idx, piece in enumerate(slices):
            # 判断是否为第一个切片，用于跳过开头部分
            skip_start = idx != 0
            # 判断是否为最后一个切片，用于跳过结尾部分
            skip_end = idx != len(slices) - 1
            # 调用 infer 函数进行推断，生成音频
            audio = infer(
                piece,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps,
                net_g=net_g,
                device=device,
                skip_start=skip_start,
                skip_end=skip_end,
                style_text=style_text,
                style_weight=style_weight,
            )
            # 将音频转换为 16 位的 WAV 格式
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            # 将转换后的音频添加到音频列表中
            audio_list.append(audio16bit)
    # 返回音频列表
    return audio_list
# 生成多语言音频
def generate_audio_multilang(
    slices,  # 切片
    sdp_ratio,  # sdp 比率
    noise_scale,  # 噪音比例
    noise_scale_w,  # 噪音比例 w
    length_scale,  # 长度比例
    speaker,  # 说话者
    language,  # 语言
    reference_audio,  # 参考音频
    emotion,  # 情感
    skip_start=False,  # 是否跳过开头
    skip_end=False,  # 是否跳过结尾
):
    audio_list = []  # 音频列表
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)  # 静音
    with torch.no_grad():  # 禁用梯度计算
        for idx, piece in enumerate(slices):  # 遍历切片
            skip_start = idx != 0  # 是否跳过开头
            skip_end = idx != len(slices) - 1  # 是否跳过结尾
            audio = infer_multilang(  # 推断多语言
                piece,  # 片段
                reference_audio=reference_audio,  # 参考音频
                emotion=emotion,  # 情感
                sdp_ratio=sdp_ratio,  # sdp 比率
                noise_scale=noise_scale,  # 噪音比例
                noise_scale_w=noise_scale_w,  # 噪音比例 w
                length_scale=length_scale,  # 长度比例
                sid=speaker,  # 说话者
                language=language[idx],  # 语言
                hps=hps,  # hps
                net_g=net_g,  # 网络 g
                device=device,  # 设备
                skip_start=skip_start,  # 是否跳过开头
                skip_end=skip_end,  # 是否跳过结尾
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)  # 转换为 16 位 wav
            audio_list.append(audio16bit)  # 添加到音频列表
    return audio_list  # 返回音频列表


def tts_split(
    text: str,  # 文本
    speaker,  # 说话者
    sdp_ratio,  # sdp 比率
    noise_scale,  # 噪音比例
    noise_scale_w,  # 噪音比例 w
    length_scale,  # 长度比例
    language,  # 语言
    cut_by_sent,  # 按句切分
    interval_between_para,  # 段落间隔
    interval_between_sent,  # 句子间隔
    reference_audio,  # 参考音频
    emotion,  # 情感
    style_text,  # 风格文本
    style_weight,  # 风格权重
):
    while text.find("\n\n") != -1:  # 当文本中存在连续两个换行符时
        text = text.replace("\n\n", "\n")  # 将连续两个换行符替换为一个换行符
    text = text.replace("|", "")  # 替换文本中的竖线
    para_list = re_matching.cut_para(text)  # 切分段落
    para_list = [p for p in para_list if p != ""]  # 去除空段落
    audio_list = []  # 音频列表
    # 遍历段落列表
    for p in para_list:
        # 如果不按句子切分
        if not cut_by_sent:
            # 处理文本，生成音频列表
            audio_list += process_text(
                p,
                speaker,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                language,
                reference_audio,
                emotion,
                style_text,
                style_weight,
            )
            # 生成静音段，长度为段落间隔时间
            silence = np.zeros((int)(44100 * interval_between_para), dtype=np.int16)
            # 将静音段添加到音频列表中
            audio_list.append(silence)
        else:
            # 初始化句子音频列表
            audio_list_sent = []
            # 使用正则表达式切分句子
            sent_list = re_matching.cut_sent(p)
            # 去除空句子
            sent_list = [s for s in sent_list if s != ""]
            # 遍历句子列表
            for s in sent_list:
                # 处理文本，生成句子音频列表
                audio_list_sent += process_text(
                    s,
                    speaker,
                    sdp_ratio,
                    noise_scale,
                    noise_scale_w,
                    length_scale,
                    language,
                    reference_audio,
                    emotion,
                    style_text,
                    style_weight,
                )
                # 生成句子间的静音段
                silence = np.zeros((int)(44100 * interval_between_sent))
                # 将静音段添加到句子音频列表中
                audio_list_sent.append(silence)
            # 如果段落间隔时间大于句子间隔时间
            if (interval_between_para - interval_between_sent) > 0:
                # 生成段落间的静音段
                silence = np.zeros(
                    (int)(44100 * (interval_between_para - interval_between_sent))
                )
                # 将静音段添加到句子音频列表中
                audio_list_sent.append(silence)
            # 将句子音频列表合并为一个完整的音频，并转换为16位wav格式
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(
                np.concatenate(audio_list_sent)
            )  # 对完整句子做音量归一
            # 将完整句子音频添加到音频列表中
            audio_list.append(audio16bit)
    # 将音频列表合并为一个完整的音频
    audio_concat = np.concatenate(audio_list)
    # 返回成功和合并后的音频数据
    return ("Success", (hps.data.sampling_rate, audio_concat))
# 处理混合文本，将文本切片中的说话者信息弹出
def process_mix(slice):
    # 弹出切片中的最后一个元素作为说话者信息
    _speaker = slice.pop()
    # 初始化文本和语言列表
    _text, _lang = [], []
    # 遍历切片中的语言和内容
    for lang, content in slice:
        # 根据 "|" 分割内容
        content = content.split("|")
        # 过滤掉空字符串
        content = [part for part in content if part != ""]
        # 如果内容为空，则跳过
        if len(content) == 0:
            continue
        # 如果文本列表为空，则将内容添加到文本和语言列表中
        if len(_text) == 0:
            _text = [[part] for part in content]
            _lang = [[lang] for part in content]
        else:
            # 否则将内容添加到文本和语言列表中
            _text[-1].append(content[0])
            _lang[-1].append(lang)
            # 如果内容长度大于1，则将剩余部分添加到文本和语言列表中
            if len(content) > 1:
                _text += [[part] for part in content[1:]]
                _lang += [[lang] for part in content[1:]]
    # 返回处理后的文本、语言和说话者信息
    return _text, _lang, _speaker


# 处理自动生成的文本，将文本按语言切分并处理
def process_auto(text):
    # 初始化文本和语言列表
    _text, _lang = [], []
    # 根据 "|" 分割文本
    for slice in text.split("|"):
        # 如果切片为空，则跳过
        if slice == "":
            continue
        # 初始化临时文本和语言列表
        temp_text, temp_lang = [], []
        # 根据指定语言切分句子
        sentences_list = split_by_language(slice, target_languages=["zh", "ja", "en"])
        # 遍历切分后的句子和语言
        for sentence, lang in sentences_list:
            # 如果句子为空，则跳过
            if sentence == "":
                continue
            # 将句子添加到临时文本列表中
            temp_text.append(sentence)
            # 如果语言为日语，则转换为"JP"
            if lang == "ja":
                lang = "jp"
            # 将语言转换为大写并添加到临时语言列表中
            temp_lang.append(lang.upper())
        # 将处理后的临时文本和语言列表添加到文本和语言列表中
        _text.append(temp_text)
        _lang.append(temp_lang)
    # 返回处理后的文本和语言列表
    return _text, _lang


# 处理文本，接收多个参数并返回音频列表
def process_text(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    reference_audio,
    emotion,
    style_text=None,
    style_weight=0,
):
    # 初始化音频列表
    audio_list = []
    # 如果语言是"mix"，则进行以下操作
    if language == "mix":
        # 使用正则表达式验证文本的有效性，并返回验证结果和验证信息
        bool_valid, str_valid = re_matching.validate_text(text)
        # 如果验证结果为假，返回验证信息和默认音频参数
        if not bool_valid:
            return str_valid, (
                hps.data.sampling_rate,
                np.concatenate([np.zeros(hps.data.sampling_rate // 2)]),
            )
        # 对文本进行匹配处理，获取每个片段
        for slice in re_matching.text_matching(text):
            # 对每个片段进行处理，获取文本、语言和说话者信息
            _text, _lang, _speaker = process_mix(slice)
            # 如果说话者信息为空，跳过当前片段
            if _speaker is None:
                continue
            # 打印文本和语言信息
            print(f"Text: {_text}\nLang: {_lang}")
            # 生成多语言音频并添加到音频列表中
            audio_list.extend(
                generate_audio_multilang(
                    _text,
                    sdp_ratio,
                    noise_scale,
                    noise_scale_w,
                    length_scale,
                    _speaker,
                    _lang,
                    reference_audio,
                    emotion,
                )
            )
    # 如果语言是"auto"，则进行以下操作
    elif language.lower() == "auto":
        # 对文本进行自动处理，获取处理后的文本和语言信息
        _text, _lang = process_auto(text)
        # 打印文本和语言信息
        print(f"Text: {_text}\nLang: {_lang}")
        # 生成多语言音频并添加到音频列表中
        audio_list.extend(
            generate_audio_multilang(
                _text,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                speaker,
                _lang,
                reference_audio,
                emotion,
            )
        )
    # 如果语言不是"mix"也不是"auto"，则进行以下操作
    else:
        # 生成音频并添加到音频列表中
        audio_list.extend(
            generate_audio(
                text.split("|"),
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                speaker,
                language,
                reference_audio,
                emotion,
                style_text,
                style_weight,
            )
        )
    # 返回音频列表
    return audio_list
# 文本转语音函数，接受多个参数
def tts_fn(
    text: str,  # 输入的文本
    speaker,  # 说话者
    sdp_ratio,  # sdp比率
    noise_scale,  # 噪音比例
    noise_scale_w,  # 噪音比例w
    length_scale,  # 长度比例
    language,  # 语言
    reference_audio,  # 参考音频
    emotion,  # 情感
    prompt_mode,  # 提示模式
    style_text=None,  # 风格文本，默认为None
    style_weight=0,  # 风格权重，默认为0
):
    # 如果风格文本为空字符串，则将其设为None
    if style_text == "":
        style_text = None
    # 如果提示模式为"Audio prompt"
    if prompt_mode == "Audio prompt":
        # 如果参考音频为None，则返回错误信息和None
        if reference_audio == None:
            return ("Invalid audio prompt", None)
        else:
            # 加载参考音频
            reference_audio = load_audio(reference_audio)[1]
    else:
        reference_audio = None

    # 处理文本，生成音频列表
    audio_list = process_text(
        text,
        speaker,
        sdp_ratio,
        noise_scale,
        noise_scale_w,
        length_scale,
        language,
        reference_audio,
        emotion,
        style_text,
        style_weight,
    )

    # 将音频列表拼接成一个音频
    audio_concat = np.concatenate(audio_list)
    # 返回成功信息和音频数据
    return "Success", (hps.data.sampling_rate, audio_concat)


# 格式化工具函数，接受文本和说话者参数
def format_utils(text, speaker):
    # 处理自动文本
    _text, _lang = process_auto(text)
    res = f"[{speaker}]"
    # 遍历处理后的文本和语言
    for lang_s, content_s in zip(_lang, _text):
        for lang, content in zip(lang_s, content_s):
            res += f"<{lang.lower()}>{content}"
        res += "|"
    # 返回混合类型和格式化后的文本
    return "mix", res[:-1]


# 加载音频函数，接受音频路径参数
def load_audio(path):
    # 使用librosa加载音频
    audio, sr = librosa.load(path, 48000)
    # 返回采样率和音频数据
    return sr, audio


# GR工具函数，接受项目参数
def gr_util(item):
    # 如果项目为"Text prompt"，返回可见和更新类型为True的字典，否则返回False
    if item == "Text prompt":
        return {"visible": True, "__type__": "update"}, {
            "visible": False,
            "__type__": "update",
        }
    else:
        return {"visible": False, "__type__": "update"}, {
            "visible": True,
            "__type__": "update",
        }


# 如果当前脚本为主程序
if __name__ == "__main__":
    # 如果webui_config中的debug为True
    if config.webui_config.debug:
        # 输出调试级别日志
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    # 从配置文件中获取超参数
    hps = utils.get_hparams_from_file(config.webui_config.config_path)
    # 如果配置文件中未指定版本，则默认为最新版本
    version = hps.version if hasattr(hps, "version") else latest_version
    # 获取生成器网络
    net_g = get_net_g(
        model_path=config.webui_config.model, version=version, device=device, hps=hps
    )
    # 获取说话者 ID 到索引的映射
    speaker_ids = hps.data.spk2id
    # 获取所有说话者的 ID 列表
    speakers = list(speaker_ids.keys())
    # 定义语言列表
    languages = ["ZH", "JP", "EN", "mix", "auto"]
    # 打印信息
    print("推理页面已开启!")
    # 在浏览器中打开指定端口的网页
    webbrowser.open(f"http://127.0.0.1:{config.webui_config.port}")
    # 启动应用程序
    app.launch(share=config.webui_config.share, server_port=config.webui_config.port)
```