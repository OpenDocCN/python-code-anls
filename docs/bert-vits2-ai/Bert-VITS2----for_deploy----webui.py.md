# `Bert-VITS2\for_deploy\webui.py`

```

# flake8: noqa: E402
# 导入所需的模块
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

# 导入其他模块和库
import torch
import utils
from infer import infer, latest_version, get_net_g, infer_multilang
import gradio as gr
import webbrowser
import numpy as np
from config import config
from tools.translate import translate
import librosa
from infer_utils import BertFeature, ClapFeature

# 初始化变量
net_g = None

# 获取设备配置
device = config.webui_config.device
if device == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 设置环境变量
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 初始化BERT特征映射
bert_feature_map = {
    "ZH": BertFeature(
        "./bert/chinese-roberta-wwm-ext-large",
        language="ZH",
    ),
    "JP": BertFeature(
        "./bert/deberta-v2-large-japanese-char-wwm",
        language="JP",
    ),
    "EN": BertFeature(
        "./bert/deberta-v3-large",
        language="EN",
    ),
}

# 初始化CLAP特征
clap_feature = ClapFeature("./emotional/clap-htsat-fused")

# 定义生成音频的函数
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
    skip_start=False,
    skip_end=False,
):
    # 初始化音频列表
    audio_list = []
    # 使用torch.no_grad()上下文管理器，关闭梯度计算
    with torch.no_grad():
        # 遍历音频片段
        for idx, piece in enumerate(slices):
            # 判断是否跳过开头和结尾
            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(slices) - 1) and skip_end
            # 推断生成音频
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
                bert=bert_feature_map,
                clap=clap_feature,
            )
            # 转换音频为16位wav格式
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            # 将音频添加到音频列表
            audio_list.append(audio16bit)
    # 返回音频列表
    return audio_list

# 定义多语言生成音频的函数
def generate_audio_multilang(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    reference_audio,
    emotion,
    skip_start=False,
    skip_end=False,
):
    # 初始化音频列表
    audio_list = []
    # 使用torch.no_grad()上下文管理器，关闭梯度计算
    with torch.no_grad():
        # 遍历音频片段
        for idx, piece in enumerate(slices):
            # 判断是否跳过开头和结尾
            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(slices) - 1) and skip_end
            # 推断生成多语言音频
            audio = infer_multilang(
                piece,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language[idx],
                hps=hps,
                net_g=net_g,
                device=device,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            # 转换音频为16位wav格式
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            # 将音频添加到音频列表
            audio_list.append(audio16bit)
    # 返回音频列表
    return audio_list

# 定义TTS分割函数
def tts_split(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    cut_by_sent,
    interval_between_para,
    interval_between_sent,
    reference_audio,
    emotion,
):
    # 如果语言为混合语言，则返回无效结果
    if language == "mix":
        return ("invalid", None)
    # 替换文本中的双换行符为单个换行符
    while text.find("\n\n") != -1:
        text = text.replace("\n\n", "\n")
    # 分割段落
    para_list = re_matching.cut_para(text)
    audio_list = []
    # 如果不按句子分割
    if not cut_by_sent:
        for idx, p in enumerate(para_list):
            skip_start = idx != 0
            skip_end = idx != len(para_list) - 1
            # 推断生成音频
            audio = infer(
                p,
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
            )
            # 转换音频为16位wav格式
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            # 将音频添加到音频列表
            audio_list.append(audio16bit)
            # 添加静音
            silence = np.zeros((int)(44100 * interval_between_para), dtype=np.int16)
            audio_list.append(silence)
    # 如果按句子分割
    else:
        for idx, p in enumerate(para_list):
            skip_start = idx != 0
            skip_end = idx != len(para_list) - 1
            audio_list_sent = []
            sent_list = re_matching.cut_sent(p)
            for idx, s in enumerate(sent_list):
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(sent_list) - 1) and skip_end
                # 推断生成音频
                audio = infer(
                    s,
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
                )
                # 将音频添加到音频列表
                audio_list_sent.append(audio)
                # 添加静音
                silence = np.zeros((int)(44100 * interval_between_sent))
                audio_list_sent.append(silence)
            if (interval_between_para - interval_between_sent) > 0:
                silence = np.zeros(
                    (int)(44100 * (interval_between_para - interval_between_sent))
                )
                audio_list_sent.append(silence)
            # 合并句子音频
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(
                np.concatenate(audio_list_sent)
            )
            # 将音频添加到音频列表
            audio_list.append(audio16bit)
    # 合并所有音频
    audio_concat = np.concatenate(audio_list)
    return ("Success", (44100, audio_concat))

# 定义TTS函数
def tts_fn(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    reference_audio,
    emotion,
    prompt_mode,
def load_audio(path):
    # 加载音频文件
    audio, sr = librosa.load(path, 48000)
    # 返回采样率和音频数据
    return sr, audio

# 定义gr_util函数
def gr_util(item):
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

```