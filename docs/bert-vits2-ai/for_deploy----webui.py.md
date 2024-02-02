# `Bert-VITS2\for_deploy\webui.py`

```py
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

# 配置日志格式和级别
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
from infer_utils import BertFeature, ClapFeature

# 初始化 net_g 为 None
net_g = None

# 获取设备配置
device = config.webui_config.device
# 如果设备为 "mps"，设置环境变量 PYTORCH_ENABLE_MPS_FALLBACK 为 "1"
if device == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 设置环境变量 OMP_NUM_THREADS 为 "1"
os.environ["OMP_NUM_THREADS"] = "1"
# 设置环境变量 MKL_NUM_THREADS 为 "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 初始化 bert_feature_map 字典，包含不同语言的 BertFeature 对象
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

# 初始化 clap_feature 为 ClapFeature 对象
clap_feature = ClapFeature("./emotional/clap-htsat-fused")

# 定义 generate_audio 函数，接受多个参数
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
    # 初始化空的音频列表
    audio_list = []
    # 初始化 silence 为长度为采样率一半的零数组
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    # 使用 torch.no_grad() 上下文管理器，关闭梯度计算
    with torch.no_grad():
        # 遍历切片列表，获取索引和切片数据
        for idx, piece in enumerate(slices):
            # 如果不是第一个切片，则根据条件更新 skip_start
            skip_start = (idx != 0) and skip_start
            # 如果不是最后一个切片，则根据条件更新 skip_end
            skip_end = (idx != len(slices) - 1) and skip_end
            # 调用 infer 函数进行推断，获取音频数据
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
            # 将音频数据转换为 16 位的 WAV 格式
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            # 将转换后的音频数据添加到音频列表中
            audio_list.append(audio16bit)
            # 将静音添加到列表中
            # audio_list.append(silence)
    # 返回音频列表
    return audio_list
# 生成多语言音频
def generate_audio_multilang(
    slices,  # 切片
    sdp_ratio,  # SDP 比例
    noise_scale,  # 噪音比例
    noise_scale_w,  # 噪音比例 W
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
            skip_start = (idx != 0) and skip_start  # 是否跳过开头
            skip_end = (idx != len(slices) - 1) and skip_end  # 是否跳过结尾
            # 推断多语言
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
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)  # 转换为 16 位 WAV
            audio_list.append(audio16bit)  # 添加到音频列表
            # audio_list.append(silence)  # 将静音添加到列表中
    return audio_list  # 返回音频列表


# TTS 分割
def tts_split(
    text: str,  # 文本
    speaker,  # 说话者
    sdp_ratio,  # SDP 比例
    noise_scale,  # 噪音比例
    noise_scale_w,  # 噪音比例 W
    length_scale,  # 长度比例
    language,  # 语言
    cut_by_sent,  # 按句切割
    interval_between_para,  # 段落间隔
    interval_between_sent,  # 句子间隔
    reference_audio,  # 参考音频
    emotion,  # 情感
):
    if language == "mix":  # 如果语言为混合
        return ("invalid", None)  # 返回无效
    while text.find("\n\n") != -1:  # 当文本中存在连续两个换行符时
        text = text.replace("\n\n", "\n")  # 替换为一个换行符
    para_list = re_matching.cut_para(text)  # 切割段落
    audio_list = []  # 音频列表
    # 如果不按句子分割，则遍历段落列表
    if not cut_by_sent:
        # 对段落列表进行遍历，获取索引和段落内容
        for idx, p in enumerate(para_list):
            # 判断是否为第一个段落，如果是则跳过开头
            skip_start = idx != 0
            # 判断是否为最后一个段落，如果是则跳过结尾
            skip_end = idx != len(para_list) - 1
            # 调用infer函数进行音频推断，传入参数包括段落内容和其他参数
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
            # 将音频转换为16位wav格式
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            # 将转换后的音频添加到音频列表中
            audio_list.append(audio16bit)
            # 生成指定长度的静音，单位为采样点数
            silence = np.zeros((int)(44100 * interval_between_para), dtype=np.int16)
            # 将静音添加到音频列表中
            audio_list.append(silence)
    else:
        # 遍历段落列表
        for idx, p in enumerate(para_list):
            # 判断是否跳过段落开头
            skip_start = idx != 0
            # 判断是否跳过段落结尾
            skip_end = idx != len(para_list) - 1
            # 初始化句子音频列表
            audio_list_sent = []
            # 使用正则表达式切分句子
            sent_list = re_matching.cut_sent(p)
            # 遍历句子列表
            for idx, s in enumerate(sent_list):
                # 判断是否跳过句子开头
                skip_start = (idx != 0) and skip_start
                # 判断是否跳过句子结尾
                skip_end = (idx != len(sent_list) - 1) and skip_end
                # 推断句子的音频
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
                # 将句子音频添加到列表中
                audio_list_sent.append(audio)
                # 添加句子间的静音
                silence = np.zeros((int)(44100 * interval_between_sent))
                audio_list_sent.append(silence)
            # 如果段落间的间隔大于句子间的间隔，添加段落间的静音
            if (interval_between_para - interval_between_sent) > 0:
                silence = np.zeros(
                    (int)(44100 * (interval_between_para - interval_between_sent))
                )
                audio_list_sent.append(silence)
            # 将句子音频合并并转换为16位wav格式
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(
                np.concatenate(audio_list_sent)
            )  # 对完整句子做音量归一
            # 将音频添加到列表中
            audio_list.append(audio16bit)
    # 将所有音频合并
    audio_concat = np.concatenate(audio_list)
    # 返回成功和合并后的音频数据
    return ("Success", (44100, audio_concat))
# 定义一个文本转语音函数，接受多个参数
def tts_fn(
    text: str,  # 输入的文本内容
    speaker,  # 说话者
    sdp_ratio,  # sdp 比例
    noise_scale,  # 噪音比例
    noise_scale_w,  # 噪音比例 w
    length_scale,  # 长度比例
    language,  # 语言
    reference_audio,  # 参考音频
    emotion,  # 情感
    prompt_mode,  # 提示模式
):
    # 如果提示模式为 "Audio prompt"
    if prompt_mode == "Audio prompt":
        # 如果没有提供参考音频，则返回错误信息和空值
        if reference_audio == None:
            return ("Invalid audio prompt", None)
        # 否则，加载参考音频的第二个值
        else:
            reference_audio = load_audio(reference_audio)[1]
    # 否则，将参考音频设为 None
    else:
        reference_audio = None
    # 初始化一个空的音频列表
    audio_list = []
    # 如果语言为"auto"，则按照指定分隔符分割文本，并遍历每个分片
    elif language.lower() == "auto":
        for idx, slice in enumerate(text.split("|")):
            # 如果分片为空，则跳过
            if slice == "":
                continue
            # 判断是否需要跳过分片的开头和结尾
            skip_start = idx != 0
            skip_end = idx != len(text.split("|")) - 1
            # 将分片按照指定语言分割成句子列表
            sentences_list = split_by_language(
                slice, target_languages=["zh", "ja", "en"]
            )
            idx = 0
            # 遍历句子列表
            while idx < len(sentences_list):
                text_to_generate = []
                lang_to_generate = []
                # 循环处理句子列表中的句子
                while True:
                    content, lang = sentences_list[idx]
                    temp_text = [content]
                    lang = lang.upper()
                    # 将日语的语言代码"JA"转换为"JP"
                    if lang == "JA":
                        lang = "JP"
                    # 将句子添加到待生成文本列表中
                    if len(text_to_generate) > 0:
                        text_to_generate[-1] += [temp_text.pop(0)]
                        lang_to_generate[-1] += [lang]
                    if len(temp_text) > 0:
                        text_to_generate += [[i] for i in temp_text]
                        lang_to_generate += [[lang]] * len(temp_text)
                    # 如果还有下一个句子，则继续处理下一个句子
                    if idx + 1 < len(sentences_list):
                        idx += 1
                    else:
                        break
                # 判断是否需要跳过分片的开头和结尾
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(sentences_list) - 1) and skip_end
                # 打印待生成文本和语言列表
                print(text_to_generate, lang_to_generate)
                # 生成多语言音频，并将结果添加到音频列表中
                audio_list.extend(
                    generate_audio_multilang(
                        text_to_generate,
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                        length_scale,
                        speaker,
                        lang_to_generate,
                        reference_audio,
                        emotion,
                        skip_start,
                        skip_end,
                    )
                )
                idx += 1
    else:
        # 如果条件不满足，则执行以下代码
        audio_list.extend(
            # 将生成的音频列表添加到现有音频列表中
            generate_audio(
                # 生成音频，传入文本、音频比例、噪音比例、噪音比例w、长度比例、说话者、语言、参考音频、情感
                text.split("|"),
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                speaker,
                language,
                reference_audio,
                emotion,
            )
        )
    # 将音频列表连接成一个数组
    audio_concat = np.concatenate(audio_list)
    # 返回成功和音频数据的采样率和连接后的音频数组
    return "Success", (hps.data.sampling_rate, audio_concat)
# 从指定路径加载音频文件，并指定采样率为48000
def load_audio(path):
    audio, sr = librosa.load(path, 48000)
    # 返回采样率和音频数据
    return sr, audio

# 根据项目类型返回对应的更新信息
def gr_util(item):
    if item == "Text prompt":
        # 如果项目类型为"Text prompt"，则返回可见和不可见的更新信息
        return {"visible": True, "__type__": "update"}, {
            "visible": False,
            "__type__": "update",
        }
    else:
        # 如果项目类型不为"Text prompt"，则返回不可见和可见的更新信息
        return {"visible": False, "__type__": "update"}, {
            "visible": True,
            "__type__": "update",
        }

# 主程序入口
if __name__ == "__main__":
    # 如果配置为调试模式，则启用DEBUG级别日志
    if config.webui_config.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    # 从配置文件中获取超参数
    hps = utils.get_hparams_from_file(config.webui_config.config_path)
    # 若config.json中未指定版本则默认为最新版本
    version = hps.version if hasattr(hps, "version") else latest_version
    # 获取生成器网络模型
    net_g = get_net_g(
        model_path=config.webui_config.model, version=version, device=device, hps=hps
    )
    # 获取说话人ID映射表
    speaker_ids = hps.data.spk2id
    # 获取所有说话人列表
    speakers = list(speaker_ids.keys())
    # 设置语言列表
    languages = ["ZH", "JP", "EN", "mix", "auto"]
    # 打印提示信息
    print("推理页面已开启!")
    # 在浏览器中打开指定端口的页面
    webbrowser.open(f"http://127.0.0.1:{config.webui_config.port}")
    # 启动应用程序
    app.launch(share=config.webui_config.share, server_port=config.webui_config.port)
```