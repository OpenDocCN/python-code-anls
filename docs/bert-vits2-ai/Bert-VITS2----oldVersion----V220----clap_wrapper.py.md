# `Bert-VITS2\oldVersion\V220\clap_wrapper.py`

```

# 导入必要的库
import sys
import torch
from transformers import ClapModel, ClapProcessor
from config import config

# 创建一个空的模型字典和处理器对象
models = dict()
processor = ClapProcessor.from_pretrained("./emotional/clap-htsat-fused")

# 定义一个函数，用于获取音频特征
def get_clap_audio_feature(audio_data, device=config.bert_gen_config.device):
    # 检查设备是否支持 MPS，并根据条件设置设备
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备为空，则默认使用 CUDA
    if not device:
        device = "cuda"
    # 如果设备不在模型字典中，则加载模型到设备
    if device not in models.keys():
        models[device] = ClapModel.from_pretrained("./emotional/clap-htsat-fused").to(
            device
        )
    # 使用模型获取音频特征
    with torch.no_grad():
        inputs = processor(
            audios=audio_data, return_tensors="pt", sampling_rate=48000
        ).to(device)
        emb = models[device].get_audio_features(**inputs)
    return emb.T

# 定义一个函数，用于获取文本特征
def get_clap_text_feature(text, device=config.bert_gen_config.device):
    # 检查设备是否支持 MPS，并根据条件设置设备
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备为空，则默认使用 CUDA
    if not device:
        device = "cuda"
    # 如果设备不在模型字典中，则加载模型到设备
    if device not in models.keys():
        models[device] = ClapModel.from_pretrained("./emotional/clap-htsat-fused").to(
            device
        )
    # 使用模型获取文本特征
    with torch.no_grad():
        inputs = processor(text=text, return_tensors="pt").to(device)
        emb = models[device].get_text_features(**inputs)
    return emb.T

```