# `Bert-VITS2\oldVersion\V220\clap_wrapper.py`

```
# 导入 sys 模块
import sys

# 导入 torch 模块
import torch
# 从 transformers 模块中导入 ClapModel 和 ClapProcessor 类
from transformers import ClapModel, ClapProcessor

# 从 config 模块中导入 config 变量
from config import config

# 创建空的模型字典
models = dict()
# 从本地预训练模型路径创建处理器对象
processor = ClapProcessor.from_pretrained("./emotional/clap-htsat-fused")

# 定义获取音频特征的函数
def get_clap_audio_feature(audio_data, device=config.bert_gen_config.device):
    # 如果是 macOS 平台且支持多进程并行计算且设备为 CPU，则将设备设置为 "mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备为空，则将设备设置为 "cuda"
    if not device:
        device = "cuda"
    # 如果设备不在模型字典中，则加载预训练模型并将其移动到指定设备
    if device not in models.keys():
        models[device] = ClapModel.from_pretrained("./emotional/clap-htsat-fused").to(
            device
        )
    # 禁用梯度计算
    with torch.no_grad():
        # 使用处理器对象处理音频数据，返回 PyTorch 张量，并将其移动到指定设备
        inputs = processor(
            audios=audio_data, return_tensors="pt", sampling_rate=48000
        ).to(device)
        # 获取音频特征
        emb = models[device].get_audio_features(**inputs)
    # 返回音频特征的转置
    return emb.T

# 定义获取文本特征的函数
def get_clap_text_feature(text, device=config.bert_gen_config.device):
    # 如果是 macOS 平台且支持多进程并行计算且设备为 CPU，则将设备设置为 "mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备为空，则将设备设置为 "cuda"
    if not device:
        device = "cuda"
    # 如果设备不在模型字典中，则加载预训练模型并将其移动到指定设备
    if device not in models.keys():
        models[device] = ClapModel.from_pretrained("./emotional/clap-htsat-fused").to(
            device
        )
    # 禁用梯度计算
    with torch.no_grad():
        # 使用处理器对象处理文本数据，返回 PyTorch 张量，并将其移动到指定设备
        inputs = processor(text=text, return_tensors="pt").to(device)
        # 获取文本特征
        emb = models[device].get_text_features(**inputs)
    # 返回文本特征的转置
    return emb.T
```