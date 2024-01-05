# `d:/src/tocomm/Bert-VITS2\oldVersion\V220\clap_wrapper.py`

```
import sys  # 导入sys模块，用于访问与Python解释器交互的变量和函数

import torch  # 导入torch模块，用于构建深度学习模型
from transformers import ClapModel, ClapProcessor  # 从transformers模块中导入ClapModel和ClapProcessor类

from config import config  # 从config模块中导入config变量

models = dict()  # 创建一个空的字典变量models
processor = ClapProcessor.from_pretrained("./emotional/clap-htsat-fused")  # 从预训练模型中加载ClapProcessor对象


def get_clap_audio_feature(audio_data, device=config.bert_gen_config.device):  # 定义一个名为get_clap_audio_feature的函数，接受音频数据和设备参数
    if (  # 如果条件判断语句开始
        sys.platform == "darwin"  # 如果当前操作系统是MacOS
        and torch.backends.mps.is_available()  # 并且torch后端支持MPS（多进程单元）
        and device == "cpu"  # 并且设备参数为CPU
    ):  # 条件判断语句结束
        device = "mps"  # 将设备参数修改为MPS
    if not device:  # 如果设备参数为空
        device = "cuda"  # 将设备参数修改为CUDA
        # 如果设备不在模型字典的键中，将其添加到模型字典中，并加载预训练模型到设备
        if device not in models.keys():
            models[device] = ClapModel.from_pretrained("./emotional/clap-htsat-fused").to(
                device
            )
        # 使用torch.no_grad()上下文管理器，确保在评估模式下执行，减少内存消耗
        with torch.no_grad():
            # 使用processor处理文本数据，返回PyTorch张量，并将其加载到设备上
            inputs = processor(
                audios=audio_data, return_tensors="pt", sampling_rate=48000
            ).to(device)
            # 获取音频特征
            emb = models[device].get_audio_features(**inputs)
        # 返回音频特征的转置
        return emb.T

    # 定义一个函数，用于获取Clap模型的文本特征
    def get_clap_text_feature(text, device=config.bert_gen_config.device):
        # 如果操作系统是macOS，并且torch的多进程服务可用，并且设备是CPU，则将设备设置为"mps"
        if (
            sys.platform == "darwin"
            and torch.backends.mps.is_available()
            and device == "cpu"
        ):
            device = "mps"
        # 如果设备未指定，则执行以下操作
        if not device:
        device = "cuda"  # 设置设备为 CUDA，用于加速模型训练和推理
    if device not in models.keys():  # 如果设备不在模型字典中
        models[device] = ClapModel.from_pretrained("./emotional/clap-htsat-fused").to(
            device  # 从预训练模型中加载模型，并将模型移动到指定设备
        )
    with torch.no_grad():  # 在推理过程中不需要计算梯度
        inputs = processor(text=text, return_tensors="pt").to(device)  # 处理输入文本并将其移动到指定设备
        emb = models[device].get_text_features(**inputs)  # 使用模型获取文本特征
    return emb.T  # 返回文本特征的转置
```