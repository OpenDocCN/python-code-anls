# `d:/src/tocomm/Bert-VITS2\oldVersion\V210\emo_gen.py`

```
import librosa  # 导入 librosa 库，用于音频处理
import numpy as np  # 导入 numpy 库，用于数值计算
import torch  # 导入 torch 库，用于深度学习
import torch.nn as nn  # 导入 torch.nn 模块，用于神经网络构建
from torch.utils.data import Dataset  # 从 torch.utils.data 模块中导入 Dataset 类，用于构建数据集
from torch.utils.data import Dataset  # 从 torch.utils.data 模块中再次导入 Dataset 类，可能是重复的代码
from transformers import Wav2Vec2Processor  # 从 transformers 库中导入 Wav2Vec2Processor 类，用于音频处理
from transformers.models.wav2vec2.modeling_wav2vec2 import (  # 从 transformers 库中导入 Wav2Vec2Model 和 Wav2Vec2PreTrainedModel 类
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from config import config  # 从 config 模块中导入 config 变量

class RegressionHead(nn.Module):  # 定义 RegressionHead 类，继承自 nn.Module 类
    r"""Classification head."""  # 类的文档字符串

    def __init__(self, config):  # 定义初始化方法，接受 config 参数
        super().__init__()  # 调用父类的初始化方法
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 创建一个全连接层，输入和输出维度都为config.hidden_size
        self.dropout = nn.Dropout(config.final_dropout)  # 创建一个dropout层，丢弃概率为config.final_dropout
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)  # 创建一个全连接层，输入维度为config.hidden_size，输出维度为config.num_labels

    def forward(self, features, **kwargs):
        x = features  # 将输入features赋值给变量x
        x = self.dropout(x)  # 对x进行dropout操作
        x = self.dense(x)  # 对x进行全连接层操作
        x = torch.tanh(x)  # 对x进行tanh激活函数操作
        x = self.dropout(x)  # 对x再次进行dropout操作
        x = self.out_proj(x)  # 对x进行输出层全连接操作

        return x  # 返回x作为输出结果


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):
        super().__init__(config)  # 调用父类的构造函数，初始化模型的配置

        self.config = config  # 将传入的配置保存到模型的属性中
        self.wav2vec2 = Wav2Vec2Model(config)  # 创建一个 Wav2Vec2 模型
        self.classifier = RegressionHead(config)  # 创建一个回归头模型
        self.init_weights()  # 初始化模型的权重

    def forward(
        self,
        input_values,
    ):
        outputs = self.wav2vec2(input_values)  # 将输入数据传入 Wav2Vec2 模型中进行前向传播
        hidden_states = outputs[0]  # 获取模型输出的隐藏状态
        hidden_states = torch.mean(hidden_states, dim=1)  # 对隐藏状态进行平均池化
        logits = self.classifier(hidden_states)  # 将平均池化后的隐藏状态传入回归头模型中得到预测结果

        return hidden_states, logits  # 返回隐藏状态和预测结果


class AudioDataset(Dataset):
    def __init__(self, list_of_wav_files, sr, processor):
        # 初始化函数，接受音频文件列表、采样率和处理器作为参数
        self.list_of_wav_files = list_of_wav_files  # 将音频文件列表存储在对象属性中
        self.processor = processor  # 将处理器存储在对象属性中
        self.sr = sr  # 将采样率存储在对象属性中

    def __len__(self):
        # 返回音频文件列表的长度
        return len(self.list_of_wav_files)

    def __getitem__(self, idx):
        # 获取指定索引位置的音频文件
        wav_file = self.list_of_wav_files[idx]
        # 使用librosa库加载音频文件，并指定采样率
        audio_data, _ = librosa.load(wav_file, sr=self.sr)
        # 使用处理器处理音频数据，获取处理后的数据
        processed_data = self.processor(audio_data, sampling_rate=self.sr)["input_values"][0]
        # 将处理后的数据转换为PyTorch张量并返回
        return torch.from_numpy(processed_data)


device = config.emo_gen_config.device  # 获取设备信息
model_name = "./emotional/wav2vec2-large-robust-12-ft-emotion-msp-dim"  # 指定模型名称
processor = Wav2Vec2Processor.from_pretrained(model_name)  # 从预训练模型中加载处理器
model = EmotionModel.from_pretrained(model_name).to(device)
# 从预训练模型中加载情感模型，并将其移动到指定的设备上

def process_func(
    x: np.ndarray,
    sampling_rate: int,
    model: EmotionModel,
    processor: Wav2Vec2Processor,
    device: str,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""
    # 将模型移动到指定的设备上
    model = model.to(device)
    # 对原始音频信号进行处理，得到处理后的结果
    y = processor(x, sampling_rate=sampling_rate)
    # 从处理后的结果中提取输入数值
    y = y["input_values"][0]
    # 将输入数值转换为张量，并移动到指定的设备上
    y = torch.from_numpy(y).unsqueeze(0).to(device)

    # 通过模型运行输入数据
    with torch.no_grad():
        # 如果需要提取嵌入，则运行模型并获取嵌入结果；否则获取情感预测结果
        y = model(y)[0 if embeddings else 1]
    # 将张量转换为 numpy 数组
    y = y.detach().cpu().numpy()

    return y


def get_emo(path):
    # 从指定路径加载音频文件，并指定采样率为 16000
    wav, sr = librosa.load(path, 16000)
    # 调用 process_func 函数处理音频数据，返回情感分析结果
    return process_func(
        np.expand_dims(wav, 0).astype(np.float64),  # 将音频数据转换为 numpy 数组，并扩展维度
        sr,  # 采样率
        model,  # 模型
        processor,  # 处理器
        device,  # 设备
        embeddings=True,  # 是否返回嵌入向量
    ).squeeze(0)  # 压缩维度为 0
```