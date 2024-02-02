# `Bert-VITS2\oldVersion\V210\emo_gen.py`

```py
# 导入所需的库
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

# 从外部文件导入配置
from config import config

# 定义回归头部模型
class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):
        super().__init__()

        # 全连接层，输入维度为隐藏层大小，输出维度为隐藏层大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 随机失活层
        self.dropout = nn.Dropout(config.final_dropout)
        # 输出投影层，输入维度为隐藏层大小，输出维度为标签数量
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x

# 定义情感识别模型
class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        # 初始化 Wav2Vec2 模型
        self.wav2vec2 = Wav2Vec2Model(config)
        # 初始化回归头部模型
        self.classifier = RegressionHead(config)
        # 初始化模型权重
        self.init_weights()

    def forward(
        self,
        input_values,
    ):
        # 将输入值传入 Wav2Vec2 模型
        outputs = self.wav2vec2(input_values)
        # 获取隐藏状态
        hidden_states = outputs[0]
        # 对隐藏状态进行平均池化
        hidden_states = torch.mean(hidden_states, dim=1)
        # 通过回归头部模型得到预测结果
        logits = self.classifier(hidden_states)

        return hidden_states, logits

# 定义音频数据集类
class AudioDataset(Dataset):
    def __init__(self, list_of_wav_files, sr, processor):
        self.list_of_wav_files = list_of_wav_files
        self.processor = processor
        self.sr = sr

    def __len__(self):
        return len(self.list_of_wav_files)
    # 定义一个特殊方法，用于获取对象中指定索引位置的元素
    def __getitem__(self, idx):
        # 获取指定索引位置的音频文件名
        wav_file = self.list_of_wav_files[idx]
        # 使用 librosa 库加载音频数据，并指定采样率
        audio_data, _ = librosa.load(wav_file, sr=self.sr)
        # 对加载的音频数据进行处理，使用 processor 方法，获取处理后的数据
        processed_data = self.processor(audio_data, sampling_rate=self.sr)["input_values"][0]
        # 将处理后的数据转换为 PyTorch 的张量，并返回
        return torch.from_numpy(processed_data)
# 从配置文件中获取设备信息
device = config.emo_gen_config.device
# 指定模型名称
model_name = "./emotional/wav2vec2-large-robust-12-ft-emotion-msp-dim"
# 从预训练模型中加载音频处理器
processor = Wav2Vec2Processor.from_pretrained(model_name)
# 从预训练模型中加载情感模型，并将其移动到指定设备上
model = EmotionModel.from_pretrained(model_name).to(device)

# 定义处理函数，用于预测情感或从原始音频信号中提取嵌入
def process_func(
    x: np.ndarray,
    sampling_rate: int,
    model: EmotionModel,
    processor: Wav2Vec2Processor,
    device: str,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""
    # 将模型移动到指定设备上
    model = model.to(device)
    # 使用音频处理器处理原始音频信号
    y = processor(x, sampling_rate=sampling_rate)
    y = y["input_values"][0]
    y = torch.from_numpy(y).unsqueeze(0).to(device)

    # 通过模型运行
    with torch.no_grad():
        # 如果需要提取嵌入，则运行模型并获取嵌入
        y = model(y)[0 if embeddings else 1]

    # 转换为 numpy 数组
    y = y.detach().cpu().numpy()

    return y

# 定义获取情感的函数，用于加载音频并获取情感
def get_emo(path):
    # 加载音频文件
    wav, sr = librosa.load(path, 16000)
    # 调用处理函数，传入音频数据和相关参数
    return process_func(
        np.expand_dims(wav, 0).astype(np.float64),
        sr,
        model,
        processor,
        device,
        embeddings=True,
    ).squeeze(0)
```