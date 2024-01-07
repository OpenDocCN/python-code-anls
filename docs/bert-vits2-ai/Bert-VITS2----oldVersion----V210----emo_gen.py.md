# `Bert-VITS2\oldVersion\V210\emo_gen.py`

```

# 导入所需的库
import librosa  # 用于音频处理
import numpy as np  # 用于数值计算
import torch  # 用于构建神经网络
import torch.nn as nn  # 用于构建神经网络
from torch.utils.data import Dataset  # 用于构建数据集
from transformers import Wav2Vec2Processor  # 用于处理音频数据
from transformers.models.wav2vec2.modeling_wav2vec2 import (  # 用于构建Wav2Vec2模型
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from config import config  # 导入配置文件中的参数


# 定义回归头部，用于情感分类
class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 全连接层
        self.dropout = nn.Dropout(config.final_dropout)  # Dropout层
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)  # 输出层

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


# 定义情感模型，继承自Wav2Vec2PreTrainedModel
class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)  # Wav2Vec2模型
        self.classifier = RegressionHead(config)  # 回归头部
        self.init_weights()

    def forward(
        self,
        input_values,
    ):
        outputs = self.wav2vec2(input_values)  # 输入数据到Wav2Vec2模型
        hidden_states = outputs[0]  # 获取隐藏状态
        hidden_states = torch.mean(hidden_states, dim=1)  # 对隐藏状态进行平均
        logits = self.classifier(hidden_states)  # 通过回归头部进行情感分类

        return hidden_states, logits


# 定义音频数据集类
class AudioDataset(Dataset):
    def __init__(self, list_of_wav_files, sr, processor):
        self.list_of_wav_files = list_of_wav_files
        self.processor = processor
        self.sr = sr

    def __len__(self):
        return len(self.list_of_wav_files)

    def __getitem__(self, idx):
        wav_file = self.list_of_wav_files[idx]
        audio_data, _ = librosa.load(wav_file, sr=self.sr)  # 加载音频数据
        processed_data = self.processor(audio_data, sampling_rate=self.sr)[
            "input_values"
        ][0]  # 处理音频数据
        return torch.from_numpy(processed_data)


# 从配置文件中获取设备信息
device = config.emo_gen_config.device
# 模型名称
model_name = "./emotional/wav2vec2-large-robust-12-ft-emotion-msp-dim"
# 从预训练模型中加载音频处理器
processor = Wav2Vec2Processor.from_pretrained(model_name)
# 从预训练模型中加载情感模型，并移动到指定设备
model = EmotionModel.from_pretrained(model_name).to(device)


# 定义处理函数，用于预测情感或提取嵌入
def process_func(
    x: np.ndarray,
    sampling_rate: int,
    model: EmotionModel,
    processor: Wav2Vec2Processor,
    device: str,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""
    model = model.to(device)
    y = processor(x, sampling_rate=sampling_rate)  # 处理音频数据
    y = y["input_values"][0]
    y = torch.from_numpy(y).unsqueeze(0).to(device)

    # 通过模型运行
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # 转换为numpy数组
    y = y.detach().cpu().numpy()

    return y


# 获取音频文件的情感信息
def get_emo(path):
    wav, sr = librosa.load(path, 16000)  # 加载音频文件
    return process_func(
        np.expand_dims(wav, 0).astype(np.float64),
        sr,
        model,
        processor,
        device,
        embeddings=True,
    ).squeeze(0)

```