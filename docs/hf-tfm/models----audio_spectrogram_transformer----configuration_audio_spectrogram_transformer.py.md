# `.\models\audio_spectrogram_transformer\configuration_audio_spectrogram_transformer.py`

```
# coding=utf-8
# Copyright 2022 Google AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Audio Spectogram Transformer (AST) model configuration
"""

# 从相应的库中导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件的映射字典，将模型名称映射到配置文件的下载链接
AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "MIT/ast-finetuned-audioset-10-10-0.4593": (
        "https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/resolve/main/config.json"
    ),
}


class ASTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ASTModel`]. It is used to instantiate an AST
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the AST
    [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 设置模型类型为音频频谱变换器
    model_type = "audio-spectrogram-transformer"
    # 初始化函数，用于初始化 Transformer 模型的参数
    def __init__(
        self,
        hidden_size=768,  # 设置隐藏层的大小，默认为768
        num_hidden_layers=12,  # Transformer 模型中的隐藏层数，默认为12
        num_attention_heads=12,  # 每个注意力头的数量，默认为12
        intermediate_size=3072,  # Transformer 中间层的大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数的选择，默认为 GELU
        hidden_dropout_prob=0.0,  # 隐藏层的 dropout 概率，默认为0.0，即不进行 dropout
        attention_probs_dropout_prob=0.0,  # 注意力层的 dropout 概率，默认为0.0，即不进行 dropout
        initializer_range=0.02,  # 参数初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # Layer normalization 的 epsilon，默认为 1e-12
        patch_size=16,  # 图像块的大小，默认为16
        qkv_bias=True,  # 是否在 QKV 层中使用偏置，默认为 True
        frequency_stride=10,  # 频率维度的步长，默认为10
        time_stride=10,  # 时间维度的步长，默认为10
        max_length=1024,  # 最大序列长度，默认为1024
        num_mel_bins=128,  # Mel 频谱的频道数，默认为128
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 将参数赋值给对象的属性
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.patch_size = patch_size
        self.qkv_bias = qkv_bias
        self.frequency_stride = frequency_stride
        self.time_stride = time_stride
        self.max_length = max_length
        self.num_mel_bins = num_mel_bins
```