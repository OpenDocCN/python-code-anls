# `.\models\vits\configuration_vits.py`

```
# coding=utf-8
# Copyright 2023 The Kakao Enterprise Authors and the HuggingFace Inc. team. All rights reserved.
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
""" VITS model configuration"""


# 引入预训练配置类 PretrainedConfig 和日志工具 logging
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预训练配置映射字典，指定预训练模型名称及其配置文件的下载链接
VITS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/mms-tts-eng": "https://huggingface.co/facebook/mms-tts-eng/resolve/main/config.json",
}


# VitsConfig 类继承自 PretrainedConfig，用于存储 VITS 模型的配置信息
class VitsConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VitsModel`]. It is used to instantiate a VITS
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the VITS
    [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import VitsModel, VitsConfig

    >>> # Initializing a "facebook/mms-tts-eng" style configuration
    >>> configuration = VitsConfig()

    >>> # Initializing a model (with random weights) from the "facebook/mms-tts-eng" style configuration
    >>> model = VitsModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    
    model_type 属性指定了模型类型为 "vits"
    """
    model_type = "vits"
    # 定义初始化方法，用于创建和初始化 TransformerTTS 模型
    def __init__(
        self,
        vocab_size=38,  # 词汇表大小，默认为 38
        hidden_size=192,  # 隐藏层大小，默认为 192
        num_hidden_layers=6,  # Transformer 中隐藏层的数量，默认为 6
        num_attention_heads=2,  # 注意力头的数量，默认为 2
        window_size=4,  # 窗口大小，默认为 4
        use_bias=True,  # 是否使用偏置，默认为 True
        ffn_dim=768,  # FeedForward 网络的维度，默认为 768
        layerdrop=0.1,  # 层丢弃率，默认为 0.1
        ffn_kernel_size=3,  # FeedForward 网络的卷积核大小，默认为 3
        flow_size=192,  # 流的大小，默认为 192
        spectrogram_bins=513,  # 频谱图的频率分辨率，默认为 513
        hidden_act="relu",  # 隐藏层激活函数，默认为 ReLU
        hidden_dropout=0.1,  # 隐藏层的 dropout 概率，默认为 0.1
        attention_dropout=0.1,  # 注意力机制的 dropout 概率，默认为 0.1
        activation_dropout=0.1,  # 激活函数的 dropout 概率，默认为 0.1
        initializer_range=0.02,  # 参数初始化范围，默认为 0.02
        layer_norm_eps=1e-5,  # Layer Normalization 的 epsilon，默认为 1e-5
        use_stochastic_duration_prediction=True,  # 是否使用随机时长预测，默认为 True
        num_speakers=1,  # 说话者的数量，默认为 1
        speaker_embedding_size=0,  # 说话者嵌入的维度，默认为 0
        upsample_initial_channel=512,  # 上采样层的初始通道数，默认为 512
        upsample_rates=[8, 8, 2, 2],  # 上采样层的上采样率列表，默认为 [8, 8, 2, 2]
        upsample_kernel_sizes=[16, 16, 4, 4],  # 上采样层的卷积核大小列表，默认为 [16, 16, 4, 4]
        resblock_kernel_sizes=[3, 7, 11],  # ResBlock 的卷积核大小列表，默认为 [3, 7, 11]
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],  # ResBlock 的扩张率列表，默认为 [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        leaky_relu_slope=0.1,  # Leaky ReLU 斜率，默认为 0.1
        depth_separable_channels=2,  # 深度可分离卷积的通道数，默认为 2
        depth_separable_num_layers=3,  # 深度可分离卷积的层数，默认为 3
        duration_predictor_flow_bins=10,  # 时长预测器的流的数量，默认为 10
        duration_predictor_tail_bound=5.0,  # 时长预测器的尾部边界，默认为 5.0
        duration_predictor_kernel_size=3,  # 时长预测器的卷积核大小，默认为 3
        duration_predictor_dropout=0.5,  # 时长预测器的 dropout 概率，默认为 0.5
        duration_predictor_num_flows=4,  # 时长预测器的流的数量，默认为 4
        duration_predictor_filter_channels=256,  # 时长预测器的卷积滤波器通道数，默认为 256
        prior_encoder_num_flows=4,  # 先验编码器的流的数量，默认为 4
        prior_encoder_num_wavenet_layers=4,  # 先验编码器的 WaveNet 层的数量，默认为 4
        posterior_encoder_num_wavenet_layers=16,  # 后验编码器的 WaveNet 层的数量，默认为 16
        wavenet_kernel_size=5,  # WaveNet 的卷积核大小，默认为 5
        wavenet_dilation_rate=1,  # WaveNet 的膨胀率，默认为 1
        wavenet_dropout=0.0,  # WaveNet 的 dropout 概率，默认为 0.0
        speaking_rate=1.0,  # 说话速率，默认为 1.0
        noise_scale=0.667,  # 噪声缩放因子，默认为 0.667
        noise_scale_duration=0.8,  # 时长噪声缩放因子，默认为 0.8
        sampling_rate=16_000,  # 采样率，默认为 16,000
        **kwargs,  # 其它未命名参数
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.window_size = window_size
        self.use_bias = use_bias
        self.ffn_dim = ffn_dim
        self.layerdrop = layerdrop
        self.ffn_kernel_size = ffn_kernel_size
        self.flow_size = flow_size
        self.spectrogram_bins = spectrogram_bins
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_stochastic_duration_prediction = use_stochastic_duration_prediction
        self.num_speakers = num_speakers
        self.speaker_embedding_size = speaker_embedding_size
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.leaky_relu_slope = leaky_relu_slope
        self.depth_separable_channels = depth_separable_channels
        self.depth_separable_num_layers = depth_separable_num_layers
        self.duration_predictor_flow_bins = duration_predictor_flow_bins
        self.duration_predictor_tail_bound = duration_predictor_tail_bound
        self.duration_predictor_kernel_size = duration_predictor_kernel_size
        self.duration_predictor_dropout = duration_predictor_dropout
        self.duration_predictor_num_flows = duration_predictor_num_flows
        self.duration_predictor_filter_channels = duration_predictor_filter_channels
        self.prior_encoder_num_flows = prior_encoder_num_flows
        self.prior_encoder_num_wavenet_layers = prior_encoder_num_wavenet_layers
        self.posterior_encoder_num_wavenet_layers = posterior_encoder_num_wavenet_layers
        self.wavenet_kernel_size = wavenet_kernel_size
        self.wavenet_dilation_rate = wavenet_dilation_rate
        self.wavenet_dropout = wavenet_dropout
        self.speaking_rate = speaking_rate
        self.noise_scale = noise_scale
        self.noise_scale_duration = noise_scale_duration
        self.sampling_rate = sampling_rate

        # 检查 `upsample_kernel_sizes` 和 `upsample_rates` 的长度是否一致，不一致则抛出 ValueError 异常
        if len(upsample_kernel_sizes) != len(upsample_rates):
            raise ValueError(
                f"The length of `upsample_kernel_sizes` ({len(upsample_kernel_sizes)}) must match the length of "
                f"`upsample_rates` ({len(upsample_rates)})"
            )

        # 调用父类的初始化方法，传入可能的关键字参数
        super().__init__(**kwargs)
```