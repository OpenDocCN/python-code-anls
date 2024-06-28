# `.\models\tvlt\configuration_tvlt.py`

```py
# coding=utf-8
# Copyright 2023 MURGe-Lab and The HuggingFace Inc. team. All rights reserved.
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
""" TVLT model configuration"""

# 导入必要的类和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# TVLT 预训练模型配置文件映射字典，指定了模型名称及其配置文件的 URL
TVLT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ZinengTang/tvlt-base": "https://huggingface.co/ZinengTang/tvlt-base/blob/main/config.json",
}

# TvltConfig 类，用于存储 TVLT 模型的配置信息，继承自 PretrainedConfig 类
class TvltConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TvltModel`]. It is used to instantiate a TVLT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the TVLT
    [ZinengTang/tvlt-base](https://huggingface.co/ZinengTang/tvlt-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```
    >>> from transformers import TvltConfig, TvltModel

    >>> # # Initializing a TVLT ZinengTang/tvlt-base style configuration
    >>> configuration = TvltConfig()

    >>> # # Initializing a model (with random weights) from the ZinengTang/tvlt-base style configuration
    >>> model = TvltModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型定义为 "tvlt"
    model_type = "tvlt"

    # 初始化函数，用于设定 TVLT 模型的各种配置参数
    def __init__(
        self,
        image_size=224,
        spectrogram_length=2048,
        frequency_length=128,
        image_patch_size=[16, 16],
        audio_patch_size=[16, 16],
        num_image_channels=3,
        num_audio_channels=1,
        num_frames=8,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        qkv_bias=True,
        use_mean_pooling=False,
        decoder_num_attention_heads=16,
        decoder_hidden_size=512,
        decoder_num_hidden_layers=8,
        decoder_intermediate_size=2048,
        pixel_mask_ratio=0.75,
        audio_mask_ratio=0.15,
        audio_mask_type="frame-level",
        task_matching=True,
        task_mae=True,
        loss_type="classification",
        **kwargs,
    ):
        """
        Initializes TvltConfig with various parameters to define the TVLT model architecture and behavior.
        """
        # 调用父类 PretrainedConfig 的初始化函数，设定通用的模型配置参数
        super().__init__(**kwargs)
        # 设置模型的特定参数，用于控制模型结构和行为
        self.image_size = image_size
        self.spectrogram_length = spectrogram_length
        self.frequency_length = frequency_length
        self.image_patch_size = image_patch_size
        self.audio_patch_size = audio_patch_size
        self.num_image_channels = num_image_channels
        self.num_audio_channels = num_audio_channels
        self.num_frames = num_frames
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.use_mean_pooling = use_mean_pooling
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.pixel_mask_ratio = pixel_mask_ratio
        self.audio_mask_ratio = audio_mask_ratio
        self.audio_mask_type = audio_mask_type
        self.task_matching = task_matching
        self.task_mae = task_mae
        self.loss_type = loss_type
        # 将额外的参数存储在 kwargs 中
        self.update(kwargs)
        ):
            super().__init__(**kwargs)
            
            # 调用父类初始化函数，并传递所有关键字参数
            if audio_mask_type not in ("frame-level", "patch_level"):
                # 检查音频遮罩类型是否为有效的值之一，如果不是则引发值错误异常
                raise ValueError(
                    "audio_mask_type must be one of two acceptable strategies - {'frame_level', 'patch-level') "
                    f"got {audio_mask_type}"
                )
    
            # 设置图像大小、频谱长度、频率长度、图像补丁大小、音频补丁大小、图像通道数、音频通道数、帧数
            self.image_size = image_size
            self.spectrogram_length = spectrogram_length
            self.frequency_length = frequency_length
            self.image_patch_size = image_patch_size
            self.audio_patch_size = audio_patch_size
            self.num_image_channels = num_image_channels
            self.num_audio_channels = num_audio_channels
            self.num_frames = num_frames
    
            # 设置隐藏层大小、隐藏层数量、注意力头数量、中间层大小、隐藏层激活函数、隐藏层丢弃率、注意力机制概率丢弃率、初始化范围、层归一化的 epsilon 值、QKV 偏置、是否使用均值池化
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.intermediate_size = intermediate_size
            self.hidden_act = hidden_act
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.qkv_bias = qkv_bias
            self.use_mean_pooling = use_mean_pooling
    
            # 设置解码器注意力头数量、解码器隐藏层大小、解码器隐藏层数量、解码器中间层大小、像素遮罩比例、音频遮罩比例、音频遮罩类型
            self.decoder_num_attention_heads = decoder_num_attention_heads
            self.decoder_hidden_size = decoder_hidden_size
            self.decoder_num_hidden_layers = decoder_num_hidden_layers
            self.decoder_intermediate_size = decoder_intermediate_size
            self.pixel_mask_ratio = pixel_mask_ratio
            self.audio_mask_ratio = audio_mask_ratio
            self.audio_mask_type = audio_mask_type
    
            # 设置任务匹配、任务均方误差、损失类型
            self.task_matching = task_matching
            self.task_mae = task_mae
            self.loss_type = loss_type
```