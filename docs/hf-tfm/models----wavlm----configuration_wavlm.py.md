# `.\models\wavlm\configuration_wavlm.py`

```py
# coding=utf-8
# Copyright 2021 The Fairseq Authors, Microsoft Research, and The HuggingFace Inc. team. All rights reserved.
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
WavLM model configuration

This module contains the configuration class `WavLMConfig` which defines the model architecture
and inherits from `PretrainedConfig`.
"""

import functools
import operator

# Import logger from utils for logging purposes
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# Get logger instance for this module
logger = logging.get_logger(__name__)

# Mapping of pretrained model names to their configuration file URLs
WAVLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/wavlm-base": "https://huggingface.co/microsoft/wavlm-base/resolve/main/config.json",
    # See all WavLM models at https://huggingface.co/models?filter=wavlm
}


class WavLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`WavLMModel`]. It is used to instantiate an WavLM
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the WavLM
    [microsoft/wavlm-base](https://huggingface.co/microsoft/wavlm-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```

    ```

    Example:

    ```
    >>> from transformers import WavLMConfig, WavLMModel

    >>> # Initializing a WavLM facebook/wavlm-base-960h style configuration
    >>> configuration = WavLMConfig()

    >>> # Initializing a model (with random weights) from the facebook/wavlm-base-960h style configuration
    >>> model = WavLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    
    # Specify the model type as "wavlm"
    model_type = "wavlm"
    # 初始化函数，用于创建一个新的对象实例，设置模型的各种参数
    def __init__(
        self,
        vocab_size=32,  # 词汇表大小，默认为32
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # Transformer模型中的隐藏层数，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        intermediate_size=3072,  # Transformer中间层的大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
        hidden_dropout=0.1,  # 隐藏层的Dropout率，默认为0.1
        activation_dropout=0.1,  # 激活函数的Dropout率，默认为0.1
        attention_dropout=0.1,  # 注意力机制的Dropout率，默认为0.1
        feat_proj_dropout=0.0,  # 特征投影层的Dropout率，默认为0.0
        final_dropout=0.1,  # 最终层的Dropout率，默认为0.1
        layerdrop=0.1,  # LayerDrop的概率，默认为0.1
        initializer_range=0.02,  # 参数初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # LayerNorm的epsilon值，默认为1e-5
        feat_extract_norm="group",  # 特征提取层的归一化方式，默认为"group"
        feat_extract_activation="gelu",  # 特征提取层的激活函数，默认为GELU
        conv_dim=(512, 512, 512, 512, 512, 512, 512),  # 卷积层的通道数，默认为(512, 512, 512, 512, 512, 512, 512)
        conv_stride=(5, 2, 2, 2, 2, 2, 2),  # 卷积层的步幅，默认为(5, 2, 2, 2, 2, 2, 2)
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),  # 卷积层的核大小，默认为(10, 3, 3, 3, 3, 2, 2)
        conv_bias=False,  # 卷积层是否使用偏置，默认为False
        num_conv_pos_embeddings=128,  # 卷积位置嵌入的数量，默认为128
        num_conv_pos_embedding_groups=16,  # 卷积位置嵌入的组数，默认为16
        num_buckets=320,  # 桶的数量，默认为320
        max_bucket_distance=800,  # 桶之间的最大距离，默认为800
        do_stable_layer_norm=False,  # 是否使用稳定的LayerNorm，默认为False
        apply_spec_augment=True,  # 是否应用音频增强，默认为True
        mask_time_prob=0.05,  # 时间掩码的概率，默认为0.05
        mask_time_length=10,  # 时间掩码的长度，默认为10
        mask_time_min_masks=2,  # 时间掩码的最小数量，默认为2
        mask_feature_prob=0.0,  # 特征掩码的概率，默认为0.0
        mask_feature_length=10,  # 特征掩码的长度，默认为10
        num_codevectors_per_group=320,  # 每组码向量的数量，默认为320
        num_codevector_groups=2,  # 码向量组的数量，默认为2
        contrastive_logits_temperature=0.1,  # 对比损失的温度参数，默认为0.1
        num_negatives=100,  # 负样本的数量，默认为100
        codevector_dim=256,  # 码向量的维度，默认为256
        proj_codevector_dim=256,  # 投影码向量的维度，默认为256
        diversity_loss_weight=0.1,  # 多样性损失的权重，默认为0.1
        ctc_loss_reduction="mean",  # CTC损失的减少方式，默认为"mean"
        ctc_zero_infinity=False,  # CTC损失中是否使用零无穷，默认为False
        use_weighted_layer_sum=False,  # 是否使用加权层求和，默认为False
        classifier_proj_size=256,  # 分类器投影层的大小，默认为256
        tdnn_dim=(512, 512, 512, 512, 1500),  # TDNN层的通道数，默认为(512, 512, 512, 512, 1500)
        tdnn_kernel=(5, 3, 3, 1, 1),  # TDNN层的核大小，默认为(5, 3, 3, 1, 1)
        tdnn_dilation=(1, 2, 3, 1, 1),  # TDNN层的扩张率，默认为(1, 2, 3, 1, 1)
        xvector_output_dim=512,  # x-vector的输出维度，默认为512
        num_ctc_classes=80,  # CTC输出的类别数，默认为80
        pad_token_id=0,  # 填充标记的ID，默认为0
        bos_token_id=1,  # 开始标记的ID，默认为1
        eos_token_id=2,  # 结束标记的ID，默认为2
        add_adapter=False,  # 是否添加适配器，默认为False
        adapter_kernel_size=3,  # 适配器的核大小，默认为3
        adapter_stride=2,  # 适配器的步幅，默认为2
        num_adapter_layers=3,  # 适配器的层数，默认为3
        output_hidden_size=None,  # 输出层的隐藏大小，默认为None
        **kwargs,  # 其他关键字参数
    ):
        # 属性：输入到logits比例
        @property
        def inputs_to_logits_ratio(self):
            # 计算输入到logits的比例，即卷积步幅的乘积
            return functools.reduce(operator.mul, self.conv_stride, 1)
```