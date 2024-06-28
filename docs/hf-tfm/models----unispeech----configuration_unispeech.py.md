# `.\models\unispeech\configuration_unispeech.py`

```
# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
UniSpeech model configuration
"""

import functools  # 导入 functools 模块，用于高阶函数操作
import operator   # 导入 operator 模块，用于函数式编程中的操作符函数

from ...configuration_utils import PretrainedConfig  # 导入预训练配置基类
from ...utils import logging  # 导入日志记录模块

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例

UNISPEECH_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/unispeech-large-1500h-cv": (
        "https://huggingface.co/microsoft/unispeech-large-1500h-cv/resolve/main/config.json"
    ),
    # 查看所有 UniSpeech 模型，请访问 https://huggingface.co/models?filter=unispeech
}


class UniSpeechConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UniSpeechModel`]. It is used to instantiate an
    UniSpeech model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the UniSpeech
    [microsoft/unispeech-large-1500h-cv](https://huggingface.co/microsoft/unispeech-large-1500h-cv) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import UniSpeechConfig, UniSpeechModel

    >>> # Initializing a UniSpeech facebook/unispeech-base-960h style configuration
    >>> configuration = UniSpeechConfig()

    >>> # Initializing a model (with random weights) from the facebook/unispeech-base-960h style configuration
    >>> model = UniSpeechModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    
    model_type = "unispeech"  # 定义模型类型为 unispeech
    # 初始化函数，用于创建一个新的实例
    def __init__(
        self,
        vocab_size=32,  # 词汇表大小，默认为32
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # Transformer模型中隐藏层的数量，默认为12
        num_attention_heads=12,  # Transformer模型中注意力头的数量，默认为12
        intermediate_size=3072,  # Transformer模型中间层的大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
        hidden_dropout=0.1,  # 隐藏层dropout比例，默认为0.1
        activation_dropout=0.1,  # 激活函数dropout比例，默认为0.1
        attention_dropout=0.1,  # 注意力机制dropout比例，默认为0.1
        feat_proj_dropout=0.0,  # 特征投影层dropout比例，默认为0.0
        feat_quantizer_dropout=0.0,  # 特征量化器dropout比例，默认为0.0
        final_dropout=0.1,  # 最终输出层dropout比例，默认为0.1
        layerdrop=0.1,  # LayerDrop比例，默认为0.1
        initializer_range=0.02,  # 初始化权重的范围，默认为0.02
        layer_norm_eps=1e-5,  # Layer normalization的epsilon值，默认为1e-5
        feat_extract_norm="group",  # 特征提取层的归一化类型，默认为"group"
        feat_extract_activation="gelu",  # 特征提取层的激活函数，默认为GELU
        conv_dim=(512, 512, 512, 512, 512, 512, 512),  # 卷积层的维度，默认为(512, 512, 512, 512, 512, 512, 512)
        conv_stride=(5, 2, 2, 2, 2, 2, 2),  # 卷积层的步长，默认为(5, 2, 2, 2, 2, 2, 2)
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),  # 卷积层的核大小，默认为(10, 3, 3, 3, 3, 2, 2)
        conv_bias=False,  # 是否使用卷积层的偏置，默认为False
        num_conv_pos_embeddings=128,  # 卷积位置嵌入的数量，默认为128
        num_conv_pos_embedding_groups=16,  # 卷积位置嵌入的分组数量，默认为16
        do_stable_layer_norm=False,  # 是否进行稳定的层归一化，默认为False
        apply_spec_augment=True,  # 是否应用语音数据增强，默认为True
        mask_time_prob=0.05,  # 时间掩码的概率，默认为0.05
        mask_time_length=10,  # 时间掩码的长度，默认为10
        mask_time_min_masks=2,  # 时间掩码的最小掩码数，默认为2
        mask_feature_prob=0.0,  # 特征掩码的概率，默认为0.0
        mask_feature_length=10,  # 特征掩码的长度，默认为10
        mask_feature_min_masks=0,  # 特征掩码的最小掩码数，默认为0
        num_codevectors_per_group=320,  # 每组的编码向量数量，默认为320
        num_codevector_groups=2,  # 编码向量的分组数量，默认为2
        contrastive_logits_temperature=0.1,  # 对比损失的温度参数，默认为0.1
        num_negatives=100,  # 对比损失中的负样本数量，默认为100
        codevector_dim=256,  # 编码向量的维度，默认为256
        proj_codevector_dim=256,  # 投影编码向量的维度，默认为256
        diversity_loss_weight=0.1,  # 多样性损失的权重，默认为0.1
        ctc_loss_reduction="mean",  # CTC损失的减少方式，默认为"mean"
        ctc_zero_infinity=False,  # CTC损失中是否使用零和无穷，默认为False
        use_weighted_layer_sum=False,  # 是否使用加权层求和，默认为False
        classifier_proj_size=256,  # 分类器投影层的大小，默认为256
        num_ctc_classes=80,  # CTC损失中的类别数量，默认为80
        pad_token_id=0,  # 填充标记的ID，默认为0
        bos_token_id=1,  # 起始标记的ID，默认为1
        eos_token_id=2,  # 终止标记的ID，默认为2
        replace_prob=0.5,  # 替换概率，默认为0.5
        **kwargs,  # 其他关键字参数
    ):
        # 返回输入到logits比例的属性值，计算conv_stride列表中所有元素的乘积
        @property
        def inputs_to_logits_ratio(self):
            return functools.reduce(operator.mul, self.conv_stride, 1)
```