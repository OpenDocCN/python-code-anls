# `.\models\ctrl\configuration_ctrl.py`

```py
# coding=utf-8
# Copyright 2018 Salesforce and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
Salesforce CTRL configuration

This module defines the configuration class `CTRLConfig` for the CTRL model. It provides a mapping of pretrained model names
to their corresponding configuration files.

The `CTRLConfig` class inherits from `PretrainedConfig` and defines parameters that control the architecture and behavior
of the CTRL model. It provides defaults that align with the Salesforce/ctrl architecture.

For more details on how to use configuration objects like `CTRLConfig` to instantiate CTRL models, refer to the
documentation of `PretrainedConfig`.
"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging

logger = logging.get_logger(__name__)

# Mapping from pretrained model names to their configuration file URLs
CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Salesforce/ctrl": "https://huggingface.co/Salesforce/ctrl/resolve/main/config.json"
}

class CTRLConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a `CTRLModel` or a `TFCTRLModel`.
    It defines parameters that control the architecture and behavior of the model when instantiated.
    Instantiating a configuration with the defaults will yield a similar configuration to that of
    the Salesforce/ctrl architecture from SalesForce.

    Configuration objects inherit from `PretrainedConfig` and can be used to control the model outputs.
    For more detailed information about configuring CTRL models, refer to the documentation of `PretrainedConfig`.
    """
    pass
    # 定义模型类型为 "ctrl"
    model_type = "ctrl"

    # 在推断阶段忽略的键列表，这些键不会在推断时被使用
    keys_to_ignore_at_inference = ["past_key_values"]

    # 属性映射字典，将模型配置中的一些属性映射到自定义的名称
    attribute_map = {
        "max_position_embeddings": "n_positions",   # 最大位置嵌入长度映射到 n_positions
        "hidden_size": "n_embd",                    # 隐藏大小映射到 n_embd
        "num_attention_heads": "n_head",            # 注意力头的数量映射到 n_head
        "num_hidden_layers": "n_layer",             # 隐藏层的数量映射到 n_layer
    }

    # 类的构造函数，初始化模型的配置参数
    def __init__(
        self,
        vocab_size=246534,                          # 词汇表大小，默认为 246534
        n_positions=256,                            # 最大序列长度，默认为 256
        n_embd=1280,                                # 嵌入和隐藏状态的维度，默认为 1280
        dff=8192,                                   # 前馈网络内部维度，默认为 8192
        n_layer=48,                                 # Transformer 编码器中的隐藏层数，默认为 48
        n_head=16,                                  # Transformer 编码器中每个注意力层的注意力头数，默认为 16
        resid_pdrop=0.1,                            # 嵌入、编码器和池化器中所有全连接层的 dropout 概率，默认为 0.1
        embd_pdrop=0.1,                             # 嵌入层的 dropout 比率，默认为 0.1
        layer_norm_epsilon=1e-6,                    # 层归一化层中使用的 epsilon，默认为 1e-6
        initializer_range=0.02,                     # 初始化所有权重矩阵时使用的截断正态初始化器的标准差，默认为 0.02
        use_cache=True,                             # 模型是否应返回最后的键/值注意力，默认为 True
        **kwargs,                                   # 允许接收任意其他关键字参数
        ):
        # 初始化Transformer模型的参数：词汇表大小
        self.vocab_size = vocab_size
        # 初始化Transformer模型的参数：位置编码的最大长度
        self.n_positions = n_positions
        # 初始化Transformer模型的参数：词嵌入的维度
        self.n_embd = n_embd
        # 初始化Transformer模型的参数：层数
        self.n_layer = n_layer
        # 初始化Transformer模型的参数：注意力头的数量
        self.n_head = n_head
        # 初始化Transformer模型的参数：前馈神经网络内部隐藏层的维度
        self.dff = dff
        # 初始化Transformer模型的参数：残差连接的dropout概率
        self.resid_pdrop = resid_pdrop
        # 初始化Transformer模型的参数：词嵌入的dropout概率
        self.embd_pdrop = embd_pdrop
        # 初始化Transformer模型的参数：层归一化的epsilon值
        self.layer_norm_epsilon = layer_norm_epsilon
        # 初始化Transformer模型的参数：初始化权重范围
        self.initializer_range = initializer_range

        # 初始化Transformer模型的参数：是否使用缓存
        self.use_cache = use_cache

        # 调用父类初始化方法，传递任意关键字参数
        super().__init__(**kwargs)
```