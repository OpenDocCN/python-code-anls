# `.\models\yoso\configuration_yoso.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" YOSO model configuration"""

# 导入所需模块和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取 logger 对象，用于记录日志信息
logger = logging.get_logger(__name__)

# YOSO 预训练模型配置文件的映射字典，将模型名称映射到配置文件的 URL
YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "uw-madison/yoso-4096": "https://huggingface.co/uw-madison/yoso-4096/resolve/main/config.json",
    # 查看所有 YOSO 模型的列表：https://huggingface.co/models?filter=yoso
}

# YosoConfig 类，继承自 PretrainedConfig 类，用于存储 YOSO 模型的配置信息
class YosoConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`YosoModel`]. It is used to instantiate an YOSO
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the YOSO
    [uw-madison/yoso-4096](https://huggingface.co/uw-madison/yoso-4096) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import YosoConfig, YosoModel

    >>> # Initializing a YOSO uw-madison/yoso-4096 style configuration
    >>> configuration = YosoConfig()

    >>> # Initializing a model (with random weights) from the uw-madison/yoso-4096 style configuration
    >>> model = YosoModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型为 "yoso"
    model_type = "yoso"

    # 初始化方法，定义了 YOSO 模型的各种配置参数
    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=4096,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        position_embedding_type="absolute",
        use_expectation=True,
        hash_code_len=9,
        num_hash=64,
        conv_window=None,
        use_fast_hash=True,
        lsh_backward=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        # 调用父类的初始化方法，设置默认参数
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            **kwargs,
        )

        # 特有的配置参数，用于定义 YOSO 模型的一些特性
        self.position_embedding_type = position_embedding_type
        self.use_expectation = use_expectation
        self.hash_code_len = hash_code_len
        self.num_hash = num_hash
        self.conv_window = conv_window
        self.use_fast_hash = use_fast_hash
        self.lsh_backward = lsh_backward
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        ):
        # 调用父类的初始化方法，设置模型的特殊标记 ID 和其他关键参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置模型的词汇表大小
        self.vocab_size = vocab_size
        # 设置模型能处理的最大位置嵌入长度
        self.max_position_embeddings = max_position_embeddings
        # 设置模型隐藏层的尺寸
        self.hidden_size = hidden_size
        # 设置模型的隐藏层层数
        self.num_hidden_layers = num_hidden_layers
        # 设置模型注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置模型中间层的大小
        self.intermediate_size = intermediate_size
        # 设置模型隐藏层的激活函数类型
        self.hidden_act = hidden_act
        # 设置模型隐藏层的 dropout 概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置模型注意力头的 dropout 概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置模型参数初始化的范围
        self.initializer_range = initializer_range
        # 设置模型的类型词汇表大小
        self.type_vocab_size = type_vocab_size
        # 设置模型层归一化的 epsilon 值
        self.layer_norm_eps = layer_norm_eps
        # 设置位置嵌入的类型
        self.position_embedding_type = position_embedding_type
        # 设置是否使用期望值
        self.use_expectation = use_expectation
        # 设置哈希编码的长度
        self.hash_code_len = hash_code_len
        # 设置哈希函数的数量
        self.num_hash = num_hash
        # 设置卷积窗口的大小
        self.conv_window = conv_window
        # 设置是否使用快速哈希
        self.use_fast_hash = use_fast_hash
        # 设置是否进行 LSH 反向传播
        self.lsh_backward = lsh_backward
```