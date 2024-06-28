# `.\models\cohere\configuration_cohere.py`

```
# coding=utf-8
# Copyright 2024 Cohere team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" Cohere model configuration"""

# 导入预训练配置类 PretrainedConfig 和日志记录工具 logging
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 初始化预训练配置文件映射字典
COHERE_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

# CohereConfig 类，继承自 PretrainedConfig 类，用于存储 CohereModel 的配置信息
class CohereConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CohereModel`]. It is used to instantiate an Cohere
    model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the [CohereForAI/c4ai-command-r-v01](https://huggingface.co/CohereForAI/c4ai-command-r-v01) model.


    ```python
    >>> from transformers import CohereModel, CohereConfig

    >>> # Initializing a Cohere model configuration
    >>> configuration = CohereConfig()

    >>> # Initializing a model from the Cohere configuration
    >>> model = CohereModel(configuration) # doctest: +SKIP

    >>> # Accessing the model configuration
    >>> configuration = model.config # doctest: +SKIP
    ```
    """

    # 模型类型
    model_type = "cohere"
    # 推理过程中需要忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]

    # 初始化方法，定义了各种模型超参数和配置项
    def __init__(
        self,
        vocab_size=256000,
        hidden_size=8192,
        intermediate_size=22528,
        logit_scale=0.0625,
        num_hidden_layers=40,
        num_attention_heads=64,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=5,
        eos_token_id=255001,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        # 调用父类 PretrainedConfig 的初始化方法
        super().__init__(**kwargs)
        # 设置模型的各种超参数和配置项
        # 词汇表大小
        self.vocab_size = vocab_size
        # 隐藏层大小
        self.hidden_size = hidden_size
        # 中间层大小
        self.intermediate_size = intermediate_size
        # logit 缩放比例
        self.logit_scale = logit_scale
        # 隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 注意力头数量
        self.num_attention_heads = num_attention_heads
        # key-value 头数量
        self.num_key_value_heads = num_key_value_heads
        # 隐藏层激活函数
        self.hidden_act = hidden_act
        # 最大位置嵌入
        self.max_position_embeddings = max_position_embeddings
        # 初始化范围
        self.initializer_range = initializer_range
        # 层归一化 epsilon
        self.layer_norm_eps = layer_norm_eps
        # 是否使用缓存
        self.use_cache = use_cache
        # 填充 token 的 id
        self.pad_token_id = pad_token_id
        # 开始 token 的 id
        self.bos_token_id = bos_token_id
        # 结束 token 的 id
        self.eos_token_id = eos_token_id
        # 是否绑定词嵌入
        self.tie_word_embeddings = tie_word_embeddings
        # rope_theta 参数
        self.rope_theta = rope_theta
        # 注意力偏置
        self.attention_bias = attention_bias
        # 注意力 dropout
        self.attention_dropout = attention_dropout
        ):
            # 初始化模型的各种参数
            self.vocab_size = vocab_size  # 词汇表大小
            self.max_position_embeddings = max_position_embeddings  # 最大位置编码数
            self.hidden_size = hidden_size  # 隐藏层大小
            self.logit_scale = logit_scale  # 对数缩放比例
            self.intermediate_size = intermediate_size  # 中间层大小
            self.num_hidden_layers = num_hidden_layers  # 隐藏层数
            self.num_attention_heads = num_attention_heads  # 注意力头数

            # 为了向后兼容性
            if num_key_value_heads is None:
                num_key_value_heads = num_attention_heads

            self.num_key_value_heads = num_key_value_heads  # 键值头数
            self.hidden_act = hidden_act  # 隐藏层激活函数
            self.initializer_range = initializer_range  # 初始化范围
            self.layer_norm_eps = layer_norm_eps  # 层归一化的 epsilon 参数
            self.use_cache = use_cache  # 是否使用缓存
            self.rope_theta = rope_theta  # 绳索 theta 参数
            self.attention_bias = attention_bias  # 注意力偏置
            self.attention_dropout = attention_dropout  # 注意力丢弃率

            # 调用父类的初始化方法，设置特殊的令牌 ID 和参数关联词嵌入
            super().__init__(
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                tie_word_embeddings=tie_word_embeddings,
                **kwargs,
            )
```