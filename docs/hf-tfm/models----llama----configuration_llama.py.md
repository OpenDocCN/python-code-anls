# `.\models\llama\configuration_llama.py`

```py
# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
""" LLaMA model configuration"""

# Importing necessary classes from transformers library
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# Getting the logger instance for logging messages related to this module
logger = logging.get_logger(__name__)

# Mapping dictionary to store pretrained configurations for LLaMA models
LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

# Configuration class inheriting from PretrainedConfig to define LLaMA model configuration
class LlamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    ```
    >>> from transformers import LlamaModel, LlamaConfig

    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # Setting model_type attribute for LLaMA model identification
    model_type = "llama"
    
    # List of keys to ignore during inference
    keys_to_ignore_at_inference = ["past_key_values"]

    # Constructor method to initialize LLaMA configuration parameters
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
        ):
        # 设置模型的参数：词汇表大小
        self.vocab_size = vocab_size
        # 设置模型的参数：最大位置编码长度
        self.max_position_embeddings = max_position_embeddings
        # 设置模型的参数：隐藏层大小
        self.hidden_size = hidden_size
        # 设置模型的参数：中间层大小
        self.intermediate_size = intermediate_size
        # 设置模型的参数：隐藏层的数量
        self.num_hidden_layers = num_hidden_layers
        # 设置模型的参数：注意力头的数量
        self.num_attention_heads = num_attention_heads

        # 兼容性处理：如果未指定键值头的数量，则默认与注意力头数量相同
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        # 设置模型的参数：键值头的数量
        self.num_key_value_heads = num_key_value_heads
        # 设置模型的参数：隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置模型的参数：初始化范围
        self.initializer_range = initializer_range
        # 设置模型的参数：RMS归一化的epsilon值
        self.rms_norm_eps = rms_norm_eps
        # 设置模型的参数：预训练类型
        self.pretraining_tp = pretraining_tp
        # 设置模型的参数：是否使用缓存
        self.use_cache = use_cache
        # 设置模型的参数：Rope模型的theta值
        self.rope_theta = rope_theta
        # 设置模型的参数：Rope模型的缩放参数
        self.rope_scaling = rope_scaling
        # 调用私有方法验证Rope模型的缩放参数是否合法
        self._rope_scaling_validation()
        # 设置模型的参数：注意力偏置
        self.attention_bias = attention_bias
        # 设置模型的参数：注意力dropout率
        self.attention_dropout = attention_dropout

        # 调用父类的初始化方法，设置模型的其他参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        # 如果没有设置Rope模型的缩放参数，则直接返回
        if self.rope_scaling is None:
            return

        # 检查Rope模型的缩放参数是否为字典且包含两个字段
        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        # 获取Rope模型的缩放类型和缩放因子
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        # 检查Rope模型的缩放类型是否合法
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        # 检查Rope模型的缩放因子是否为浮点数且大于1
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
```