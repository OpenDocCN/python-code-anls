# `.\models\gpt_bigcode\configuration_gpt_bigcode.py`

```
# coding=utf-8
# Copyright 2023 The BigCode team and HuggingFace Inc. team.
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
GPTBigCode configuration

This module contains the configuration for the GPTBigCode model, specifying how to instantiate and customize it.
"""

# Importing necessary modules from the parent directories
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# Setting up logging for the current module
logger = logging.get_logger(__name__)

# Mapping from model identifier to its corresponding configuration file URL
GPT_BIGCODE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bigcode/gpt_bigcode-santacoder": "https://huggingface.co/bigcode/gpt_bigcode-santacoder/resolve/main/config.json",
}

# Configuration class for GPTBigCode model, inherits from PretrainedConfig
class GPTBigCodeConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`GPTBigCodeModel`]. It is used to instantiate a
    GPTBigCode model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GPTBigCode
    [gpt_bigcode](https://huggingface.co/gpt_bigcode) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 定义 GPT-2 模型的配置类 GPTBigCodeConfig，包含了各种可选参数
    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            GPT-2 模型的词汇表大小，定义了可以表示的不同标记数量
        n_positions (`int`, *optional*, defaults to 1024):
            模型可能使用的最大序列长度。通常设置为一个较大的值，例如 512、1024 或 2048
        n_embd (`int`, *optional*, defaults to 768):
            嵌入和隐藏状态的维度
        n_layer (`int`, *optional*, defaults to 12):
            Transformer 编码器中的隐藏层数量
        n_head (`int`, *optional*, defaults to 12):
            Transformer 编码器中每个注意力层的注意头数量
        n_inner (`int`, *optional*, defaults to None):
            内部前馈层的维度。如果为 `None`，将设置为 4 倍的 n_embd
        activation_function (`str`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            激活函数，可在列表 `["relu", "silu", "gelu", "tanh", "gelu_new", "gelu_pytorch_tanh"]` 中选择
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            嵌入、编码器和池化器中所有全连接层的 dropout 概率
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            嵌入的 dropout 比率
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            注意力的 dropout 比率
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            层归一化层使用的 epsilon
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态分布的标准差
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            是否通过除以 sqrt(hidden_size) 来缩放注意力权重
        use_cache (`bool`, *optional*, defaults to `True`):
            模型是否应返回最后的键/值注意力（不是所有模型都使用）
        attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
            是否在 float32 中调用融合 softmax
        scale_attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
            是否在 float32 中缩放注意力 softmax
        attention_type (`bool`, *optional*, defaults to `True`):
            是否使用多查询注意力（True）或多头注意力（False）
    Example:

    ```python
    >>> from transformers import GPTBigCodeConfig, GPTBigCodeModel

    >>> # 初始化一个 GPTBigCodeConfig 配置对象
    >>> configuration = GPTBigCodeConfig()

    >>> # 根据配置初始化一个具有随机权重的模型
    >>> model = GPTBigCodeModel(configuration)
    ```
    # 访问模型配置信息
    configuration = model.config



    # 设置模型类型为"gpt_bigcode"
    model_type = "gpt_bigcode"



    # 在推断过程中忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]



    # 属性映射，将模型配置的名称映射到内部使用的名称
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }



    # 初始化函数，用于设置模型的各种参数和默认值
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_pytorch_tanh",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        attention_softmax_in_fp32=True,
        scale_attention_softmax_in_fp32=True,
        multi_query=True,
        **kwargs,
    ):



        # 初始化模型的各个参数
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = scale_attention_softmax_in_fp32
        self.multi_query = multi_query

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # 调用父类的初始化方法，设置起始和结束标记的 token ID
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
```