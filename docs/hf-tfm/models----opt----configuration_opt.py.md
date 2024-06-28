# `.\models\opt\configuration_opt.py`

```py
# coding=utf-8
# Copyright 2022 The Metaseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" OPT model configuration"""

# 导入预训练配置类 PretrainedConfig 和日志工具 logging
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取 logger 对象，用于记录日志信息
logger = logging.get_logger(__name__)

# 预训练配置映射字典，将模型名称映射到其配置文件的 URL
OPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/opt-125m": "https://huggingface.co/facebook/opt-125m/blob/main/config.json",
    "facebook/opt-350m": "https://huggingface.co/facebook/opt-350m/blob/main/config.json",
    "facebook/opt-1.3b": "https://huggingface.co/facebook/opt-1.3b/blob/main/config.json",
    "facebook/opt-2.7b": "https://huggingface.co/facebook/opt-2.7b/blob/main/config.json",
    "facebook/opt-6.7b": "https://huggingface.co/facebook/opt-6.7b/blob/main/config.json",
    "facebook/opt-13b": "https://huggingface.co/facebook/opt-13b/blob/main/config.json",
}

# OPTConfig 类，继承自 PretrainedConfig，用于存储 OPTModel 的配置信息
class OPTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OPTModel`]. It is used to instantiate a OPT model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the OPT
    [facebook/opt-350m](https://huggingface.co/facebook/opt-350m) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 定义 OPT 模型的配置类 OPTConfig，用于设置模型的各种参数和选项
    Args:
        vocab_size (`int`, *optional*, defaults to 50272):
            OPT 模型的词汇表大小，定义了在调用 OPTModel 时 `inputs_ids` 可以表示的不同标记数量
        hidden_size (`int`, *optional*, defaults to 768):
            层和池化层的维度
        num_hidden_layers (`int`, *optional*, defaults to 12):
            解码器层的数量
        ffn_dim (`int`, *optional*, defaults to 3072):
            解码器中“中间”（通常称为前馈）层的维度
        num_attention_heads (`int`, *optional*, defaults to 12):
            Transformer 解码器中每个注意力层的注意力头数量
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            编码器和池化器中的非线性激活函数（函数或字符串），支持的字符串有 `"gelu"`, `"relu"`, `"silu"` 和 `"gelu_new"`
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            模型可能使用的最大序列长度，通常设置为较大的值（例如 512、1024 或 2048）
        do_layer_norm_before (`bool`, *optional*, defaults to `True`):
            是否在注意力块之前执行层归一化
        word_embed_proj_dim (`int`, *optional*):
            可以设置为缩小词嵌入的维度，例如 `opt-350m`。默认为 `hidden_size`
        dropout (`float`, *optional*, defaults to 0.1):
            所有嵌入层、编码器和池化器中完全连接层的 dropout 概率
        attention_dropout (`float`, *optional*, defaults to 0.0):
            注意力概率的 dropout 比率
        layerdrop (`float`, *optional*, defaults to 0.0):
            LayerDrop 概率。参见 LayerDrop 论文以获取更多细节（https://arxiv.org/abs/1909.11556）
        init_std (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差
        use_cache (`bool`, *optional*, defaults to `True`):
            模型是否应返回最后一组键/值注意力（并非所有模型都使用）
        enable_bias (`bool`, *optional*, defaults to `True`):
            注意力块中线性层是否应使用偏置项
        layer_norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            层归一化是否应具有可学习参数

    Example:

    ```
    >>> from transformers import OPTConfig, OPTModel

    >>> # Initializing a OPT facebook/opt-large style configuration
    >>> configuration = OPTConfig()
    ```
    # 使用 Facebook 的 OPTModel 类初始化一个模型实例，使用给定的配置初始化模型参数（包括随机权重）
    model = OPTModel(configuration)

    # 访问模型的配置信息，获取配置对象
    configuration = model.config
```