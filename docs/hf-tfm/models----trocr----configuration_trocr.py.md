# `.\models\trocr\configuration_trocr.py`

```py
# coding=utf-8
# 上面的行指定了源文件的编码格式为UTF-8，确保可以正确处理各种字符

# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
# 以下几行是版权声明，指明此代码的版权归HuggingFace Inc.团队所有

# Licensed under the Apache License, Version 2.0 (the "License");
# 使用Apache License, Version 2.0授权许可，允许在符合许可条件下使用本代码

# you may not use this file except in compliance with the License.
# 除非符合许可条件，否则不得使用此文件

# You may obtain a copy of the License at
# 您可以在上述链接获取许可证的副本

# http://www.apache.org/licenses/LICENSE-2.0
# 许可证详情请访问该网址

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# distributed的代码基于"AS IS"基础分发，即无任何形式的明示或暗示保证

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有任何形式的明示或暗示保证，包括但不限于适销性或特定用途适用性的保证

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查看许可证了解权限的具体限制

""" TrOCR model configuration"""
# 下面是对TrOCR模型配置的简短描述

from ...configuration_utils import PretrainedConfig
from ...utils import logging
# 导入所需的模块

logger = logging.get_logger(__name__)
# 获取日志记录器实例

TROCR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/trocr-base-handwritten": (
        "https://huggingface.co/microsoft/trocr-base-handwritten/resolve/main/config.json"
    ),
    # 用于存储TrOCR预训练模型的配置映射，指定了模型名称和其配置文件的URL
    # 可在https://huggingface.co/models?filter=trocr查看所有TrOCR模型
}


class TrOCRConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TrOCRForCausalLM`]. It is used to instantiate an
    TrOCR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the TrOCR
    [microsoft/trocr-base-handwritten](https://huggingface.co/microsoft/trocr-base-handwritten) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # TrOCRConfig类用于存储TrOCR模型的配置信息，继承自PretrainedConfig类
    # 用于根据指定的参数实例化TrOCR模型，定义模型的架构
    # 使用默认参数实例化配置将生成与TrOCR microsoft/trocr-base-handwritten架构相似的配置
    # TrOCR 模型的配置类，定义了模型的各种参数和选项
    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            TrOCR 模型的词汇表大小，定义了在调用 `TrOCRForCausalLM` 时可以表示的不同标记数量。
        d_model (`int`, *optional*, defaults to 1024):
            层和池化层的维度。
        decoder_layers (`int`, *optional*, defaults to 12):
            解码器层数。
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Transformer 解码器中每个注意力层的注意力头数。
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            解码器中“中间”（通常称为前馈）层的维度。
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            池化器中的非线性激活函数（函数或字符串）。支持的字符串包括 "gelu"、"relu"、"silu" 和 "gelu_new"。
        max_position_embeddings (`int`, *optional*, defaults to 512):
            模型可能使用的最大序列长度。通常设置为一个很大的值（例如 512、1024 或 2048）。
        dropout (`float`, *optional*, defaults to 0.1):
            嵌入层和池化器中所有全连接层的dropout概率。
        attention_dropout (`float`, *optional*, defaults to 0.0):
            注意力概率的dropout比率。
        activation_dropout (`float`, *optional*, defaults to 0.0):
            全连接层内部激活的dropout比率。
        init_std (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准偏差。
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            解码器的层丢弃概率。详见 LayerDrop 论文(https://arxiv.org/abs/1909.11556) 了解更多细节。
        use_cache (`bool`, *optional*, defaults to `True`):
            模型是否应返回最后的键/值注意力（并非所有模型都使用）。
        scale_embedding (`bool`, *optional*, defaults to `False`):
            是否对词嵌入进行 sqrt(d_model) 的缩放。
        use_learned_position_embeddings (`bool`, *optional*, defaults to `True`):
            是否使用学习到的位置嵌入。如果不使用，则使用正弦位置嵌入。
        layernorm_embedding (`bool`, *optional*, defaults to `True`):
            是否在词 + 位置嵌入后使用 layernorm。
    
    Example:

    ```
    >>> from transformers import TrOCRConfig, TrOCRForCausalLM

    >>> # Initializing a TrOCR-base style configuration
    >>> configuration = TrOCRConfig()
    ```
    # 初始化一个 TrOCRForCausalLM 模型实例，使用指定的配置 (configuration)。模型权重是随机的。
    model = TrOCRForCausalLM(configuration)
    
    # 访问模型的配置信息
    configuration = model.config
```