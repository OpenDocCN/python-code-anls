# `.\models\mra\configuration_mra.py`

```
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# 上面是版权声明和编码声明

# 导入必要的模块和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 定义一个映射，将预训练模型名称映射到其配置文件的URL
MRA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "uw-madison/mra-base-512-4": "https://huggingface.co/uw-madison/mra-base-512-4/resolve/main/config.json",
}

# MraConfig类继承自PretrainedConfig类，用于存储MRA模型的配置信息
class MraConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MraModel`]. It is used to instantiate an MRA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Mra
    [uw-madison/mra-base-512-4](https://huggingface.co/uw-madison/mra-base-512-4) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 定义 Mra 模型的配置类，包含 Transformer 编码器的各种参数设置
    
    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Mra 模型的词汇表大小，定义了在调用 [`MraModel`] 时输入 `inputs_ids` 可以表示的不同标记数量。
        hidden_size (`int`, *optional*, defaults to 768):
            编码器层和池化层的维度大小。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer 编码器中隐藏层的数量。
        num_attention_heads (`int`, *optional*, defaults to 12):
            Transformer 编码器中每个注意力层的注意头数量。
        intermediate_size (`int`, *optional*, defaults to 3072):
            Transformer 编码器中“中间”（即前馈）层的维度大小。
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。如果是字符串，支持 `"gelu"`, `"relu"`, `"selu"` 和 `"gelu_new"`。
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            嵌入层、编码器和池化器中所有全连接层的 dropout 概率。
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            注意力概率的 dropout 比率。
        max_position_embeddings (`int`, *optional*, defaults to 512):
            该模型可能使用的最大序列长度。通常设置一个大值（例如 512、1024 或 2048）以防万一。
        type_vocab_size (`int`, *optional*, defaults to 1):
            在调用 [`MraModel`] 时传递的 `token_type_ids` 的词汇表大小。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            层归一化层使用的 epsilon。
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            位置嵌入的类型。选择 `"absolute"`, `"relative_key"`, `"relative_key_query"` 之一。
        block_per_row (`int`, *optional*, defaults to 4):
            用于设置高分辨率比例的预算。
        approx_mode (`str`, *optional*, defaults to `"full"`):
            控制是否使用低分辨率和高分辨率的逼近。设置为 `"full"` 表示同时使用低分辨率和高分辨率，设置为 `"sparse"` 表示仅使用低分辨率。
        initial_prior_first_n_blocks (`int`, *optional*, defaults to 0):
            最初使用高分辨率的块数。
        initial_prior_diagonal_n_blocks (`int`, *optional*, defaults to 0):
            使用高分辨率的对角块数。
    
    Example:
    >>> from transformers import MraConfig, MraModel
    
    # 初始化一个 MRA 模型的配置，使用默认参数
    >>> configuration = MraConfig()
    
    # 根据给定的配置初始化一个 MRA 模型（权重随机初始化）
    >>> model = MraModel(configuration)
    
    # 获取模型的配置信息
    >>> configuration = model.config
```