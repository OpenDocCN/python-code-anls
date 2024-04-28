# `.\transformers\models\nystromformer\configuration_nystromformer.py`

```
# coding=utf-8
# 版权 2022 UW-Madison 和 The HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版（“许可证”）授权;
# 除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 无论是明示还是暗示的，都不作任何保证或条件。
# 有关特定语言的权限，请参阅许可证。
""" Nystromformer 模型配置"""

# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置存档映射
NYSTROMFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "uw-madison/nystromformer-512": "https://huggingface.co/uw-madison/nystromformer-512/resolve/main/config.json",
    # 查看所有 Nystromformer 模型：https://huggingface.co/models?filter=nystromformer
}

# Nystromformer 配置类，继承自 PretrainedConfig
class NystromformerConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`NystromformerModel`] 的配置。它用于根据指定参数实例化
    一个 Nystromformer 模型，定义模型架构。使用默认值实例化配置将产生与 Nystromformer
    [uw-madison/nystromformer-512](https://huggingface.co/uw-madison/nystromformer-512) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。请阅读
    [`PretrainedConfig`] 的文档以获取更多信息。
    Args:
        # 词汇表大小，用于Nystromformer模型。定义了在调用NystromformerModel时可以表示的不同标记数量
        vocab_size (`int`, *optional*, defaults to 30000):
        # 编码器层和池化层的维度大小
        hidden_size (`int`, *optional*, defaults to 768):
        # Transformer编码器中隐藏层的数量
        num_hidden_layers (`int`, *optional*, defaults to 12):
        # Transformer编码器中每个注意力层的注意力头数量
        num_attention_heads (`int`, *optional*, defaults to 12):
        # Transformer编码器中"中间"（即，前馈）层的维度
        intermediate_size (`int`, *optional*, defaults to 3072):
        # 编码器和池化层中的非线性激活函数（函数或字符串）。如果是字符串，则支持"gelu"、"relu"、"selu"和"gelu_new"
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
        # 在所有全连接层中的dropout概率，包括嵌入层、编码器和池化层
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        # 注意概率的dropout比率
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        # 模型可能在其中使用的最大序列长度。通常将其设置为较大的值（例如512、1024或2048）
        max_position_embeddings (`int`, *optional*, defaults to 512):
        # 在调用NystromformerModel时传递的token_type_ids的词汇表大小
        type_vocab_size (`int`, *optional*, defaults to 2):
        # segment-means中使用的序列长度
        segment_means_seq_len (`int`, *optional*, defaults to 64):
        # 在Nystrom近似softmax自注意力矩阵中使用的地标点（或Nystrom点）的数量
        num_landmarks (`int`, *optional*, defaults to 64):
        # Nystrom近似中使用的深度卷积的卷积核大小
        conv_kernel_size (`int`, *optional*, defaults to 65):
        # 是否使用确切系数计算来计算矩阵的Moore-Penrose逆的初始值的迭代方法
        inv_coeff_init_option (`bool`, *optional*, defaults to `False`):
        # 初始化所有权重矩阵的截断正态初始化器的标准差
        initializer_range (`float`, *optional*, defaults to 0.02):
        # 层归一化层使用的epsilon
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):

    Example:

    ```python
    >>> from transformers import NystromformerModel, NystromformerConfig
    # 初始化一个 Nystromformer uw-madison/nystromformer-512 风格的配置
    configuration = NystromformerConfig()
    
    # 根据 uw-madison/nystromformer-512 风格的配置初始化一个模型
    model = NystromformerModel(configuration)
    
    # 获取模型的配置信息
    configuration = model.config
```