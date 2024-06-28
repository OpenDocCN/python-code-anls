# `.\models\mpnet\configuration_mpnet.py`

```
# coding=utf-8
# 声明文件编码格式为UTF-8

# Copyright 2018 The HuggingFace Inc. team, Microsoft Corporation.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# 版权声明，版权归HuggingFace Inc.团队、Microsoft Corporation和NVIDIA CORPORATION所有

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证 2.0 版本授权使用本文件

# you may not use this file except in compliance with the License.
# 除非遵守许可证，否则不得使用此文件

# You may obtain a copy of the License at
# 可以在以下网址获取许可证副本

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 除非适用法律要求或书面同意，本软件按"原样"分发，不附带任何明示或暗示的担保或条件
# 详见许可证中的特定语言及限制条件

""" MPNet model configuration"""

# 引入必要的模块和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取logger对象，用于记录日志信息
logger = logging.get_logger(__name__)

# 预训练配置文件映射，指定预训练模型的名称及其对应的配置文件URL
MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/mpnet-base": "https://huggingface.co/microsoft/mpnet-base/resolve/main/config.json",
}


class MPNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MPNetModel`] or a [`TFMPNetModel`]. It is used to
    instantiate a MPNet model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MPNet
    [microsoft/mpnet-base](https://huggingface.co/microsoft/mpnet-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # MPNetConfig类用于存储MPNet模型或TFMPNet模型的配置信息。
    # 通过指定的参数实例化一个MPNet模型，定义模型架构。使用默认配置将生成与MPNet microsoft/mpnet-base架构类似的配置。

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化函数，调用父类的初始化方法
    # 指定模型的配置参数，定义了MPNet模型的各种参数选项，默认值见文档描述
    Args:
        vocab_size (`int`, *optional*, defaults to 30527):
            MPNet模型的词汇表大小，决定了在调用`MPNetModel`或`TFMPNetModel`时，`inputs_ids`可以表示的不同token数量。
        hidden_size (`int`, *optional*, defaults to 768):
            编码器层和池化层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer编码器中的隐藏层数量。
        num_attention_heads (`int`, *optional*, defaults to 12):
            Transformer编码器中每个注意力层的注意力头数量。
        intermediate_size (`int`, *optional*, defaults to 3072):
            Transformer编码器中"中间"（通常称为前馈）层的维度。
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。支持的字符串有："gelu"、"relu"、"silu"和"gelu_new"。
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            嵌入层、编码器和池化器中所有全连接层的dropout概率。
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            注意力概率的dropout比例。
        max_position_embeddings (`int`, *optional*, defaults to 512):
            模型可能使用的最大序列长度。通常设置为一个较大的值（例如512、1024或2048），以防万一。
        initializer_range (`float`, *optional*, defaults to 0.02):
            初始化所有权重矩阵的截断正态初始化器的标准差。
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            层归一化层使用的epsilon值。
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            每个注意力层使用的桶的数量。
    
    Examples:
        
        # 导入MPNetModel和MPNetConfig
        >>> from transformers import MPNetModel, MPNetConfig
        
        # 初始化一个MPNet mpnet-base风格的配置
        >>> configuration = MPNetConfig()
        
        # 使用mpnet-base风格的配置初始化一个模型
        >>> model = MPNetModel(configuration)
        
        # 访问模型配置
        >>> configuration = model.config
    # 初始化函数，用于创建一个新的实例对象
    def __init__(
        self,
        vocab_size=30527,                     # 词汇表大小，默认为30527
        hidden_size=768,                      # 隐藏层大小，默认为768
        num_hidden_layers=12,                 # 隐藏层的数量，默认为12
        num_attention_heads=12,               # 注意力头的数量，默认为12
        intermediate_size=3072,               # 中间层大小，默认为3072
        hidden_act="gelu",                    # 隐藏层激活函数，默认为gelu
        hidden_dropout_prob=0.1,              # 隐藏层的dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,     # 注意力概率的dropout概率，默认为0.1
        max_position_embeddings=512,          # 最大位置嵌入数，默认为512
        initializer_range=0.02,               # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,                 # 层归一化的epsilon，默认为1e-12
        relative_attention_num_buckets=32,    # 相对注意力的桶数量，默认为32
        pad_token_id=1,                       # 填充标记ID，默认为1
        bos_token_id=0,                       # 起始标记ID，默认为0
        eos_token_id=2,                       # 结束标记ID，默认为2
        **kwargs,
    ):
        # 调用父类的初始化函数，传入填充、起始和结束标记ID，以及其他关键字参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 初始化对象的属性
        self.vocab_size = vocab_size                           # 设置词汇表大小属性
        self.hidden_size = hidden_size                         # 设置隐藏层大小属性
        self.num_hidden_layers = num_hidden_layers             # 设置隐藏层数量属性
        self.num_attention_heads = num_attention_heads         # 设置注意力头数量属性
        self.hidden_act = hidden_act                           # 设置隐藏层激活函数属性
        self.intermediate_size = intermediate_size             # 设置中间层大小属性
        self.hidden_dropout_prob = hidden_dropout_prob         # 设置隐藏层dropout概率属性
        self.attention_probs_dropout_prob = attention_probs_dropout_prob   # 设置注意力dropout概率属性
        self.max_position_embeddings = max_position_embeddings # 设置最大位置嵌入数属性
        self.initializer_range = initializer_range             # 设置初始化范围属性
        self.layer_norm_eps = layer_norm_eps                   # 设置层归一化的epsilon属性
        self.relative_attention_num_buckets = relative_attention_num_buckets   # 设置相对注意力的桶数量属性
```