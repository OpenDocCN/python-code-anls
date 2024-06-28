# `.\models\cpmant\configuration_cpmant.py`

```py
# coding=utf-8
# Copyright 2022 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
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
""" CPMAnt model configuration"""

# 从configuration_utils模块导入PretrainedConfig类
from ...configuration_utils import PretrainedConfig
# 从utils模块导入logging函数
from ...utils import logging

# 获取logger对象，用于日志记录
logger = logging.get_logger(__name__)

# CPMAnt预训练配置文件的映射字典，指定模型名称和其对应的配置文件URL
CPMANT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openbmb/cpm-ant-10b": "https://huggingface.co/openbmb/cpm-ant-10b/blob/main/config.json"
    # 查看所有CPMAnt模型：https://huggingface.co/models?filter=cpmant
}


class CpmAntConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CpmAntModel`]. It is used to instantiate an
    CPMAnt model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CPMAnt
    [openbmb/cpm-ant-10b](https://huggingface.co/openbmb/cpm-ant-10b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 模型类型设定为 "cpmant"
    model_type = "cpmant"
    # 初始化函数，用于初始化一个 Transformer 模型的参数和配置
    def __init__(
        self,
        vocab_size: int = 30720,  # 词汇表大小，默认为 30720
        hidden_size: int = 4096,  # 隐藏层的尺寸，默认为 4096
        num_attention_heads: int = 32,  # 注意力头的数量，默认为 32
        dim_head: int = 128,  # 注意力头的维度，默认为 128
        dim_ff: int = 10240,  # FeedForward 层的尺寸，默认为 10240
        num_hidden_layers: int = 48,  # Transformer 层的数量，默认为 48
        dropout_p: int = 0.0,  # Dropout 概率，默认为 0.0，即无 dropout
        position_bias_num_buckets: int = 512,  # 位置偏置的哈希桶数量，默认为 512
        position_bias_max_distance: int = 2048,  # 位置偏置的最大距离，默认为 2048
        eps: int = 1e-6,  # 避免除零的小数，默认为 1e-6
        init_std: float = 1.0,  # 参数初始化的标准差，默认为 1.0
        prompt_types: int = 32,  # 提示类型的数量，默认为 32
        prompt_length: int = 32,  # 提示长度，默认为 32
        segment_types: int = 32,  # 段落类型的数量，默认为 32
        use_cache: bool = True,  # 是否使用缓存，默认为 True
        **kwargs,  # 其他额外的参数，以字典形式接收
    ):
        # 调用父类的初始化方法，传递额外的关键字参数
        super().__init__(**kwargs)
        # 初始化特定于 Prompt 的参数
        self.prompt_types = prompt_types
        self.prompt_length = prompt_length
        self.segment_types = segment_types
        # 初始化通用的 Transformer 参数
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_hidden_layers = num_hidden_layers
        self.position_bias_num_buckets = position_bias_num_buckets
        self.position_bias_max_distance = position_bias_max_distance
        self.dropout_p = dropout_p
        self.eps = eps
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.init_std = init_std
```