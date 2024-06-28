# `.\models\splinter\configuration_splinter.py`

```
# coding=utf-8
# Copyright 2021 Tel AViv University, AllenAI and The HuggingFace Inc. team. All rights reserved.
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
Splinter model configuration
"""

# 导入预训练配置类 PretrainedConfig 和日志工具 logging
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取 logger 对象，用于记录日志
logger = logging.get_logger(__name__)

# 定义预训练配置文件映射字典，将模型名称映射到其配置文件的 URL
SPLINTER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "tau/splinter-base": "https://huggingface.co/tau/splinter-base/resolve/main/config.json",
    "tau/splinter-base-qass": "https://huggingface.co/tau/splinter-base-qass/resolve/main/config.json",
    "tau/splinter-large": "https://huggingface.co/tau/splinter-large/resolve/main/config.json",
    "tau/splinter-large-qass": "https://huggingface.co/tau/splinter-large-qass/resolve/main/config.json",
    # 查看所有 Splinter 模型：https://huggingface.co/models?filter=splinter
}

# 定义 SplinterConfig 类，继承自 PretrainedConfig 类
class SplinterConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SplinterModel`]. It is used to instantiate an
    Splinter model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Splinter
    [tau/splinter-base](https://huggingface.co/tau/splinter-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    # 构造函数注释省略，用于实例化 SplinterConfig 对象时使用
    # 设置模型类型为 "splinter"
    model_type = "splinter"
    # 初始化函数，用于初始化一个 Transformer 模型的配置参数
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为30522
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # Transformer 模型的隐藏层数，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        intermediate_size=3072,  # Feedforward 层的中间维度大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为 GELU
        hidden_dropout_prob=0.1,  # 隐藏层的dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力概率的dropout概率，默认为0.1
        max_position_embeddings=512,  # 最大位置嵌入大小，默认为512
        type_vocab_size=2,  # 类型词汇表的大小，默认为2
        initializer_range=0.02,  # 参数初始化的范围，默认为0.02
        layer_norm_eps=1e-12,  # 层归一化的epsilon值，默认为1e-12
        use_cache=True,  # 是否使用缓存，默认为True
        pad_token_id=0,  # 填充token的id，默认为0
        question_token_id=104,  # 问题token的id，默认为104
        **kwargs,
    ):
        # 调用父类的初始化函数，传入填充token的id和其他关键字参数
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # 设置类的属性，将参数值赋给对应的属性
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.question_token_id = question_token_id
```