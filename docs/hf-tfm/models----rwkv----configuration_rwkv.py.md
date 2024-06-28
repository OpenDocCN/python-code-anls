# `.\models\rwkv\configuration_rwkv.py`

```py
# coding=utf-8
# Copyright 2023 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" RWKV configuration"""

# 导入配置基类 PretrainedConfig 和日志工具 logging
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义 RWKV 预训练模型的配置文件映射字典
RWKV_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "RWKV/rwkv-4-169m-pile": "https://huggingface.co/RWKV/rwkv-4-169m-pile/resolve/main/config.json",
    "RWKV/rwkv-4-430m-pile": "https://huggingface.co/RWKV/rwkv-4-430m-pile/resolve/main/config.json",
    "RWKV/rwkv-4-1b5-pile": "https://huggingface.co/RWKV/rwkv-4-1b5-pile/resolve/main/config.json",
    "RWKV/rwkv-4-3b-pile": "https://huggingface.co/RWKV/rwkv-4-3b-pile/resolve/main/config.json",
    "RWKV/rwkv-4-7b-pile": "https://huggingface.co/RWKV/rwkv-4-7b-pile/resolve/main/config.json",
    "RWKV/rwkv-4-14b-pile": "https://huggingface.co/RWKV/rwkv-4-14b-pile/resolve/main/config.json",
    "RWKV/rwkv-raven-1b5": "https://huggingface.co/RWKV/rwkv-raven-1b5/resolve/main/config.json",
    "RWKV/rwkv-raven-3b": "https://huggingface.co/RWKV/rwkv-raven-3b/resolve/main/config.json",
    "RWKV/rwkv-raven-7b": "https://huggingface.co/RWKV/rwkv-raven-7b/resolve/main/config.json",
    "RWKV/rwkv-raven-14b": "https://huggingface.co/RWKV/rwkv-raven-14b/resolve/main/config.json",
}

# RWKV 配置类，用于存储 RWKV 模型的配置信息
class RwkvConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`RwkvModel`]. It is used to instantiate a RWKV
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the RWVK-4
    [RWKV/rwkv-4-169m-pile](https://huggingface.co/RWKV/rwkv-4-169m-pile) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 定义 RWKV 模型类型
    model_type = "rwkv"
    # 映射模型属性，将 "max_position_embeddings" 映射到类中的 "context_length"
    attribute_map = {"max_position_embeddings": "context_length"}
    
    # RWKV 模型的配置类，包含了模型的各种参数设置
    def __init__(
        self,
        vocab_size=50277,  # 词汇表大小，默认为 50277
        context_length=1024,  # 模型可以处理的最大序列长度，默认为 1024
        hidden_size=4096,  # 嵌入层和隐藏状态的维度
        num_hidden_layers=32,  # 模型中的隐藏层数量，默认为 32
        attention_hidden_size=None,  # 注意力机制隐藏状态的维度，默认为 hidden_size
        intermediate_size=None,  # 内部前馈层的维度，默认为 hidden_size 的四倍
        layer_norm_epsilon=1e-5,  # 层归一化层使用的 epsilon 值，默认为 1e-5
        bos_token_id=0,  # 词汇表中句子开头 token 的 id，默认为 0
        eos_token_id=0,  # 词汇表中句子结尾 token 的 id，默认为 0
        rescale_every=6,  # 推断时，每隔多少层将隐藏状态和对应输出层的权重除以 2，默认为 6
        tie_word_embeddings=False,  # 是否将词嵌入与输入 token 的嵌入进行绑定，默认为 False
        use_cache=True,  # 模型是否应返回最后状态，默认为 True
        **kwargs,  # 允许接受任意其他参数
    ):
        ):
        # 初始化模型的参数：词汇表大小、上下文长度、隐藏层大小、隐藏层数量、注意力隐藏大小
        # 如果注意力隐藏大小未指定，则使用隐藏层大小作为默认值
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attention_hidden_size = attention_hidden_size if attention_hidden_size is not None else hidden_size
        # 如果中间层大小未指定，则使用隐藏层大小的四倍作为默认值
        self.intermediate_size = intermediate_size if intermediate_size is not None else 4 * hidden_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.rescale_every = rescale_every
        self.use_cache = use_cache

        # 设置模型的特殊令牌（起始和结束令牌）的标识符
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # 调用父类的初始化方法，传递一些参数，如是否共享词嵌入、起始和结束令牌的标识符等
        super().__init__(
            tie_word_embeddings=tie_word_embeddings, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs
        )
```