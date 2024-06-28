# `.\models\qdqbert\configuration_qdqbert.py`

```
# coding=utf-8
# Copyright 2021 NVIDIA Corporation and The HuggingFace Team. All rights reserved.
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
QDQBERT model configuration
"""

# 导入预训练配置类 PretrainedConfig 和日志工具 logging
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取 logger 对象用于记录日志
logger = logging.get_logger(__name__)

# QDQBERT 预训练配置文件映射字典，将模型名称映射到其配置文件的 URL
QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google-bert/bert-base-uncased": "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/config.json",
    # QDQBERT 模型可以从任何 BERT 检查点加载，这些检查点可在 https://huggingface.co/models?filter=bert 找到
}

# QDQBERT 配置类，继承自 PretrainedConfig
class QDQBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`QDQBertModel`]. It is used to instantiate an
    QDQBERT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the BERT
    [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 定义模型类型为 "qdqbert"
    model_type = "qdqbert"
        # 初始化函数，用于创建一个新的实例
        def __init__(
            self,
            vocab_size=30522,                          # 词汇表大小，默认为30522
            hidden_size=768,                           # 隐藏层大小，默认为768
            num_hidden_layers=12,                      # 隐藏层的数量，默认为12
            num_attention_heads=12,                    # 注意力头的数量，默认为12
            intermediate_size=3072,                    # 中间层大小，默认为3072
            hidden_act="gelu",                         # 隐藏层激活函数，默认为gelu
            hidden_dropout_prob=0.1,                   # 隐藏层的Dropout概率，默认为0.1
            attention_probs_dropout_prob=0.1,          # 注意力概率的Dropout概率，默认为0.1
            max_position_embeddings=512,               # 最大位置嵌入数，默认为512
            type_vocab_size=2,                         # 类型词汇表大小，默认为2
            initializer_range=0.02,                    # 初始化范围，默认为0.02
            layer_norm_eps=1e-12,                      # Layer Norm的epsilon值，默认为1e-12
            use_cache=True,                            # 是否使用缓存，默认为True
            pad_token_id=1,                            # 填充token的ID，默认为1
            bos_token_id=0,                            # 开始token的ID，默认为0
            eos_token_id=2,                            # 结束token的ID，默认为2
            **kwargs,
        ):
            # 调用父类的初始化函数，设置特殊token的ID和额外的关键字参数
            super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

            # 初始化类的成员变量，设置模型的各种参数
            self.vocab_size = vocab_size                # 设置词汇表大小
            self.max_position_embeddings = max_position_embeddings  # 设置最大位置嵌入数
            self.hidden_size = hidden_size              # 设置隐藏层大小
            self.num_hidden_layers = num_hidden_layers  # 设置隐藏层的数量
            self.num_attention_heads = num_attention_heads  # 设置注意力头的数量
            self.intermediate_size = intermediate_size  # 设置中间层大小
            self.hidden_act = hidden_act                # 设置隐藏层激活函数
            self.hidden_dropout_prob = hidden_dropout_prob  # 设置隐藏层的Dropout概率
            self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 设置注意力概率的Dropout概率
            self.initializer_range = initializer_range  # 设置初始化范围
            self.type_vocab_size = type_vocab_size      # 设置类型词汇表大小
            self.layer_norm_eps = layer_norm_eps        # 设置Layer Norm的epsilon值
            self.use_cache = use_cache                  # 设置是否使用缓存
```