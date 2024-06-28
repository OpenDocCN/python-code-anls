# `.\models\deprecated\retribert\configuration_retribert.py`

```py
# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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
""" RetriBERT model configuration"""

# 导入预训练配置类 PretrainedConfig 和日志记录工具 logging
from ....configuration_utils import PretrainedConfig
from ....utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件的映射字典，键为模型名称，值为配置文件的下载链接
RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "yjernite/retribert-base-uncased": (
        "https://huggingface.co/yjernite/retribert-base-uncased/resolve/main/config.json"
    ),
}

# RetriBertConfig 类，继承自 PretrainedConfig，用于存储 RetriBertModel 的配置信息
class RetriBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RetriBertModel`]. It is used to instantiate a
    RetriBertModel model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the RetriBERT
    [yjernite/retribert-base-uncased](https://huggingface.co/yjernite/retribert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 定义模型类型为 "retribert"
    model_type = "retribert"
    
    # 定义 RetriBertModel 类的构造函数，初始化模型的各种参数
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为 30522
        hidden_size=768,  # 编码器层和池化层的维度
        num_hidden_layers=8,  # Transformer 编码器中隐藏层的数量
        num_attention_heads=12,  # 每个注意力层中的注意力头数
        intermediate_size=3072,  # Transformer 编码器中“中间”（通常称为前馈）层的维度
        hidden_act="gelu",  # 编码器和池化器中的非线性激活函数
        hidden_dropout_prob=0.1,  # 嵌入层、编码器和池化层中所有全连接层的 dropout 概率
        attention_probs_dropout_prob=0.1,  # 注意力概率的 dropout 比率
        max_position_embeddings=512,  # 模型可能使用的最大序列长度
        type_vocab_size=2,  # 传递给 BertModel 的 token_type_ids 的词汇表大小
        initializer_range=0.02,  # 用于初始化所有权重矩阵的截断正态分布的标准差
        layer_norm_eps=1e-12,  # 层归一化层使用的 epsilon
        share_encoders=True,  # 是否使用相同的 Bert 类型编码器来处理查询和文档
        projection_dim=128,  # 投影后的查询和文档表示的最终维度
        pad_token_id=0,  # 用于填充的 token ID
        **kwargs,  # 其他未命名参数
    ):
        ):
            # 调用父类的初始化方法，设置填充标记的 ID 和其他可选参数
            super().__init__(pad_token_id=pad_token_id, **kwargs)

            # 设置词汇表大小
            self.vocab_size = vocab_size
            # 设置隐藏层大小
            self.hidden_size = hidden_size
            # 设置隐藏层的数量
            self.num_hidden_layers = num_hidden_layers
            # 设置注意力头的数量
            self.num_attention_heads = num_attention_heads
            # 设置隐藏层激活函数类型
            self.hidden_act = hidden_act
            # 设置中间层大小
            self.intermediate_size = intermediate_size
            # 设置隐藏层的 dropout 概率
            self.hidden_dropout_prob = hidden_dropout_prob
            # 设置注意力机制的 dropout 概率
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            # 设置最大位置编码的长度
            self.max_position_embeddings = max_position_embeddings
            # 设置类型词汇表的大小
            self.type_vocab_size = type_vocab_size
            # 设置初始化范围
            self.initializer_range = initializer_range
            # 设置层归一化的 epsilon 值
            self.layer_norm_eps = layer_norm_eps
            # 是否共享编码器的标志
            self.share_encoders = share_encoders
            # 设置投影维度
            self.projection_dim = projection_dim
```