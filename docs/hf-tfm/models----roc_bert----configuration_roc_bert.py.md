# `.\models\roc_bert\configuration_roc_bert.py`

```py
# coding=utf-8
# Copyright 2022 WeChatAI and The HuggingFace Inc. team. All rights reserved.
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
""" RoCBert model configuration"""

# 从 transformers 库中导入预训练配置的基类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 从 transformers 库中导入日志记录工具 logging
from ...utils import logging

# 获取 logger 实例用于日志记录
logger = logging.get_logger(__name__)

# RoCBert 预训练模型配置的映射字典，指定了模型名称到配置文件的 URL 映射
ROC_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "weiweishi/roc-bert-base-zh": "https://huggingface.co/weiweishi/roc-bert-base-zh/resolve/main/config.json",
}

# RoCBertConfig 类，继承自 PretrainedConfig 类，用于存储 RoCBert 模型的配置信息
class RoCBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RoCBertModel`]. It is used to instantiate a
    RoCBert model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the RoCBert
    [weiweishi/roc-bert-base-zh](https://huggingface.co/weiweishi/roc-bert-base-zh) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    ```
    >>> from transformers import RoCBertModel, RoCBertConfig

    >>> # Initializing a RoCBert weiweishi/roc-bert-base-zh style configuration
    >>> configuration = RoCBertConfig()

    >>> # Initializing a model from the weiweishi/roc-bert-base-zh style configuration
    >>> model = RoCBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 指定模型类型为 "roc_bert"
    model_type = "roc_bert"

    # RoCBertConfig 类的初始化函数，定义了模型的各项配置参数
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_cache=True,
        pad_token_id=0,
        position_embedding_type="absolute",
        classifier_dropout=None,
        enable_pronunciation=True,
        enable_shape=True,
        pronunciation_embed_dim=768,
        pronunciation_vocab_size=910,
        shape_embed_dim=512,
        shape_vocab_size=24858,
        concat_input=True,
        **kwargs,
        ):
        # 初始化模型参数
        self.vocab_size = vocab_size
        # 最大位置编码长度
        self.max_position_embeddings = max_position_embeddings
        # 隐藏层大小
        self.hidden_size = hidden_size
        # 隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 注意力头数量
        self.num_attention_heads = num_attention_heads
        # 中间层大小
        self.intermediate_size = intermediate_size
        # 隐藏层激活函数类型
        self.hidden_act = hidden_act
        # 隐藏层Dropout概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 注意力层Dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 初始化范围
        self.initializer_range = initializer_range
        # 类型词汇表大小
        self.type_vocab_size = type_vocab_size
        # 层标准化的epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 是否使用缓存
        self.use_cache = use_cache
        # 是否启用发音特征
        self.enable_pronunciation = enable_pronunciation
        # 是否启用形状特征
        self.enable_shape = enable_shape
        # 发音嵌入维度
        self.pronunciation_embed_dim = pronunciation_embed_dim
        # 发音词汇表大小
        self.pronunciation_vocab_size = pronunciation_vocab_size
        # 形状嵌入维度
        self.shape_embed_dim = shape_embed_dim
        # 形状词汇表大小
        self.shape_vocab_size = shape_vocab_size
        # 是否将输入串联起来
        self.concat_input = concat_input
        # 位置编码类型
        self.position_embedding_type = position_embedding_type
        # 分类器Dropout概率
        self.classifier_dropout = classifier_dropout
        # 调用父类的初始化方法，设置填充标记ID和其他关键字参数
        super().__init__(pad_token_id=pad_token_id, **kwargs)
```