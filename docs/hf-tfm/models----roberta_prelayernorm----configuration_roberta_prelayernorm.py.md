# `.\models\roberta_prelayernorm\configuration_roberta_prelayernorm.py`

```
# coding=utf-8
# Copyright 2022 The Google AI Language Team Authors and The HuggingFace Inc. team.
# All rights reserved.
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

""" RoBERTa-PreLayerNorm configuration"""

# 从 collections 模块导入 OrderedDict 类
from collections import OrderedDict
# 从 typing 模块导入 Mapping 类型
from typing import Mapping

# 从配置工具中导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 从 ONNX 配置中导入 OnnxConfig
from ...onnx import OnnxConfig
# 从工具模块中导入日志记录功能 logging
from ...utils import logging

# 获取当前模块的日志记录器 logger
logger = logging.get_logger(__name__)

# 定义 RoBERTa-PreLayerNorm 模型预训练配置文件的映射字典
ROBERTA_PRELAYERNORM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "andreasmadsen/efficient_mlm_m0.40": (
        "https://huggingface.co/andreasmadsen/efficient_mlm_m0.40/resolve/main/config.json"
    ),
}


# 定义 RoBERTa-PreLayerNormConfig 类，继承自 PretrainedConfig
# 这个类用于存储 RoBERTa-PreLayerNorm 模型的配置信息
class RobertaPreLayerNormConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RobertaPreLayerNormModel`] or a [`TFRobertaPreLayerNormModel`]. It is
    used to instantiate a RoBERTa-PreLayerNorm model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the RoBERTa-PreLayerNorm
    [andreasmadsen/efficient_mlm_m0.40](https://huggingface.co/andreasmadsen/efficient_mlm_m0.40) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Examples:

    ```python
    >>> from transformers import RobertaPreLayerNormConfig, RobertaPreLayerNormModel

    >>> # Initializing a RoBERTa-PreLayerNorm configuration
    >>> configuration = RobertaPreLayerNormConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = RobertaPreLayerNormModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    
    # 模型类型标识为 "roberta-prelayernorm"
    model_type = "roberta-prelayernorm"
    # 初始化方法，用于创建一个新的实例对象
    def __init__(
        self,
        vocab_size=50265,  # 词汇表大小，默认为50265
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层的数量，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
        hidden_dropout_prob=0.1,  # 隐藏层的dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力机制的dropout概率，默认为0.1
        max_position_embeddings=512,  # 最大位置嵌入数，默认为512
        type_vocab_size=2,  # 类型词汇表的大小，默认为2
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 层归一化的epsilon值，默认为1e-12
        pad_token_id=1,  # 填充标记的ID，默认为1
        bos_token_id=0,  # 开始标记的ID，默认为0
        eos_token_id=2,  # 结束标记的ID，默认为2
        position_embedding_type="absolute",  # 位置嵌入类型，默认为绝对位置嵌入
        use_cache=True,  # 是否使用缓存，默认为True
        classifier_dropout=None,  # 分类器的dropout，默认为None
        **kwargs,
    ):
        # 调用父类的初始化方法，传递填充、开始和结束标记的ID，以及其他参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 将传入的参数赋值给对象的属性
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
# 从 transformers.models.roberta.configuration_roberta.RobertaOnnxConfig 复制代码，并将名称中的 Roberta 修改为 RobertaPreLayerNorm
class RobertaPreLayerNormOnnxConfig(OnnxConfig):
    
    # 定义 inputs 属性，返回一个映射，表示模型输入的动态轴
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多项选择 ("multiple-choice")，设置动态轴为 {0: "batch", 1: "choice", 2: "sequence"}
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则设置动态轴为 {0: "batch", 1: "sequence"}
            dynamic_axis = {0: "batch", 1: "sequence"}
        
        # 返回一个有序字典，包含模型输入的名称和对应的动态轴
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),       # 模型输入的 token IDs，使用动态轴
                ("attention_mask", dynamic_axis),  # 模型输入的注意力遮罩，使用动态轴
            ]
        )
```