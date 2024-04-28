# `.\transformers\models\big_bird\configuration_big_bird.py`

```
# coding=utf-8
# 版权声明和许可证信息，用于标识代码版权及使用许可
# Copyright 2021 Google Research and The HuggingFace Inc. team. All rights reserved.
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

# 导入必要的库和模块
""" BigBird model configuration"""
from collections import OrderedDict
from typing import Mapping

# 导入配置工具和日志记录
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件地址映射
BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/bigbird-roberta-base": "https://huggingface.co/google/bigbird-roberta-base/resolve/main/config.json",
    "google/bigbird-roberta-large": "https://huggingface.co/google/bigbird-roberta-large/resolve/main/config.json",
    "google/bigbird-base-trivia-itc": "https://huggingface.co/google/bigbird-base-trivia-itc/resolve/main/config.json",
    # See all BigBird models at https://huggingface.co/models?filter=big_bird
}

# BigBird 配置类，继承自预训练配置类
class BigBirdConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BigBirdModel`]. It is used to instantiate an
    BigBird model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the BigBird
    [google/bigbird-roberta-base](https://huggingface.co/google/bigbird-roberta-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import BigBirdConfig, BigBirdModel

    >>> # Initializing a BigBird google/bigbird-roberta-base style configuration
    >>> configuration = BigBirdConfig()

    >>> # Initializing a model (with random weights) from the google/bigbird-roberta-base style configuration
    >>> model = BigBirdModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    # 模型类型
    model_type = "big_bird"
    # 初始化函数，用于初始化Transformer模型的参数
    def __init__(
        self,
        vocab_size=50358,  # 词汇表大小，默认为50358
        hidden_size=768,   # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层的数量，默认为12
        num_attention_heads=12, # 注意力头的数量，默认为12
        intermediate_size=3072, # 中间层大小，默认为3072
        hidden_act="gelu_new",  # 激活函数，默认为gelu_new
        hidden_dropout_prob=0.1, # 隐藏层的dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1, # 注意力机制的dropout概率，默认为0.1
        max_position_embeddings=4096,     # 最大位置嵌入数，默认为4096
        type_vocab_size=2,                # 类型词汇表大小，默认为2
        initializer_range=0.02,           # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,             # 层归一化的epsilon值，默认为1e-12
        use_cache=True,                   # 是否使用缓存，默认为True
        pad_token_id=0,                   # 填充标记ID，默认为0
        bos_token_id=1,                   # 起始标记ID，默认为1
        eos_token_id=2,                   # 结束标记ID，默认为2
        sep_token_id=66,                  # 分隔标记ID，默认为66
        attention_type="block_sparse",    # 注意力类型，默认为block_sparse
        use_bias=True,                    # 是否使用偏置，默认为True
        rescale_embeddings=False,         # 是否重新缩放嵌入，默认为False
        block_size=64,                    # 块大小，默认为64
        num_random_blocks=3,              # 随机块的数量，默认为3
        classifier_dropout=None,          # 分类器的dropout，默认为None
        **kwargs,
    ):
        # 调用父类的初始化函数，设置填充、起始、结束和分隔标记ID
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )

        # 设置模型的词汇表大小
        self.vocab_size = vocab_size
        # 设置模型的最大位置嵌入数
        self.max_position_embeddings = max_position_embeddings
        # 设置模型的隐藏层大小
        self.hidden_size = hidden_size
        # 设置模型的隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置模型的注意力头数量
        self.num_attention_heads = num_attention_heads
        # 设置模型的中间层大小
        self.intermediate_size = intermediate_size
        # 设置模型的隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置模型的隐藏层dropout概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置模型的注意力机制的dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置模型的初始化范围
        self.initializer_range = initializer_range
        # 设置模型的类型词汇表大小
        self.type_vocab_size = type_vocab_size
        # 设置模型的层归一化epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 设置模型是否使用缓存
        self.use_cache = use_cache

        # 设置是否重新缩放嵌入
        self.rescale_embeddings = rescale_embeddings
        # 设置注意力类型
        self.attention_type = attention_type
        # 设置是否使用偏置
        self.use_bias = use_bias
        # 设置块大小
        self.block_size = block_size
        # 设置随机块的数量
        self.num_random_blocks = num_random_blocks
        # 设置分类器的dropout
        self.classifier_dropout = classifier_dropout
# 定义 BigBirdOnnxConfig 类，继承自 OnnxConfig 类
class BigBirdOnnxConfig(OnnxConfig):
    # 定义 inputs 属性，返回一个映射，将输入名称映射到动态轴的字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多项选择
        if self.task == "multiple-choice":
            # 动态轴为 {0: "batch", 1: "choice", 2: "sequence"}
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则动态轴为 {0: "batch", 1: "sequence"}
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回有序字典，将输入名称映射到动态轴的字典
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),        # 输入名称为 input_ids，动态轴为 dynamic_axis
                ("attention_mask", dynamic_axis),  # 输入名称为 attention_mask，动态轴为 dynamic_axis
            ]
        )
```