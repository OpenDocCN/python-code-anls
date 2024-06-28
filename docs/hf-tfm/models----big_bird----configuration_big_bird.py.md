# `.\models\big_bird\configuration_big_bird.py`

```
# coding=utf-8
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

""" BigBird model configuration"""

# 导入所需模块
from collections import OrderedDict
from typing import Mapping

# 从相对路径导入必要的配置和工具类
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件的映射，映射了模型名称到配置文件的 URL
BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/bigbird-roberta-base": "https://huggingface.co/google/bigbird-roberta-base/resolve/main/config.json",
    "google/bigbird-roberta-large": "https://huggingface.co/google/bigbird-roberta-large/resolve/main/config.json",
    "google/bigbird-base-trivia-itc": "https://huggingface.co/google/bigbird-base-trivia-itc/resolve/main/config.json",
    # 查看所有 BigBird 模型的列表：https://huggingface.co/models?filter=big_bird
}

# 定义 BigBirdConfig 类，继承自 PretrainedConfig
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
    ```
    """

    # 定义模型类型为 "big_bird"
    model_type = "big_bird"
    # 初始化函数，用于初始化一个 Transformer 模型对象
    def __init__(
        self,
        vocab_size=50358,  # 设置词汇表大小，默认为50358
        hidden_size=768,  # 设置隐藏层大小，默认为768
        num_hidden_layers=12,  # 设置隐藏层数，默认为12
        num_attention_heads=12,  # 设置注意力头数，默认为12
        intermediate_size=3072,  # 设置中间层大小，默认为3072
        hidden_act="gelu_new",  # 设置隐藏层激活函数，默认为"gelu_new"
        hidden_dropout_prob=0.1,  # 设置隐藏层的dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 设置注意力概率dropout概率，默认为0.1
        max_position_embeddings=4096,  # 设置最大位置嵌入数，默认为4096
        type_vocab_size=2,  # 设置类型词汇表大小，默认为2
        initializer_range=0.02,  # 设置初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 设置层归一化epsilon，默认为1e-12
        use_cache=True,  # 是否使用缓存，默认为True
        pad_token_id=0,  # 设置填充标记的ID，默认为0
        bos_token_id=1,  # 设置开始标记的ID，默认为1
        eos_token_id=2,  # 设置结束标记的ID，默认为2
        sep_token_id=66,  # 设置分隔标记的ID，默认为66
        attention_type="block_sparse",  # 设置注意力类型，默认为"block_sparse"
        use_bias=True,  # 是否使用偏置，默认为True
        rescale_embeddings=False,  # 是否重新缩放嵌入，默认为False
        block_size=64,  # 设置块大小，默认为64
        num_random_blocks=3,  # 设置随机块数，默认为3
        classifier_dropout=None,  # 分类器的dropout率，默认为None
        **kwargs,  # 其他关键字参数
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            **kwargs,  # 调用父类的初始化函数，并传递相应的参数
        )

        self.vocab_size = vocab_size  # 初始化模型的词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 初始化最大位置嵌入数
        self.hidden_size = hidden_size  # 初始化隐藏层大小
        self.num_hidden_layers = num_hidden_layers  # 初始化隐藏层数
        self.num_attention_heads = num_attention_heads  # 初始化注意力头数
        self.intermediate_size = intermediate_size  # 初始化中间层大小
        self.hidden_act = hidden_act  # 初始化隐藏层激活函数
        self.hidden_dropout_prob = hidden_dropout_prob  # 初始化隐藏层的dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 初始化注意力概率dropout概率
        self.initializer_range = initializer_range  # 初始化初始化范围
        self.type_vocab_size = type_vocab_size  # 初始化类型词汇表大小
        self.layer_norm_eps = layer_norm_eps  # 初始化层归一化epsilon
        self.use_cache = use_cache  # 初始化是否使用缓存

        self.rescale_embeddings = rescale_embeddings  # 初始化是否重新缩放嵌入
        self.attention_type = attention_type  # 初始化注意力类型
        self.use_bias = use_bias  # 初始化是否使用偏置
        self.block_size = block_size  # 初始化块大小
        self.num_random_blocks = num_random_blocks  # 初始化随机块数
        self.classifier_dropout = classifier_dropout  # 初始化分类器的dropout率
# 定义一个 BigBirdOnnxConfig 类，继承自 OnnxConfig 类
class BigBirdOnnxConfig(OnnxConfig):
    
    # 定义 inputs 属性，返回一个映射结构，其键为字符串，值为映射到字符串的字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务类型是多选，则动态轴包含 batch、choice 和 sequence
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则动态轴只包含 batch 和 sequence
            dynamic_axis = {0: "batch", 1: "sequence"}
        
        # 返回一个有序字典，包含两个键值对
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),      # 键为 "input_ids"，值为 dynamic_axis
                ("attention_mask", dynamic_axis), # 键为 "attention_mask"，值为 dynamic_axis
            ]
        )
```