# `.\models\mobilebert\configuration_mobilebert.py`

```
# coding=utf-8
# 指定文件编码为 UTF-8

# Copyright 2020 The HuggingFace Team. All rights reserved.
# 版权声明，保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 进行许可，允许在特定条件下使用、复制、修改和分发本软件
# you may not use this file except in compliance with the License.
# 除非符合许可协议，否则不能使用此文件

# You may obtain a copy of the License at
# 您可以在以下网址获取许可协议的副本
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则依据 "原样" 分发本软件，不提供任何明示或暗示的保证或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 查阅许可协议，了解权限和限制

""" MobileBERT model configuration"""
# MobileBERT 模型配置

from collections import OrderedDict
# 导入 OrderedDict 类，用于有序字典的支持
from typing import Mapping
# 导入 Mapping 类型提示，用于支持映射类型的提示

from ...configuration_utils import PretrainedConfig
# 从配置工具中导入预训练配置类 PretrainedConfig
from ...onnx import OnnxConfig
# 从 onnx 模块导入 OnnxConfig
from ...utils import logging
# 从 utils 中导入 logging 模块

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/mobilebert-uncased": "https://huggingface.co/google/mobilebert-uncased/resolve/main/config.json"
}
# 预训练模型配置存档映射，提供模型名称到预训练配置文件的 URL 映射

class MobileBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MobileBertModel`] or a [`TFMobileBertModel`]. It
    is used to instantiate a MobileBERT model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the MobileBERT
    [google/mobilebert-uncased](https://huggingface.co/google/mobilebert-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Examples:

    ```python
    >>> from transformers import MobileBertConfig, MobileBertModel

    >>> # Initializing a MobileBERT configuration
    >>> configuration = MobileBertConfig()

    >>> # Initializing a model (with random weights) from the configuration above
    >>> model = MobileBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    Attributes: pretrained_config_archive_map (Dict[str, str]): A dictionary containing all the available pre-trained
    checkpoints.
    """
    # MobileBERT 配置类，用于存储 [`MobileBertModel`] 或 [`TFMobileBertModel`] 的配置。
    # 根据指定参数实例化 MobileBERT 模型，定义模型架构。
    # 使用默认配置实例化将产生与 MobileBERT [google/mobilebert-uncased](https://huggingface.co/google/mobilebert-uncased) 架构类似的配置。

    # 配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。详细信息请阅读 [`PretrainedConfig`] 的文档。

    pretrained_config_archive_map = MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
    # 预训练模型配置存档映射，存储所有可用的预训练检查点

    model_type = "mobilebert"
    # 模型类型设定为 "mobilebert"
    # 初始化函数，用于初始化一个多头注意力模型的参数
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为30522
        hidden_size=512,  # 隐藏层大小，默认为512
        num_hidden_layers=24,  # 隐藏层的数量，默认为24层
        num_attention_heads=4,  # 注意力头的数量，默认为4个
        intermediate_size=512,  # 中间层大小，默认为512
        hidden_act="relu",  # 隐藏层激活函数，默认为ReLU
        hidden_dropout_prob=0.0,  # 隐藏层的dropout概率，默认为0.0（不使用）
        attention_probs_dropout_prob=0.1,  # 注意力机制的dropout概率，默认为0.1
        max_position_embeddings=512,  # 最大位置嵌入大小，默认为512
        type_vocab_size=2,  # 类型词汇表大小，默认为2
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 层归一化的epsilon值，默认为1e-12
        pad_token_id=0,  # 填充token的ID，默认为0
        embedding_size=128,  # 嵌入大小，默认为128
        trigram_input=True,  # 是否使用trigram输入，默认为True
        use_bottleneck=True,  # 是否使用瓶颈结构，默认为True
        intra_bottleneck_size=128,  # 瓶颈内部大小，默认为128
        use_bottleneck_attention=False,  # 是否使用瓶颈的注意力，默认为False
        key_query_shared_bottleneck=True,  # 键和查询是否共享瓶颈，默认为True
        num_feedforward_networks=4,  # 前馈网络的数量，默认为4
        normalization_type="no_norm",  # 归一化类型，默认为"no_norm"
        classifier_activation=True,  # 分类器是否激活，默认为True
        classifier_dropout=None,  # 分类器的dropout概率，默认为None
        **kwargs,
    ):
        # 调用父类的初始化方法，传递填充token的ID和其他关键字参数
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # 初始化模型的各种参数
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
        self.embedding_size = embedding_size
        self.trigram_input = trigram_input
        self.use_bottleneck = use_bottleneck
        self.intra_bottleneck_size = intra_bottleneck_size
        self.use_bottleneck_attention = use_bottleneck_attention
        self.key_query_shared_bottleneck = key_query_shared_bottleneck
        self.num_feedforward_networks = num_feedforward_networks
        self.normalization_type = normalization_type
        self.classifier_activation = classifier_activation

        # 根据是否使用瓶颈结构来确定真实的隐藏层大小
        if self.use_bottleneck:
            self.true_hidden_size = intra_bottleneck_size
        else:
            self.true_hidden_size = hidden_size

        # 分类器的dropout概率
        self.classifier_dropout = classifier_dropout
# 从 transformers.models.bert.configuration_bert.BertOnnxConfig 复制的代码，创建了 MobileBertOnnxConfig 类，用于配置 MobileBert 模型的 ONNX 格式设置。
class MobileBertOnnxConfig(OnnxConfig):
    # 定义 inputs 属性，返回一个映射，其中键为字符串，值为映射，映射的键为整数，值为字符串。
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多选项问题 ("multiple-choice")，则定义动态轴为 {0: "batch", 1: "choice", 2: "sequence"}。
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        # 否则，定义动态轴为 {0: "batch", 1: "sequence"}。
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回一个有序字典，包含输入名称到动态轴的映射。
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),  # 输入名称 "input_ids" 映射到 dynamic_axis 中的值。
                ("attention_mask", dynamic_axis),  # 输入名称 "attention_mask" 映射到 dynamic_axis 中的值。
                ("token_type_ids", dynamic_axis),  # 输入名称 "token_type_ids" 映射到 dynamic_axis 中的值。
            ]
        )
```