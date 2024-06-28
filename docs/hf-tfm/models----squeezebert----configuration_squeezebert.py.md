# `.\models\squeezebert\configuration_squeezebert.py`

```py
# coding=utf-8
# Copyright 2020 The SqueezeBert authors and The HuggingFace Inc. team.
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

""" SqueezeBERT model configuration"""

from collections import OrderedDict  # 导入有序字典模块
from typing import Mapping  # 导入 Mapping 类型

from ...configuration_utils import PretrainedConfig  # 导入预训练配置工具
from ...onnx import OnnxConfig  # 导入 ONNX 配置
from ...utils import logging  # 导入日志工具

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

# 定义 SqueezeBERT 预训练模型配置文件映射
SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "squeezebert/squeezebert-uncased": (
        "https://huggingface.co/squeezebert/squeezebert-uncased/resolve/main/config.json"
    ),
    "squeezebert/squeezebert-mnli": "https://huggingface.co/squeezebert/squeezebert-mnli/resolve/main/config.json",
    "squeezebert/squeezebert-mnli-headless": (
        "https://huggingface.co/squeezebert/squeezebert-mnli-headless/resolve/main/config.json"
    ),
}

# 定义 SqueezeBertConfig 类，继承自 PretrainedConfig 类
class SqueezeBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SqueezeBertModel`]. It is used to instantiate a
    SqueezeBERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SqueezeBERT
    [squeezebert/squeezebert-uncased](https://huggingface.co/squeezebert/squeezebert-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Examples:

    ```
    >>> from transformers import SqueezeBertConfig, SqueezeBertModel

    >>> # Initializing a SqueezeBERT configuration
    >>> configuration = SqueezeBertConfig()

    >>> # Initializing a model (with random weights) from the configuration above
    >>> model = SqueezeBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    Attributes: pretrained_config_archive_map (Dict[str, str]): A dictionary containing all the available pre-trained
    checkpoints.
    """

    # 预训练配置文件映射
    pretrained_config_archive_map = SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
    # 模型类型
    model_type = "squeezebert"
    # 初始化函数，用于初始化一个 Transformer 模型的配置参数
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为 30522
        hidden_size=768,  # 隐藏层大小，默认为 768
        num_hidden_layers=12,  # Transformer 模型的隐藏层层数，默认为 12
        num_attention_heads=12,  # 注意力头的数量，默认为 12
        intermediate_size=3072,  # 中间层的大小，默认为 3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为 gelu
        hidden_dropout_prob=0.1,  # 隐藏层的 dropout 概率，默认为 0.1
        attention_probs_dropout_prob=0.1,  # 注意力概率的 dropout 概率，默认为 0.1
        max_position_embeddings=512,  # 最大位置嵌入数，默认为 512
        type_vocab_size=2,  # 类型词汇表大小，默认为 2
        initializer_range=0.02,  # 初始化范围，默认为 0.02
        layer_norm_eps=1e-12,  # 层归一化的 epsilon，默认为 1e-12
        pad_token_id=0,  # 填充标记的 ID，默认为 0
        embedding_size=768,  # 嵌入大小，默认为 768
        q_groups=4,  # 查询张量的分组数，默认为 4
        k_groups=4,  # 键张量的分组数，默认为 4
        v_groups=4,  # 值张量的分组数，默认为 4
        post_attention_groups=1,  # 注意力后处理的分组数，默认为 1
        intermediate_groups=4,  # 中间层的分组数，默认为 4
        output_groups=4,  # 输出层的分组数，默认为 4
        **kwargs,  # 其它关键字参数，用于接收未知的额外参数
    ):
        # 调用父类的初始化方法，传递填充标记 ID 和其他未知关键字参数
        super().__init__(pad_token_id=pad_token_id, **kwargs)
    
        # 初始化 Transformer 模型的各种配置参数
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
        self.q_groups = q_groups
        self.k_groups = k_groups
        self.v_groups = v_groups
        self.post_attention_groups = post_attention_groups
        self.intermediate_groups = intermediate_groups
        self.output_groups = output_groups
# # 从 transformers.models.bert.configuration_bert.BertOnxxConfig 复制并修改为 SqueezeBertOnnxConfig
class SqueezeBertOnnxConfig(OnnxConfig):
    # 定义 inputs 属性，返回一个映射，表示模型输入的动态轴
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多项选择，则设置动态轴为 {0: "batch", 1: "choice", 2: "sequence"}
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则，设置动态轴为 {0: "batch", 1: "sequence"}
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回一个有序字典，包含输入名称到动态轴的映射
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),        # 输入为 input_ids，对应动态轴为 dynamic_axis
                ("attention_mask", dynamic_axis),  # 输入为 attention_mask，对应动态轴为 dynamic_axis
                ("token_type_ids", dynamic_axis),  # 输入为 token_type_ids，对应动态轴为 dynamic_axis
            ]
        )
```