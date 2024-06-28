# `.\models\xlm_roberta\configuration_xlm_roberta.py`

```
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" XLM-RoBERTa configuration"""

# 从 collections 模块导入 OrderedDict 类
from collections import OrderedDict
# 从 typing 模块导入 Mapping 类型
from typing import Mapping

# 从 transformers 的相关模块中导入所需的类和函数
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 获取 logger 对象，用于日志记录
logger = logging.get_logger(__name__)

# XLM-RoBERTa 预训练模型的配置文件映射表，包含不同模型及其配置文件的 URL
XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "FacebookAI/xlm-roberta-base": "https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/main/config.json",
    "FacebookAI/xlm-roberta-large": "https://huggingface.co/FacebookAI/xlm-roberta-large/resolve/main/config.json",
    "FacebookAI/xlm-roberta-large-finetuned-conll02-dutch": (
        "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll02-dutch/resolve/main/config.json"
    ),
    "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish": (
        "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll02-spanish/resolve/main/config.json"
    ),
    "FacebookAI/xlm-roberta-large-finetuned-conll03-english": (
        "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll03-english/resolve/main/config.json"
    ),
    "FacebookAI/xlm-roberta-large-finetuned-conll03-german": (
        "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll03-german/resolve/main/config.json"
    ),
}

# XLMRoBERTaConfig 类，继承自 PretrainedConfig 类，用于存储 XLM-RoBERTa 模型的配置信息
class XLMRobertaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XLMRobertaModel`] or a [`TFXLMRobertaModel`]. It
    is used to instantiate a XLM-RoBERTa model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the XLMRoBERTa
    [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Examples:

    ```python
    >>> from transformers import XLMRobertaConfig, XLMRobertaModel

    >>> # Initializing a XLM-RoBERTa FacebookAI/xlm-roberta-base style configuration
    >>> configuration = XLMRobertaConfig()

    >>> # Initializing a model (with random weights) from the FacebookAI/xlm-roberta-base style configuration

    """
    >>> model = XLMRobertaModel(configuration)

    >>> # 访问模型配置信息
    >>> configuration = model.config
    ```

    model_type = "xlm-roberta"

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
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        # 调用父类的构造函数，初始化基类的参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置模型的各种参数
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
# 从 transformers.models.roberta.configuration_roberta.RobertaOnnxConfig 复制而来，修改为 XLMRobertaOnnxConfig
class XLMRobertaOnnxConfig(OnnxConfig):
    # 定义 inputs 属性，返回一个映射，表示输入的结构
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 根据任务类型确定动态轴的设置
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回有序字典，包含 input_ids 和 attention_mask 的动态轴设置
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )
```