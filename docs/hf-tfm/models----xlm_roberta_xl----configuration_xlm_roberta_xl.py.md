# `.\models\xlm_roberta_xl\configuration_xlm_roberta_xl.py`

```
# coding=utf-8
# 声明文件的编码格式为 UTF-8

# Copyright 2022 The HuggingFace Inc. team.
# 版权声明，版权归 HuggingFace 公司所有，日期为 2022 年

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 许可证授权使用本文件

# you may not use this file except in compliance with the License.
# 除非符合许可证，否则不得使用本文件

# You may obtain a copy of the License at
# 可以在上述许可证链接获取许可证的副本

# http://www.apache.org/licenses/LICENSE-2.0
# 许可证链接地址

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则按"原样"分发软件，不附带任何明示或暗示的保证或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 查看许可证以了解具体的语言控制权限和限制

""" XLM_ROBERTA_XL configuration"""

# 导入必要的模块
from collections import OrderedDict  # 导入 OrderedDict 类
from typing import Mapping  # 导入 Mapping 类型提示

# 导入配置工具函数和 ONNX 配置
from ...configuration_utils import PretrainedConfig  # 导入 PretrainedConfig 类
from ...onnx import OnnxConfig  # 导入 OnnxConfig 类
from ...utils import logging  # 导入 logging 模块

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 预训练模型及其配置文件映射
XLM_ROBERTA_XL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/xlm-roberta-xl": "https://huggingface.co/facebook/xlm-roberta-xl/resolve/main/config.json",
    "facebook/xlm-roberta-xxl": "https://huggingface.co/facebook/xlm-roberta-xxl/resolve/main/config.json",
    # 查看所有 XLM-RoBERTa-XL 模型的链接地址
}

# XLMRoertaXLConfig 类的定义，继承自 PretrainedConfig
class XLMRobertaXLConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XLMRobertaXLModel`] or a [`TFXLMRobertaXLModel`].
    It is used to instantiate a XLM_ROBERTA_XL model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    XLM_ROBERTA_XL [facebook/xlm-roberta-xl](https://huggingface.co/facebook/xlm-roberta-xl) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Examples:

    ```python
    >>> from transformers import XLMRobertaXLConfig, XLMRobertaXLModel

    >>> # Initializing a XLM_ROBERTA_XL google-bert/bert-base-uncased style configuration
    >>> configuration = XLMRobertaXLConfig()

    >>> # Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration
    >>> model = XLMRobertaXLModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    
    # 模型类型定义为 "xlm-roberta-xl"
    model_type = "xlm-roberta-xl"
    # 定义一个初始化方法，用于初始化一个 Transformer 模型的参数
    def __init__(
        self,
        vocab_size=250880,  # 词汇表大小，默认为 250880
        hidden_size=2560,  # 隐藏层大小，默认为 2560
        num_hidden_layers=36,  # 隐藏层的数量，默认为 36
        num_attention_heads=32,  # 注意力头的数量，默认为 32
        intermediate_size=10240,  # 中间层的大小，默认为 10240
        hidden_act="gelu",  # 隐藏层的激活函数，默认为 GELU
        hidden_dropout_prob=0.1,  # 隐藏层的 dropout 概率，默认为 0.1
        attention_probs_dropout_prob=0.1,  # 注意力概率的 dropout 概率，默认为 0.1
        max_position_embeddings=514,  # 最大位置编码数，默认为 514
        type_vocab_size=1,  # 类型词汇表大小，默认为 1
        initializer_range=0.02,  # 初始化范围，默认为 0.02
        layer_norm_eps=1e-05,  # 层归一化的 epsilon，默认为 1e-05
        pad_token_id=1,  # 填充 token 的 id，默认为 1
        bos_token_id=0,  # 起始 token 的 id，默认为 0
        eos_token_id=2,  # 结束 token 的 id，默认为 2
        position_embedding_type="absolute",  # 位置嵌入的类型，默认为绝对位置编码
        use_cache=True,  # 是否使用缓存，默认为 True
        classifier_dropout=None,  # 分类器的 dropout，初始为 None，可以后续设置
        **kwargs,  # 其他未明确指定的参数
    ):
        # 调用父类的初始化方法，设置填充、起始和结束 token 的 id
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
# 从 transformers.models.roberta.configuration_roberta.RobertaOnnxConfig 复制代码，并将 Roberta 替换为 XLMRobertaXL
class XLMRobertaXLOnnxConfig(OnnxConfig):
    
    # 定义 inputs 属性，返回一个字典，其中包含动态轴的映射关系
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多选题 ("multiple-choice")，设置动态轴为 {0: "batch", 1: "choice", 2: "sequence"}
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        # 否则，设置动态轴为 {0: "batch", 1: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        
        # 返回有序字典，包含 input_ids 和 attention_mask 作为键，对应的动态轴作为值
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )
```