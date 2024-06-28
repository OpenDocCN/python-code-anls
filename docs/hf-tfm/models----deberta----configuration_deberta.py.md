# `.\models\deberta\configuration_deberta.py`

```
# coding=utf-8
# Copyright 2020, Microsoft and the HuggingFace Inc. team.
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
""" DeBERTa model configuration"""

# 导入所需模块
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

# 从 Transformers 库中导入必要的配置和工具函数
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 如果是类型检查，导入相关类
if TYPE_CHECKING:
    from ... import FeatureExtractionMixin, PreTrainedTokenizerBase, TensorType

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 定义 DeBERTa 预训练配置文件映射表
DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/deberta-base": "https://huggingface.co/microsoft/deberta-base/resolve/main/config.json",
    "microsoft/deberta-large": "https://huggingface.co/microsoft/deberta-large/resolve/main/config.json",
    "microsoft/deberta-xlarge": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/config.json",
    "microsoft/deberta-base-mnli": "https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/config.json",
    "microsoft/deberta-large-mnli": "https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/config.json",
    "microsoft/deberta-xlarge-mnli": "https://huggingface.co/microsoft/deberta-xlarge-mnli/resolve/main/config.json",
}

# DeBERTa 的配置类，继承自 PretrainedConfig
class DebertaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DebertaModel`] or a [`TFDebertaModel`]. It is
    used to instantiate a DeBERTa model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the DeBERTa
    [microsoft/deberta-base](https://huggingface.co/microsoft/deberta-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import DebertaConfig, DebertaModel

    >>> # Initializing a DeBERTa microsoft/deberta-base style configuration
    >>> configuration = DebertaConfig()

    >>> # Initializing a model (with random weights) from the microsoft/deberta-base style configuration
    >>> model = DebertaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型为 "deberta"
    model_type = "deberta"
        # 初始化函数，用于创建一个新的实例
        def __init__(
            self,
            vocab_size=50265,                      # 词汇表大小，默认为50265
            hidden_size=768,                       # 隐藏层大小，默认为768
            num_hidden_layers=12,                  # 隐藏层的数量，默认为12
            num_attention_heads=12,                # 注意力头的数量，默认为12
            intermediate_size=3072,                # 中间层大小，默认为3072
            hidden_act="gelu",                     # 隐藏层激活函数，默认为GELU
            hidden_dropout_prob=0.1,               # 隐藏层的dropout概率，默认为0.1
            attention_probs_dropout_prob=0.1,      # 注意力概率的dropout概率，默认为0.1
            max_position_embeddings=512,           # 最大位置嵌入数，默认为512
            type_vocab_size=0,                     # 类型词汇表大小，默认为0
            initializer_range=0.02,                # 初始化范围，默认为0.02
            layer_norm_eps=1e-7,                   # 层归一化的epsilon，默认为1e-7
            relative_attention=False,              # 是否使用相对注意力，默认为False
            max_relative_positions=-1,             # 最大相对位置，默认为-1
            pad_token_id=0,                        # 填充标记的ID，默认为0
            position_biased_input=True,            # 是否使用位置偏置的输入，默认为True
            pos_att_type=None,                     # 位置注意力的类型，默认为None
            pooler_dropout=0,                      # 汇集器的dropout概率，默认为0
            pooler_hidden_act="gelu",              # 汇集器的隐藏层激活函数，默认为GELU
            **kwargs,
        ):
            super().__init__(**kwargs)

            self.hidden_size = hidden_size                       # 设置隐藏层大小
            self.num_hidden_layers = num_hidden_layers           # 设置隐藏层的数量
            self.num_attention_heads = num_attention_heads       # 设置注意力头的数量
            self.intermediate_size = intermediate_size           # 设置中间层大小
            self.hidden_act = hidden_act                         # 设置隐藏层激活函数
            self.hidden_dropout_prob = hidden_dropout_prob       # 设置隐藏层的dropout概率
            self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 设置注意力概率的dropout概率
            self.max_position_embeddings = max_position_embeddings    # 设置最大位置嵌入数
            self.type_vocab_size = type_vocab_size               # 设置类型词汇表大小
            self.initializer_range = initializer_range           # 设置初始化范围
            self.relative_attention = relative_attention         # 设置是否使用相对注意力
            self.max_relative_positions = max_relative_positions  # 设置最大相对位置
            self.pad_token_id = pad_token_id                     # 设置填充标记的ID
            self.position_biased_input = position_biased_input   # 设置是否使用位置偏置的输入

            # 向后兼容性
            if isinstance(pos_att_type, str):
                pos_att_type = [x.strip() for x in pos_att_type.lower().split("|")]

            self.pos_att_type = pos_att_type                     # 设置位置注意力的类型
            self.vocab_size = vocab_size                         # 设置词汇表大小
            self.layer_norm_eps = layer_norm_eps                 # 设置层归一化的epsilon

            self.pooler_hidden_size = kwargs.get("pooler_hidden_size", hidden_size)   # 设置汇集器的隐藏层大小
            self.pooler_dropout = pooler_dropout                 # 设置汇集器的dropout概率
            self.pooler_hidden_act = pooler_hidden_act           # 设置汇集器的隐藏层激活函数
# 从 transformers.models.deberta_v2.configuration_deberta_v2.DebertaV2OnnxConfig 复制而来的类定义，继承自 OnnxConfig
class DebertaOnnxConfig(OnnxConfig):
    
    # 定义 inputs 属性，返回输入的结构化映射，键为字符串，值为映射到字符串的整数
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务类型是 "multiple-choice"
        if self.task == "multiple-choice":
            # 设置动态轴的结构为 {0: "batch", 1: "choice", 2: "sequence"}
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则设置为 {0: "batch", 1: "sequence"}
            dynamic_axis = {0: "batch", 1: "sequence"}
        
        # 如果配置对象的 type_vocab_size 大于 0
        if self._config.type_vocab_size > 0:
            # 返回有序字典，包含 "input_ids", "attention_mask", "token_type_ids" 三个键，值为 dynamic_axis
            return OrderedDict(
                [("input_ids", dynamic_axis), ("attention_mask", dynamic_axis), ("token_type_ids", dynamic_axis)]
            )
        else:
            # 返回有序字典，包含 "input_ids", "attention_mask" 两个键，值为 dynamic_axis
            return OrderedDict([("input_ids", dynamic_axis), ("attention_mask", dynamic_axis)])

    # 定义 default_onnx_opset 属性，返回默认的 ONNX 操作集版本号，为整数 12
    @property
    def default_onnx_opset(self) -> int:
        return 12

    # 定义 generate_dummy_inputs 方法，生成虚拟输入数据的字典
    def generate_dummy_inputs(
        self,
        preprocessor: Union["PreTrainedTokenizerBase", "FeatureExtractionMixin"],
        batch_size: int = -1,
        seq_length: int = -1,
        num_choices: int = -1,
        is_pair: bool = False,
        framework: Optional["TensorType"] = None,
        num_channels: int = 3,
        image_width: int = 40,
        image_height: int = 40,
        tokenizer: "PreTrainedTokenizerBase" = None,
    ) -> Mapping[str, Any]:
        # 调用父类的 generate_dummy_inputs 方法生成初始的虚拟输入数据
        dummy_inputs = super().generate_dummy_inputs(preprocessor=preprocessor, framework=framework)
        
        # 如果配置对象的 type_vocab_size 为 0 并且 dummy_inputs 中包含 "token_type_ids"
        if self._config.type_vocab_size == 0 and "token_type_ids" in dummy_inputs:
            # 删除 dummy_inputs 中的 "token_type_ids" 键
            del dummy_inputs["token_type_ids"]
        
        # 返回更新后的 dummy_inputs 字典
        return dummy_inputs
```