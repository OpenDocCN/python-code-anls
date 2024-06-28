# `.\models\deberta_v2\configuration_deberta_v2.py`

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
"""
DeBERTa-v2 model configuration
"""
from collections import OrderedDict  # 导入有序字典类
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union  # 导入类型检查、类型声明相关模块

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...onnx import OnnxConfig  # 导入ONNX配置类
from ...utils import logging  # 导入日志工具


if TYPE_CHECKING:
    from ... import FeatureExtractionMixin, PreTrainedTokenizerBase, TensorType  # 如果是类型检查模式，则导入特征提取、预训练分词器基类和张量类型

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 预训练配置的映射字典，将预训练模型名称映射到配置文件URL
DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/deberta-v2-xlarge": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/config.json",
    "microsoft/deberta-v2-xxlarge": "https://huggingface.co/microsoft/deberta-v2-xxlarge/resolve/main/config.json",
    "microsoft/deberta-v2-xlarge-mnli": (
        "https://huggingface.co/microsoft/deberta-v2-xlarge-mnli/resolve/main/config.json"
    ),
    "microsoft/deberta-v2-xxlarge-mnli": (
        "https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli/resolve/main/config.json"
    ),
}


class DebertaV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DebertaV2Model`]. It is used to instantiate a
    DeBERTa-v2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the DeBERTa
    [microsoft/deberta-v2-xlarge](https://huggingface.co/microsoft/deberta-v2-xlarge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import DebertaV2Config, DebertaV2Model

    >>> # Initializing a DeBERTa-v2 microsoft/deberta-v2-xlarge style configuration
    >>> configuration = DebertaV2Config()

    >>> # Initializing a model (with random weights) from the microsoft/deberta-v2-xlarge style configuration
    >>> model = DebertaV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "deberta-v2"  # 模型类型为deberta-v2
        # 初始化函数，用于初始化模型参数
        def __init__(
            self,
            vocab_size=128100,  # 词汇表大小，默认为128100
            hidden_size=1536,  # 隐藏层大小，默认为1536
            num_hidden_layers=24,  # 隐藏层的数量，默认为24
            num_attention_heads=24,  # 注意力头的数量，默认为24
            intermediate_size=6144,  # 中间层大小，默认为6144
            hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
            hidden_dropout_prob=0.1,  # 隐藏层dropout概率，默认为0.1
            attention_probs_dropout_prob=0.1,  # 注意力概率dropout概率，默认为0.1
            max_position_embeddings=512,  # 最大位置嵌入数，默认为512
            type_vocab_size=0,  # 类型词汇表大小，默认为0
            initializer_range=0.02,  # 初始化范围，默认为0.02
            layer_norm_eps=1e-7,  # 层归一化的epsilon值，默认为1e-7
            relative_attention=False,  # 是否使用相对注意力，默认为False
            max_relative_positions=-1,  # 最大相对位置，默认为-1
            pad_token_id=0,  # 填充标记ID，默认为0
            position_biased_input=True,  # 位置偏置输入，默认为True
            pos_att_type=None,  # 位置注意力类型，默认为None
            pooler_dropout=0,  # 汇聚层dropout概率，默认为0
            pooler_hidden_act="gelu",  # 汇聚层隐藏层激活函数，默认为GELU
            **kwargs,
        ):
            super().__init__(**kwargs)  # 调用父类的初始化函数

            self.hidden_size = hidden_size  # 设置隐藏层大小
            self.num_hidden_layers = num_hidden_layers  # 设置隐藏层数量
            self.num_attention_heads = num_attention_heads  # 设置注意力头数量
            self.intermediate_size = intermediate_size  # 设置中间层大小
            self.hidden_act = hidden_act  # 设置隐藏层激活函数
            self.hidden_dropout_prob = hidden_dropout_prob  # 设置隐藏层dropout概率
            self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 设置注意力概率dropout概率
            self.max_position_embeddings = max_position_embeddings  # 设置最大位置嵌入数
            self.type_vocab_size = type_vocab_size  # 设置类型词汇表大小
            self.initializer_range = initializer_range  # 设置初始化范围
            self.relative_attention = relative_attention  # 设置是否使用相对注意力
            self.max_relative_positions = max_relative_positions  # 设置最大相对位置
            self.pad_token_id = pad_token_id  # 设置填充标记ID
            self.position_biased_input = position_biased_input  # 设置位置偏置输入

            # 兼容性处理
            if isinstance(pos_att_type, str):  # 如果位置注意力类型为字符串
                pos_att_type = [x.strip() for x in pos_att_type.lower().split("|")]  # 将其分割为小写后的列表

            self.pos_att_type = pos_att_type  # 设置位置注意力类型
            self.vocab_size = vocab_size  # 设置词汇表大小
            self.layer_norm_eps = layer_norm_eps  # 设置层归一化的epsilon值

            self.pooler_hidden_size = kwargs.get("pooler_hidden_size", hidden_size)  # 设置汇聚层隐藏大小，默认为隐藏层大小
            self.pooler_dropout = pooler_dropout  # 设置汇聚层dropout概率
            self.pooler_hidden_act = pooler_hidden_act  # 设置汇聚层隐藏层激活函数
# 定义一个 DebertaV2OnnxConfig 类，继承自 OnnxConfig 类
class DebertaV2OnnxConfig(OnnxConfig):
    
    # 定义 inputs 属性，返回一个映射，其键为字符串，值为映射，其值为整数到字符串的映射
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务类型是多项选择
        if self.task == "multiple-choice":
            # 动态轴设置为 {0: "batch", 1: "choice", 2: "sequence"}
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则动态轴设置为 {0: "batch", 1: "sequence"}
            dynamic_axis = {0: "batch", 1: "sequence"}
        
        # 如果配置中的 type_vocab_size 大于 0
        if self._config.type_vocab_size > 0:
            # 返回一个有序字典，包含键为 "input_ids", "attention_mask", "token_type_ids"，值为 dynamic_axis 的条目
            return OrderedDict(
                [("input_ids", dynamic_axis), ("attention_mask", dynamic_axis), ("token_type_ids", dynamic_axis)]
            )
        else:
            # 否则返回一个有序字典，包含键为 "input_ids", "attention_mask"，值为 dynamic_axis 的条目
            return OrderedDict([("input_ids", dynamic_axis), ("attention_mask", dynamic_axis)])
    
    # 定义 default_onnx_opset 属性，返回整数 12
    @property
    def default_onnx_opset(self) -> int:
        return 12
    
    # 定义 generate_dummy_inputs 方法，用于生成虚拟输入数据
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
        # 调用父类的 generate_dummy_inputs 方法生成虚拟输入
        dummy_inputs = super().generate_dummy_inputs(preprocessor=preprocessor, framework=framework)
        
        # 如果配置中的 type_vocab_size 为 0 并且 dummy_inputs 中包含 "token_type_ids"
        if self._config.type_vocab_size == 0 and "token_type_ids" in dummy_inputs:
            # 从 dummy_inputs 中删除 "token_type_ids" 条目
            del dummy_inputs["token_type_ids"]
        
        # 返回生成的虚拟输入
        return dummy_inputs
```