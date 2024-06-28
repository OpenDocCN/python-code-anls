# `.\models\ibert\configuration_ibert.py`

```
# coding=utf-8
# 声明编码格式为UTF-8
# Copyright 2021 The I-BERT Authors (Sehoon Kim, Amir Gholami, Zhewei Yao,
# Michael Mahoney, Kurt Keutzer - UC Berkeley) and The HuggingFace Inc. team.
# 版权声明，包括作者信息和版权信息
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# 版权声明，包括年份和版权所有者信息
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 进行许可
# you may not use this file except in compliance with the License.
# 除非遵循 Apache License, Version 2.0，否则不得使用此文件
# You may obtain a copy of the License at
# 可以在以下链接获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有任何明示或暗示的担保或条件，软件在分发时是基于“按原样”分发的
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# See the License for the specific language governing permissions and
# limitations under the License.
# 在许可下限制的特定语言和限制
""" I-BERT configuration"""
# 模块级文档字符串，描述本文件是关于 I-BERT 的配置信息

from collections import OrderedDict
# 导入 OrderedDict 类，用于有序字典
from typing import Mapping
# 导入 Mapping 类型提示

from ...configuration_utils import PretrainedConfig
# 从配置工具中导入预训练配置类 PretrainedConfig
from ...onnx import OnnxConfig
# 从 onnx 模块中导入 OnnxConfig
from ...utils import logging
# 从 utils 模块中导入 logging 模块

logger = logging.get_logger(__name__)
# 获取当前模块的 logger 对象

IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "kssteven/ibert-roberta-base": "https://huggingface.co/kssteven/ibert-roberta-base/resolve/main/config.json",
    "kssteven/ibert-roberta-large": "https://huggingface.co/kssteven/ibert-roberta-large/resolve/main/config.json",
    "kssteven/ibert-roberta-large-mnli": (
        "https://huggingface.co/kssteven/ibert-roberta-large-mnli/resolve/main/config.json"
    ),
}
# 定义一个字典，映射预训练模型名称到其配置文件的 URL

class IBertConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`IBertModel`]. It is used to instantiate a I-BERT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the IBERT
    [kssteven/ibert-roberta-base](https://huggingface.co/kssteven/ibert-roberta-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    """
    # IBertConfig 类，用于存储 IBERT 模型的配置信息

    model_type = "ibert"
    # 模型类型为 ibert

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
        quant_mode=False,
        force_dequant="none",
        **kwargs,
    ):
        """
        Initializes an IBertConfig object with default values for its parameters.
        构造函数，初始化 IBertConfig 对象，设置各个参数的默认值。
        """
        ):
            # 调用父类的构造函数，设置模型的特定参数和超参数
            super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

            # 设置模型的词汇表大小
            self.vocab_size = vocab_size
            # 设置模型的隐藏层大小
            self.hidden_size = hidden_size
            # 设置模型的隐藏层数量
            self.num_hidden_layers = num_hidden_layers
            # 设置模型的注意力头数量
            self.num_attention_heads = num_attention_heads
            # 设置模型的隐藏层激活函数
            self.hidden_act = hidden_act
            # 设置模型的中间层大小（全连接层）
            self.intermediate_size = intermediate_size
            # 设置模型的隐藏层dropout概率
            self.hidden_dropout_prob = hidden_dropout_prob
            # 设置模型的注意力层dropout概率
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            # 设置模型的最大位置嵌入长度
            self.max_position_embeddings = max_position_embeddings
            # 设置模型的类型词汇表大小
            self.type_vocab_size = type_vocab_size
            # 设置模型的初始化范围
            self.initializer_range = initializer_range
            # 设置模型的层归一化epsilon值
            self.layer_norm_eps = layer_norm_eps
            # 设置模型的位置嵌入类型
            self.position_embedding_type = position_embedding_type
            # 设置模型的量化模式
            self.quant_mode = quant_mode
            # 设置模型的强制去量化标志
            self.force_dequant = force_dequant
# 定义一个名为 IBertOnnxConfig 的类，它继承自 OnnxConfig 类
class IBertOnnxConfig(OnnxConfig):
    
    # 定义一个 inputs 属性，返回一个字典，键为字符串，值为映射（字典，键为整数，值为字符串）
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        
        # 如果任务是多项选择 ("multiple-choice")，则设置动态轴为三维：批量（batch）、选择（choice）、序列（sequence）
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则，设置动态轴为二维：批量（batch）、序列（sequence）
            dynamic_axis = {0: "batch", 1: "sequence"}
        
        # 返回一个有序字典，包含两个条目：("input_ids", dynamic_axis) 和 ("attention_mask", dynamic_axis)
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )
```