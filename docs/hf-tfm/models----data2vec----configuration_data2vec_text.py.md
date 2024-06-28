# `.\models\data2vec\configuration_data2vec_text.py`

```
# coding=utf-8
# 文件编码声明，使用 UTF-8 编码格式
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# 版权声明，版权归 HuggingFace Inc. 团队所有
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 授权协议声明，使用 Apache License, Version 2.0 版本
# you may not use this file except in compliance with the License.
# 除非遵循 Apache License 2.0 版本，否则不得使用此文件
# You may obtain a copy of the License at
# 可以获取协议的副本链接
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，否则软件
# distributed under the License is distributed on an "AS IS" BASIS,
# 根据协议分发软件，按"现状"提供，无任何担保
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 无论是明示还是暗示的，均不对软件的任何担保或条件
# See the License for the specific language governing permissions and
# 详细信息请查阅协议
# limitations under the License.
# 限制的详细内容请查阅协议
""" Data2VecText configuration"""
# 模块说明，Data2VecText 配置
from collections import OrderedDict
# 导入 OrderedDict，有序字典类
from typing import Mapping
# 导入 Mapping，映射类型

from ...configuration_utils import PretrainedConfig
# 导入预训练配置类 PretrainedConfig
from ...onnx import OnnxConfig
# 导入 OnnxConfig，ONNX 配置类
from ...utils import logging
# 导入 logging 模块，日志模块

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/data2vec-text-base": "https://huggingface.co/data2vec/resolve/main/config.json",
}
# 预训练配置映射表，将预训练模型名称映射到其配置文件的 URL

class Data2VecTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Data2VecTextModel`] and [`Data2VecTextModel`]. It
    is used to instantiate a Data2VecText model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Data2VecText
    [facebook/data2vec-text-base](https://huggingface.co/facebook/data2vec-text-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # Data2VecText 配置类，用于存储 Data2VecTextModel 的配置

    model_type = "data2vec-text"
    # 模型类型标识为 "data2vec-text"

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
        # 初始化方法，配置 Data2VecText 模型的各项参数
        pass
        # 占位符，未实际执行操作，保留该方法以后可能的参数扩展
        ):
            # 调用父类初始化方法，传递相关参数并设置默认的特殊标记的 ID
            super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

            # 初始化模型的词汇表大小
            self.vocab_size = vocab_size
            # 初始化隐藏层的大小
            self.hidden_size = hidden_size
            # 初始化隐藏层的数量
            self.num_hidden_layers = num_hidden_layers
            # 初始化注意力头的数量
            self.num_attention_heads = num_attention_heads
            # 初始化隐藏层激活函数
            self.hidden_act = hidden_act
            # 初始化中间层的大小
            self.intermediate_size = intermediate_size
            # 初始化隐藏层的 dropout 概率
            self.hidden_dropout_prob = hidden_dropout_prob
            # 初始化注意力矩阵的 dropout 概率
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            # 初始化最大位置嵌入的长度
            self.max_position_embeddings = max_position_embeddings
            # 初始化类型词汇表的大小
            self.type_vocab_size = type_vocab_size
            # 初始化初始化范围
            self.initializer_range = initializer_range
            # 初始化层归一化的 epsilon 值
            self.layer_norm_eps = layer_norm_eps
            # 初始化位置嵌入的类型
            self.position_embedding_type = position_embedding_type
            # 初始化是否使用缓存
            self.use_cache = use_cache
            # 初始化分类器的 dropout 概率
            self.classifier_dropout = classifier_dropout
# 定义一个继承自OnnxConfig的Data2VecTextOnnxConfig类，用于配置ONNX模型的输入规格
class Data2VecTextOnnxConfig(OnnxConfig):
    
    # 定义一个inputs属性，返回一个映射，描述模型的输入
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        
        # 如果任务是"multiple-choice"
        if self.task == "multiple-choice":
            # 定义动态轴的映射，包含批处理、选择和序列三个轴
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则，定义动态轴的映射，包含批处理和序列两个轴
            dynamic_axis = {0: "batch", 1: "sequence"}
        
        # 返回一个有序字典，描述模型的输入，包括input_ids和attention_mask两个输入
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),         # input_ids使用动态轴定义
                ("attention_mask", dynamic_axis),    # attention_mask使用动态轴定义
            ]
        )
```