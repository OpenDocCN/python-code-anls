# `.\models\rembert\configuration_rembert.py`

```py
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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

""" RemBERT model configuration"""

from collections import OrderedDict  # 导入有序字典
from typing import Mapping  # 导入类型提示 Mapping

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...onnx import OnnxConfig  # 导入ONNX配置
from ...utils import logging  # 导入日志工具


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# RemBERT预训练配置文件映射字典，指定不同模型的配置文件下载链接
REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/rembert": "https://huggingface.co/google/rembert/resolve/main/config.json",
    # 查看所有RemBERT模型：https://huggingface.co/models?filter=rembert
}


class RemBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RemBertModel`]. It is used to instantiate an
    RemBERT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the RemBERT
    [google/rembert](https://huggingface.co/google/rembert) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```
    >>> from transformers import RemBertModel, RemBertConfig

    >>> # Initializing a RemBERT rembert style configuration
    >>> configuration = RemBertConfig()

    >>> # Initializing a model from the rembert style configuration
    >>> model = RemBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rembert"  # 模型类型为rembert

    def __init__(
        self,
        vocab_size=250300,  # 词汇表大小，默认为250300
        hidden_size=1152,  # 隐藏层大小，默认为1152
        num_hidden_layers=32,  # 隐藏层层数，默认为32
        num_attention_heads=18,  # 注意力头数，默认为18
        input_embedding_size=256,  # 输入嵌入大小，默认为256
        output_embedding_size=1664,  # 输出嵌入大小，默认为1664
        intermediate_size=4608,  # 中间层大小，默认为4608
        hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
        hidden_dropout_prob=0.0,  # 隐藏层Dropout概率，默认为0.0
        attention_probs_dropout_prob=0.0,  # 注意力Dropout概率，默认为0.0
        classifier_dropout_prob=0.1,  # 分类器Dropout概率，默认为0.1
        max_position_embeddings=512,  # 最大位置嵌入数，默认为512
        type_vocab_size=2,  # 类型词汇表大小，默认为2
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 层归一化的epsilon，默认为1e-12
        use_cache=True,  # 是否使用缓存，默认为True
        pad_token_id=0,  # 填充token的ID，默认为0
        bos_token_id=312,  # 开始token的ID，默认为312
        eos_token_id=313,  # 结束token的ID，默认为313
        **kwargs,  # 其他关键字参数
        # 调用父类的初始化方法，设置模型的特殊 token ID 和其他关键参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置模型的词汇表大小
        self.vocab_size = vocab_size
        # 设置输入词嵌入的维度大小
        self.input_embedding_size = input_embedding_size
        # 设置输出词嵌入的维度大小
        self.output_embedding_size = output_embedding_size
        # 设置最大位置嵌入的数量
        self.max_position_embeddings = max_position_embeddings
        # 设置隐藏层的大小
        self.hidden_size = hidden_size
        # 设置隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置中间层的大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层的激活函数
        self.hidden_act = hidden_act
        # 设置隐藏层的丢弃率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力概率的丢弃率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置分类器的丢弃率
        self.classifier_dropout_prob = classifier_dropout_prob
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置类型词汇表的大小
        self.type_vocab_size = type_vocab_size
        # 设置层归一化的 epsilon 值
        self.layer_norm_eps = layer_norm_eps
        # 设置是否使用缓存
        self.use_cache = use_cache
        # 设置是否将词嵌入进行绑定
        self.tie_word_embeddings = False
# 定义一个自定义的配置类 RemBertOnnxConfig，继承自 OnnxConfig 类
class RemBertOnnxConfig(OnnxConfig):

    # 定义一个属性 inputs，返回一个映射，其键为字符串，值为映射（键为整数，值为字符串）
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务类型为 "multiple-choice"
        if self.task == "multiple-choice":
            # 设置动态轴 dynamic_axis 为 {0: "batch", 1: "choice", 2: "sequence"}
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则设置动态轴 dynamic_axis 为 {0: "batch", 1: "sequence"}
            dynamic_axis = {0: "batch", 1: "sequence"}
        
        # 返回一个有序字典，包含三个键值对，分别是 ("input_ids", dynamic_axis)，("attention_mask", dynamic_axis)，("token_type_ids", dynamic_axis)
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )

    # 定义一个属性 atol_for_validation，返回一个浮点数，表示验证时的绝对容差
    @property
    def atol_for_validation(self) -> float:
        # 返回绝对容差的数值，设定为 1e-4
        return 1e-4
```