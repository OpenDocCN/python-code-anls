# `.\models\plbart\configuration_plbart.py`

```py
# coding=utf-8
# Copyright 2022, UCLA NLP, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
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
""" PLBART model configuration"""
# 导入需要的模块和类
from collections import OrderedDict
from typing import Mapping

# 导入配置工具和ONNX配置
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件映射
PLBART_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "uclanlp/plbart-base": "https://huggingface.co/uclanlp/plbart-base/resolve/main/config.json",
    # 查看所有PLBART模型的列表 https://huggingface.co/models?filter=plbart
}

# 定义PLBartConfig类，继承自PretrainedConfig类
class PLBartConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PLBartModel`]. It is used to instantiate an
    PLBART model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the PLBART
    [uclanlp/plbart-base](https://huggingface.co/uclanlp/plbart-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```
    >>> from transformers import PLBartConfig, PLBartModel

    >>> # Initializing a PLBART uclanlp/plbart-base style configuration
    >>> configuration = PLBartConfig()

    >>> # Initializing a model (with random weights) from the uclanlp/plbart-base style configuration
    >>> model = PLBartModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    
    # 模型类型为plbart
    model_type = "plbart"
    # 推理过程中需要忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射字典，用于将配置中的属性名映射到模型架构中使用的名称
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    # 初始化函数，用于初始化Transformer模型的各种参数和配置
    def __init__(
        self,
        vocab_size=50005,  # 词汇表大小，默认为50005
        max_position_embeddings=1024,  # 最大位置编码长度，默认为1024
        encoder_layers=6,  # 编码器层数，默认为6层
        encoder_ffn_dim=3072,  # 编码器中FFN（Feed Forward Network）层的维度，默认为3072
        encoder_attention_heads=12,  # 编码器中注意力头的数量，默认为12
        decoder_layers=6,  # 解码器层数，默认为6层
        decoder_ffn_dim=3072,  # 解码器中FFN层的维度，默认为3072
        decoder_attention_heads=12,  # 解码器中注意力头的数量，默认为12
        encoder_layerdrop=0.0,  # 编码器层的丢弃概率，默认为0.0
        decoder_layerdrop=0.0,  # 解码器层的丢弃概率，默认为0.0
        use_cache=True,  # 是否使用缓存，默认为True
        is_encoder_decoder=True,  # 是否是编码器-解码器结构，默认为True
        activation_function="gelu",  # 激活函数类型，默认为GELU
        d_model=768,  # 模型维度，默认为768
        dropout=0.1,  # 全局Dropout概率，默认为0.1
        attention_dropout=0.1,  # 注意力机制的Dropout概率，默认为0.1
        activation_dropout=0.0,  # 激活函数Dropout概率，默认为0.0
        init_std=0.02,  # 初始化标准差，默认为0.02
        classifier_dropout=0.0,  # 分类器层的Dropout概率，默认为0.0
        scale_embedding=True,  # 是否缩放嵌入，默认为True；如果为True，则缩放因子为sqrt(d_model)
        pad_token_id=1,  # 填充标记的ID，默认为1
        bos_token_id=0,  # 起始标记的ID，默认为0
        eos_token_id=2,  # 结束标记的ID，默认为2
        forced_eos_token_id=2,  # 强制结束标记的ID，默认为2
        **kwargs,  # 其他未明确列出的参数，用于接收和处理其他未命名参数
    ):
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置编码长度
        self.d_model = d_model  # 设置模型维度
        self.encoder_ffn_dim = encoder_ffn_dim  # 设置编码器中FFN层的维度
        self.encoder_layers = encoder_layers  # 设置编码器层数
        self.encoder_attention_heads = encoder_attention_heads  # 设置编码器中注意力头的数量
        self.decoder_ffn_dim = decoder_ffn_dim  # 设置解码器中FFN层的维度
        self.decoder_layers = decoder_layers  # 设置解码器层数
        self.decoder_attention_heads = decoder_attention_heads  # 设置解码器中注意力头的数量
        self.dropout = dropout  # 设置全局Dropout概率
        self.attention_dropout = attention_dropout  # 设置注意力机制的Dropout概率
        self.activation_dropout = activation_dropout  # 设置激活函数Dropout概率
        self.activation_function = activation_function  # 设置激活函数类型
        self.init_std = init_std  # 设置初始化标准差
        self.encoder_layerdrop = encoder_layerdrop  # 设置编码器层的丢弃概率
        self.decoder_layerdrop = decoder_layerdrop  # 设置解码器层的丢弃概率
        self.classifier_dropout = classifier_dropout  # 设置分类器层的Dropout概率
        self.use_cache = use_cache  # 设置是否使用缓存
        self.num_hidden_layers = encoder_layers  # 设置隐藏层的数量为编码器的层数
        self.scale_embedding = scale_embedding  # 设置是否缩放嵌入

        super().__init__(  # 调用父类的初始化函数
            pad_token_id=pad_token_id,  # 设置填充标记的ID
            bos_token_id=bos_token_id,  # 设置起始标记的ID
            eos_token_id=eos_token_id,  # 设置结束标记的ID
            is_encoder_decoder=is_encoder_decoder,  # 设置是否是编码器-解码器结构
            forced_eos_token_id=forced_eos_token_id,  # 设置强制结束标记的ID
            **kwargs,  # 传递其他未明确列出的参数
        )
# 定义一个继承自 OnnxConfigWithPast 的配置类，用于配置 PLBart 模型的 ONNX 格式导出参数
class PLBartOnnxConfig(OnnxConfigWithPast):
    
    # 定义一个属性方法 inputs，返回一个有序字典，描述了模型的输入格式
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),  # 输入的 input_ids 的格式描述
                ("attention_mask", {0: "batch", 1: "sequence"}),  # 输入的 attention_mask 的格式描述
            ]
        )

    # 定义一个属性方法 outputs，返回一个有序字典，描述了模型的输出格式
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.use_past:
            return OrderedDict(
                [
                    ("last_hidden_state", {0: "batch", 1: "sequence"}),  # 输出的 last_hidden_state 的格式描述
                    ("past_keys", {0: "batch", 2: "sequence"}),  # 输出的 past_keys 的格式描述
                    ("encoder_last_hidden_state", {0: "batch", 1: "sequence"}),  # 输出的 encoder_last_hidden_state 的格式描述
                ]
            )
        else:
            return OrderedDict(
                [
                    ("last_hidden_state", {0: "batch", 1: "sequence"}),  # 输出的 last_hidden_state 的格式描述
                    ("encoder_last_hidden_state", {0: "batch", 1: "sequence"}),  # 输出的 encoder_last_hidden_state 的格式描述
                ]
            )
```