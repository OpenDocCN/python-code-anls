# `.\models\led\configuration_led.py`

```py
# coding=utf-8
# Copyright 2021 Iz Beltagy, Matthew E. Peters, Arman Cohan and The HuggingFace Inc. team. All rights reserved.
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

""" LED model configuration"""

from typing import List, Union

# 从配置工具中导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 从工具中导入日志记录功能
from ...utils import logging

# 获取logger对象，用于记录日志信息
logger = logging.get_logger(__name__)

# 预训练配置存档映射，将模型名称映射到其配置文件的URL
LED_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/config.json",
    # 查看所有LED模型：https://huggingface.co/models?filter=led
}

# LED配置类，继承自预训练配置类PretrainedConfig
class LEDConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LEDModel`]. It is used to instantiate an LED
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LED
    [allenai/led-base-16384](https://huggingface.co/allenai/led-base-16384) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```
    >>> from transformers import LEDModel, LEDConfig

    >>> # Initializing a LED allenai/led-base-16384 style configuration
    >>> configuration = LEDConfig()

    >>> # Initializing a model from the allenai/led-base-16384 style configuration
    >>> model = LEDModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    
    # 模型类型为LED
    model_type = "led"
    # 属性映射字典，将LED配置中的一些属性名称映射到标准的transformers配置名称
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "attention_probs_dropout_prob": "attention_dropout",
        "initializer_range": "init_std",
    }
    # 初始化方法，用于创建一个新的Transformer模型实例
    def __init__(
        self,
        vocab_size=50265,  # 词汇表大小，默认为50265
        max_encoder_position_embeddings=16384,  # 编码器最大位置嵌入数量，默认为16384
        max_decoder_position_embeddings=1024,   # 解码器最大位置嵌入数量，默认为1024
        encoder_layers=12,  # 编码器层数，默认为12层
        encoder_ffn_dim=4096,  # 编码器中FFN层的维度，默认为4096
        encoder_attention_heads=16,  # 编码器中注意力头的数量，默认为16
        decoder_layers=12,  # 解码器层数，默认为12层
        decoder_ffn_dim=4096,  # 解码器中FFN层的维度，默认为4096
        decoder_attention_heads=16,  # 解码器中注意力头的数量，默认为16
        encoder_layerdrop=0.0,  # 编码器层的丢弃率，默认为0.0
        decoder_layerdrop=0.0,  # 解码器层的丢弃率，默认为0.0
        use_cache=True,  # 是否使用缓存，默认为True
        is_encoder_decoder=True,  # 是否为编码-解码结构，默认为True
        activation_function="gelu",  # 激活函数类型，默认为gelu
        d_model=1024,  # 模型的维度，默认为1024
        dropout=0.1,  # 总体的丢弃率，默认为0.1
        attention_dropout=0.0,  # 注意力丢弃率，默认为0.0
        activation_dropout=0.0,  # 激活函数的丢弃率，默认为0.0
        init_std=0.02,  # 参数初始化的标准差，默认为0.02
        decoder_start_token_id=2,  # 解码器起始标记ID，默认为2
        classifier_dropout=0.0,  # 分类器的丢弃率，默认为0.0
        pad_token_id=1,  # 填充标记的ID，默认为1
        bos_token_id=0,  # 开始序列标记的ID，默认为0
        eos_token_id=2,  # 结束序列标记的ID，默认为2
        attention_window: Union[List[int], int] = 512,  # 注意力窗口的大小或者列表，默认为512
        **kwargs,  # 其他关键字参数
    ):
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.max_encoder_position_embeddings = max_encoder_position_embeddings  # 设置编码器最大位置嵌入数量
        self.max_decoder_position_embeddings = max_decoder_position_embeddings  # 设置解码器最大位置嵌入数量
        self.d_model = d_model  # 设置模型的维度
        self.encoder_ffn_dim = encoder_ffn_dim  # 设置编码器中FFN层的维度
        self.encoder_layers = encoder_layers  # 设置编码器层数
        self.encoder_attention_heads = encoder_attention_heads  # 设置编码器中注意力头的数量
        self.decoder_ffn_dim = decoder_ffn_dim  # 设置解码器中FFN层的维度
        self.decoder_layers = decoder_layers  # 设置解码器层数
        self.decoder_attention_heads = decoder_attention_heads  # 设置解码器中注意力头的数量
        self.dropout = dropout  # 设置总体的丢弃率
        self.attention_dropout = attention_dropout  # 设置注意力丢弃率
        self.activation_dropout = activation_dropout  # 设置激活函数的丢弃率
        self.activation_function = activation_function  # 设置激活函数类型
        self.init_std = init_std  # 设置参数初始化的标准差
        self.encoder_layerdrop = encoder_layerdrop  # 设置编码器层的丢弃率
        self.decoder_layerdrop = decoder_layerdrop  # 设置解码器层的丢弃率
        self.classifier_dropout = classifier_dropout  # 设置分类器的丢弃率
        self.use_cache = use_cache  # 设置是否使用缓存
        self.num_hidden_layers = encoder_layers  # 设置隐藏层的数量为编码器层数
        self.attention_window = attention_window  # 设置注意力窗口的大小或者列表

        super().__init__(  # 调用父类的初始化方法
            pad_token_id=pad_token_id,  # 设置填充标记的ID
            bos_token_id=bos_token_id,  # 设置开始序列标记的ID
            eos_token_id=eos_token_id,  # 设置结束序列标记的ID
            is_encoder_decoder=is_encoder_decoder,  # 设置是否为编码-解码结构
            decoder_start_token_id=decoder_start_token_id,  # 设置解码器起始标记ID
            **kwargs,  # 传递其他关键字参数
        )
```