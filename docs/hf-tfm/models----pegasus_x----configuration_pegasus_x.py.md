# `.\models\pegasus_x\configuration_pegasus_x.py`

```
# coding=utf-8
# Copyright 2022, Google and The HuggingFace Inc. team. All rights reserved.
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
""" PEGASUS-X model configuration"""

# 导入配置基类 PretrainedConfig 和 logging 工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# PEGASUS-X 预训练模型配置文件映射表，提供预训练模型的配置文件 URL
PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/pegasus-x-base": "https://huggingface.co/google/pegasus-x-base/resolve/main/config.json",
    "google/pegasus-x-large": "https://huggingface.co/google/pegasus-x-large/resolve/main/config.json",
    # 查看所有 PEGASUS-X 模型的列表，访问 https://huggingface.co/models?filter=pegasus-x
}

# PegasusXConfig 类，用于存储 PEGASUS-X 模型的配置信息，继承自 PretrainedConfig
class PegasusXConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PegasusXModel`]. It is used to instantiate a
    PEGASUS-X model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the PEGASUS-X
    [google/pegasus-x-large](https://huggingface.co/google/pegasus-x-large) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import PegasusXConfig, PegasusXModel

    >>> # Initializing a PEGASUS google/pegasus-x-large style configuration
    >>> configuration = PegasusXConfig()

    >>> # Initializing a model (with random weights) from the google/pegasus-x-large style configuration
    >>> model = PegasusXModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型标识为 "pegasus_x"
    model_type = "pegasus_x"
    # 推断时要忽略的键列表，这里忽略 "past_key_values"
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射表，将一些通用名称映射到 PEGASUS-X 模型配置中的特定名称
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    # 初始化函数，用于初始化一个 Transformer 模型的参数和配置
    def __init__(
        self,
        vocab_size=96103,  # 词汇表大小，默认为96103
        max_position_embeddings=16384,  # 最大位置嵌入数，默认为16384
        encoder_layers=16,  # 编码器层数，默认为16层
        encoder_ffn_dim=4096,  # 编码器中 Feed-Forward 层的维度，默认为4096
        encoder_attention_heads=16,  # 编码器中注意力头的数量，默认为16
        decoder_layers=16,  # 解码器层数，默认为16层
        decoder_ffn_dim=4096,  # 解码器中 Feed-Forward 层的维度，默认为4096
        decoder_attention_heads=16,  # 解码器中注意力头的数量，默认为16
        encoder_layerdrop=0.0,  # 编码器层间的随机删除率，默认为0.0
        decoder_layerdrop=0.0,  # 解码器层间的随机删除率，默认为0.0
        use_cache=True,  # 是否使用缓存，默认为True
        is_encoder_decoder=True,  # 是否是编码解码结构，默认为True
        activation_function="gelu",  # 激活函数类型，默认为GELU
        d_model=1024,  # 模型的维度，默认为1024
        dropout=0.1,  # 总体的Dropout概率，默认为0.1
        attention_dropout=0.0,  # 注意力Dropout概率，默认为0.0
        activation_dropout=0.0,  # 激活函数Dropout概率，默认为0.0
        init_std=0.02,  # 初始化参数的标准差，默认为0.02
        decoder_start_token_id=0,  # 解码器的起始标记ID，默认为0
        scale_embedding=True,  # 是否对嵌入进行缩放，默认为True；如果为True，缩放因子为sqrt(d_model)
        pad_token_id=0,  # 填充标记的ID，默认为0
        eos_token_id=1,  # 结束标记的ID，默认为1
        forced_eos_token_id=1,  # 强制结束标记的ID，默认为1
        num_global_tokens=32,  # 全局标记的数量，默认为32
        block_size=512,  # 块大小，默认为512
        stagger_local_blocks=True,  # 是否交错本地块，默认为True
        **kwargs,  # 其他关键字参数，用于传递给父类的初始化函数
    ):
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置嵌入数
        self.d_model = d_model  # 设置模型的维度
        self.encoder_ffn_dim = encoder_ffn_dim  # 设置编码器中 Feed-Forward 层的维度
        self.encoder_layers = encoder_layers  # 设置编码器的层数
        self.encoder_attention_heads = encoder_attention_heads  # 设置编码器中注意力头的数量
        self.decoder_ffn_dim = decoder_ffn_dim  # 设置解码器中 Feed-Forward 层的维度
        self.decoder_layers = decoder_layers  # 设置解码器的层数
        self.decoder_attention_heads = decoder_attention_heads  # 设置解码器中注意力头的数量
        self.dropout = dropout  # 设置总体的Dropout概率
        self.attention_dropout = attention_dropout  # 设置注意力Dropout概率
        self.activation_dropout = activation_dropout  # 设置激活函数Dropout概率
        self.activation_function = activation_function  # 设置激活函数类型
        self.init_std = init_std  # 设置初始化参数的标准差
        self.encoder_layerdrop = encoder_layerdrop  # 设置编码器层间的随机删除率
        self.decoder_layerdrop = decoder_layerdrop  # 设置解码器层间的随机删除率
        self.use_cache = use_cache  # 设置是否使用缓存
        self.num_hidden_layers = encoder_layers  # 设置隐藏层的数量（与编码器层数相同）
        self.scale_embedding = scale_embedding  # 设置是否缩放嵌入

        self.num_global_tokens = num_global_tokens  # 设置全局标记的数量
        self.block_size = block_size  # 设置块大小
        self.stagger_local_blocks = stagger_local_blocks  # 设置是否交错本地块

        super().__init__(  # 调用父类的初始化函数
            pad_token_id=pad_token_id,  # 传递填充标记的ID
            eos_token_id=eos_token_id,  # 传递结束标记的ID
            is_encoder_decoder=is_encoder_decoder,  # 传递是否是编码解码结构
            decoder_start_token_id=decoder_start_token_id,  # 传递解码器起始标记ID
            forced_eos_token_id=forced_eos_token_id,  # 传递强制结束标记的ID
            **kwargs,  # 传递其他关键字参数
        )

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads  # 返回编码器中注意力头的数量

    @property
    def hidden_size(self) -> int:
        return self.d_model  # 返回模型的维度
```