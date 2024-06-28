# `.\models\vilt\configuration_vilt.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" VilT model configuration"""

# 导入所需模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取logger对象用于记录日志
logger = logging.get_logger(__name__)

# 预训练模型配置文件的映射字典，指定模型名称及其对应的配置文件URL
VILT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "dandelin/vilt-b32-mlm": "https://huggingface.co/dandelin/vilt-b32-mlm/blob/main/config.json"
}


class ViltConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ViLTModel`]. It is used to instantiate an ViLT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ViLT
    [dandelin/vilt-b32-mlm](https://huggingface.co/dandelin/vilt-b32-mlm) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import ViLTModel, ViLTConfig

    >>> # Initializing a ViLT dandelin/vilt-b32-mlm style configuration
    >>> configuration = ViLTConfig()

    >>> # Initializing a model from the dandelin/vilt-b32-mlm style configuration
    >>> model = ViLTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    model_type = "vilt"

    def __init__(
        self,
        vocab_size=30522,
        type_vocab_size=2,
        modality_type_vocab_size=2,
        max_position_embeddings=40,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=384,
        patch_size=32,
        num_channels=3,
        qkv_bias=True,
        max_image_length=-1,
        tie_word_embeddings=False,
        num_images=-1,
        **kwargs,
        ):
        # 调用父类的构造函数，初始化模型参数和超参数
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        # 设置模型的词汇表大小
        self.vocab_size = vocab_size
        # 设置模型的类型词汇表大小
        self.type_vocab_size = type_vocab_size
        # 设置模型的模态类型词汇表大小
        self.modality_type_vocab_size = modality_type_vocab_size
        # 设置模型的最大位置嵌入长度
        self.max_position_embeddings = max_position_embeddings

        # 设置模型的隐藏层大小
        self.hidden_size = hidden_size
        # 设置模型的隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置模型的注意力头数量
        self.num_attention_heads = num_attention_heads
        # 设置模型的中间层大小
        self.intermediate_size = intermediate_size
        # 设置模型的隐藏层激活函数类型
        self.hidden_act = hidden_act
        # 设置模型的隐藏层的丢弃率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置模型的注意力机制的概率丢弃率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置模型的初始化范围
        self.initializer_range = initializer_range
        # 设置模型的层归一化 epsilon 参数
        self.layer_norm_eps = layer_norm_eps

        # 设置模型的图像输入大小
        self.image_size = image_size
        # 设置模型的图像块的大小
        self.patch_size = patch_size
        # 设置模型的图像通道数量
        self.num_channels = num_channels
        # 设置模型的注意力中的查询、键、值是否包含偏置
        self.qkv_bias = qkv_bias
        # 设置模型的最大图像长度
        self.max_image_length = max_image_length
        # 设置模型处理的图像数量
        self.num_images = num_images
```