# `.\models\vit_msn\configuration_vit_msn.py`

```py
# coding=utf-8
# 指定文件编码格式为UTF-8

# Copyright 2022 Facebook AI and The HuggingFace Inc. team. All rights reserved.
# 版权声明，版权归Facebook AI和HuggingFace Inc.团队所有

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据Apache License 2.0许可协议授权使用该文件

# you may not use this file except in compliance with the License.
# 除非遵守许可协议，否则不能使用该文件

# You may obtain a copy of the License at
# 您可以在此处获取许可协议的副本

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0 的链接

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有任何形式的明示或暗示的担保或条件，包括但不限于

# See the License for the specific language governing permissions and
# limitations under the License.
# 许可协议详细说明了授权的特定语言和限制条件

""" ViT MSN model configuration"""
# ViT MSN模型的配置信息

# Import necessary libraries
# 导入必要的库
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# Get logger instance
# 获取日志记录器实例
logger = logging.get_logger(__name__)

# Dictionary mapping model names to their respective config.json file URLs
# 字典，将模型名称映射到其相应的config.json文件的URL
VIT_MSN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "sayakpaul/vit-msn-base": "https://huggingface.co/sayakpaul/vit-msn-base/resolve/main/config.json",
    # See all ViT MSN models at https://huggingface.co/models?filter=vit_msn
}

# Configuration class for ViT MSN model inheriting PretrainedConfig
# ViT MSN模型的配置类，继承自PretrainedConfig
class ViTMSNConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ViTMSNModel`]. It is used to instantiate an ViT
    MSN model according to the specified arguments, defining the model architecture. Instantiating a configuration with
    the defaults will yield a similar configuration to that of the ViT
    [facebook/vit_msn_base](https://huggingface.co/facebook/vit_msn_base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 这是用于存储ViTMSNModel配置的配置类。根据指定的参数实例化ViT MSN模型，定义模型架构。
    # 使用默认参数实例化配置将产生与ViT facebook/vit_msn_base架构类似的配置。

    # Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    # 配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。阅读[`PretrainedConfig`]的文档获取更多信息。
    # 设定模型类型为 "vit_msn"
    model_type = "vit_msn"

    # 定义初始化方法，接受多个可选参数
    def __init__(
        self,
        hidden_size=768,  # 编码器层和池化层的维度大小，默认为768
        num_hidden_layers=12,  # Transformer 编码器中隐藏层的数量，默认为12
        num_attention_heads=12,  # Transformer 编码器中每个注意力层的注意头数量，默认为12
        intermediate_size=3072,  # Transformer 编码器中"中间"（即前馈）层的维度，默认为3072
        hidden_act="gelu",  # 编码器和池化器中的非线性激活函数，默认为"gelu"
        hidden_dropout_prob=0.0,  # 嵌入层、编码器和池化器中所有全连接层的dropout概率，默认为0.0
        attention_probs_dropout_prob=0.0,  # 注意力概率的dropout比率，默认为0.0
        initializer_range=0.02,  # 用于初始化所有权重矩阵的截断正态分布的标准差，默认为0.02
        layer_norm_eps=1e-06,  # 层归一化层使用的 epsilon，默认为1e-06
        image_size=224,  # 每个图像的大小（分辨率），默认为224
        patch_size=16,  # 每个补丁的大小（分辨率），默认为16
        num_channels=3,  # 输入通道的数量，默认为3
        qkv_bias=True,  # 是否向查询、键和值中添加偏置，默认为True
        **kwargs,  # 其他可选参数
    ):
        ):
        # 调用父类的初始化方法，传递所有关键字参数
        super().__init__(**kwargs)

        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置中间层大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置隐藏层的dropout概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力概率的dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置层归一化的epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 设置图像大小
        self.image_size = image_size
        # 设置patch（补丁）的大小
        self.patch_size = patch_size
        # 设置通道数
        self.num_channels = num_channels
        # 设置qkv偏置
        self.qkv_bias = qkv_bias
```