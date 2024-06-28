# `.\models\mgp_str\configuration_mgp_str.py`

```
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" MGP-STR model configuration"""

# Importing necessary modules from the Transformers library
from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...utils import logging  # 导入日志记录工具

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射字典，指定了模型名称到其配置文件的映射关系
MGP_STR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "alibaba-damo/mgp-str-base": "https://huggingface.co/alibaba-damo/mgp-str-base/resolve/main/config.json",
}

# MgpstrConfig 类，继承自 PretrainedConfig，用于存储 MGP-STR 模型的配置信息
class MgpstrConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`MgpstrModel`]. It is used to instantiate an
    MGP-STR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MGP-STR
    [alibaba-damo/mgp-str-base](https://huggingface.co/alibaba-damo/mgp-str-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 定义默认的图像大小为 [32, 128]
    Args:
        image_size (`List[int]`, *optional*, defaults to `[32, 128]`):
            The size (resolution) of each image.
        # 定义每个补丁的大小，默认为 4
        patch_size (`int`, *optional*, defaults to 4):
            The size (resolution) of each patch.
        # 定义输入通道数，默认为 3
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        # 定义输出令牌的最大数量，默认为 27
        max_token_length (`int`, *optional*, defaults to 27):
            The max number of output tokens.
        # 定义字符头的类别数量，默认为 38
        num_character_labels (`int`, *optional*, defaults to 38):
            The number of classes for character head .
        # 定义bpe头的类别数量，默认为 50257
        num_bpe_labels (`int`, *optional*, defaults to 50257):
            The number of classes for bpe head .
        # 定义wordpiece头的类别数量，默认为 30522
        num_wordpiece_labels (`int`, *optional*, defaults to 30522):
            The number of classes for wordpiece head .
        # 定义嵌入维度，默认为 768
        hidden_size (`int`, *optional*, defaults to 768):
            The embedding dimension.
        # 定义Transformer编码器中的隐藏层数量，默认为 12
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        # 定义Transformer编码器中每个注意力层的注意头数量，默认为 12
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        # 定义mlp隐藏维度与嵌入维度的比率，默认为 4.0
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of mlp hidden dim to embedding dim.
        # 定义是否向查询、键和值添加偏置，默认为 True
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        # 定义模型是否包含蒸馏令牌和头部，如DeiT模型，默认为 False
        distilled (`bool`, *optional*, defaults to `False`):
            Model includes a distillation token and head as in DeiT models.
        # 定义层归一化层使用的 epsilon，默认为 1e-05
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        # 定义所有全连接层的丢弃概率，包括嵌入和编码器，默认为 0.0
        drop_rate (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder.
        # 定义注意力概率的丢弃比率，默认为 0.0
        attn_drop_rate (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        # 定义随机深度的丢弃率，默认为 0.0
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The stochastic depth rate.
        # 定义是否返回A^3模块注意力的布尔值，默认为 False
        output_a3_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not the model should returns A^3 module attentions.
        # 定义所有权重矩阵初始化时的截断正态分布的标准差，默认为 0.02
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import MgpstrConfig, MgpstrForSceneTextRecognition

    >>> # Initializing a Mgpstr mgp-str-base style configuration
    >>> configuration = MgpstrConfig()

    >>> # Initializing a model (with random weights) from the mgp-str-base style configuration
    >>> model = MgpstrForSceneTextRecognition(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    # 设置模型类型为 "mgp-str"
    model_type = "mgp-str"
    # 定义一个初始化函数，初始化一个模型对象
    def __init__(
        self,
        image_size=[32, 128],  # 图像大小，默认为[32, 128]
        patch_size=4,          # 补丁大小，默认为4
        num_channels=3,        # 图像通道数，默认为3
        max_token_length=27,   # 最大标记长度，默认为27
        num_character_labels=38,  # 字符标签数，默认为38
        num_bpe_labels=50257,      # BPE标签数，默认为50257
        num_wordpiece_labels=30522,  # WordPiece标签数，默认为30522
        hidden_size=768,        # 隐藏层大小，默认为768
        num_hidden_layers=12,   # 隐藏层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        mlp_ratio=4.0,          # MLP（多层感知机）比例，默认为4.0
        qkv_bias=True,          # 是否在QKV转换中使用偏置，默认为True
        distilled=False,        # 是否为蒸馏模型，默认为False
        layer_norm_eps=1e-5,    # 层归一化的epsilon值，默认为1e-5
        drop_rate=0.0,          # dropout比率，默认为0.0
        attn_drop_rate=0.0,     # 注意力dropout比率，默认为0.0
        drop_path_rate=0.0,     # 路径dropout比率，默认为0.0
        output_a3_attentions=False,  # 是否输出A3注意力，默认为False
        initializer_range=0.02,  # 初始化范围，默认为0.02
        **kwargs,               # 其他关键字参数
    ):
        super().__init__(**kwargs)  # 调用父类的初始化方法

        self.image_size = image_size  # 初始化图像大小属性
        self.patch_size = patch_size  # 初始化补丁大小属性
        self.num_channels = num_channels  # 初始化图像通道数属性
        self.max_token_length = max_token_length  # 初始化最大标记长度属性
        self.num_character_labels = num_character_labels  # 初始化字符标签数属性
        self.num_bpe_labels = num_bpe_labels  # 初始化BPE标签数属性
        self.num_wordpiece_labels = num_wordpiece_labels  # 初始化WordPiece标签数属性
        self.hidden_size = hidden_size  # 初始化隐藏层大小属性
        self.num_hidden_layers = num_hidden_layers  # 初始化隐藏层数属性
        self.num_attention_heads = num_attention_heads  # 初始化注意力头数属性
        self.mlp_ratio = mlp_ratio  # 初始化MLP比例属性
        self.distilled = distilled  # 初始化蒸馏模型属性
        self.layer_norm_eps = layer_norm_eps  # 初始化层归一化epsilon属性
        self.drop_rate = drop_rate  # 初始化dropout比率属性
        self.qkv_bias = qkv_bias  # 初始化QKV偏置属性
        self.attn_drop_rate = attn_drop_rate  # 初始化注意力dropout比率属性
        self.drop_path_rate = drop_path_rate  # 初始化路径dropout比率属性
        self.output_a3_attentions = output_a3_attentions  # 初始化是否输出A3注意力属性
        self.initializer_range = initializer_range  # 初始化初始化范围属性
```