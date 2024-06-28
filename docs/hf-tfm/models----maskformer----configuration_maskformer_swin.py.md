# `.\models\maskformer\configuration_maskformer_swin.py`

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
"""
MaskFormer Swin Transformer model configuration
"""

# 导入必要的配置类和工具函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 MaskFormerSwinConfig 类，继承自 BackboneConfigMixin 和 PretrainedConfig
class MaskFormerSwinConfig(BackboneConfigMixin, PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MaskFormerSwinModel`]. It is used to instantiate
    a Donut model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Swin
    [microsoft/swin-tiny-patch4-window7-224](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import MaskFormerSwinConfig, MaskFormerSwinModel

    >>> # Initializing a microsoft/swin-tiny-patch4-window7-224 style configuration
    >>> configuration = MaskFormerSwinConfig()

    >>> # Initializing a model (with random weights) from the microsoft/swin-tiny-patch4-window7-224 style configuration
    >>> model = MaskFormerSwinModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型标识
    model_type = "maskformer-swin"

    # 属性映射字典，将配置中的参数映射到实际使用的参数名称
    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    # 初始化方法，定义了模型的配置参数
    def __init__(
        self,
        image_size=224,                         # 图像大小
        patch_size=4,                           # 补丁大小
        num_channels=3,                         # 输入通道数
        embed_dim=96,                           # 嵌入维度
        depths=[2, 2, 6, 2],                    # 每个阶段的深度
        num_heads=[3, 6, 12, 24],               # 每个阶段的注意力头数
        window_size=7,                          # 窗口大小
        mlp_ratio=4.0,                          # MLP 的尺度比率
        qkv_bias=True,                          # 是否在 QKV 中使用偏置
        hidden_dropout_prob=0.0,                # 隐藏层的dropout概率
        attention_probs_dropout_prob=0.0,       # 注意力层的dropout概率
        drop_path_rate=0.1,                     # DropPath 的概率
        hidden_act="gelu",                      # 隐藏层激活函数
        use_absolute_embeddings=False,          # 是否使用绝对位置嵌入
        initializer_range=0.02,                 # 初始化范围
        layer_norm_eps=1e-5,                    # LayerNorm 的 epsilon 值
        out_features=None,                      # 输出特征
        out_indices=None,                       # 输出索引
        **kwargs,                               # 其他参数
    ):
        super().__init__(**kwargs)
        # 初始化方法体，设置模型的各种参数配置
        # （具体初始化方法体内的内容未提供，但注释已经涵盖了参数的功能和用途）
        ):
            # 调用父类的初始化方法，传入所有关键字参数
            super().__init__(**kwargs)

            # 设置图像大小属性
            self.image_size = image_size
            # 设置补丁大小属性
            self.patch_size = patch_size
            # 设置通道数属性
            self.num_channels = num_channels
            # 设置嵌入维度属性
            self.embed_dim = embed_dim
            # 设置每个阶段的深度列表
            self.depths = depths
            # 计算阶段数目
            self.num_layers = len(depths)
            # 设置注意力头数目
            self.num_heads = num_heads
            # 设置窗口大小属性
            self.window_size = window_size
            # 设置MLP扩展比例属性
            self.mlp_ratio = mlp_ratio
            # 设置注意力机制中的query/key/value是否带偏置
            self.qkv_bias = qkv_bias
            # 设置隐藏层dropout概率属性
            self.hidden_dropout_prob = hidden_dropout_prob
            # 设置注意力概率dropout概率属性
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            # 设置dropout路径丢弃率属性
            self.drop_path_rate = drop_path_rate
            # 设置隐藏层激活函数属性
            self.hidden_act = hidden_act
            # 设置是否使用绝对位置嵌入属性
            self.use_absolute_embeddings = use_absolute_embeddings
            # 设置层归一化epsilon值属性
            self.layer_norm_eps = layer_norm_eps
            # 设置初始化范围属性
            self.initializer_range = initializer_range
            # 设置隐藏大小属性，以便使Swin与VisionEncoderDecoderModel兼容
            # 这表示模型最后阶段后的通道维度
            self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
            # 设置阶段名称列表，包括"stem"和"stage1"到"stageN"
            self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]
            # 获取对齐的输出特征和输出索引，确保与阶段名称对齐
            self._out_features, self._out_indices = get_aligned_output_features_output_indices(
                out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
            )
```