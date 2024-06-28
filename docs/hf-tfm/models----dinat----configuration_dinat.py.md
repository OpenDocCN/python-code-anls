# `.\models\dinat\configuration_dinat.py`

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
Dilated Neighborhood Attention Transformer model configuration
"""

# 导入预训练配置类和日志工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging
# 导入Backbone配置混合类和获取对齐输出特征输出索引的实用函数
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 获取模块级别的日志记录器
logger = logging.get_logger(__name__)

# DINAT预训练配置文件映射，指定不同预训练模型对应的配置文件URL
DINAT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "shi-labs/dinat-mini-in1k-224": "https://huggingface.co/shi-labs/dinat-mini-in1k-224/resolve/main/config.json",
    # 查看所有Dinat模型：https://huggingface.co/models?filter=dinat
}

# DinatConfig类继承自BackboneConfigMixin和PretrainedConfig类
class DinatConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DinatModel`]. It is used to instantiate a Dinat
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Dinat
    [shi-labs/dinat-mini-in1k-224](https://huggingface.co/shi-labs/dinat-mini-in1k-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import DinatConfig, DinatModel

    >>> # Initializing a Dinat shi-labs/dinat-mini-in1k-224 style configuration
    >>> configuration = DinatConfig()

    >>> # Initializing a model (with random weights) from the shi-labs/dinat-mini-in1k-224 style configuration
    >>> model = DinatModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型为'dinat'
    model_type = "dinat"

    # 属性映射字典，将num_attention_heads映射到num_heads，将num_hidden_layers映射到num_layers
    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }
        # 调用父类的初始化方法，继承父类的属性和方法
        super().__init__(**kwargs)

        # 设定模型的 patch 大小（用于图像分割中的每个小块的尺寸）
        self.patch_size = patch_size
        # 输入图像的通道数
        self.num_channels = num_channels
        # 嵌入维度，即每个位置的特征向量的维度
        self.embed_dim = embed_dim
        # 每个阶段的深度列表，指定每个阶段有多少个注意力层
        self.depths = depths
        # 阶段（层）的数量
        self.num_layers = len(depths)
        # 每个注意力头的数量列表，每个阶段可以有多个注意力头
        self.num_heads = num_heads
        # 卷积核大小，用于卷积操作的核的尺寸
        self.kernel_size = kernel_size
        # 不同阶段的空洞卷积的扩张率列表
        self.dilations = dilations
        # MLP 部分的宽度倍率
        self.mlp_ratio = mlp_ratio
        # 是否在注意力计算中使用偏置项
        self.qkv_bias = qkv_bias
        # 隐藏层的 dropout 概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 注意力概率的 dropout 概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 随机删除路径的比率，用于随机丢弃路径
        self.drop_path_rate = drop_path_rate
        # 隐藏层激活函数的类型
        self.hidden_act = hidden_act
        # 层归一化的 epsilon 值
        self.layer_norm_eps = layer_norm_eps
        # 初始化权重的范围
        self.initializer_range = initializer_range

        # 隐藏层大小，即在模型的最后阶段的通道维度
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))

        # 层缩放的初始值，用于缩放每个阶段的输出
        self.layer_scale_init_value = layer_scale_init_value

        # 阶段的名称列表，包括 'stem'（干部阶段）和每个阶段的编号
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]

        # 获得对齐的输出特征和输出索引，用于确保与给定阶段名称对齐的输出特征和索引
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
```