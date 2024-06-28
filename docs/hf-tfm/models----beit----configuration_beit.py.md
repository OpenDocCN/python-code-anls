# `.\models\beit\configuration_beit.py`

```py
# coding=utf-8
# Copyright Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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

""" BEiT model configuration"""

from collections import OrderedDict  # 导入OrderedDict类，用于创建有序字典
from typing import Mapping  # 导入Mapping类型，用于类型提示

from packaging import version  # 导入version模块，用于版本处理

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...onnx import OnnxConfig  # 导入Onnx配置类
from ...utils import logging  # 导入日志工具模块
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices  # 导入背骨网络工具和特征对齐索引获取函数

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/beit-base-patch16-224-pt22k": (
        "https://huggingface.co/microsoft/beit-base-patch16-224-pt22k/resolve/main/config.json"
    ),
    # See all BEiT models at https://huggingface.co/models?filter=beit
}

class BeitConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BeitModel`]. It is used to instantiate an BEiT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the BEiT
    [microsoft/beit-base-patch16-224-pt22k](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k) architecture.

    Example:

    ```
    >>> from transformers import BeitConfig, BeitModel

    >>> # Initializing a BEiT beit-base-patch16-224-pt22k style configuration
    >>> configuration = BeitConfig()

    >>> # Initializing a model (with random weights) from the beit-base-patch16-224-pt22k style configuration
    >>> model = BeitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "beit"  # 设置模型类型为 "beit"
    # 初始化函数，用于创建一个新的模型实例
    def __init__(
        self,
        vocab_size=8192,  # 词汇表大小，默认为8192
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层的数量，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
        hidden_dropout_prob=0.0,  # 隐藏层的dropout概率，默认为0.0
        attention_probs_dropout_prob=0.0,  # 注意力概率的dropout概率，默认为0.0
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 层归一化的epsilon值，默认为1e-12
        image_size=224,  # 图像大小，默认为224
        patch_size=16,  # 补丁大小，默认为16
        num_channels=3,  # 图像通道数，默认为3
        use_mask_token=False,  # 是否使用mask token，默认为False
        use_absolute_position_embeddings=False,  # 是否使用绝对位置嵌入，默认为False
        use_relative_position_bias=False,  # 是否使用相对位置偏置，默认为False
        use_shared_relative_position_bias=False,  # 是否共享相对位置偏置，默认为False
        layer_scale_init_value=0.1,  # 层缩放初始化值，默认为0.1
        drop_path_rate=0.1,  # drop path的概率，默认为0.1
        use_mean_pooling=True,  # 是否使用均值池化，默认为True
        pool_scales=[1, 2, 3, 6],  # 池化尺度列表，默认为[1, 2, 3, 6]
        use_auxiliary_head=True,  # 是否使用辅助头，默认为True
        auxiliary_loss_weight=0.4,  # 辅助损失权重，默认为0.4
        auxiliary_channels=256,  # 辅助头的通道数，默认为256
        auxiliary_num_convs=1,  # 辅助头的卷积层数，默认为1
        auxiliary_concat_input=False,  # 辅助头是否将输入进行拼接，默认为False
        semantic_loss_ignore_index=255,  # 语义损失忽略的索引，默认为255
        out_features=None,  # 输出特征，默认为None
        out_indices=None,  # 输出索引，默认为None
        add_fpn=False,  # 是否添加特征金字塔网络，默认为False
        reshape_hidden_states=True,  # 是否重塑隐藏状态，默认为True
        **kwargs,  # 其他关键字参数
        ):
            super().__init__(**kwargs)
    
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.intermediate_size = intermediate_size
            self.hidden_act = hidden_act
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
    
            self.image_size = image_size
            self.patch_size = patch_size
            self.num_channels = num_channels
            self.use_mask_token = use_mask_token
            self.use_absolute_position_embeddings = use_absolute_position_embeddings
            self.use_relative_position_bias = use_relative_position_bias
            self.use_shared_relative_position_bias = use_shared_relative_position_bias
            self.layer_scale_init_value = layer_scale_init_value
            self.drop_path_rate = drop_path_rate
            self.use_mean_pooling = use_mean_pooling
            # decode head attributes (semantic segmentation)
            self.pool_scales = pool_scales
            # auxiliary head attributes (semantic segmentation)
            self.use_auxiliary_head = use_auxiliary_head
            self.auxiliary_loss_weight = auxiliary_loss_weight
            self.auxiliary_channels = auxiliary_channels
            self.auxiliary_num_convs = auxiliary_num_convs
            self.auxiliary_concat_input = auxiliary_concat_input
            self.semantic_loss_ignore_index = semantic_loss_ignore_index
    
            # handle backwards compatibility
            如果传入参数中包含"segmentation_indices"，发出警告，建议使用"out_indices"代替
            if "segmentation_indices" in kwargs:
                logger.warning(
                    "The `segmentation_indices` argument is deprecated and will be removed in a future version, use `out_indices` instead.",
                    FutureWarning,
                )
                将"segmentation_indices"参数从kwargs中移除
                out_indices = kwargs.pop("segmentation_indices")
    
            # backbone attributes
            构建阶段名称列表，从"stem"开始，然后是每个隐藏层的阶段名
            self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
            根据输出特征和输出索引，以及阶段名称，获取对齐的输出特征和输出索引
            self._out_features, self._out_indices = get_aligned_output_features_output_indices(
                out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
            )
            是否添加特征金字塔网络（FPN）
            self.add_fpn = add_fpn
            是否重新整形隐藏状态
            self.reshape_hidden_states = reshape_hidden_states
# 从transformers.models.vit.configuration_vit.ViTOnnxConfig复制而来的类定义，继承自OnnxConfig类
class BeitOnnxConfig(OnnxConfig):
    # 设定torch_onnx_minimum_version属性为1.11版本
    torch_onnx_minimum_version = version.parse("1.11")

    # inputs属性的getter方法，返回一个有序字典，描述了输入数据的索引映射关系
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                # 指定输入名称为"pixel_values"，并定义其维度索引映射关系
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # atol_for_validation属性的getter方法，返回一个浮点数，指定验证时的绝对误差限制
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```